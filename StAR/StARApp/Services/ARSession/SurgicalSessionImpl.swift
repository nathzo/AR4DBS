// SurgicalSessionImpl — the live ARKit session + lock/recalibrate state machine.
//
// Ported from the legacy C++/Qt reference implementation:
//   • app/AppController.cpp        — onARFrame (per-frame streak machine),
//                                     resetARRegistration, onTrackingQualityChanged
//                                     (drift guard), the 10-frame SO(3) averaging.
//   • platform/ios/ARKitSession.mm — ARSession/ARWorldTrackingConfiguration setup
//                                     and the ARCamera trackingState → quality map.
//
// Differences from the legacy port (per WORKPLAN WP2):
//   • No ARKit→OpenCV Y/Z flip. We work in ARKit world space directly because
//     MarkerFusion consumes world_T_marker straight from ARImageAnchor.transform.
//   • Marker detection is ARKit image tracking, not OpenCV ArUco; meetsInitConditions
//     (≥8 corners / face-on / reproj) collapses into anchor.isTracked +
//     estimatedScaleFactor≈1 + MarkerFusion's geometric agreement gate.
//   • The drift guard prefers ARCamera.trackingState over rawFeaturePoints counts
//     (feature counts are meaningless with LiDAR / image anchors).
//   • The legacy QAtomicInt busy-guard is unnecessary: per-frame work runs on the
//     main actor and is synchronous, so fusion is naturally serialized.

import ARKit
import Foundation
import Observation
import simd

@MainActor
@Observable
public final class SurgicalSessionImpl: SurgicalSession {

    // MARK: - Observable contract state

    public var registration: RegistrationState = .awaitingMarkers
    public var trackingQuality: TrackingQuality = .unavailable

    // MARK: - Collaborators / configuration

    private let fusion = MarkerFusion()
    /// B2: native ArUco detection + solvePnP on the camera frame (replaces ARKit
    /// image anchors). Produces [MarkerID: world_T_marker] from each ARFrame.
    private let arucoTracker = ArucoMarkerTracker()
    /// Throttle for the (heavy) per-frame CV during calibration; detection is skipped
    /// entirely once locked (world tracking carries the overlay).
    private static let detectEveryNFrames = 2
    private var frameCounter = 0
    private var parameters: RegistrationParameters
    private var planGeometry: PlanGeometry?

    /// The ARView's ARSession (owned by WP7). Weak: WP7 owns the lifecycle.
    private weak var session: ARSession?
    private let coordinator = SessionCoordinator()

    // MARK: - Streak / drift state
    //
    // The locked transform itself lives inside `registration == .locked(_)` — that
    // enum is the single source of truth for the frozen world_T_leksell.

    /// Accumulating world_T_leksell candidates during calibration (legacy m_streakPoses).
    private var streak: [simd_float4x4] = []
    /// Tracking quality recorded at lock time (legacy m_anchorTrackingState).
    /// The drift guard only re-calibrates when current quality drops below this.
    private var anchorQuality: TrackingQuality = .normal


    // MARK: - Init

    public init(parameters: RegistrationParameters = .init()) {
        self.parameters = parameters
        coordinator.owner = self
    }

    // MARK: - Concrete (beyond-protocol) API for WP7

    /// Inject the ARView's single ARSession. There is exactly ONE ARSession in the
    /// app (the ARView's); WP2 never creates a second one for the live path.
    public func attach(session: ARSession) {
        self.session = session
        session.delegateQueue = .main
        session.delegate = coordinator
    }

    /// Live-update tunables from Settings (WP6) without tearing down the session.
    public func updateParameters(_ parameters: RegistrationParameters) {
        self.parameters = parameters
    }

    // MARK: - SurgicalSession

    public func start() {
        guard let session else { return }
        guard let config = Self.makeConfiguration() else {
            // Unsupported (e.g. simulator): no-op so the app still runs.
            trackingQuality = .unavailable
            registration = .awaitingMarkers
            return
        }
        session.run(config, options: [.resetTracking, .removeExistingAnchors])
    }

    public func stop() {
        session?.pause()
    }

    /// The "Recalibrer" action: drop the lock and re-enter calibration.
    /// (Ported from AppController::resetARRegistration.)
    public func resetRegistration() {
        streak.removeAll(keepingCapacity: true)
        anchorQuality = .normal
        registration = .awaitingMarkers
        // Clear stale anchors so a fresh streak starts cleanly.
        if let session, let config = session.configuration {
            session.run(config, options: [.removeExistingAnchors])
        }
    }

    /// Retain the active plan geometry. WP3 renders it; WP2 just holds it.
    public func setPlan(_ geometry: PlanGeometry) {
        planGeometry = geometry
    }

    // MARK: - Configuration (ported from ARKitSession.mm)

    private static func makeConfiguration() -> ARWorldTrackingConfiguration? {
        guard ARWorldTrackingConfiguration.isSupported else { return nil }
        let config = ARWorldTrackingConfiguration()
        // B2 detects ArUco tags on the camera frame (ArucoMarkerTracker), so no ARKit
        // detectionImages are configured. World tracking + LiDAR mesh + scene depth
        // are still needed (world anchor, occlusion, incision raycast).
        if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
            config.sceneReconstruction = .mesh
        }
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics.insert(.sceneDepth)
        }
        config.planeDetection = []
        return config
    }

    // MARK: - Per-frame state machine (ported from AppController::onARFrame)

    /// Called from the delegate on the main thread for every ARFrame.
    fileprivate func handleFrame(_ frame: ARFrame) {
        // 1. Tracking quality from the camera state.
        let quality = Self.quality(from: frame.camera.trackingState)
        trackingQuality = quality

        // 2. Already locked: world tracking maintains the overlay — skip the (heavy)
        // per-frame detection entirely. Drift guard (ported onTrackingQualityChanged,
        // adapted to prefer trackingState): drop the lock if quality falls below the
        // level recorded at lock time.
        if case .locked = registration {
            if quality < anchorQuality {
                resetRegistration()
            }
            return
        }

        // 3. Calibrating: detect the ArUco markers (B2), throttled so the main actor
        // doesn't hitch. (Offloading detection to a background queue is a future perf
        // step; during calibration the surgeon holds steady and detection stops once
        // locked, so synchronous main-actor CV is acceptable for now.)
        frameCounter &+= 1
        guard frameCounter % Self.detectEveryNFrames == 0 else { return }
        let markers = arucoTracker.markers(in: frame)

        // 4. Fuse and advance/break the streak.
        guard let result = fusion.fuse(anchors: markers, parameters: parameters),
              result.qualifies else {
            // Streak broken — reset accumulator (legacy: m_streakPoses.clear()).
            if !streak.isEmpty {
                streak.removeAll(keepingCapacity: true)
            }
            registration = .awaitingMarkers
            return
        }

        // Qualifying frame: accumulate.
        streak.append(result.worldToLeksell)

        if streak.count >= parameters.lockStreakFrames {
            // SO(3)+R³ average the streak → frozen world_T_leksell.
            let fused = MarkerFusion.average(streak)
            anchorQuality = quality
            streak.removeAll(keepingCapacity: true)
            registration = .locked(worldToLeksell: fused)
        } else {
            registration = .calibrating(progress: streak.count,
                                        total: parameters.lockStreakFrames)
        }
    }

    /// ARCamera.TrackingState → TrackingQuality (legacy ARKitSession.mm mapping:
    /// 0 = unavailable, 1 = limited, 2 = normal).
    private static func quality(from state: ARCamera.TrackingState) -> TrackingQuality {
        switch state {
        case .notAvailable: return .unavailable
        case .limited:      return .limited
        case .normal:       return .normal
        @unknown default:   return .unavailable
        }
    }
}

// MARK: - ARSessionDelegate bridge (Swift 6 concurrency)
//
// ARSessionDelegate methods are nonisolated. Because we set delegateQueue = .main
// in attach(session:), the callbacks run on the main thread, so we can safely
// MainActor.assumeIsolated and forward to the @MainActor owner. The coordinator is
// @unchecked Sendable because its only stored state is the weak owner reference,
// which is only touched on the main thread.
private final class SessionCoordinator: NSObject, ARSessionDelegate, @unchecked Sendable {

    weak var owner: SurgicalSessionImpl?

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        MainActor.assumeIsolated {
            owner?.handleFrame(frame)
        }
    }
}
