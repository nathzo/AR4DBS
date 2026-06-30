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

    /// Reference-image name → MarkerID. Matches the WP8 "LeksellMarkers" group.
    private static let markerNameMap: [String: CoordinateConventions.MarkerID] = [
        "LeksellMarkerLeft": .left,
        "LeksellMarkerRight": .right
    ]

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
        config.detectionImages = ARReferenceImage.referenceImages(
            inGroupNamed: "LeksellMarkers", bundle: .main)
        config.maximumNumberOfTrackedImages = 2
        config.automaticImageScaleEstimationEnabled = true
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
    fileprivate func handleFrame(camera: ARCamera, anchors: [ARAnchor]) {
        // 1. Tracking quality from the camera state.
        let quality = Self.quality(from: camera.trackingState)
        trackingQuality = quality

        // 2. Currently-tracked image anchors → [MarkerID: world_T_marker].
        var markers: [CoordinateConventions.MarkerID: simd_float4x4] = [:]
        for case let image as ARImageAnchor in anchors where image.isTracked {
            // Skip anchors whose estimated scale is far from 1 (mis-detection / wrong
            // physical size) — guards against a bad world_T_marker poisoning fusion.
            if abs(image.estimatedScaleFactor - 1.0) > 0.1 { continue }
            guard let id = Self.markerNameMap[image.referenceImage.name ?? ""] else { continue }
            markers[id] = image.transform
        }

        // 3 & 4: branch on whether we are already locked.
        if case .locked = registration {
            // Locked: keep the frozen transform; world tracking maintains the overlay.
            // Drift guard (ported onTrackingQualityChanged, adapted to prefer
            // trackingState over feature counts): if quality drops below the level
            // recorded at lock time, drop the lock and re-calibrate.
            if quality < anchorQuality {
                resetRegistration()
            }
            return
        }

        // Not locked: try to fuse and advance/break the streak.
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
            owner?.handleFrame(camera: frame.camera, anchors: frame.anchors)
        }
    }
}
