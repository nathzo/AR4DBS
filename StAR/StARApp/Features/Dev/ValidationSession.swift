//  ValidationSession.swift
//  DEV-ONLY registration-validation harness — live ARKit driver.
//
//  Runs ITS OWN ARSession (independent of the production SurgicalSessionImpl) so a
//  surgeon/engineer can, with the two physical Leksell markers on a measured jig
//  and a LiDAR iPhone Pro, empirically resolve four open audit issues
//  (StAR/docs/v1-v2-equivalence-audit.md): S1-01 (marker rotation), REG-01
//  (face-on angle), REG-03 (range vs lateral error) and S3-02 (LiDAR confidence at
//  the incision hit). It publishes an @Observable per-frame snapshot consumed by
//  ValidationView and logs captured samples to a CSV for off-device analysis.
//
//  The ARSessionDelegate concurrency pattern is COPIED from
//  Services/ARSession/SurgicalSessionImpl.swift: a private final Coordinator that
//  is `@unchecked Sendable`, holds a `weak owner`, runs on `delegateQueue = .main`,
//  and forwards via `MainActor.assumeIsolated`. All decision math is delegated to
//  the pure `RegistrationDiagnostics` so this file stays thin.
//
//  Wrapped in #if DEBUG so it is excluded from release builds. Guarded by
//  `ARWorldTrackingConfiguration.isSupported` so it no-ops on the simulator.
//
//  Single module "StAR": Core/* / Services/* types are visible.

#if DEBUG
import ARKit
import RealityKit
import Foundation
import Observation
import simd

// MARK: - Snapshot value types

/// LiDAR confidence bucket (ARConfidenceLevel 0/1/2 → low/med/high).
public enum DevConfidence: String, Sendable {
    case low, medium, high, unavailable

    init(_ level: ARConfidenceLevel) {
        switch level {
        case .low:    self = .low
        case .medium: self = .medium
        case .high:   self = .high
        @unknown default: self = .unavailable
        }
    }
}

/// One tracked image anchor, decoded for the diagnostics overlay.
public struct DevAnchorInfo: Sendable, Identifiable {
    public let id: Int                      // MarkerID.rawValue (0 = left, 1 = right)
    public let name: String
    public let isTracked: Bool
    public let estimatedScaleFactor: Float
    public let axisAnglesDeg: SIMD3<Float>  // camera-forward vs anchor (x,y,z) axes
    public let faceOnAngleDeg: Float        // assuming +z is the plane normal (see note)
    public let worldTransform: simd_float4x4
}

/// A complete per-frame diagnostics snapshot. Plain value type so the UI just reads.
public struct ValidationSnapshot: Sendable {
    public var arSupported: Bool = false
    public var trackingStateLabel: String = "—"
    /// How many reference images loaded from the "LeksellMarkers" group (0 ⇒ the
    /// asset group is missing/misnamed; 2 ⇒ images loaded, so a no-detection issue
    /// is physical: size/lighting/print, not the bundle).
    public var referenceImageCount: Int = 0
    /// How many ARImageAnchors ARKit reports this frame, regardless of name (so you
    /// can tell "ARKit sees an image but the name doesn't map" from "sees nothing").
    public var rawImageAnchorCount: Int = 0
    public var anchors: [DevAnchorInfo] = []

    /// Fused world_T_leksell for the CURRENTLY SELECTED candidate rotation (nil if
    /// no markers tracked this frame).
    public var fusedWorldToLeksell: simd_float4x4?
    public var selectedRotationName: String = RegistrationDiagnostics.candidateRotations[0].name

    public var interMarkerOriginMM: Float = 0
    public var interMarkerOrientationDeg: Float = 0

    /// Incision raycast hit (world) for the default-test left trajectory + its
    /// LiDAR confidence at the projected pixel.
    public var incisionHitWorld: SIMD3<Float>?
    public var incisionConfidence: DevConfidence = .unavailable

    public var cameraPositionWorld: SIMD3<Float> = .zero
    public var cameraForwardWorld: SIMD3<Float> = SIMD3(0, 0, -1)
}

/// A captured sample (snapshot + operator note) for the CSV log.
public struct ValidationSample: Sendable {
    public let timestamp: Date
    public let note: String
    public let snapshot: ValidationSnapshot
}

// MARK: - Session

@MainActor
@Observable
public final class ValidationSession {

    // Published state the UI observes.
    public private(set) var snapshot = ValidationSnapshot()
    public private(set) var samples: [ValidationSample] = []

    /// Reference images loaded from the "LeksellMarkers" group at start (diagnostic).
    public private(set) var referenceImageCount: Int = 0

    /// Index into `RegistrationDiagnostics.candidateRotations`. The UI changes this
    /// live (the cycler) and the next frame republishes the fused frame + triad
    /// under the new rotation — THE tool to pin S1-01.
    /// Plain stored property — deliberately NO didSet. A self-assigning didSet on an
    /// @Observable class can re-enter the macro-generated setter and recurse into a
    /// stack overflow when the picker changes it; reads are clamped instead.
    public var selectedRotationIndex: Int = 0

    private var clampedRotationIndex: Int {
        max(0, min(candidates.count - 1, selectedRotationIndex))
    }
    public var selectedRotation: simd_quatf { candidates[clampedRotationIndex].quat }
    public var selectedRotationName: String { candidates[clampedRotationIndex].name }

    private let candidates = RegistrationDiagnostics.candidateRotations

    /// Reference-image name → MarkerID (matches the production "LeksellMarkers" group).
    private static let markerNameMap: [String: CoordinateConventions.MarkerID] = [
        "LeksellMarkerLeft": .left,
        "LeksellMarkerRight": .right
    ]

    private weak var arView: ARView?
    private weak var session: ARSession?
    private let coordinator = Coordinator()

    /// The trajectory whose incision hit + confidence we probe (default-test plan).
    private let probeTrajectory = Trajectory.fromLeksell(SurgicalPlanDTO.defaultTest.left)

    public init() {
        coordinator.owner = self
    }

    // MARK: Lifecycle

    /// Inject the dev ARView's single ARSession (called by ValidationView on appear).
    public func attach(to arView: ARView) {
        self.arView = arView
        let s = arView.session
        self.session = s
        s.delegateQueue = .main
        s.delegate = coordinator
        // LiDAR mesh occlusion so the incision raycast has a surface to hit.
        arView.environment.sceneUnderstanding.options.insert(.occlusion)
    }

    public func start() {
        snapshot.arSupported = ARWorldTrackingConfiguration.isSupported
        // Load the marker group explicitly so the UI can report how many images
        // actually loaded — the key "no detection" diagnostic.
        let refs = ARReferenceImage.referenceImages(inGroupNamed: "LeksellMarkers", bundle: .main)
        referenceImageCount = refs?.count ?? 0
        snapshot.referenceImageCount = referenceImageCount

        guard let session, ARWorldTrackingConfiguration.isSupported else {
            return  // Unsupported (simulator): UI shows the placeholder.
        }
        let config = ARWorldTrackingConfiguration()
        config.detectionImages = refs
        config.maximumNumberOfTrackedImages = 2
        config.automaticImageScaleEstimationEnabled = true
        if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
            config.sceneReconstruction = .mesh
        }
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics.insert(.sceneDepth)
        }
        config.planeDetection = []
        session.run(config, options: [.resetTracking, .removeExistingAnchors])
    }

    public func stop() {
        session?.pause()
    }

    /// Append the current snapshot to the in-memory log with an operator note.
    public func captureSample(note: String) {
        samples.append(ValidationSample(timestamp: Date(), note: note, snapshot: snapshot))
    }

    // MARK: Per-frame handling (main actor)

    fileprivate func handleFrame(_ frame: ARFrame) {
        var snap = ValidationSnapshot()
        snap.arSupported = true
        snap.trackingStateLabel = Self.label(for: frame.camera.trackingState)
        snap.selectedRotationName = selectedRotationName
        snap.referenceImageCount = referenceImageCount
        snap.rawImageAnchorCount = frame.anchors.lazy.compactMap { $0 as? ARImageAnchor }.count

        // Camera pose: position (translation) and forward (−z of camera transform).
        let camT = frame.camera.transform
        let camPos = camT.translation
        let camForward = -SIMD3<Float>(camT.columns.2.x, camT.columns.2.y, camT.columns.2.z)
        snap.cameraPositionWorld = camPos
        snap.cameraForwardWorld = camForward

        // Tracked image anchors → diagnostics + [MarkerID: world_T_marker].
        var markers: [CoordinateConventions.MarkerID: simd_float4x4] = [:]
        var anchorInfos: [DevAnchorInfo] = []
        for case let image as ARImageAnchor in frame.anchors {
            let name = image.referenceImage.name ?? "?"
            guard let id = Self.markerNameMap[name] else { continue }
            let axis = RegistrationDiagnostics.anchorAxisAnglesDeg(
                anchorTransform: image.transform, cameraForwardWorld: camForward)
            // Face-on assuming the anchor's local +z is the plane normal (the
            // production assumption the harness is here to confirm/refute).
            let normalZ = SIMD3<Float>(image.transform.columns.2.x,
                                       image.transform.columns.2.y,
                                       image.transform.columns.2.z)
            let faceOn = RegistrationDiagnostics.faceOnAngleDeg(
                markerNormalWorld: normalZ, cameraForwardWorld: camForward)
            anchorInfos.append(DevAnchorInfo(
                id: id.rawValue,
                name: name,
                isTracked: image.isTracked,
                estimatedScaleFactor: Float(image.estimatedScaleFactor),
                axisAnglesDeg: SIMD3(axis.x, axis.y, axis.z),
                faceOnAngleDeg: faceOn,
                worldTransform: image.transform))
            if image.isTracked { markers[id] = image.transform }
        }
        anchorInfos.sort { $0.id < $1.id }
        snap.anchors = anchorInfos

        // Fused world_T_leksell for the selected candidate rotation.
        snap.fusedWorldToLeksell = RegistrationDiagnostics.fuse(
            anchors: markers, rotation: selectedRotation)

        // Inter-marker disagreement (needs both markers' implied Leksell frames).
        if markers.count >= 2 {
            let frames = markers.map { id, w in
                w * RegistrationDiagnostics.leksellToMarker(id, rotation: selectedRotation).inverse
            }
            snap.interMarkerOriginMM = RegistrationDiagnostics.interMarkerOriginDisagreementMM(frames)
            snap.interMarkerOrientationDeg = RegistrationDiagnostics.interMarkerOrientationDisagreementDeg(frames)
        }

        // S3-02: incision raycast hit + LiDAR confidence at its projected pixel.
        if let w2l = snap.fusedWorldToLeksell {
            let (hit, conf) = incisionHitAndConfidence(worldToLeksell: w2l, frame: frame)
            snap.incisionHitWorld = hit
            snap.incisionConfidence = conf
        }

        snapshot = snap
    }

    /// Raycast the LiDAR mesh from the probe target along its trajectory, then read
    /// the sceneDepth confidence at the hit's projected pixel. (S3-02.)
    private func incisionHitAndConfidence(worldToLeksell: simd_float4x4,
                                          frame: ARFrame)
    -> (SIMD3<Float>?, DevConfidence) {
        guard let arView else { return (nil, .unavailable) }

        let originWorld = (worldToLeksell * SIMD4<Float>(probeTrajectory.target, 1)).xyz
        let dirWorld = simd_normalize(worldToLeksell.upperLeft3x3 * probeTrajectory.direction)
        guard dirWorld.x.isFinite, dirWorld.y.isFinite, dirWorld.z.isFinite else {
            return (nil, .unavailable)
        }
        let length = max(simd_length(probeTrajectory.lineEnd - probeTrajectory.target), 0.001)

        let hits = arView.scene.raycast(origin: originWorld,
                                        direction: dirWorld,
                                        length: length,
                                        query: .nearest,
                                        mask: .sceneUnderstanding,
                                        relativeTo: nil)
        guard let nearest = hits.first else { return (nil, .unavailable) }
        let hitWorld = nearest.position

        let conf = confidence(atWorldPoint: hitWorld, frame: frame)
        return (hitWorld, conf)
    }

    /// Sample `frame.sceneDepth?.confidenceMap` at the pixel where `worldPoint`
    /// projects. Returns `.unavailable` when there is no depth/confidence map
    /// (simulator) or the point projects off-screen. (S3-02.)
    private func confidence(atWorldPoint worldPoint: SIMD3<Float>,
                            frame: ARFrame) -> DevConfidence {
        guard let confidenceMap = frame.sceneDepth?.confidenceMap else { return .unavailable }

        // Project the 3D world point to image (capture) space.
        let viewport = arView?.bounds.size ?? .zero
        let orientation: UIInterfaceOrientation = .portrait
        let projected = frame.camera.projectPoint(worldPoint,
                                                   orientation: orientation,
                                                   viewportSize: viewport)
        guard projected.x.isFinite, projected.y.isFinite else { return .unavailable }

        CVPixelBufferLockBaseAddress(confidenceMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(confidenceMap, .readOnly) }
        let width = CVPixelBufferGetWidth(confidenceMap)
        let height = CVPixelBufferGetHeight(confidenceMap)
        guard width > 0, height > 0, viewport.width > 0, viewport.height > 0 else {
            return .unavailable
        }
        // Map the viewport pixel into the (typically smaller) confidence buffer.
        let nx = Double(projected.x) / Double(viewport.width)
        let ny = Double(projected.y) / Double(viewport.height)
        guard nx >= 0, nx < 1, ny >= 0, ny < 1 else { return .unavailable }
        let cx = Int(nx * Double(width))
        let cy = Int(ny * Double(height))

        guard let base = CVPixelBufferGetBaseAddress(confidenceMap) else { return .unavailable }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(confidenceMap)
        let row = base.advanced(by: cy * bytesPerRow)
        let raw = row.advanced(by: cx).assumingMemoryBound(to: UInt8.self).pointee
        guard let level = ARConfidenceLevel(rawValue: Int(raw)) else { return .unavailable }
        return DevConfidence(level)
    }

    private static func label(for state: ARCamera.TrackingState) -> String {
        switch state {
        case .notAvailable: return "notAvailable"
        case .limited(let reason):
            switch reason {
            case .initializing:        return "limited(initializing)"
            case .excessiveMotion:     return "limited(motion)"
            case .insufficientFeatures:return "limited(features)"
            case .relocalizing:        return "limited(relocalizing)"
            @unknown default:          return "limited(?)"
            }
        case .normal: return "normal"
        @unknown default: return "?"
        }
    }

    // MARK: - CSV export

    /// Write the captured samples to a CSV in Documents and return its URL (for a
    /// share sheet). Returns nil if there is nothing to write or the write fails.
    public func exportCSV() -> URL? {
        guard !samples.isEmpty else { return nil }

        var rows: [String] = []
        rows.append([
            "timestamp", "note", "selectedRotation", "trackingState",
            "anchorCount", "interMarkerOriginMM", "interMarkerOrientationDeg",
            "fused_tx", "fused_ty", "fused_tz",
            "left_faceOnDeg", "left_axisX", "left_axisY", "left_axisZ", "left_scale",
            "right_faceOnDeg", "right_axisX", "right_axisY", "right_axisZ", "right_scale",
            "incisionConfidence", "incision_x", "incision_y", "incision_z",
            "cam_x", "cam_y", "cam_z"
        ].joined(separator: ","))

        let iso = ISO8601DateFormatter()
        func f(_ v: Float) -> String { String(format: "%.5f", v) }
        func anchor(_ s: ValidationSnapshot, _ id: Int) -> DevAnchorInfo? {
            s.anchors.first { $0.id == id }
        }

        for sample in samples {
            let s = sample.snapshot
            let fused = s.fusedWorldToLeksell?.translation ?? SIMD3<Float>(repeating: .nan)
            let left = anchor(s, CoordinateConventions.MarkerID.left.rawValue)
            let right = anchor(s, CoordinateConventions.MarkerID.right.rawValue)
            let inc = s.incisionHitWorld ?? SIMD3<Float>(repeating: .nan)

            func axisCols(_ a: DevAnchorInfo?) -> [String] {
                guard let a else { return ["", "", "", "", ""] }
                return [f(a.faceOnAngleDeg), f(a.axisAnglesDeg.x), f(a.axisAnglesDeg.y),
                        f(a.axisAnglesDeg.z), f(a.estimatedScaleFactor)]
            }

            var cols: [String] = [
                iso.string(from: sample.timestamp),
                "\"" + sample.note.replacingOccurrences(of: "\"", with: "'") + "\"",
                "\"" + s.selectedRotationName + "\"",
                s.trackingStateLabel,
                "\(s.anchors.count)",
                f(s.interMarkerOriginMM), f(s.interMarkerOrientationDeg),
                f(fused.x), f(fused.y), f(fused.z),
            ]
            cols += axisCols(left)
            cols += axisCols(right)
            cols += [s.incisionConfidence.rawValue, f(inc.x), f(inc.y), f(inc.z),
                     f(s.cameraPositionWorld.x), f(s.cameraPositionWorld.y), f(s.cameraPositionWorld.z)]
            rows.append(cols.joined(separator: ","))
        }

        let csv = rows.joined(separator: "\n")
        let stamp = ISO8601DateFormatter().string(from: Date())
            .replacingOccurrences(of: ":", with: "-")
        let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("RegistrationValidation-\(stamp).csv")
        do {
            try csv.data(using: .utf8)?.write(to: url, options: .atomic)
            return url
        } catch {
            return nil
        }
    }

    // MARK: - ARSessionDelegate bridge (copied pattern)

    private final class Coordinator: NSObject, ARSessionDelegate, @unchecked Sendable {
        weak var owner: ValidationSession?
        func session(_ session: ARSession, didUpdate frame: ARFrame) {
            MainActor.assumeIsolated {
                owner?.handleFrame(frame)
            }
        }
    }
}
#endif
