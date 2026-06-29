import simd

/// The lifecycle of surgical-frame registration, surfaced to the UI.
/// Ported conceptually from the AprilTag "lock streak" state machine in
/// `AppController.cpp` (onARFrame), adapted to ARKit image anchors.
public enum RegistrationState: Equatable, Sendable {
    /// No usable marker detection yet. UI: "Calibration requise…".
    case awaitingMarkers
    /// Markers seen and accumulating consecutive qualifying frames.
    /// UI: "Calibration en cours…" + progress n/total.
    case calibrating(progress: Int, total: Int)
    /// Locked: `worldToLeksell` is frozen; overlay runs from world tracking.
    /// UI: "Calibration réussie, repères verrouillés".
    case locked(worldToLeksell: simd_float4x4)
}

/// ARKit tracking quality mirror (matches ARCamera.TrackingState ordering used
/// by the legacy debug overlay): 0 = unavailable, 1 = limited, 2 = normal.
public enum TrackingQuality: Int, Comparable, Sendable {
    case unavailable = 0, limited = 1, normal = 2
    public static func < (a: TrackingQuality, b: TrackingQuality) -> Bool { a.rawValue < b.rawValue }
}

/// Tunable registration thresholds (persisted via Settings). Defaults match the
/// legacy constants in `AppController.h`.
public struct RegistrationParameters: Codable, Equatable, Sendable {
    /// Consecutive qualifying frames required before locking.
    public var lockStreakFrames: Int = 10
    /// Max allowed disagreement (metres) between the two markers' implied
    /// Leksell origins for a frame to qualify.
    public var maxMarkerDisagreementM: Float = 0.003
    /// Translation delta (m) that counts as camera movement (drift guard).
    public var moveTransThreshM: Float = 0.010
    /// Rotation delta (deg) that counts as camera movement (drift guard).
    public var moveRotThreshDeg: Float = 1.0
    /// Max allowed orientation disagreement (radians) between the two markers'
    /// independently-implied Leksell frames for a frame to qualify for the lock
    /// streak. Origin agreement alone (maxMarkerDisagreementM) misses a correlated
    /// tilt where both markers' frames are rotated the same wrong way; this gate
    /// catches it (audit S1-02 / partial substitute for v1's reproj/face-on gate).
    /// ~5° default; convention-independent (compares the two markers to each other).
    public var maxMarkerOrientationDisagreementRad: Float = 0.0873

    /// Field-correctable physical marker positions in the Leksell frame (metres).
    /// Default to the baked `tag_config.json` layout; editable in Settings so the
    /// printed marker spacing can be corrected without a rebuild (audit S4-D2).
    /// `MarkerFusion` uses these as the translation of `leksell_T_marker`.
    public var markerLeftOffsetM: SIMD3<Float> = CoordinateConventions.defaultMarkerTranslation(.left)
    public var markerRightOffsetM: SIMD3<Float> = CoordinateConventions.defaultMarkerTranslation(.right)

    public init() {}
}
