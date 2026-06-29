//  RegistrationDiagnostics.swift
//  DEV-ONLY registration-validation harness — PURE math layer.
//
//  This file backs the on-device "registration validation harness"
//  (ValidationView / ValidationSession) used to empirically resolve four open
//  issues from the v1→v2 equivalence audit (StAR/docs/v1-v2-equivalence-audit.md):
//
//    • S1-01  — the marker→Leksell rotation Rz(π) is analytically derived but
//               UNVALIDATED (Apple's ARImageAnchor basis is uncertain, possibly
//               ~90° off). `candidateRotations` is the live-cyclable set the user
//               sweeps against the physical jig to pin the correct rotation.
//    • REG-01 — v1's ~5° face-on lock gate was dropped. `faceOnAngleDeg` /
//               `anchorAxisAnglesDeg` surface the per-marker face-on angle.
//    • REG-03 — v2 lost v1's joint solvePnP depth constraint. `placementError`
//               decomposes a placed point's error into RANGE vs LATERAL.
//    • S3-02  — LiDAR-confidence hair-rejection was dropped. (Confidence sampling
//               lives in ValidationSession; this layer stays pure.)
//
//  EVERYTHING here is PURE & deterministic — only `simd` is imported, no ARKit /
//  RealityKit — so it is unconditionally unit-testable (see
//  StAR/Tests/RegistrationTests/RegistrationDiagnosticsTests.swift). Intentionally
//  NOT wrapped in #if DEBUG: it is harmless and we want it always tested.
//
//  Single module "StAR": CoordinateConventions / MarkerFusion are visible.

import simd

/// Pure, deterministic decision-math for the dev registration harness.
public enum RegistrationDiagnostics {

    public typealias MarkerID = CoordinateConventions.MarkerID

    // MARK: - S1-01: candidate marker→Leksell rotations

    /// The candidate `R_leksell_marker` rotations the user cycles live to pin
    /// S1-01. Each is composed from `simd_quatf` products (quaternion composition:
    /// `a * b` applies `b` first, then `a`). The list always starts with the
    /// current production rotation so index 0 == "no change from shipping code".
    ///
    /// Composition note: `CoordinateConventions.markerRotation` is `Rz(π)`; the two
    /// leading suspects from the audit are `Rz(π)·Rx(±90°)` (the ARImageAnchor
    /// plane-normal is +y, not +z, so the basis is tilted 90° about x). `Ry(π)` is
    /// the legacy ArUco value; `identity` is the null hypothesis / sanity baseline.
    public static let candidateRotations: [(name: String, quat: simd_quatf)] = {
        let rzPi = CoordinateConventions.markerRotation                       // Rz(π)
        let rxPlus  = simd_quatf(angle:  .pi / 2, axis: SIMD3<Float>(1, 0, 0)) // Rx(+90°)
        let rxMinus = simd_quatf(angle: -.pi / 2, axis: SIMD3<Float>(1, 0, 0)) // Rx(−90°)
        let ryPi    = simd_quatf(angle:  .pi,     axis: SIMD3<Float>(0, 1, 0)) // Ry(π)
        let identity = simd_quatf(angle: 0,       axis: SIMD3<Float>(0, 0, 1))
        return [
            ("Rz(π) [current]",   rzPi),
            ("Rz(π)·Rx(+90°)",    rzPi * rxPlus),
            ("Rz(π)·Rx(−90°)",    rzPi * rxMinus),
            ("Ry(π) [legacy ArUco]", ryPi),
            ("identity",          identity),
        ]
    }()

    // MARK: - Registration frame helpers

    /// `leksell_T_marker` for `id` using the candidate `rotation` and the baked
    /// physical translation (`CoordinateConventions.defaultMarkerTranslation`).
    /// Mirrors `CoordinateConventions.leksellToMarker` but with a swappable rotation.
    public static func leksellToMarker(_ id: MarkerID,
                                       rotation: simd_quatf) -> simd_float4x4 {
        simd_float4x4(rotation: rotation,
                      translation: CoordinateConventions.defaultMarkerTranslation(id))
    }

    /// Fuse the currently-tracked anchors into `world_T_leksell` for a candidate
    /// rotation:  world_T_leksell = average over markers of
    /// `world_T_marker · (leksell_T_marker(id, rotation))⁻¹`.
    /// Reuses `MarkerFusion.average`. Returns nil when no markers are present.
    public static func fuse(anchors: [MarkerID: simd_float4x4],
                            rotation: simd_quatf) -> simd_float4x4? {
        guard !anchors.isEmpty else { return nil }
        let frames: [simd_float4x4] = anchors.map { id, worldToMarker in
            worldToMarker * leksellToMarker(id, rotation: rotation).inverse
        }
        return MarkerFusion.average(frames)
    }

    // MARK: - REG-01: face-on angle

    /// Angle (degrees) between the marker plane normal and the camera viewing
    /// direction. 0° = perfectly face-on (the surgeon looks straight down the
    /// normal); 90° = grazing.
    ///
    /// SIGN / CONVENTION: both inputs are world-space directions. We take the
    /// UNSIGNED angle between the marker normal and the camera-forward axis, folding
    /// the front/back ambiguity via `abs(dot)` — a normal pointing toward OR away
    /// from the camera both read as "face-on" (the printed image can be detected
    /// from either declared normal sense). Result is in [0°, 90°]. Inputs need not
    /// be unit length; they are normalised here.
    public static func faceOnAngleDeg(markerNormalWorld: SIMD3<Float>,
                                      cameraForwardWorld: SIMD3<Float>) -> Float {
        let n = safeNormalize(markerNormalWorld)
        let f = safeNormalize(cameraForwardWorld)
        let d = min(1, abs(simd_dot(n, f)))
        return acos(d) * 180 / .pi
    }

    // MARK: - S1-01 diagnostic: which anchor axis is the plane normal?

    /// Angle (degrees) between camera-forward and each of the anchor's three local
    /// axes, expressed in world space. Held FACE-ON, exactly one of these reads
    /// near 0° or 180° — that axis is the ARImageAnchor plane normal. This is the
    /// empirical key to S1-01: it tells the user whether the normal is +x, +y, or
    /// +z in Apple's basis, which directly fixes the `Rz(π)` vs `Rz(π)·Rx(±90°)`
    /// question.
    ///
    /// These are SIGNED in [0°, 180°] (not folded) so the user can see 0° vs 180°
    /// and read off the axis SENSE, not just the line.
    public static func anchorAxisAnglesDeg(anchorTransform: simd_float4x4,
                                           cameraForwardWorld: SIMD3<Float>)
    -> (x: Float, y: Float, z: Float) {
        let f = safeNormalize(cameraForwardWorld)
        let r = anchorTransform.upperLeft3x3
        let ax = safeNormalize(r.columns.0)
        let ay = safeNormalize(r.columns.1)
        let az = safeNormalize(r.columns.2)
        func ang(_ v: SIMD3<Float>) -> Float {
            acos(max(-1, min(1, simd_dot(v, f)))) * 180 / .pi
        }
        return (ang(ax), ang(ay), ang(az))
    }

    // MARK: - REG-03: range vs lateral placement error

    /// Decompose the error of a placed point into the component ALONG the
    /// camera→point line of sight (RANGE / depth error) vs PERPENDICULAR to it
    /// (LATERAL error). REG-03: v2 lost v1's joint solvePnP depth constraint, so we
    /// expect range error to dominate; this quantifies it.
    ///
    /// `rangeAlongCamera` and `lateral` are magnitudes (≥ 0). They satisfy
    /// `range² + lateral² == total²` (up to float error). The viewing axis is
    /// `(groundTruth − cameraPosition)` normalised; if the camera sits ON the point
    /// the axis is undefined and the whole error is reported as lateral.
    public static func placementError(estimatedWorld: SIMD3<Float>,
                                      groundTruthWorld: SIMD3<Float>,
                                      cameraPositionWorld: SIMD3<Float>)
    -> (total: Float, rangeAlongCamera: Float, lateral: Float) {
        let err = estimatedWorld - groundTruthWorld
        let total = simd_length(err)
        let viewVec = groundTruthWorld - cameraPositionWorld
        let viewLen = simd_length(viewVec)
        guard viewLen > 1e-9 else { return (total, 0, total) }
        let axis = viewVec / viewLen
        let range = abs(simd_dot(err, axis))               // signed projection magnitude
        let lateralVec = err - simd_dot(err, axis) * axis
        let lateral = simd_length(lateralVec)
        return (total, range, lateral)
    }

    // MARK: - Inter-marker disagreement (reuses MarkerFusion's pairwise helpers)

    /// Largest pairwise distance between the supplied frames' origins, in MILLIMETRES.
    public static func interMarkerOriginDisagreementMM(_ frames: [simd_float4x4]) -> Float {
        guard frames.count >= 2 else { return 0 }
        return MarkerFusion.maxPairwiseDistance(frames.map(\.translation)) * 1000
    }

    /// Largest pairwise geodesic orientation angle between the supplied frames, in
    /// DEGREES.
    public static func interMarkerOrientationDisagreementDeg(_ frames: [simd_float4x4]) -> Float {
        guard frames.count >= 2 else { return 0 }
        return MarkerFusion.maxPairwiseAngle(frames.map(\.upperLeft3x3)) * 180 / .pi
    }

    // MARK: - Internal

    /// Normalise, returning a finite zero vector for a (near-)zero input so callers
    /// never see NaNs from `simd_normalize(0)`.
    private static func safeNormalize(_ v: SIMD3<Float>) -> SIMD3<Float> {
        let len = simd_length(v)
        return len > 1e-9 ? v / len : SIMD3<Float>(0, 0, 0)
    }
}
