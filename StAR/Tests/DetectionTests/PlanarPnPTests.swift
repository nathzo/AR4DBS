//  PlanarPnPTests.swift
//  B2 — synthetic round-trip + degeneracy tests for the pure planar pose solver.
//
//  Strategy: pick a known `cam_T_marker` (OpenCV camera convention: +x right,
//  +y down, +z forward), project the 4 marker corners through it + K to synthesize
//  the detected pixel corners, run `PlanarPnP.solve`, and assert the recovered pose
//  matches truth. Because the data is noise-free, the bounds are tight.

import Testing
import simd
@testable import StAR

struct PlanarPnPTests {

    // Intrinsics used throughout (a plausible iPhone-class wide camera).
    private static let fx: Float = 1400, fy: Float = 1400, cx: Float = 960, cy: Float = 720
    private static let h: Float = 0.015   // marker half-side (30 mm marker)

    /// Marker corners on the z = 0 plane, ordered TL, TR, BR, BL (+z out of marker).
    private static let objectPoints: [SIMD3<Float>] = [
        SIMD3(-h,  h, 0),   // TL
        SIMD3( h,  h, 0),   // TR
        SIMD3( h, -h, 0),   // BR
        SIMD3(-h, -h, 0),   // BL
    ]

    /// Build a `cam_T_marker` from a rotation quaternion and translation.
    private static func makePose(_ q: simd_quatf, _ t: SIMD3<Float>) -> simd_float4x4 {
        let R = simd_float3x3(q)
        return simd_float4x4(
            SIMD4(R.columns.0, 0),
            SIMD4(R.columns.1, 0),
            SIMD4(R.columns.2, 0),
            SIMD4(t, 1)
        )
    }

    /// Project the 4 object corners through `camTmarker` + K → pixel corners.
    private static func project(_ camTmarker: simd_float4x4) -> [SIMD2<Float>] {
        objectPoints.map { p in
            let pc = (camTmarker * SIMD4(p, 1))
            return SIMD2(fx * pc.x / pc.z + cx, fy * pc.y / pc.z + cy)
        }
    }

    private static func rotation(_ m: simd_float4x4) -> simd_quatf {
        simd_quatf(simd_float3x3(
            SIMD3(m.columns.0.x, m.columns.0.y, m.columns.0.z),
            SIMD3(m.columns.1.x, m.columns.1.y, m.columns.1.z),
            SIMD3(m.columns.2.x, m.columns.2.y, m.columns.2.z)
        ))
    }

    private static func translation(_ m: simd_float4x4) -> SIMD3<Float> {
        SIMD3(m.columns.3.x, m.columns.3.y, m.columns.3.z)
    }

    /// Geodesic angle (degrees) between two rotations (sign-agnostic: q ≡ -q).
    private static func angleBetween(_ a: simd_quatf, _ b: simd_quatf) -> Float {
        let d = min(1, abs(simd_dot(a.vector, b.vector)))
        return 2 * acos(d) * 180 / .pi
    }

    private static func reprojectionRMS(_ pose: simd_float4x4, _ img: [SIMD2<Float>]) -> Float {
        let pred = project(pose)
        var sum: Float = 0
        for i in 0..<4 { sum += simd_length_squared(pred[i] - img[i]) }
        return (sum / 4).squareRoot()
    }

    // MARK: - Synthetic round-trip

    /// For each known pose: reproject error < 0.5 px, |Δt| < 1 mm, |Δrot| < 1°.
    /// Bounds rationale (noise-free synthetic data):
    ///   • 0.5 px  — generous vs. the ~1e-3 px we actually achieve; guards regressions.
    ///   • 1 mm    — well above the µm-level error of the noise-free round-trip.
    ///   • 1°      — far above the ~0.05° we achieve; tolerates any sign/quantization wobble.
    @Test(arguments: [
        // (label, rx-deg, ry-deg, tz-m)
        ("frontal",     Float(0),  Float(0),  Float(0.40)),
        ("oblique_x20", Float(20), Float(0),  Float(0.30)),
        ("oblique_x40", Float(40), Float(0),  Float(0.30)),
        ("oblique_y20", Float(0),  Float(20), Float(0.25)),
        ("oblique_y40", Float(0),  Float(40), Float(0.55)),
        ("oblique_xy",  Float(25), Float(-30), Float(0.35)),
    ])
    func roundTrip(label: String, rxDeg: Float, ryDeg: Float, tz: Float) throws {
        let qx = simd_quatf(angle: rxDeg * .pi / 180, axis: SIMD3(1, 0, 0))
        let qy = simd_quatf(angle: ryDeg * .pi / 180, axis: SIMD3(0, 1, 0))
        let q = simd_normalize(qy * qx)
        let t = SIMD3<Float>(0.012, -0.018, tz)   // small lateral offset
        let truth = Self.makePose(q, t)
        let img = Self.project(truth)

        let est = try #require(
            PlanarPnP.solve(imagePoints: img, objectPoints: Self.objectPoints,
                            fx: Self.fx, fy: Self.fy, cx: Self.cx, cy: Self.cy),
            "solve returned nil for \(label)"
        )

        // Recovered pose must reproject the corners onto the measured pixels.
        let rms = Self.reprojectionRMS(est, img)
        #expect(rms < 0.5, "\(label): reprojection RMS \(rms) px ≥ 0.5")

        // Translation within 1 mm of truth.
        let dt = simd_distance(Self.translation(est), t)
        #expect(dt < 0.001, "\(label): translation error \(dt) m ≥ 1 mm")

        // Rotation within ~1° of truth.
        let dAng = Self.angleBetween(Self.rotation(est), q)
        #expect(dAng < 1.0, "\(label): rotation error \(dAng)° ≥ 1°")

        // Sanity: the marker is in front of the camera (z > 0).
        #expect(Self.translation(est).z > 0, "\(label): recovered z must be positive")
    }

    // MARK: - Corner ordering matters

    /// Re-ordering the corners changes the recovered pose. A square marker is
    /// rotationally symmetric in its corner *positions*, so a 90° corner-cycle leaves
    /// the centroid (translation) unchanged but rotates the frame ~90° about the
    /// marker normal — we assert the rotation differs substantially.
    @Test func cornerOrderingChangesPose() throws {
        let q = simd_normalize(simd_quatf(angle: 30 * .pi / 180, axis: SIMD3(1, 0, 0)))
        let t = SIMD3<Float>(0.02, 0.0, 0.35)
        let img = Self.project(Self.makePose(q, t))

        let correct = try #require(
            PlanarPnP.solve(imagePoints: img, objectPoints: Self.objectPoints,
                            fx: Self.fx, fy: Self.fy, cx: Self.cx, cy: Self.cy))

        // Cycle the corner order by one (TL,TR,BR,BL → TR,BR,BL,TL).
        let shuffled = [img[1], img[2], img[3], img[0]]
        let wrong = try #require(
            PlanarPnP.solve(imagePoints: shuffled, objectPoints: Self.objectPoints,
                            fx: Self.fx, fy: Self.fy, cx: Self.cx, cy: Self.cy))

        let dAng = Self.angleBetween(Self.rotation(correct), Self.rotation(wrong))
        // A one-step cycle of a square corresponds to ~90° about the marker normal.
        #expect(dAng > 30, "shuffled corner order should change rotation; got \(dAng)°")
    }

    // MARK: - Degenerate input

    @Test func duplicateCornersReturnNil() {
        // Two coincident corners → no valid homography.
        let img: [SIMD2<Float>] = [
            SIMD2(800, 600), SIMD2(800, 600),   // duplicate
            SIMD2(1100, 800), SIMD2(820, 810),
        ]
        #expect(PlanarPnP.solve(imagePoints: img, objectPoints: Self.objectPoints,
                                fx: Self.fx, fy: Self.fy, cx: Self.cx, cy: Self.cy) == nil)
    }

    @Test func collinearCornersReturnNil() {
        // All four corners on a single line → degenerate (zero quad area).
        let img: [SIMD2<Float>] = [
            SIMD2(800, 600), SIMD2(900, 600),
            SIMD2(1000, 600), SIMD2(1100, 600),
        ]
        #expect(PlanarPnP.solve(imagePoints: img, objectPoints: Self.objectPoints,
                                fx: Self.fx, fy: Self.fy, cx: Self.cx, cy: Self.cy) == nil)
    }

    @Test func wrongCountReturnsNil() {
        let img: [SIMD2<Float>] = [SIMD2(800, 600), SIMD2(900, 600), SIMD2(1000, 700)]
        #expect(PlanarPnP.solve(imagePoints: img, objectPoints: Self.objectPoints,
                                fx: Self.fx, fy: Self.fy, cx: Self.cx, cy: Self.cy) == nil)
    }
}
