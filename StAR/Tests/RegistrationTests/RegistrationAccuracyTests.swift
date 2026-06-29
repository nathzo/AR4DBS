// WP8 — Synthetic registration-accuracy harness.
//
// Ported in intent from `app/AppController.cpp` (`meetsInitConditions`: the
// legacy accuracy gates — ≥8 corners, face-on within 5°, reproj ≤ 3 px) and the
// 10-frame SO(3) streak averaging in `onARFrame`. We cannot run ARKit in this
// environment, so this suite feeds SYNTHETIC `world_T_marker` transforms through
// the SAME math the device path uses (CoordinateConventions.leksellToMarker +
// MarkerFusion) and asserts the registration recovers ground truth, trips the
// disagreement gate, and stays bounded under noise.
//
// IMPORTANT: these numbers prove the *fusion/geometry* is correct given accurate
// anchors. They do NOT prove ARKit image-anchor poses are accurate on hardware —
// see StAR/docs/WP8-accuracy.md for the honest go/no-go framing.

import Testing
import simd
@testable import StAR

struct RegistrationAccuracyTests {

    private let fusion = MarkerFusion()
    private let params = RegistrationParameters()

    /// world_T_marker producing fused world_T_leksell == `expected`.
    /// world_T_leksell = world_T_marker · (leksell_T_marker)⁻¹  ⇒
    /// world_T_marker = expected · leksell_T_marker.
    private func anchor(_ id: CoordinateConventions.MarkerID,
                        worldToLeksell expected: simd_float4x4) -> simd_float4x4 {
        expected * CoordinateConventions.leksellToMarker(id)
    }

    // MARK: (a) two perfectly-consistent markers recover ground truth & qualify

    @Test func twoPerfectMarkersRecoverOriginAndQualify() {
        // A realistic head-on registration pose ~0.5 m from the camera.
        let truth = simd_float4x4(rotation: Rotations.aboutY(0.2),
                                  translation: SIMD3(0.12, -0.05, 0.55))

        let result = fusion.fuse(
            anchors: [.left:  anchor(.left,  worldToLeksell: truth),
                      .right: anchor(.right, worldToLeksell: truth)],
            parameters: params)

        let r = try! #require(result)
        // Fused Leksell origin matches ground-truth origin to < 1e-4 m.
        let originErr = simd_distance(r.worldToLeksell.translation, truth.translation)
        #expect(originErr < 1e-4)
        #expect(r.qualifies == true)
    }

    // MARK: (b) displacing one marker past the gate fails to qualify

    @Test func markerDisagreementTripsGate() {
        // Truth for the right marker.
        let truth = simd_float4x4(rotation: Rotations.aboutY(0),
                                  translation: SIMD3(0, 0, 0.5))
        // Left marker's implied origin shifted by 5 mm > maxMarkerDisagreementM (3 mm).
        let shift: Float = 0.005
        #expect(shift > params.maxMarkerDisagreementM)
        let truthLeftShifted = simd_float4x4(rotation: Rotations.aboutY(0),
                                             translation: SIMD3(shift, 0, 0.5))

        let result = fusion.fuse(
            anchors: [.left:  anchor(.left,  worldToLeksell: truthLeftShifted),
                      .right: anchor(.right, worldToLeksell: truth)],
            parameters: params)

        let r = try! #require(result)
        #expect(r.qualifies == false)   // gate trips on >3 mm inter-marker disagreement
    }

    @Test func smallDisagreementInsideGateStillQualifies() {
        let truth = simd_float4x4(rotation: Rotations.aboutY(0),
                                  translation: SIMD3(0, 0, 0.5))
        // 1 mm disagreement < 3 mm gate → still qualifies.
        let small: Float = 0.001
        #expect(small < params.maxMarkerDisagreementM)
        let truthLeft = simd_float4x4(rotation: Rotations.aboutY(0),
                                      translation: SIMD3(small, 0, 0.5))
        let r = try! #require(fusion.fuse(
            anchors: [.left: anchor(.left, worldToLeksell: truthLeft),
                      .right: anchor(.right, worldToLeksell: truth)],
            parameters: params))
        #expect(r.qualifies == true)
    }

    // MARK: (c) jitter test — bounded spread under synthetic per-frame noise

    @Test func averagedJitterStaysBounded() {
        // Ground-truth registration.
        let truth = simd_float4x4(rotation: Rotations.aboutY(-0.1),
                                  translation: SIMD3(0.10, 0.0, 0.50))

        // Synthetic per-anchor translational noise budget: ±0.5 mm per axis on
        // each marker per frame (a deliberately pessimistic stand-in for ARKit
        // image-anchor jitter — real device numbers are UNKNOWN, see docs).
        let noiseAmp: Float = 0.0005
        let frameCount = 60

        // Deterministic LCG so the bound assertion is reproducible.
        var seed: UInt64 = 0x9E3779B97F4A7C15
        func nextNoise() -> Float {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            let u = Float(seed >> 40) / Float(1 << 24)   // [0,1)
            return (u * 2 - 1) * noiseAmp                 // [-amp, amp]
        }
        func jitterT(_ base: simd_float4x4) -> simd_float4x4 {
            var m = base
            m.columns.3.x += nextNoise()
            m.columns.3.y += nextNoise()
            m.columns.3.z += nextNoise()
            return m
        }

        // Per-frame fuse the two (independently jittered) markers, then average the
        // fused origins across the streak window (mirrors onARFrame's 10-frame mean,
        // generalised to `frameCount`).
        var sum = SIMD3<Float>(repeating: 0)
        var maxFrameErr: Float = 0
        for _ in 0..<frameCount {
            let aLeft  = jitterT(anchor(.left,  worldToLeksell: truth))
            let aRight = jitterT(anchor(.right, worldToLeksell: truth))
            let r = try! #require(fusion.fuse(anchors: [.left: aLeft, .right: aRight],
                                              parameters: params))
            let o = r.worldToLeksell.translation
            sum += o
            maxFrameErr = max(maxFrameErr, simd_distance(o, truth.translation))
        }
        let mean = sum / Float(frameCount)
        let meanErr = simd_distance(mean, truth.translation)

        // BOUNDS (documented):
        //  • Any single-frame fused origin must stay within 2 mm of truth given a
        //    ±0.5 mm/axis/marker noise budget (per-axis worst case ≈ √3·0.5 mm
        //    ≈ 0.87 mm; 2 mm leaves generous headroom).
        #expect(maxFrameErr < 0.002)
        //  • Averaging over the window must drive the residual well under 0.5 mm —
        //    zero-mean noise cancels, so the windowed mean is far tighter than any
        //    single frame.
        #expect(meanErr < 0.0005)
        //  • And the average must be at least as good as the worst single frame.
        #expect(meanErr <= maxFrameErr)
    }

    // MARK: (d) orientation-disagreement gate (audit S1-02)

    @Test func markerOrientationDisagreementTripsGate() {
        // Both markers imply the SAME Leksell origin but orientations differing by
        // 10° (> the ~5° gate) — origin-only agreement would have passed; the
        // orientation gate must trip.
        let t = SIMD3<Float>(0.05, 0, 0.40)
        let truthL = simd_float4x4(rotation: Rotations.aboutY(0), translation: t)
        let truthR = simd_float4x4(rotation: Rotations.aboutY(10 * .pi / 180), translation: t)
        let r = try! #require(fusion.fuse(
            anchors: [.left:  anchor(.left,  worldToLeksell: truthL),
                      .right: anchor(.right, worldToLeksell: truthR)],
            parameters: params))
        #expect(r.qualifies == false)
    }

    @Test func smallOrientationDisagreementStillQualifies() {
        // 3° orientation disagreement (< gate), origins identical → still qualifies.
        let t = SIMD3<Float>(0, 0, 0.40)
        let truthL = simd_float4x4(rotation: Rotations.aboutY(0), translation: t)
        let truthR = simd_float4x4(rotation: Rotations.aboutY(3 * .pi / 180), translation: t)
        let r = try! #require(fusion.fuse(
            anchors: [.left:  anchor(.left,  worldToLeksell: truthL),
                      .right: anchor(.right, worldToLeksell: truthR)],
            parameters: params))
        #expect(r.qualifies == true)
    }
}
