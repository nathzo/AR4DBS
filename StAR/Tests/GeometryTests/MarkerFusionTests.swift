import Testing
import simd
@testable import StAR   // adjust to WP0's product module name if different

struct MarkerFusionTests {

    private let fusion = MarkerFusion()
    private var params = RegistrationParameters()

    /// world_T_marker that makes the fused world_T_leksell come out as `expected`.
    /// Since world_T_leksell = world_T_marker · (leksell_T_marker)⁻¹,
    /// we set world_T_marker = expected · leksell_T_marker.
    private func anchor(_ id: CoordinateConventions.MarkerID,
                        worldToLeksell expected: simd_float4x4) -> simd_float4x4 {
        expected * CoordinateConventions.leksellToMarker(id)
    }

    @Test func singleMarkerRecoversFrameButDoesNotQualify() {
        let expected = simd_float4x4(rotation: Rotations.aboutY(0.3),
                                     translation: SIMD3(0.1, -0.2, 0.5))
        let result = fusion.fuse(anchors: [.left: anchor(.left, worldToLeksell: expected)],
                                 parameters: params)
        let r = try! #require(result)
        #expect(transformsClose(r.worldToLeksell, expected))
        #expect(r.qualifies == false)   // a single marker never qualifies for lock
    }

    @Test func twoConsistentMarkersQualify() {
        let expected = simd_float4x4(rotation: Rotations.aboutY(-0.5),
                                     translation: SIMD3(0.2, 0.05, 1.0))
        let result = fusion.fuse(
            anchors: [.left:  anchor(.left,  worldToLeksell: expected),
                      .right: anchor(.right, worldToLeksell: expected)],
            parameters: params)
        let r = try! #require(result)
        #expect(transformsClose(r.worldToLeksell, expected))
        #expect(r.qualifies == true)
    }

    @Test func disagreeingMarkersDoNotQualify() {
        let a = simd_float4x4(rotation: Rotations.aboutY(0), translation: SIMD3(0, 0, 0.5))
        // Shift one marker's implied origin by 2 cm — well past the 3 mm gate.
        let b = simd_float4x4(rotation: Rotations.aboutY(0), translation: SIMD3(0.02, 0, 0.5))
        let result = fusion.fuse(
            anchors: [.left:  anchor(.left,  worldToLeksell: a),
                      .right: anchor(.right, worldToLeksell: b)],
            parameters: params)
        let r = try! #require(result)
        #expect(r.qualifies == false)
    }

    @Test func emptyAnchorsReturnNil() {
        #expect(fusion.fuse(anchors: [:], parameters: params) == nil)
    }

    @Test func averagedRotationStaysOrthonormal() {
        let t1 = simd_float4x4(rotation: Rotations.aboutY(0.10), translation: .zero)
        let t2 = simd_float4x4(rotation: Rotations.aboutY(0.20), translation: .zero)
        let avg = MarkerFusion.average([t1, t2]).upperLeft3x3
        let shouldBeIdentity = avg.transpose * avg
        #expect(transforms3Close(shouldBeIdentity, matrix_identity_float3x3))
        #expect(abs(simd_determinant(avg) - 1) < 1e-4)
    }

    // MARK: - helpers
    private func transformsClose(_ a: simd_float4x4, _ b: simd_float4x4, tol: Float = 1e-4) -> Bool {
        simd_distance(a.columns.0, b.columns.0) <= tol &&
        simd_distance(a.columns.1, b.columns.1) <= tol &&
        simd_distance(a.columns.2, b.columns.2) <= tol &&
        simd_distance(a.columns.3, b.columns.3) <= tol
    }
    private func transforms3Close(_ a: simd_float3x3, _ b: simd_float3x3, tol: Float = 1e-4) -> Bool {
        simd_distance(a.columns.0, b.columns.0) <= tol &&
        simd_distance(a.columns.1, b.columns.1) <= tol &&
        simd_distance(a.columns.2, b.columns.2) <= tol
    }
}
