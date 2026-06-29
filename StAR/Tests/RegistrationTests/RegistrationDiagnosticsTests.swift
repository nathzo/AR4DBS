// DEV harness — pure-math tests for RegistrationDiagnostics.
//
// Covers the PURE helpers behind the on-device registration validation harness
// (StAR/StARApp/Features/Dev/RegistrationDiagnostics.swift), which exists to
// empirically resolve audit issues S1-01 / REG-01 / REG-03 / S3-02
// (StAR/docs/v1-v2-equivalence-audit.md). RegistrationDiagnostics is intentionally
// NOT #if DEBUG-gated (it is pure & simd-only), so these run in the sim gate.
//
// We assert: faceOnAngleDeg endpoints; placementError range/lateral decomposition;
// every candidate rotation is a unit/proper rotation; and fuse() recovers a known
// world_T_leksell for two synthetic consistent markers under identity and Rz(π).

import Testing
import simd
@testable import StAR

struct RegistrationDiagnosticsTests {

    // MARK: faceOnAngleDeg (REG-01)

    @Test func faceOnIsZeroWhenNormalParallelToCameraForward() {
        // Camera looks along −z; a normal pointing toward the camera (+z) is face-on.
        let angle = RegistrationDiagnostics.faceOnAngleDeg(
            markerNormalWorld: SIMD3(0, 0, 1),
            cameraForwardWorld: SIMD3(0, 0, -1))
        #expect(angle < 0.001)
    }

    @Test func faceOnIsZeroRegardlessOfNormalSense() {
        // abs(dot) folds the front/back ambiguity → antiparallel is also 0°.
        let angle = RegistrationDiagnostics.faceOnAngleDeg(
            markerNormalWorld: SIMD3(0, 0, -1),
            cameraForwardWorld: SIMD3(0, 0, -1))
        #expect(angle < 0.001)
    }

    @Test func faceOnIsNinetyWhenPerpendicular() {
        let angle = RegistrationDiagnostics.faceOnAngleDeg(
            markerNormalWorld: SIMD3(1, 0, 0),
            cameraForwardWorld: SIMD3(0, 0, -1))
        #expect(abs(angle - 90) < 0.001)
    }

    @Test func faceOnHandlesNonUnitInputs() {
        let angle = RegistrationDiagnostics.faceOnAngleDeg(
            markerNormalWorld: SIMD3(0, 0, 5),
            cameraForwardWorld: SIMD3(0, 0, -3))
        #expect(angle < 0.001)
    }

    // MARK: placementError decomposition (REG-03)

    @Test func placementErrorPureRange() {
        // Camera at origin looking down +z; truth 0.5 m ahead; estimate is 2 cm
        // FURTHER along the same line of sight → all error is range, no lateral.
        let cam = SIMD3<Float>(0, 0, 0)
        let truth = SIMD3<Float>(0, 0, 0.5)
        let est = SIMD3<Float>(0, 0, 0.52)
        let e = RegistrationDiagnostics.placementError(
            estimatedWorld: est, groundTruthWorld: truth, cameraPositionWorld: cam)
        #expect(abs(e.rangeAlongCamera - 0.02) < 1e-5)
        #expect(e.lateral < 1e-5)
        #expect(abs(e.total - 0.02) < 1e-5)
    }

    @Test func placementErrorPureLateral() {
        // Estimate displaced 2 cm perpendicular to the line of sight → all lateral.
        let cam = SIMD3<Float>(0, 0, 0)
        let truth = SIMD3<Float>(0, 0, 0.5)
        let est = SIMD3<Float>(0.02, 0, 0.5)
        let e = RegistrationDiagnostics.placementError(
            estimatedWorld: est, groundTruthWorld: truth, cameraPositionWorld: cam)
        #expect(e.rangeAlongCamera < 1e-5)
        #expect(abs(e.lateral - 0.02) < 1e-5)
        #expect(abs(e.total - 0.02) < 1e-5)
    }

    @Test func placementErrorPythagorean() {
        // Mixed error: range² + lateral² must equal total².
        let cam = SIMD3<Float>(0, 0, 0)
        let truth = SIMD3<Float>(0, 0, 0.5)
        let est = SIMD3<Float>(0.03, 0, 0.54)   // 4 cm range, 3 cm lateral
        let e = RegistrationDiagnostics.placementError(
            estimatedWorld: est, groundTruthWorld: truth, cameraPositionWorld: cam)
        #expect(abs(e.rangeAlongCamera - 0.04) < 1e-5)
        #expect(abs(e.lateral - 0.03) < 1e-5)
        let recombined = (e.rangeAlongCamera * e.rangeAlongCamera + e.lateral * e.lateral).squareRoot()
        #expect(abs(recombined - e.total) < 1e-5)
    }

    @Test func placementErrorDegenerateCameraOnPoint() {
        // Camera coincident with truth → axis undefined → all error is lateral.
        let p = SIMD3<Float>(1, 2, 3)
        let est = SIMD3<Float>(1.01, 2, 3)
        let e = RegistrationDiagnostics.placementError(
            estimatedWorld: est, groundTruthWorld: p, cameraPositionWorld: p)
        #expect(e.rangeAlongCamera == 0)
        #expect(abs(e.lateral - 0.01) < 1e-5)
    }

    // MARK: candidateRotations validity (S1-01)

    @Test func candidateRotationsAreUnitProperRotations() {
        #expect(RegistrationDiagnostics.candidateRotations.count >= 5)
        for (name, q) in RegistrationDiagnostics.candidateRotations {
            // Unit quaternion.
            let mag = simd_length(q.vector)
            #expect(abs(mag - 1) < 1e-4, "\(name) not unit (|q|=\(mag))")
            // Proper rotation: det(R) == +1, orthonormal columns.
            let r = simd_float3x3(q)
            let det = simd_determinant(r)
            #expect(abs(det - 1) < 1e-4, "\(name) det=\(det)")
            let rtr = r.transpose * r
            let identity = matrix_identity_float3x3
            #expect(simd_distance(rtr.columns.0, identity.columns.0) < 1e-3)
            #expect(simd_distance(rtr.columns.1, identity.columns.1) < 1e-3)
            #expect(simd_distance(rtr.columns.2, identity.columns.2) < 1e-3)
        }
    }

    @Test func firstCandidateIsProductionRotation() {
        let first = RegistrationDiagnostics.candidateRotations[0].quat
        let prod = CoordinateConventions.markerRotation
        #expect(simd_length(first.vector - prod.vector) < 1e-5
                || simd_length(first.vector + prod.vector) < 1e-5)
    }

    // MARK: fuse() recovers a known world_T_leksell (S1-01)

    /// world_T_marker producing the given fused world_T_leksell under `rotation`:
    /// world_T_leksell = world_T_marker · leksellToMarker⁻¹  ⇒
    /// world_T_marker = world_T_leksell · leksellToMarker.
    private func anchor(_ id: RegistrationDiagnostics.MarkerID,
                        worldToLeksell: simd_float4x4,
                        rotation: simd_quatf) -> simd_float4x4 {
        worldToLeksell * RegistrationDiagnostics.leksellToMarker(id, rotation: rotation)
    }

    @Test func fuseRecoversKnownFrameUnderIdentity() {
        let rot = RegistrationDiagnostics.candidateRotations.first { $0.name == "identity" }!.quat
        let truth = simd_float4x4(rotation: Rotations.aboutY(0.15),
                                  translation: SIMD3(0.10, -0.04, 0.50))
        let fused = RegistrationDiagnostics.fuse(
            anchors: [.left:  anchor(.left,  worldToLeksell: truth, rotation: rot),
                      .right: anchor(.right, worldToLeksell: truth, rotation: rot)],
            rotation: rot)
        let f = try! #require(fused)
        #expect(simd_distance(f.translation, truth.translation) < 1e-4)
    }

    @Test func fuseRecoversKnownFrameUnderRzPi() {
        let rot = CoordinateConventions.markerRotation     // Rz(π), the current candidate
        let truth = simd_float4x4(rotation: Rotations.aboutY(-0.2),
                                  translation: SIMD3(0.12, 0.0, 0.55))
        let fused = RegistrationDiagnostics.fuse(
            anchors: [.left:  anchor(.left,  worldToLeksell: truth, rotation: rot),
                      .right: anchor(.right, worldToLeksell: truth, rotation: rot)],
            rotation: rot)
        let f = try! #require(fused)
        #expect(simd_distance(f.translation, truth.translation) < 1e-4)
        // Orientation recovered too (geodesic angle ~0).
        let qT = simd_quatf(truth.upperLeft3x3)
        let qF = simd_quatf(f.upperLeft3x3)
        let dot = min(1, abs(simd_dot(qT.vector, qF.vector)))
        #expect(2 * acos(dot) < 1e-3)
    }

    @Test func fuseReturnsNilWithNoAnchors() {
        #expect(RegistrationDiagnostics.fuse(anchors: [:],
                                             rotation: CoordinateConventions.markerRotation) == nil)
    }

    // MARK: inter-marker disagreement helpers

    @Test func interMarkerOriginDisagreementMM() {
        let a = simd_float4x4(rotation: simd_quatf(angle: 0, axis: SIMD3(0, 0, 1)),
                              translation: SIMD3(0, 0, 0))
        let b = simd_float4x4(rotation: simd_quatf(angle: 0, axis: SIMD3(0, 0, 1)),
                              translation: SIMD3(0.005, 0, 0))   // 5 mm apart
        let mm = RegistrationDiagnostics.interMarkerOriginDisagreementMM([a, b])
        #expect(abs(mm - 5) < 1e-3)
    }

    @Test func interMarkerOrientationDisagreementDeg() {
        let a = simd_float4x4(rotation: Rotations.aboutY(0), translation: .zero)
        let b = simd_float4x4(rotation: Rotations.aboutY(10 * .pi / 180), translation: .zero)
        let deg = RegistrationDiagnostics.interMarkerOrientationDisagreementDeg([a, b])
        #expect(abs(deg - 10) < 0.01)
    }
}
