import simd

/// Pure marker fusion. Ported from `AppController::fusePoses` and the 10-frame
/// SO(3) averaging in `AppController::onARFrame` (C++), adapted for ARKit image
/// anchors. No ARKit imports — inputs are plain transforms, so this is fully
/// unit-testable (see Tests/GeometryTests).
///
/// Per marker:  world_T_leksell = world_T_marker · (leksell_T_marker)⁻¹
/// Multiple markers are fused by quaternion averaging (rotation) + mean
/// (translation). Quaternion averaging replaces the legacy `cv::SVD` rotation
/// averaging — equivalent for the small disagreements expected when two markers
/// are correctly registered, and avoids an SVD dependency.
///
/// The caller (WP2) must pass only currently-**tracked** anchors with a sane
/// `estimatedScaleFactor`; this type judges geometric agreement, not ARKit state.
public struct MarkerFusion: MarkerFusing {

    public init() {}

    public func fuse(anchors: [CoordinateConventions.MarkerID: simd_float4x4],
                     parameters: RegistrationParameters)
    -> (worldToLeksell: simd_float4x4, qualifies: Bool)? {

        guard !anchors.isEmpty else { return nil }

        // Each marker independently establishes the full Leksell frame, using the
        // (possibly field-corrected) marker translation from the parameters.
        let frames: [simd_float4x4] = anchors.map { id, worldToMarker in
            let offset = (id == .left) ? parameters.markerLeftOffsetM : parameters.markerRightOffsetM
            return worldToMarker * CoordinateConventions.leksellToMarker(id, translation: offset).inverse
        }

        let fused = MarkerFusion.average(frames)

        // A frame qualifies for the lock streak when BOTH markers are present and
        // their independently-implied Leksell frames agree — in BOTH origin
        // (translation) AND orientation. The orientation check (audit S1-02) catches
        // a correlated tilt that leaves the two origins close but both frames rotated
        // the same wrong way, which the origin-only gate would have passed.
        var qualifies = false
        if anchors.count >= 2 {
            let originDisagreement = MarkerFusion.maxPairwiseDistance(frames.map(\.translation))
            let angleDisagreement  = MarkerFusion.maxPairwiseAngle(frames.map(\.upperLeft3x3))
            qualifies = originDisagreement < parameters.maxMarkerDisagreementM
                && angleDisagreement < parameters.maxMarkerOrientationDisagreementRad
        }

        return (fused, qualifies)
    }

    /// SO(3)+R³ average of rigid transforms via hemisphere-aligned quaternion
    /// mean (nlerp) and translation mean.
    static func average(_ transforms: [simd_float4x4]) -> simd_float4x4 {
        precondition(!transforms.isEmpty)
        if transforms.count == 1 { return transforms[0] }

        let q0 = simd_quatf(transforms[0].upperLeft3x3)
        var quatAccum = q0.vector
        var tAccum = transforms[0].translation

        for t in transforms.dropFirst() {
            var q = simd_quatf(t.upperLeft3x3)
            // Align to the same hemisphere as q0 so the average doesn't cancel.
            if simd_dot(q.vector, q0.vector) < 0 { q = simd_quatf(vector: -q.vector) }
            quatAccum += q.vector
            tAccum += t.translation
        }

        let qAvg = simd_quatf(vector: simd_normalize(quatAccum))
        let tAvg = tAccum / Float(transforms.count)
        return simd_float4x4(rotation: qAvg, translation: tAvg)
    }

    /// Largest distance between any pair of points (for 2 points: their distance).
    static func maxPairwiseDistance(_ points: [SIMD3<Float>]) -> Float {
        var maxD: Float = 0
        for i in 0..<points.count {
            for j in (i + 1)..<points.count {
                maxD = max(maxD, simd_distance(points[i], points[j]))
            }
        }
        return maxD
    }

    /// Largest rotation angle (radians) between any pair of orientations.
    /// (For 2 rotations: the geodesic angle between them.)
    static func maxPairwiseAngle(_ rotations: [simd_float3x3]) -> Float {
        let quats = rotations.map { simd_quatf($0) }
        var maxA: Float = 0
        for i in 0..<quats.count {
            for j in (i + 1)..<quats.count {
                // |dot| folds the double-cover; angle = 2·acos(|q1·q2|).
                let d = min(1, abs(simd_dot(quats[i].vector, quats[j].vector)))
                maxA = max(maxA, 2 * acos(d))
            }
        }
        return maxA
    }
}
