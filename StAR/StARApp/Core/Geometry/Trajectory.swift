import simd

/// A DBS trajectory in the Leksell frame (metres): the deep target, the
/// skull-side endpoint, and the unit direction between them.
/// Ported from `IncisionLine` (C++). Owned by WP1; consumed by WP3 (rendering).
public struct Trajectory: Equatable, Sendable {
    public let target: SIMD3<Float>    // DBS target (deep end), metres
    public let lineEnd: SIMD3<Float>   // skull-side end of the visualised line
    public let direction: SIMD3<Float> // unit vector target → skull

    public init(target: SIMD3<Float>, lineEnd: SIMD3<Float>, direction: SIMD3<Float>) {
        self.target = target; self.lineEnd = lineEnd; self.direction = direction
    }

    /// Build from one Leksell target. `length` = visualised line length (m) from
    /// target toward the skull. Default 0.30 m matches `IncisionLine::fromLeksell`.
    public static func fromLeksell(_ t: LeksellTarget, length: Float = 0.30) -> Trajectory {
        let target = SIMD3<Float>(Float(t.xMM), Float(t.yMM), Float(t.zMM)) / 1000
        let dir = CoordinateConventions.trajectoryDirection(arcDeg: t.arcDeg, ringDeg: t.ringDeg)
        return Trajectory(target: target, lineEnd: target + dir * length, direction: dir)
    }
}

/// Both sides of a plan resolved into geometry. `nil` side = inactive.
public struct PlanGeometry: Equatable, Sendable {
    public let left: Trajectory?
    public let right: Trajectory?
    public init(left: Trajectory?, right: Trajectory?) { self.left = left; self.right = right }

    public static func from(_ plan: SurgicalPlanDTO) -> PlanGeometry {
        PlanGeometry(left:  plan.hasLeft  ? .fromLeksell(plan.left)  : nil,
                     right: plan.hasRight ? .fromLeksell(plan.right) : nil)
    }
}
