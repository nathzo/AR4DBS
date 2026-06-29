import simd

/// Pure trajectory-sampling helpers used by the overlay (WP3) and incision
/// locking. Ported from the ray walk in `AppController::renderOverlayOnto` /
/// `renderWithOcclusion` (C++): points run from the skull-side end to the deep
/// target, parameter `t ∈ [0, 1]` with `t = 0` at `lineEnd`, `t = 1` at `target`.
public enum IncisionGeometry {

    public static let defaultSamples = 60

    /// `count + 1` evenly spaced points along the trajectory, skull → target,
    /// expressed in the Leksell frame (metres).
    public static func samples(_ trajectory: Trajectory,
                               count: Int = defaultSamples) -> [SIMD3<Float>] {
        let diff = trajectory.target - trajectory.lineEnd
        return (0...count).map { i in
            let t = Float(i) / Float(count)
            return trajectory.lineEnd + diff * t
        }
    }

    /// Parametric position `t` of an arbitrary Leksell-frame point projected onto
    /// the `lineEnd → target` segment. Used to clamp the drawn line to a locked
    /// incision point (legacy `t_inc` computation in `renderWithOcclusion`).
    public static func parameter(of point: SIMD3<Float>,
                                 on trajectory: Trajectory) -> Float {
        let diff = trajectory.target - trajectory.lineEnd
        let len2 = simd_length_squared(diff)
        guard len2 > 1e-12 else { return 1 }
        return simd_dot(point - trajectory.lineEnd, diff) / len2
    }
}
