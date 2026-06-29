import Testing
import simd
@testable import StAR   // adjust to WP0's product module name if different

// Verifies the Leksell → 3-D trajectory math (ported from IncisionLine.cpp).
struct TrajectoryTests {

    @Test func directionMatchesLegacyFormula() {
        // d = (−cos Arc, sin Arc · cos Ring, −sin Arc · sin Ring)
        let arc = 71.0, ring = 74.2
        let d = CoordinateConventions.trajectoryDirection(arcDeg: arc, ringDeg: ring)
        let a = Float(arc) * .pi / 180, r = Float(ring) * .pi / 180
        #expect(abs(d.x - (-cos(a))) < 1e-5)
        #expect(abs(d.y - (sin(a) * cos(r))) < 1e-5)
        #expect(abs(d.z - (-sin(a) * sin(r))) < 1e-5)
    }

    @Test func directionIsUnitLength() {
        let d = CoordinateConventions.trajectoryDirection(arcDeg: 111.1, ringDeg: 66.8)
        #expect(abs(simd_length(d) - 1) < 1e-5)
    }

    @Test func targetConvertsMMToMetres() {
        let t = LeksellTarget(xMM: 140.4, yMM: 114.6, zMM: 80.0,
                              ringDeg: 74.2, arcDeg: 71.0, isValid: true)
        let traj = Trajectory.fromLeksell(t)
        #expect(abs(traj.target.x - 0.1404) < 1e-6)
        #expect(abs(traj.target.y - 0.1146) < 1e-6)
        #expect(abs(traj.target.z - 0.0800) < 1e-6)
    }

    @Test func lineEndIsLengthAwayAlongDirection() {
        let t = LeksellTarget(xMM: 0, yMM: 0, zMM: 0,
                              ringDeg: 0, arcDeg: 0, isValid: true)
        let traj = Trajectory.fromLeksell(t, length: 0.30)
        #expect(abs(simd_distance(traj.target, traj.lineEnd) - 0.30) < 1e-5)
    }

    @Test func planGeometryDropsInvalidSides() {
        var plan = SurgicalPlanDTO.defaultTest
        plan.right.isValid = false
        let geo = PlanGeometry.from(plan)
        #expect(geo.left != nil)
        #expect(geo.right == nil)
    }
}
