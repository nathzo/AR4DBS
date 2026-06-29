import Testing
@testable import StAR   // adjust to WP0's product module name if different

// Mirrors the C++ PlanScanner parsing expectations.
struct PlanTextParserTests {

    private func line(_ s: String, _ c: Float = 0.9) -> OCRLine { OCRLine(text: s, confidence: c) }

    @Test func parsesBothSidesWithHeaders() {
        let lines = [
            line("Gauche"),
            line("X (mm) 140.4"), line("Y (mm) 114.6"), line("Z (mm) 80.0"),
            line("Ring 74.2"), line("Arc 71.0"),
            line("Droite"),
            line("X (mm) 66.2"), line("Y (mm) 118.2"), line("Z (mm) 77.4"),
            line("Ring 66.8"), line("Arc 111.1"),
        ]
        let plan = PlanTextParser.parse(lines)
        #expect(plan.left.isValid)
        #expect(plan.right.isValid)
        #expect(abs(plan.left.xMM - 140.4) < 1e-6)
        #expect(abs(plan.left.arcDeg - 71.0) < 1e-6)
        #expect(abs(plan.right.zMM - 77.4) < 1e-6)
        #expect(abs(plan.right.ringDeg - 66.8) < 1e-6)
    }

    @Test func onlyOneHeaderLeavesOtherSideEmpty() {
        let lines = [
            line("Gauche"),
            line("X (mm) 10.0"), line("Y (mm) 20.0"), line("Z (mm) 30.0"),
            line("Ring 40.0"), line("Arc 50.0"),
        ]
        let plan = PlanTextParser.parse(lines)
        #expect(plan.left.isValid)
        #expect(plan.right.isValid == false)   // no mirroring
    }

    @Test func missingFieldMarksSideInvalid() {
        let lines = [
            line("Gauche"),
            line("X (mm) 10.0"), line("Y (mm) 20.0"), line("Z (mm) 30.0"),
            line("Ring 40.0"),    // Arc missing
        ]
        let plan = PlanTextParser.parse(lines)
        #expect(plan.left.isValid == false)
        #expect(plan.left.confidence(for: .arc) == nil)
        #expect(plan.left.confidence(for: .x) != nil)
    }

    @Test func confidenceComesFromSourceLine() {
        let lines = [
            line("Gauche", 0.95),
            line("X (mm) 12.3", 0.42),
            line("Y (mm) 1.0", 0.9), line("Z (mm) 1.0", 0.9),
            line("Ring 1.0", 0.9), line("Arc 1.0", 0.9),
        ]
        let plan = PlanTextParser.parse(lines)
        let xConf = try! #require(plan.left.confidence(for: .x))
        #expect(abs(xConf - 0.42) < 1e-5)
    }
}
