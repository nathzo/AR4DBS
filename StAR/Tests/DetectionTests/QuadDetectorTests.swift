//  QuadDetectorTests.swift
//  B2 — verify the pure quad-geometry helpers (the Vision request is device-tested).

import Testing
import simd
@testable import StAR

struct QuadDetectorTests {

    private let square: [SIMD2<Float>] = [
        SIMD2(0, 0), SIMD2(100, 0), SIMD2(100, 100), SIMD2(0, 100),
    ]

    @Test func signedAreaMatchesShoelace() {
        #expect(abs(abs(QuadDetector.signedArea(square)) - 10_000) < 1e-3)
    }

    @Test func convexSquareIsConvex() {
        #expect(QuadDetector.isConvex(square))
    }

    @Test func selfIntersectingQuadIsNotConvex() {
        // A "bowtie" (swapped two corners) is non-convex.
        let bowtie: [SIMD2<Float>] = [SIMD2(0, 0), SIMD2(100, 100), SIMD2(100, 0), SIMD2(0, 100)]
        #expect(!QuadDetector.isConvex(bowtie))
    }

    @Test func validQuadGate() {
        #expect(QuadDetector.isValidQuad(square))
        // Below the area floor → rejected.
        let tiny: [SIMD2<Float>] = [SIMD2(0, 0), SIMD2(5, 0), SIMD2(5, 5), SIMD2(0, 5)]
        #expect(!QuadDetector.isValidQuad(tiny))
        // Wrong count → rejected.
        #expect(!QuadDetector.isValidQuad([SIMD2(0, 0), SIMD2(1, 0), SIMD2(1, 1)]))
    }

    @Test func orderClockwiseGivesConsistentWinding() {
        // Any input order of the same 4 corners yields a convex, consistently-wound quad.
        let shuffled: [SIMD2<Float>] = [square[2], square[0], square[3], square[1]]
        let ordered = QuadDetector.orderClockwise(shuffled)
        #expect(ordered.count == 4)
        #expect(QuadDetector.isConvex(ordered))
        #expect(abs(abs(QuadDetector.signedArea(ordered)) - 10_000) < 1e-3)
    }
}
