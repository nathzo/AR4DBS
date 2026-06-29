//  ArucoDictionaryTests.swift
//  B2 — verify the DICT_4X4_50 table + rotation-aware matcher (pure, sim-testable).

import Testing
@testable import StAR

struct ArucoDictionaryTests {

    @Test func tableIsComplete() {
        #expect(ArucoDictionary.codes.count == 50)
        #expect(Set(ArucoDictionary.codes).count == 50)   // all distinct
        #expect(ArucoDictionary.maxCorrectionBits == 1)
    }

    @Test func everyCanonicalCodeMatchesItsOwnIdAtRotationZero() {
        for id in 0..<ArucoDictionary.codes.count {
            let m = ArucoDictionary.match(ArucoDictionary.codes[id])
            #expect(m?.id == id)
            #expect(m?.rotations == 0)
        }
    }

    @Test func fourCWRotationsReturnToOriginal() {
        for id in [0, 13, 27, 49] {
            var c = ArucoDictionary.codes[id]
            for _ in 0..<4 { c = ArucoDictionary.rotate90CW(c) }
            #expect(c == ArucoDictionary.codes[id])
        }
    }

    @Test func rotatedObservationsStillIdentifyTheId() {
        // A physically rotated marker still resolves to the same id (orientation is
        // carried in the rotation count, exercised by the pose stage).
        for id in 0..<ArucoDictionary.codes.count {
            var code = ArucoDictionary.codes[id]
            for _ in 0..<4 {
                #expect(ArucoDictionary.match(code)?.id == id)
                code = ArucoDictionary.rotate90CW(code)
            }
        }
    }

    @Test func nonMarkerCodeIsRejected() {
        // The all-ones code is not a DICT_4X4_50 marker; with maxCorrectionBits=1 it
        // must not be accepted (would indicate the dictionary min-distance is broken).
        #expect(ArucoDictionary.match(0xFFFF) == nil)
    }
}
