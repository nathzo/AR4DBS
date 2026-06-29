//  ArucoCode.swift
//  B2 — pure matching over the DICT_4X4_50 dictionary (no OpenCV).
//
//  Rotation-aware identification of an observed 4x4 code. The decoder samples the
//  marker grid into a packed UInt16 (same layout as `ArucoDictionary.codes`) and
//  asks `match(_:)` for the id; the returned rotation count tells the pose stage how
//  to re-order the detected quad corners onto the canonical TL→TR→BR→BL order.

import Foundation

extension ArucoDictionary {

    /// Rotate a packed 4x4 code 90° clockwise: cell (r,c) → (c, 3-r).
    static func rotate90CW(_ code: UInt16) -> UInt16 {
        var out: UInt16 = 0
        for r in 0..<4 {
            for c in 0..<4 where (code >> (r * 4 + c)) & 1 == 1 {
                let nr = c, nc = 3 - r
                out |= UInt16(1) << (nr * 4 + nc)
            }
        }
        return out
    }

    /// Best-match an observed 4x4 code against the dictionary, trying all four
    /// rotations. Returns the marker id and the number of 90°-CW rotations applied to
    /// the OBSERVED code to bring it onto the canonical code (= the marker's
    /// orientation, used to re-order corners), or nil if no id is within
    /// `maxCorrectionBits` over any rotation.
    static func match(_ observed: UInt16) -> (id: Int, rotations: Int)? {
        var best: (id: Int, rotations: Int, dist: Int)?
        var code = observed
        for rot in 0..<4 {
            for id in 0..<codes.count {
                let dist = (code ^ codes[id]).nonzeroBitCount
                if dist <= maxCorrectionBits, best == nil || dist < best!.dist {
                    best = (id, rot, dist)
                }
            }
            code = rotate90CW(code)
        }
        guard let b = best else { return nil }
        return (b.id, b.rotations)
    }
}
