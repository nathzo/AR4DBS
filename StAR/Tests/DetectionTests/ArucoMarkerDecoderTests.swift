//  ArucoMarkerDecoderTests.swift
//  B2 — verify quad → marker decode + canonical corner ordering (synthetic images).

import Testing
import simd
@testable import StAR

struct ArucoMarkerDecoderTests {

    /// Render marker `id` into the axis-aligned rectangle [x0,y0]–[x1,y1] of a
    /// `size`×`size` white image: outer 6×6 ring black, inner 4×4 from the dictionary
    /// code (white = 1). Returns the buffer + a GrayImage view.
    private func renderMarker(id: Int, size: Int,
                              x0: Float, y0: Float, x1: Float, y1: Float)
    -> GrayImage {
        var buf = [UInt8](repeating: 255, count: size * size)   // white background
        let code = ArucoDictionary.codes[id]
        let cw = (x1 - x0) / 6, ch = (y1 - y0) / 6
        func cellValue(_ r: Int, _ c: Int) -> UInt8 {
            if r == 0 || r == 5 || c == 0 || c == 5 { return 0 }      // black border
            let ir = r - 1, ic = c - 1
            return (code >> (ir * 4 + ic)) & 1 == 1 ? 255 : 0
        }
        for py in 0..<size {
            for px in 0..<size {
                let fx = Float(px), fy = Float(py)
                guard fx >= x0, fx < x1, fy >= y0, fy < y1 else { continue }
                let c = min(5, Int((fx - x0) / cw)), r = min(5, Int((fy - y0) / ch))
                buf[py * size + px] = cellValue(r, c)
            }
        }
        return GrayImage(width: size, height: size) { x, y in
            buf[min(max(y, 0), size - 1) * size + min(max(x, 0), size - 1)]
        }
    }

    @Test func decodesIdAndReturnsCanonicalCorners() {
        let size = 300
        let (x0, y0, x1, y1): (Float, Float, Float, Float) = (60, 60, 240, 240)
        let tl = SIMD2<Float>(x0, y0), tr = SIMD2<Float>(x1, y0)
        let br = SIMD2<Float>(x1, y1), bl = SIMD2<Float>(x0, y1)
        for id in [0, 1, 7, 23, 49] {
            let img = renderMarker(id: id, size: size, x0: x0, y0: y0, x1: x1, y1: y1)
            let out = ArucoMarkerDecoder.decode(image: img, quad: [tl, tr, br, bl])
            #expect(out?.id == id)
            // corners come back canonical TL,TR,BR,BL.
            if let c = out?.corners {
                #expect(simd_distance(c[0], tl) < 2)
                #expect(simd_distance(c[1], tr) < 2)
                #expect(simd_distance(c[2], br) < 2)
                #expect(simd_distance(c[3], bl) < 2)
            }
        }
    }

    @Test func cornerOrderIsRecoveredFromAnyStart() {
        // Whatever corner the detector starts the quad at, decode must return the
        // SAME canonical TL,TR,BR,BL ordering.
        let size = 300
        let (x0, y0, x1, y1): (Float, Float, Float, Float) = (60, 60, 240, 240)
        let tl = SIMD2<Float>(x0, y0), tr = SIMD2<Float>(x1, y0)
        let br = SIMD2<Float>(x1, y1), bl = SIMD2<Float>(x0, y1)
        let canonical = [tl, tr, br, bl]
        let id = 13
        let img = renderMarker(id: id, size: size, x0: x0, y0: y0, x1: x1, y1: y1)
        for start in 0..<4 {
            let quad = (0..<4).map { canonical[(start + $0) % 4] }
            let out = ArucoMarkerDecoder.decode(image: img, quad: quad)
            #expect(out?.id == id)
            if let c = out?.corners {
                for i in 0..<4 { #expect(simd_distance(c[i], canonical[i]) < 2) }
            }
        }
    }

    @Test func rejectsNonMarkerQuad() {
        // A uniform white region: no black border → reject.
        let size = 100
        let img = GrayImage(width: size, height: size) { _, _ in 255 }
        let q: [SIMD2<Float>] = [SIMD2(10, 10), SIMD2(90, 10), SIMD2(90, 90), SIMD2(10, 90)]
        #expect(ArucoMarkerDecoder.decode(image: img, quad: q) == nil)
    }
}
