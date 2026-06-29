//  ArucoMarkerDecoder.swift
//  B2 — decode a candidate quad into a DICT_4X4_50 marker id + ordered corners.
//
//  Given a grayscale image and a 4-corner quad (from the QuadDetector), this:
//    1. builds the unit-square → quad perspective homography (Heckbert closed form),
//    2. samples the 6×6 ArUco grid cell centres,
//    3. validates the all-black outer border (rejects non-marker quads),
//    4. reads the inner 4×4 code and matches it (ArucoDictionary, rotation-aware),
//    5. re-orders the quad's corners onto the canonical TL→TR→BR→BL order that
//       PlanarPnP's object points expect.
//
//  Pure (simd only) and unit-testable with synthetic images — no ARKit/CoreVideo.

import simd

/// Minimal grayscale image view. The accessor returns the 8-bit luma at integer
/// (x, y); callers (tests use a `[UInt8]` buffer, the tracker a CVPixelBuffer luma
/// plane). Out-of-range coordinates are the caller's responsibility — `sample`
/// clamps before calling.
public struct GrayImage {
    public let width: Int
    public let height: Int
    public let at: (Int, Int) -> UInt8
    public init(width: Int, height: Int, at: @escaping (Int, Int) -> UInt8) {
        self.width = width; self.height = height; self.at = at
    }
}

public enum ArucoMarkerDecoder {

    /// Decode a candidate quad. `quad` is 4 image-space corners in consistent winding
    /// order (the start corner may be any of the four — rotation is recovered). On
    /// success returns the marker id and its 4 corners re-ordered to canonical
    /// TL, TR, BR, BL. Returns nil if the border isn't black or the code isn't a
    /// DICT_4X4_50 marker.
    public static func decode(image: GrayImage, quad: [SIMD2<Float>])
    -> (id: Int, corners: [SIMD2<Float>])? {
        guard quad.count == 4 else { return nil }
        guard let H = squareToQuad(quad) else { return nil }

        // Sample the 6×6 grid (cell centres in the unit square → image via H).
        let n = 6
        var samples = [Float](repeating: 0, count: n * n)
        for r in 0..<n {
            for c in 0..<n {
                let u = (Float(c) + 0.5) / Float(n)
                let v = (Float(r) + 0.5) / Float(n)
                let p = apply(H, SIMD2(u, v))
                samples[r * n + c] = bilinear(image, p)
            }
        }

        // Threshold at the mid-point of the sampled range (markers are bimodal).
        let lo = samples.min() ?? 0, hi = samples.max() ?? 255
        let thresh = (lo + hi) * 0.5
        // Guard against a flat patch (no real black/white contrast → not a marker).
        guard hi - lo > 30 else { return nil }
        func white(_ r: Int, _ c: Int) -> Bool { samples[r * n + c] >= thresh }

        // The outer ring (border) must be entirely black.
        for i in 0..<n {
            if white(0, i) || white(n - 1, i) || white(i, 0) || white(i, n - 1) {
                return nil
            }
        }

        // Inner 4×4 code, packed row-major (white = 1) — same layout as the dictionary.
        var code: UInt16 = 0
        for ir in 0..<4 {
            for ic in 0..<4 where white(ir + 1, ic + 1) {
                code |= UInt16(1) << (ir * 4 + ic)
            }
        }

        guard let m = ArucoDictionary.match(code) else { return nil }

        // Re-order corners to canonical TL,TR,BR,BL. The input quad corners are in
        // as-read CW order (unit (0,0),(1,0),(1,1),(0,1) → quad[0..3]); the matcher's
        // `rotations` = CW steps to bring the as-read code onto canonical, so the true
        // canonical-TL corner sits at index (4 - rotations) % 4.
        let shift = (4 - m.rotations) % 4
        let corners = (0..<4).map { quad[(shift + $0) % 4] }
        return (m.id, corners)
    }

    // MARK: - Geometry

    /// Heckbert's closed-form homography mapping the unit square corners
    /// (0,0),(1,0),(1,1),(0,1) → q[0],q[1],q[2],q[3]. Returns nil if degenerate.
    static func squareToQuad(_ q: [SIMD2<Float>]) -> simd_float3x3? {
        let p0 = q[0], p1 = q[1], p2 = q[2], p3 = q[3]
        let dx1 = p1.x - p2.x, dx2 = p3.x - p2.x, dx3 = p0.x - p1.x + p2.x - p3.x
        let dy1 = p1.y - p2.y, dy2 = p3.y - p2.y, dy3 = p0.y - p1.y + p2.y - p3.y

        let a, b, c, d, e, f, g, h: Float
        if abs(dx3) < 1e-9 && abs(dy3) < 1e-9 {
            // Affine (parallelogram).
            a = p1.x - p0.x; b = p3.x - p0.x; c = p0.x
            d = p1.y - p0.y; e = p3.y - p0.y; f = p0.y
            g = 0; h = 0
        } else {
            let den = dx1 * dy2 - dx2 * dy1
            guard abs(den) > 1e-12 else { return nil }
            g = (dx3 * dy2 - dx2 * dy3) / den
            h = (dx1 * dy3 - dx3 * dy1) / den
            a = p1.x - p0.x + g * p1.x; b = p3.x - p0.x + h * p3.x; c = p0.x
            d = p1.y - p0.y + g * p1.y; e = p3.y - p0.y + h * p3.y; f = p0.y
        }
        // column-major simd_float3x3
        return simd_float3x3(columns: (SIMD3(a, d, g), SIMD3(b, e, h), SIMD3(c, f, 1)))
    }

    /// Apply a homography to a 2-D point (with the perspective divide).
    static func apply(_ H: simd_float3x3, _ p: SIMD2<Float>) -> SIMD2<Float> {
        let v = H * SIMD3(p.x, p.y, 1)
        let w = abs(v.z) > 1e-12 ? v.z : 1
        return SIMD2(v.x / w, v.y / w)
    }

    /// Bilinear luma sample at a (clamped) sub-pixel image location.
    static func bilinear(_ img: GrayImage, _ p: SIMD2<Float>) -> Float {
        let x = min(max(p.x, 0), Float(img.width - 1))
        let y = min(max(p.y, 0), Float(img.height - 1))
        let x0 = Int(x.rounded(.down)), y0 = Int(y.rounded(.down))
        let x1 = min(x0 + 1, img.width - 1), y1 = min(y0 + 1, img.height - 1)
        let fx = x - Float(x0), fy = y - Float(y0)
        let a = Float(img.at(x0, y0)), b = Float(img.at(x1, y0))
        let c = Float(img.at(x0, y1)), d = Float(img.at(x1, y1))
        return a * (1 - fx) * (1 - fy) + b * fx * (1 - fy)
             + c * (1 - fx) * fy + d * fx * fy
    }
}
