//  QuadDetector.swift
//  B2 — find candidate marker quads in a camera frame (native Vision, no OpenCV).
//
//  Uses `VNDetectContoursRequest` (dark-on-light) to trace contours, simplifies each
//  to a polygon, and keeps convex 4-gons of sufficient area as marker candidates.
//  Output corners are in CAPTURED-IMAGE pixel coordinates (origin top-left, +y down)
//  — the same space as `frame.camera.intrinsics` and the luma `GrayImage` the decoder
//  samples — so the decode + PlanarPnP chain stays in one consistent frame.
//
//  The geometry helpers (area / convexity / winding) are pure and unit-tested; the
//  Vision request itself is exercised on-device.

import Vision
import CoreVideo
import simd

enum QuadDetector {

    /// Minimum quad area in px² to be considered a marker (rejects specks/noise).
    static let minAreaPx: Float = 400

    /// Detect candidate marker quads in `pixelBuffer`. Corners are returned in
    /// captured-image pixel coords (top-left origin), wound clockwise.
    static func detectQuads(in pixelBuffer: CVPixelBuffer) -> [[SIMD2<Float>]] {
        let request = VNDetectContoursRequest()
        request.detectsDarkOnLight = true          // dark markers on a light field
        request.contrastAdjustment = 1.5
        request.maximumImageDimension = 1024       // downscale for speed (coords stay normalized)

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                            orientation: .up, options: [:])
        guard (try? handler.perform([request])) != nil,
              let obs = request.results?.first as? VNContoursObservation else { return [] }

        let w = Float(CVPixelBufferGetWidth(pixelBuffer))
        let h = Float(CVPixelBufferGetHeight(pixelBuffer))

        var quads: [[SIMD2<Float>]] = []
        for i in 0..<obs.contourCount {
            guard let contour = try? obs.contour(at: i),
                  let poly = try? contour.polygonApproximation(epsilon: 0.03) else { continue }
            let pts = poly.normalizedPoints
            guard pts.count == 4 else { continue }
            // Vision normalized points: origin bottom-left → captured-image pixels
            // (origin top-left) with a y-flip.
            let quad = pts.map { SIMD2<Float>($0.x * w, (1 - $0.y) * h) }
            if isValidQuad(quad) { quads.append(orderClockwise(quad)) }
        }
        return quads
    }

    // MARK: - Pure geometry (unit-tested)

    static func isValidQuad(_ q: [SIMD2<Float>]) -> Bool {
        guard q.count == 4 else { return false }
        guard abs(signedArea(q)) >= minAreaPx else { return false }
        return isConvex(q)
    }

    /// Signed area (shoelace); >0 for counter-clockwise winding in a y-down frame.
    static func signedArea(_ q: [SIMD2<Float>]) -> Float {
        var s: Float = 0
        for i in 0..<q.count {
            let a = q[i], b = q[(i + 1) % q.count]
            s += a.x * b.y - b.x * a.y
        }
        return s * 0.5
    }

    /// True if the polygon is convex (all cross-products share a sign).
    static func isConvex(_ q: [SIMD2<Float>]) -> Bool {
        guard q.count == 4 else { return false }
        var sign = 0
        for i in 0..<4 {
            let a = q[i], b = q[(i + 1) % 4], c = q[(i + 2) % 4]
            let cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x)
            let s = cross > 0 ? 1 : (cross < 0 ? -1 : 0)
            if s != 0 {
                if sign == 0 { sign = s } else if s != sign { return false }
            }
        }
        return true
    }

    /// Re-wind 4 corners clockwise (in the y-down image frame) starting from the
    /// corner closest to the image origin, giving the decoder a consistent winding.
    static func orderClockwise(_ q: [SIMD2<Float>]) -> [SIMD2<Float>] {
        // Centroid; sort by angle (clockwise in y-down = increasing atan2(y,x) negated).
        let cx = (q[0].x + q[1].x + q[2].x + q[3].x) / 4
        let cy = (q[0].y + q[1].y + q[2].y + q[3].y) / 4
        let sorted = q.sorted { a, b in
            atan2(a.y - cy, a.x - cx) < atan2(b.y - cy, b.x - cx)
        }
        // `sorted` is counter-clockwise in a y-up frame = clockwise in our y-down image.
        return sorted
    }
}
