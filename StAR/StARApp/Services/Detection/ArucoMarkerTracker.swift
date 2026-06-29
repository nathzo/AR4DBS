//  ArucoMarkerTracker.swift
//  B2 — turn an ARFrame into [MarkerID: world_T_marker] via ArUco detection + PnP.
//
//  This is the v1-faithful iOS pipeline (app/AppController.cpp onARFrame +
//  ARKitSession.mm), reimplemented natively: detect the physical DICT_4X4_50 tags on
//  the camera image, recover each marker's pose with PlanarPnP, and lift it into
//  ARKit world space. The result feeds the EXISTING MarkerFusing → streak → lock
//  stack unchanged (SurgicalSessionImpl), so registration/overlay are untouched.
//
//  Coordinate chain (the crux):
//    • PlanarPnP returns camCV_T_marker in the OpenCV camera convention (+x right,
//      +y down, +z forward) using the marker object points TL,TR,BR,BL with +z OUT
//      of the marker — identical to v1's fusePoses object points.
//    • ARKit camera.transform is world_T_camera in the ARKit camera convention
//      (+x right, +y up, +z toward the viewer), landscape-aligned like the
//      intrinsics. The two cameras differ by F = diag(1,-1,-1,1) (v1's kARKitFlip).
//    • world_T_marker = world_T_camera · F · camCV_T_marker.
//  Because the marker frame matches v1's, `leksell_T_marker` uses v1's PROVEN Ry(π)
//  (see CoordinateConventions) — this is what makes the B2 path resolve S1-01.

import ARKit
import CoreVideo
import Foundation
import simd

struct ArucoMarkerTracker {

    /// solvePnP object points (marker frame, metres): TL,TR,BR,BL on z=0, +z out.
    private let objectPoints: [SIMD3<Float>]

    /// DICT_4X4_50 id → MarkerID (id 0 = left, id 1 = right; per tag_config.json).
    private static let idToMarker: [Int: CoordinateConventions.MarkerID] = [0: .left, 1: .right]

    /// ARKit camera → OpenCV camera basis change (v1 kARKitFlip): diag(1,-1,-1,1).
    private static let flip = simd_float4x4(diagonal: SIMD4<Float>(1, -1, -1, 1))

    init(markerSideMeters: Float = 0.03) {
        let h = markerSideMeters / 2
        objectPoints = [SIMD3(-h, h, 0), SIMD3(h, h, 0), SIMD3(h, -h, 0), SIMD3(-h, -h, 0)]
    }

    /// Detect both markers in the frame and return their poses in ARKit world space.
    func markers(in frame: ARFrame) -> [CoordinateConventions.MarkerID: simd_float4x4] {
        let pb = frame.capturedImage
        guard let gray = GrayImage.fromLumaPlane(pb) else { return [:] }

        let quads = QuadDetector.detectQuads(in: pb)
        guard !quads.isEmpty else { return [:] }

        let K = frame.camera.intrinsics
        let fx = K.columns.0.x, fy = K.columns.1.y, cx = K.columns.2.x, cy = K.columns.2.y
        let worldTcam = frame.camera.transform

        var out: [CoordinateConventions.MarkerID: (pose: simd_float4x4, area: Float)] = [:]
        for quad in quads {
            guard let dec = ArucoMarkerDecoder.decode(image: gray, quad: quad),
                  let markerID = Self.idToMarker[dec.id],
                  let camCVtMarker = PlanarPnP.solve(imagePoints: dec.corners,
                                                     objectPoints: objectPoints,
                                                     fx: fx, fy: fy, cx: cx, cy: cy)
            else { continue }

            let worldTmarker = worldTcam * Self.flip * camCVtMarker
            // If the same id is decoded from multiple quads, keep the largest
            // (nearest / most reliable) detection.
            let area = abs(QuadDetector.signedArea(quad))
            if let existing = out[markerID], existing.area >= area { continue }
            out[markerID] = (worldTmarker, area)
        }
        return out.mapValues(\.pose)
    }
}

// MARK: - Luma bridge

extension GrayImage {
    /// Build a `GrayImage` from a CVPixelBuffer's luma plane (plane 0 of the ARKit
    /// 420 YpCbCr capture). Copies the plane so the view outlives the buffer lock.
    static func fromLumaPlane(_ pb: CVPixelBuffer) -> GrayImage? {
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddressOfPlane(pb, 0) else { return nil }
        let w = CVPixelBufferGetWidthOfPlane(pb, 0)
        let h = CVPixelBufferGetHeightOfPlane(pb, 0)
        let stride = CVPixelBufferGetBytesPerRowOfPlane(pb, 0)
        guard w > 0, h > 0 else { return nil }

        var buf = [UInt8](repeating: 0, count: w * h)
        let src = base.assumingMemoryBound(to: UInt8.self)
        buf.withUnsafeMutableBufferPointer { dst in
            guard let d = dst.baseAddress else { return }
            for y in 0..<h { memcpy(d + y * w, src + y * stride, w) }
        }
        return GrayImage(width: w, height: h) { x, y in
            buf[min(max(y, 0), h - 1) * w + min(max(x, 0), w - 1)]
        }
    }
}
