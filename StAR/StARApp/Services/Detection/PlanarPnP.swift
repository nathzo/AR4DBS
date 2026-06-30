//  PlanarPnP.swift
//  B2 — pure planar pose solver (solvePnP for a flat square ArUco marker).
//
//  Recovers the marker pose from its 4 detected corner pixels, replacing ARKit
//  image tracking. No OpenCV, no third-party deps: a normalized-DLT homography is
//  built and decomposed analytically, and the planar two-fold ambiguity is broken
//  by picking the candidate with the lower reprojection error.
//
//  ── Conventions ─────────────────────────────────────────────────────────────
//  CAMERA frame is the OpenCV pinhole convention:
//     +x right · +y DOWN · +z FORWARD (into the scene, away from the lens).
//     A point P_cam = (X,Y,Z) projects to pixel (u,v):
//        u = fx·X/Z + cx ,  v = fy·Y/Z + cy ,  with Z > 0 in front of the camera.
//
//  OBJECT (marker) frame: corners lie on the z = 0 plane, +z OUT of the marker
//  toward the camera. The caller supplies, for half-side h, the corners in this
//  order (matching imagePoints):
//     index 0 = TL (-h, +h, 0)
//     index 1 = TR (+h, +h, 0)
//     index 2 = BR (+h, -h, 0)
//     index 3 = BL (-h, -h, 0)
//
//  IMAGE points are pixel coordinates in the same TL,TR,BR,BL order.
//
//  Returned `camTmarker` (simd_float4x4) is `cam_T_marker`: it maps a marker-frame
//  point to the camera frame. For a frontal marker its translation has z > 0 and
//  its rotation faces the camera.
//
//  ── Algorithm ───────────────────────────────────────────────────────────────
//   a. Normalize image points: xn = K⁻¹·[u,v,1].
//   b. DLT homography H (object-plane [X,Y,1] → normalized image [xn,yn,1]) from
//      the 4 correspondences: assemble the 8×9 system A·h = 0 and take the right
//      singular vector of A for the smallest singular value (LAPACK dgesvd_).
//   c. Decompose H = [h1 h2 h3]: λ = 2/(‖h1‖+‖h2‖); r1 = λ·h1, r2 = λ·h2,
//      t = λ·h3; r3 = r1 × r2; orthonormalize R via SVD (R = U·Vᵀ, det = +1).
//   d. Chirality: if t.z < 0, negate (t, r1, r2) and recompute r3 (marker in front).
//   e. Planar ambiguity: a planar marker admits two poses. Build the twin (mirror
//      the marker-normal about the optical axis) and return whichever has the lower
//      reprojection RMS against imagePoints.
//
//  All linear algebra is done in Double for conditioning; the result converts to
//  Float at the very end. Rotations are carried as `simd_double3x3` (column-major:
//  column k is the image of marker basis vector k in the camera frame).

import simd
import Accelerate

public enum PlanarPnP {

    /// Solve the planar pose of a square marker from its 4 detected corners.
    ///
    /// - Parameters:
    ///   - imagePoints:  4 corner pixel coordinates, ordered TL, TR, BR, BL.
    ///   - objectPoints: the matching 4 corners on the marker's z = 0 plane,
    ///                   ordered TL, TR, BR, BL (+z out of the marker).
    ///   - fx, fy, cx, cy: pinhole intrinsics in pixels.
    /// - Returns: `cam_T_marker` in the OpenCV camera convention (z > 0 in front),
    ///   or `nil` on degenerate input (wrong count, collinear/duplicate points,
    ///   singular linear systems).
    public static func solve(
        imagePoints: [SIMD2<Float>],
        objectPoints: [SIMD3<Float>],
        fx: Float, fy: Float, cx: Float, cy: Float
    ) -> simd_float4x4? {
        guard imagePoints.count == 4, objectPoints.count == 4 else { return nil }
        guard fx > 0, fy > 0, fx.isFinite, fy.isFinite, cx.isFinite, cy.isFinite else { return nil }

        // ── (a) Normalize image points: xn = K⁻¹·[u,v,1]. ────────────────────
        let fxd = Double(fx), fyd = Double(fy), cxd = Double(cx), cyd = Double(cy)
        var obj = [SIMD2<Double>](repeating: .zero, count: 4)   // (X,Y) on z=0 plane
        var img = [SIMD2<Double>](repeating: .zero, count: 4)   // pixel (u,v)
        var nrm = [SIMD2<Double>](repeating: .zero, count: 4)   // normalized (xn,yn)
        for i in 0..<4 {
            // Object points must be planar (z ≈ 0); reject otherwise.
            if abs(Double(objectPoints[i].z)) > 1e-6 { return nil }
            let X = Double(objectPoints[i].x), Y = Double(objectPoints[i].y)
            let u = Double(imagePoints[i].x), v = Double(imagePoints[i].y)
            guard u.isFinite, v.isFinite, X.isFinite, Y.isFinite else { return nil }
            obj[i] = SIMD2(X, Y)
            img[i] = SIMD2(u, v)
            nrm[i] = SIMD2((u - cxd) / fxd, (v - cyd) / fyd)
        }

        // Reject degenerate quads (duplicate or collinear corners) on either side.
        if isDegenerateQuad(obj) || isDegenerateQuad(img) { return nil }

        // ── (b) DLT homography (object [X,Y,1] → normalized [xn,yn,1]). ──────
        guard let H = computeHomography(object: obj, normalized: nrm) else { return nil }

        // ── (c)+(d) Decompose into the primary pose candidate. ──────────────
        guard let primary = poseFromHomography(H) else { return nil }

        // ── (e) Build the planar twin and pick the lower-reprojection pose. ─
        let twin = twinPose(primary)

        let e0 = reprojectionRMS(pose: primary, obj: obj, img: img,
                                 fx: fxd, fy: fyd, cx: cxd, cy: cyd)
        var best = primary
        var bestErr = e0
        if let twin {
            let e1 = reprojectionRMS(pose: twin, obj: obj, img: img,
                                     fx: fxd, fy: fyd, cx: cxd, cy: cyd)
            if e1 < bestErr { best = twin; bestErr = e1 }
        }
        guard bestErr.isFinite else { return nil }

        return makeFloat4x4(pose: best)
    }

    // MARK: - Pose model

    /// A pose candidate: rotation R (column-major; col k = marker axis k in camera
    /// frame) and translation t (marker origin in camera frame).
    private struct Pose { var R: simd_double3x3; var t: SIMD3<Double> }

    // MARK: - Homography (normalized DLT)

    /// Solve A·h = 0 for the 3×3 homography mapping object plane → normalized image.
    /// Returns H as a `simd_double3x3` with H·[X,Y,1] giving the (unnormalized)
    /// homogeneous normalized-image point. Stored column-major.
    private static func computeHomography(
        object: [SIMD2<Double>], normalized: [SIMD2<Double>]
    ) -> simd_double3x3? {
        // Assemble the 8×9 DLT matrix A (row-major), 2 rows per correspondence.
        //   For (X,Y) → (x,y):
        //     [ -X -Y -1   0  0  0   xX  xY  x ]
        //     [  0  0  0  -X -Y -1   yX  yY  y ]
        var A = [Double](repeating: 0, count: 8 * 9)
        for i in 0..<4 {
            let X = object[i].x, Y = object[i].y
            let x = normalized[i].x, y = normalized[i].y
            let r0 = (2 * i) * 9
            A[r0 + 0] = -X; A[r0 + 1] = -Y; A[r0 + 2] = -1
            A[r0 + 6] = x * X; A[r0 + 7] = x * Y; A[r0 + 8] = x
            let r1 = (2 * i + 1) * 9
            A[r1 + 3] = -X; A[r1 + 4] = -Y; A[r1 + 5] = -1
            A[r1 + 6] = y * X; A[r1 + 7] = y * Y; A[r1 + 8] = y
        }

        // h = right singular vector of A for the smallest singular value =
        //     the last row of Vᵀ (the 9th right singular vector).
        guard let vt = svdRightVectors(A: A, rows: 8, cols: 9) else { return nil }
        // vt is Vᵀ row-major (9×9); the smallest-σ vector is its last row.
        let base = 8 * 9
        let h = (0..<9).map { vt[base + $0] }

        // Guard against an all-zero / NaN solution.
        let norm = h.reduce(0) { $0 + $1 * $1 }
        guard norm > 1e-18, h.allSatisfy({ $0.isFinite }) else { return nil }

        // h is the row-major 3×3 [h0 h1 h2; h3 h4 h5; h6 h7 h8].
        // simd_double3x3(columns:) wants columns: col j = (row0_j, row1_j, row2_j).
        return simd_double3x3(columns: (
            SIMD3(h[0], h[3], h[6]),   // column 0
            SIMD3(h[1], h[4], h[7]),   // column 1
            SIMD3(h[2], h[5], h[8])    // column 2
        ))
    }

    // MARK: - Pose from homography

    /// Decompose H into a metric pose (step c + chirality d).
    private static func poseFromHomography(_ H: simd_double3x3) -> Pose? {
        // Columns of H are h1, h2, h3 directly.
        let h1 = H.columns.0
        let h2 = H.columns.1
        let h3 = H.columns.2

        let n1 = simd_length(h1), n2 = simd_length(h2)
        guard n1 > 1e-12, n2 > 1e-12 else { return nil }

        // Scale so that the average of ‖r1‖,‖r2‖ is 1.
        let lambda = 2.0 / (n1 + n2)

        var r1 = lambda * h1
        var r2 = lambda * h2
        var t  = lambda * h3

        // Chirality (d): keep the marker in front of the camera (t.z > 0).
        if t.z < 0 {
            r1 = -r1; r2 = -r2; t = -t
        }

        // r3 completes a right-handed frame, then orthonormalize R via SVD.
        let r3 = simd_cross(r1, r2)
        guard let R = nearestRotation(simd_double3x3(columns: (r1, r2, r3))) else { return nil }
        return Pose(R: R, t: t)
    }

    /// The planar twin: the second valid pose of a flat marker. A planar marker's
    /// homography is consistent with two surface tilts that mirror through the
    /// camera optical axis. We reflect the marker normal (R's 3rd column) about the
    /// plane perpendicular to the viewing axis, rotate the whole frame by the same
    /// Rodrigues rotation, re-orthonormalize, and keep the translation. Reprojection
    /// then selects the physically correct one.
    private static func twinPose(_ p: Pose) -> Pose? {
        let n = p.R.columns.2                       // marker +z in camera frame
        let viewDir = SIMD3<Double>(0, 0, 1)        // optical axis (unit)
        let nDotV = simd_dot(n, viewDir)
        let nTwin = n - 2 * nDotV * viewDir         // reflect through the view plane
        let axis = simd_cross(n, nTwin)
        let axisLen = simd_length(axis)
        if axisLen < 1e-9 { return nil }            // normal ∥ optical axis → no twin
        let k = axis / axisLen
        let cosT = max(-1.0, min(1.0, simd_dot(simd_normalize(n), simd_normalize(nTwin))))
        let theta = acos(cosT)
        let Rtw = rodrigues(axis: k, angle: theta)

        // Rotate every column of R by Rtw (apply the same world-frame rotation).
        let twinR = Rtw * p.R
        guard let Rorth = nearestRotation(twinR) else { return nil }

        var t = p.t
        if t.z < 0 { t = -t }
        return Pose(R: Rorth, t: t)
    }

    // MARK: - Reprojection

    /// RMS pixel error of projecting `obj` through (R,t)+K versus the measured `img`.
    private static func reprojectionRMS(
        pose: Pose, obj: [SIMD2<Double>], img: [SIMD2<Double>],
        fx: Double, fy: Double, cx: Double, cy: Double
    ) -> Double {
        let R = pose.R, t = pose.t
        var sum = 0.0
        for i in 0..<obj.count {
            // P_cam = R·[X,Y,0] + t  (object z = 0 ⇒ only columns 0,1 of R used).
            let X = obj[i].x, Y = obj[i].y
            let pc = R.columns.0 * X + R.columns.1 * Y + t
            if pc.z <= 1e-9 { return .infinity }    // behind the camera → invalid
            let u = fx * pc.x / pc.z + cx
            let v = fy * pc.y / pc.z + cy
            let du = u - img[i].x, dv = v - img[i].y
            sum += du * du + dv * dv
        }
        return (sum / Double(obj.count)).squareRoot()
    }

    // MARK: - Linear-algebra helpers

    /// Nearest rotation matrix to `M` (Procrustes): SVD M = U·Σ·Vᵀ ⇒ R = U·Vᵀ with
    /// the sign of the smallest singular value adjusted so det(R) = +1.
    private static func nearestRotation(_ M: simd_double3x3) -> simd_double3x3? {
        // Flatten to row-major for the LAPACK wrapper.
        let m = [M.columns.0.x, M.columns.1.x, M.columns.2.x,
                 M.columns.0.y, M.columns.1.y, M.columns.2.y,
                 M.columns.0.z, M.columns.1.z, M.columns.2.z]
        guard let (U, Vt) = svdUV3(m) else { return nil }
        var R = matMul3(U, Vt)
        if det3(R) < 0 {
            // R = U·diag(1,1,-1)·Vᵀ ⇒ negate the 3rd column of U then remultiply.
            var U2 = U
            U2[2] = -U2[2]; U2[5] = -U2[5]; U2[8] = -U2[8]
            R = matMul3(U2, Vt)
        }
        guard R.allSatisfy({ $0.isFinite }) else { return nil }
        // R is row-major; build column-major simd matrix.
        return simd_double3x3(columns: (
            SIMD3(R[0], R[3], R[6]),
            SIMD3(R[1], R[4], R[7]),
            SIMD3(R[2], R[5], R[8])
        ))
    }

    // MARK: - LAPACK SVD wrappers (column-major C interop)

    /// Full SVD of an m×n matrix (row-major input), returning Vᵀ row-major (n×n).
    /// Used for the 8×9 DLT null-space.
    private static func svdRightVectors(A: [Double], rows m: Int, cols n: Int) -> [Double]? {
        // LAPACK is column-major; transpose our row-major A into a column-major buffer.
        var a = [Double](repeating: 0, count: m * n)
        for i in 0..<m {
            for j in 0..<n {
                a[j * m + i] = A[i * n + j]   // column-major (col j, row i)
            }
        }

        var jobu: Int8 = 0x41   // 'A' — full U (m×m).
        var jobvt: Int8 = 0x41  // 'A' — full Vᵀ (n×n).
        var M = Int32(m)
        var N = Int32(n)
        var lda = Int32(m)
        var ldu = Int32(m)
        var ldvt = Int32(n)
        var s = [Double](repeating: 0, count: min(m, n))
        var u = [Double](repeating: 0, count: m * m)
        var vt = [Double](repeating: 0, count: n * n)   // column-major Vᵀ (n×n)
        var info = Int32(0)

        // Workspace query.
        var wkopt = 0.0
        var lwork = Int32(-1)
        dgesvd_(&jobu, &jobvt, &M, &N, &a, &lda, &s, &u, &ldu, &vt, &ldvt,
                &wkopt, &lwork, &info)
        guard info == 0 else { return nil }
        lwork = Int32(wkopt)
        var work = [Double](repeating: 0, count: max(1, Int(lwork)))
        dgesvd_(&jobu, &jobvt, &M, &N, &a, &lda, &s, &u, &ldu, &vt, &ldvt,
                &work, &lwork, &info)
        guard info == 0 else { return nil }

        // `vt` is Vᵀ in column-major (n×n). Convert to row-major (n×n).
        var vtRow = [Double](repeating: 0, count: n * n)
        for r in 0..<n {
            for c in 0..<n {
                vtRow[r * n + c] = vt[c * n + r]
            }
        }
        return vtRow
    }

    /// 3×3 SVD: returns (U, Vᵀ) both row-major. Input M is row-major 3×3.
    private static func svdUV3(_ M: [Double]) -> (U: [Double], Vt: [Double])? {
        var a = [Double](repeating: 0, count: 9)
        for i in 0..<3 { for j in 0..<3 { a[j * 3 + i] = M[i * 3 + j] } }  // col-major

        var jobu: Int8 = 0x41   // 'A'
        var jobvt: Int8 = 0x41  // 'A'
        var rows = Int32(3)   // M and N held separately to avoid &n aliasing.
        var cols = Int32(3)
        var lda = Int32(3)
        var ldu = Int32(3)
        var ldvt = Int32(3)
        var s = [Double](repeating: 0, count: 3)
        var u = [Double](repeating: 0, count: 9)
        var vt = [Double](repeating: 0, count: 9)
        var info = Int32(0)

        var wkopt = 0.0
        var lwork = Int32(-1)
        dgesvd_(&jobu, &jobvt, &rows, &cols, &a, &lda, &s, &u, &ldu, &vt, &ldvt,
                &wkopt, &lwork, &info)
        guard info == 0 else { return nil }
        lwork = Int32(wkopt)
        var work = [Double](repeating: 0, count: max(1, Int(lwork)))
        dgesvd_(&jobu, &jobvt, &rows, &cols, &a, &lda, &s, &u, &ldu, &vt, &ldvt,
                &work, &lwork, &info)
        guard info == 0 else { return nil }

        var Urow = [Double](repeating: 0, count: 9)
        var VtRow = [Double](repeating: 0, count: 9)
        for i in 0..<3 {
            for j in 0..<3 {
                Urow[i * 3 + j] = u[j * 3 + i]
                VtRow[i * 3 + j] = vt[j * 3 + i]
            }
        }
        return (Urow, VtRow)
    }

    // MARK: - Tiny 3×3 utilities (row-major Double arrays)

    private static func matMul3(_ A: [Double], _ B: [Double]) -> [Double] {
        var C = [Double](repeating: 0, count: 9)
        for i in 0..<3 {
            for j in 0..<3 {
                var s = 0.0
                for k in 0..<3 { s += A[i * 3 + k] * B[k * 3 + j] }
                C[i * 3 + j] = s
            }
        }
        return C
    }

    private static func det3(_ A: [Double]) -> Double {
        A[0] * (A[4] * A[8] - A[5] * A[7])
      - A[1] * (A[3] * A[8] - A[5] * A[6])
      + A[2] * (A[3] * A[7] - A[4] * A[6])
    }

    /// Rodrigues rotation matrix (column-major simd) about unit `axis` by `angle`.
    private static func rodrigues(axis k: SIMD3<Double>, angle: Double) -> simd_double3x3 {
        let c = cos(angle), s = sin(angle), v = 1 - c
        let (x, y, z) = (k.x, k.y, k.z)
        // Row-major entries, then pack as columns.
        let r00 = c + x * x * v,     r01 = x * y * v - z * s, r02 = x * z * v + y * s
        let r10 = y * x * v + z * s, r11 = c + y * y * v,     r12 = y * z * v - x * s
        let r20 = z * x * v - y * s, r21 = z * y * v + x * s, r22 = c + z * z * v
        return simd_double3x3(columns: (
            SIMD3(r00, r10, r20),
            SIMD3(r01, r11, r21),
            SIMD3(r02, r12, r22)
        ))
    }

    // MARK: - Result assembly & degeneracy checks

    /// Build `cam_T_marker` (Float) from a pose. simd_float4x4 is column-major:
    /// column k = k-th basis image; column 3 = translation.
    private static func makeFloat4x4(pose: Pose) -> simd_float4x4 {
        let R = pose.R, t = pose.t
        func f(_ v: SIMD3<Double>) -> SIMD3<Float> {
            SIMD3<Float>(Float(v.x), Float(v.y), Float(v.z))
        }
        let col0 = f(R.columns.0)
        let col1 = f(R.columns.1)
        let col2 = f(R.columns.2)
        let tr   = f(t)
        return simd_float4x4(
            SIMD4<Float>(col0, 0),
            SIMD4<Float>(col1, 0),
            SIMD4<Float>(col2, 0),
            SIMD4<Float>(tr, 1)
        )
    }

    /// A quad is degenerate if any two corners coincide or any three are collinear.
    private static func isDegenerateQuad(_ p: [SIMD2<Double>]) -> Bool {
        let eps = 1e-9
        for i in 0..<4 {
            for j in (i + 1)..<4 where simd_distance(p[i], p[j]) < eps {
                return true
            }
        }
        // Collinearity of any triple (triangle area ≈ 0 relative to its scale).
        for i in 0..<4 {
            for j in (i + 1)..<4 {
                for k in (j + 1)..<4 {
                    let a = p[j] - p[i], b = p[k] - p[i]
                    let area2 = abs(a.x * b.y - a.y * b.x)
                    let scale = max(simd_length(a), simd_length(b))
                    if scale > eps && area2 < 1e-9 * scale * scale {
                        return true
                    }
                }
            }
        }
        return false
    }
}
