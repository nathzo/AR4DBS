import simd

/// Single source of truth for every coordinate convention in the app.
/// Read this before touching registration, geometry, or rendering code.
///
/// ── Leksell frame (the surgical coordinate frame) ───────────────────────────
///   x: 0 = right (marker "right"), 0.200 m = left  (marker "left")
///   y: 0 = posterior,              0.200 m = anterior
///   z: 0 = superior (toward camera / out of marker plane), increases inferior
///   Units inside the geometry layer are METRES (Leksell screen values are mm → /1000).
///
///   Arc  sweeps right ear → top of skull → left ear:
///     Arc=0°   → right ear (−x)
///     Arc=90°  → top of skull (−z) at Ring=90°
///     Arc=180° → left ear (+x)
///   Ring sweeps nose → top of skull → back:
///     Ring=0°   → nose (+y)
///     Ring=90°  → top of skull (−z)
///     Ring=180° → back of head (−y)
///
///   Trajectory direction (unit vector, target → skull entry):
///     d = ( −cos(Arc),  sin(Arc)·cos(Ring),  −sin(Arc)·sin(Ring) )
///   (Ported from `IncisionLine.cpp`.)
///
/// ── ARKit world frame ───────────────────────────────────────────────────────
///   Right-handed, y-up, gravity-aligned. `ARImageAnchor.transform` is
///   `world_T_marker`. NOTE: ARKit image-anchor axes are NOT the same as the
///   legacy OpenCV/ArUco axes. `markerToLeksell(_:)` below MUST be re-derived
///   and validated against the physical rig in WP1/WP8 — do not assume the old
///   `tag_config.json` Ry(π) values transfer unchanged.
public enum CoordinateConventions {

    /// Unit direction vector (Leksell frame) from DBS target toward skull entry.
    @inlinable
    public static func trajectoryDirection(arcDeg: Double, ringDeg: Double) -> SIMD3<Float> {
        let arc = Float(arcDeg) * .pi / 180
        let ring = Float(ringDeg) * .pi / 180
        return SIMD3<Float>(
            -cos(arc),
             sin(arc) * cos(ring),
            -sin(arc) * sin(ring)
        )
    }

    /// Identifies which physical marker established a coordinate frame.
    public enum MarkerID: Int, CaseIterable, Sendable {
        case right = 1   // tag id 1 — near Leksell x≈0
        case left  = 0   // tag id 0 — near Leksell x≈0.200 m
    }

    /// `leksell_T_marker`: the fixed transform placing each marker in the
    /// Leksell frame.
    ///
    /// ── SUPERSEDED (history) — ARImageAnchor-basis derivation (WP8) ──────────────
    /// B2 detects the real ArUco tags via solvePnP, so the marker frame is now the
    /// proven v1 ArUco frame and the rotation below is v1's Ry(π), NOT this Rz(π).
    /// This block is kept only as the record of the former ARImageAnchor guess.
    ///
    /// We re-derive `leksell_T_marker`'s rotation for the ARKit image-anchor axis
    /// convention. (Ported/re-derived from `resources/tag_config.json` — the
    /// legacy `Ry(π)` note — and `core/tracking/AprilTagTracker.cpp`.)
    ///
    /// ARImageAnchor basis (the standard ARKit image-anchor convention):
    ///   • origin at the printed image **centre**
    ///   • the printed image lies in the anchor's local **x/y plane**
    ///   • marker-local +x = image **right**
    ///   • marker-local +y = image **up**
    ///   • marker-local +z = **out of the printed surface, toward the viewer**
    /// (Right-handed: x × y = z.)
    ///
    /// ArUco/OpenCV basis (what the legacy `Ry(π)` was derived against):
    ///   • marker-local +x = image right
    ///   • marker-local +y = image **down**
    ///   • marker-local +z = **into** the marker (away from the camera)
    /// So `arkit = aruco · Rx(π)`: the ARKit basis is the ArUco basis flipped
    /// about its x-axis (y and z both negate).
    ///
    /// Leksell axes (per the file header and `tag_config.json`):
    ///   +x → toward the "left" marker · +y → anterior · +z → inferior.
    ///
    /// The legacy comment states `R_leksell_aruco = Ry(π)` (ArUco +z toward the
    /// camera maps to Leksell −z; ArUco +x right maps to Leksell −x). Composing:
    ///
    ///   R_leksell_arkit = R_leksell_aruco · Rx(π)
    ///                   = Ry(π) · Rx(π)
    ///                   = diag(-1, 1, -1) · diag(1, -1, -1)
    ///                   = diag(-1, -1,  1)
    ///                   = Rz(π)            (orthonormal, det = +1 — a valid rotation)
    ///
    /// Axis mapping under Rz(π) (marker axis → Leksell axis):
    ///   ARKit +x (image right)   → Leksell −x  (toward the "right" marker)
    ///   ARKit +y (image up)      → Leksell −y  (posterior)
    ///   ARKit +z (out of surface)→ Leksell +z  (inferior)
    ///
    /// TRANSLATIONS are fixed physical jig measurements and carry over unchanged
    /// from `tag_config.json` (metres):
    ///   left  = (0.2325, 0.100, 0.171)   right = (-0.0325, 0.100, 0.171)
    ///
    /// TODO(WP8 on-device): rotation is analytically derived for the ARImageAnchor
    /// basis but NOT yet measured on the physical jig — validate sign/handedness on
    /// an iPhone Pro before clinical trust. The two suspect signs are the z-flip
    /// (ARKit +z out-of-surface vs. Leksell inferior) and the overall handedness;
    /// any correction is a fixed re-orthonormal rotation premultiplying the result.
    /// The marker→Leksell ROTATION, shared by every marker. With the B2 ArUco +
    /// solvePnP path the marker frame is the standard ArUco/OpenCV solvePnP frame
    /// (identical to v1's object points), so this uses v1's PROVEN value
    /// Ry(π) = diag(-1, 1, -1) (orthonormal, det = +1) — NOT the superseded
    /// ARImageAnchor Rz(π) guess above. v1 validated Ry(π) clinically (resolves S1-01).
    public static let markerRotation = simd_quatf(angle: .pi, axis: SIMD3<Float>(0, 1, 0))

    /// Default (baked) Leksell-frame translation of each marker (metres), from the
    /// fixed physical measurements in `tag_config.json`. These are the defaults for
    /// `RegistrationParameters.marker*OffsetM`, which can field-correct them.
    public static func defaultMarkerTranslation(_ id: MarkerID) -> SIMD3<Float> {
        switch id {
        case .left:  return SIMD3(0.2325, 0.100, 0.171)
        case .right: return SIMD3(-0.0325, 0.100, 0.171)
        }
    }

    public static func leksellToMarker(_ id: MarkerID) -> simd_float4x4 {
        leksellToMarker(id, translation: defaultMarkerTranslation(id))
    }

    /// `leksell_T_marker` with a caller-supplied translation (a field-corrected
    /// marker position, e.g. from Settings); rotation is always `markerRotation`.
    public static func leksellToMarker(_ id: MarkerID, translation: SIMD3<Float>) -> simd_float4x4 {
        simd_float4x4(rotation: markerRotation, translation: translation)
    }
}
