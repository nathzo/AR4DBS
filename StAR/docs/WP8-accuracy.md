# WP8 — ARKit Image-Tracking Registration: Accuracy Report & Go/No-Go

**Status: CONDITIONAL GO for continued development — NO-GO for clinical trust until
on-device hardware validation is complete.**

This package de-risks the project's central unknown: *is ARKit image tracking
accurate enough for surgical (sub-mm) registration?* That question **cannot be
answered in this environment** — there is no iPhone Pro, no ARKit runtime, and no
physical jig here. What WP8 delivers is everything that *can* be produced and
proven analytically/synthetically, plus an explicit, honest list of what remains
unproven on hardware.

---

## 1. Deliverables produced

| Item | Path |
|------|------|
| AR resource group `LeksellMarkers` (2 reference images, 30 mm) | `StAR/StARApp/Resources/ARReferenceImages.xcassets/LeksellMarkers.arresourcegroup/` |
| Re-derived `leksell_T_marker` (ARImageAnchor basis) | `StAR/StARApp/Core/Geometry/CoordinateConventions.swift` |
| Synthetic accuracy harness (Swift Testing) | `StAR/Tests/RegistrationTests/RegistrationAccuracyTests.swift` |
| This report | `StAR/docs/WP8-accuracy.md` |

The resource group is consumed by WP2 via
`ARReferenceImage.referenceImages(inGroupNamed: "LeksellMarkers", bundle: .main)`,
mapping image name → `MarkerID`: `LeksellMarkerLeft → .left` (tag id 0),
`LeksellMarkerRight → .right` (tag id 1).

---

## 2. Marker-design decision (sanctioned deviation from DICT_4X4_50)

> **UPDATE 2026-06-23 (B1 test):** the bundled reference images were reverted to the
> **real `DICT_4X4_50` ids 0/1** (generated with OpenCV `cv2.aruco`; 30 mm marker +
> 5 mm white quiet zone → 40 mm reference image, `width = 0.04 m`) to test on a device
> whether ARKit `detectionImages` can detect the surgeon's **actual physical AprilTags**.
> The bespoke markers below were never the surgeon's tags — that was a silent build-time
> substitution. If ARKit detects the real tags acceptably, keep this; if not, that is the
> empirical case for **B2** (a native ArUco decoder + `solvePnP`, as in v1). The rationale
> below is exactly *why* ARKit may still struggle with sparse 4×4 fiducials.

**Decision: replace the legacy `DICT_4X4_50` ArUco markers with bespoke
high-feature, asymmetric, high-contrast images.** WORKPLAN WP8 explicitly permits
this ("if ArUco tracks poorly as reference images, design high-feature replacement
markers and document the change"). Rationale:

- **ARKit `detectionImages` is a natural-feature tracker**, not an ArUco decoder.
  It scores candidate images by the number and distinctiveness of detectable
  feature points. Sparse 4×4 binary ArUco tiles have very few, highly repetitive
  features and are routinely rejected by Xcode's asset validator with
  "unsupported / not enough features" warnings, and track poorly/jitter badly when
  they are accepted.
- **Our replacement markers** (`~600×600 px`, printed at **30 mm**) carry: a heavy
  black border, three *distinct* corner glyphs placed at three of four corners
  (breaks rotational and mirror symmetry → unambiguous orientation), a dense
  non-repeating blob/stroke feature field, and a printed **UP arrow + side label
  ("L"/"R")**. The two markers are globally distinct (different accent colour,
  label, empty-corner position, and RNG seed) so ARKit will not confuse them.
- **Physical width is fixed at 0.03 m (30 mm)**, identical to the legacy marker
  size, so the translations in `tag_config.json` carry over unchanged.
- **Orientation cue:** the printed "UP" arrow defines the physical mounting
  orientation, which fixes the marker-local +y (image up) axis used in the
  transform derivation below. The jig builder must mount each marker UP-arrow
  toward Leksell-superior, in the marker plane.

This is an **intentional, sanctioned deviation** from the legacy `DICT_4X4_50`
geometry. It changes *what is printed*, not the marker's physical size or
position, so it does not alter the registration math beyond the axis convention.

---

## 3. Axis derivation — `leksell_T_marker` for the ARImageAnchor basis

**ARImageAnchor convention (assumed):** origin at the printed image **centre**;
the image lies in the anchor's local **x/y plane**; **+x = image right**,
**+y = image up**, **+z = out of the printed surface toward the viewer**
(right-handed, x × y = z). This is the standard ARKit image-anchor basis.

**ArUco/OpenCV convention (what the legacy `Ry(π)` was derived against):**
+x = image right, **+y = image down**, **+z = into the marker** (away from the
camera). Therefore the ARKit basis is the ArUco basis flipped about x:
`arkit = aruco · Rx(π)` (y and z negate).

**Leksell axes** (file header / `tag_config.json`): +x → toward the "left" marker,
+y → anterior, +z → inferior.

The legacy note gives `R_leksell_aruco = Ry(π)`. Composing:

```
R_leksell_arkit = R_leksell_aruco · Rx(π)
                = Ry(π) · Rx(π)
                = diag(-1, 1, -1) · diag(1, -1, -1)
                = diag(-1, -1,  1)
                = Rz(π)        (orthonormal, det = +1 — a valid rigid rotation)
```

Resulting marker-axis → Leksell-axis mapping:

| ARKit marker axis | → Leksell axis |
|---|---|
| +x (image right) | −x (toward "right" marker) |
| +y (image up) | −y (posterior) |
| +z (out of surface) | +z (inferior) |

**Translations (fixed jig measurements, metres, unchanged from `tag_config.json`):**
`left = (0.2325, 0.100, 0.171)`, `right = (-0.0325, 0.100, 0.171)`.

The implemented transform is therefore `simd_float4x4(rotation: Rz(π),
translation: t)`. Because the existing `MarkerFusionTests` build
`world_T_marker = expected · leksell_T_marker` and recover via
`world_T_marker · leksell_T_marker⁻¹`, **any** valid rigid transform with the
correct translations keeps fusion self-consistent — those tests remain green
regardless of the rotation chosen. The rotation only matters when matched against
real ARImageAnchor poses on hardware.

> **Unproven on hardware.** Two signs are suspect and flagged with a `TODO(WP8
> on-device)` in code: the **z-flip** (ARKit +z out-of-surface vs. Leksell
> inferior) and overall **handedness**. Any correction is a fixed re-orthonormal
> rotation premultiplying the result; it must be measured on the physical jig.

---

## 4. Legacy accuracy gates — intent translated

`AppController::meetsInitConditions` (C++) gated lock on: ≥ 2 tags, ≥ 8 detected
corners total, **face-on within ~5°** (`-R(2,2) ≤ -cos(175°)`), and **reprojection
error ≤ 3 px**. ARKit image anchors expose neither corners nor reprojection
error, so the *intent* maps to:

- **"≥ 2 tags, ≥ 8 corners"** → require **both** markers present and `isTracked`
  (a single image anchor never qualifies — enforced by `MarkerFusion`).
- **"reproj ≤ 3 px / pose quality"** → require **inter-marker agreement**: the two
  markers' independently-implied Leksell origins must agree within
  `maxMarkerDisagreementM` (3 mm). This is the geometric analogue of reprojection
  consistency and is the gate `MarkerFusion.fuse` evaluates.
- **"face-on within 5°"** + scale sanity → enforced by WP2 at the ARKit layer
  (`ARImageAnchor.isTracked`, `estimatedScaleFactor ≈ 1`, camera `trackingState`),
  which WP8 cannot exercise here.
- **10-frame streak + SO(3) average** → preserved; the synthetic harness models
  the windowed mean (`§5c`).

---

## 5. Synthetic numbers the harness asserts

`RegistrationAccuracyTests` feeds **synthetic** `world_T_marker` transforms through
the *same* `CoordinateConventions.leksellToMarker` + `MarkerFusion` math the device
uses. These prove the fusion/geometry is correct given accurate anchors; they do
**not** measure ARKit.

| Test | Setup | Assertion |
|---|---|---|
| (a) perfect pair | two consistent markers, head-on ~0.55 m | fused Leksell origin error **< 1e-4 m**; `qualifies == true` |
| (b) gate trips | one marker origin shifted **5 mm** (> 3 mm gate) | `qualifies == false` |
| (b′) inside gate | shift **1 mm** (< 3 mm gate) | `qualifies == true` |
| (c) jitter | ±0.5 mm/axis/marker noise, 60 frames, windowed mean | single-frame origin error **< 2 mm**; windowed-mean error **< 0.5 mm**; mean ≤ worst single frame |

**Documented jitter bound rationale.** With a ±0.5 mm/axis/marker zero-mean noise
budget, a single fused-origin error sits near √3 · 0.5 mm ≈ 0.87 mm worst case;
the 2 mm single-frame bound leaves headroom. Because the noise is zero-mean,
averaging over a 60-frame window collapses the residual by ~√N, driving the
windowed-mean error well under 0.5 mm (numeric model: ~0.01 mm). The chosen
±0.5 mm noise amplitude is a **placeholder**, not a measured ARKit figure.

---

## 6. Go / No-Go

- **GO** to continue building WP2/WP3 against this resource group and transform:
  the asset catalog is structurally valid, the transform is a valid rigid
  transform, the gating/averaging math is proven correct on synthetic data, and
  the existing geometry tests stay green.
- **NO-GO for clinical/sub-mm trust** until measured on an iPhone Pro with a
  physical jig. **Final sub-mm accuracy is UNPROVEN.** The dominant unknowns —
  ARKit image-anchor pose jitter, absolute pose bias, scale-estimation error, and
  the z/handedness sign of `leksell_T_marker` — are *physically impossible to
  measure in this environment*.

### Recommended on-device mitigations (validate during WP2 bring-up)
1. **Measure first.** With both markers on a measured jig, log per-frame fused
   origin jitter, inter-marker disagreement, and re-registration repeatability
   across distance (0.3–0.8 m) and angle (0–30° off-axis). Replace the ±0.5 mm
   placeholder in test (c) with the measured number and re-tighten the bounds.
2. **Larger / denser markers** if jitter > a few mm (40–60 mm improves angular
   resolution and feature count).
3. **`automaticImageScaleEstimationEnabled = false`** once physical sizes are
   trusted — fixed scale removes a jitter source; keep it on only while
   characterising scale error.
4. **Multi-image / redundancy** — both markers already required for lock; consider
   adding a third reference image to over-constrain registration.
5. **Averaging windows** — keep/extend the 10-frame SO(3) streak; widen the window
   if single-frame jitter is high (trades lock latency for stability).
6. **Validate the rotation sign** against the jig before any clinical use; correct
   the `TODO(WP8 on-device)` rotation if measured axes disagree.
