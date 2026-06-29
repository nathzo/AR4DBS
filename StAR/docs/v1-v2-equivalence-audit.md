# v1 → v2 equivalence audit (2026-06-23)

A differential audit comparing the proven C++/Qt app (**v1**, under `app/ core/
platform/ resources/`) against the Swift rewrite (**v2**, under `StAR/StARApp/`),
focused on whether v2's safety-critical math is *behaviorally equivalent* to v1.
16 candidate divergences were flagged; each was adversarially verified. 9 were
confirmed real, 7 refuted (benign idiom or already-correct — notably the incision
ray **direction** and the line-clamp **t-convention** were verified equivalent to v1).

## Applied (safe, faithful to v1, verified by compile + tests)

| ID | Sev | Fix |
|----|-----|-----|
| **S4-D1** | High | Scan mode hid the per-side *Activer* toggle, forcing both electrodes (a single-electrode case had to fill all 10 fields). Restored the toggle in both modes (`ConfirmPlanView.swift`); `SideFormModel`/`readBack` already honor `isEnabled`. |
| **S1-02b** | High | Lock gate checked only marker-origin agreement, missing a *correlated tilt* where both markers' frames are rotated the same wrong way. Added a marker-to-marker **orientation** disagreement gate (`maxMarkerOrientationDisagreementRad`, ~5°) in `MarkerFusion.fuse` (+ `maxPairwiseAngle` helper). Convention-independent (compares the two markers to each other). Covered by two new tests in `RegistrationAccuracyTests`. |
| **S3-01** | High | The incision raycast/lock ran **once** at lock (`ARScreen` calls `update()` only on registration *change*), so the 5-frame lock streak could never complete and the marker could fail to appear. `RealityKitOverlay` now subscribes to `SceneEvents.Update` and re-runs the per-frame incision logic from cached inputs every rendered frame. |

## Deferred to on-device validation (NOT changed — would be a guess)

| ID | Sev | Why deferred |
|----|-----|--------------|
| **S1-01** | High | `leksell_T_marker` rotation (`Rz(π)`) assumes the `ARImageAnchor` image lies in the local **x/y plane with +z out**; Apple's documented convention is the **x/z plane with +y as the surface normal** — a ~90° difference that would mis-aim the trajectory *attitude* while leaving the origin plausible. The corrected rotation is roughly `Rz(π)·Rx(±π/2)`, **but the sign/handedness cannot be settled analytically** (the file's own `TODO(WP8)` says so). Fixing this on a guess is as risky as leaving it. **Must be pinned on an iPhone Pro against the physical Leksell jig.** |
| **REG-01** | High | v1's **face-on angle gate** (~5°) was dropped. The natural ARKit analogue depends on the marker-plane normal — i.e. the *same* unvalidated axis convention as S1-01 — so it is resolved together with S1-01 on the jig, not guessed now. (The S1-02b orientation gate restores part of the lost quality bar in a convention-independent way.) |
| **REG-02 / REG-03** | Med | v1's absolute **reprojection-error** bound (≤3 px) and the unified **8-corner solvePnP** depth constraint cannot be reproduced — `ARImageAnchor` exposes no corner pixels. This is an inherent accuracy tradeoff of the ArUco→image-anchor swap; document and weigh against measured device accuracy (WP8). |
| **S3-02** | Med | v1's `checkIncisionQuality` LiDAR-**confidence / hair-rejection** gate was dropped. Restoring it needs a 3D→confidence-map projection that cannot be verified without a device; a wrong projection could make the incision never lock. Recommended, but to be implemented and validated on hardware. |
| **S4-D2** | Med | Per-tag marker positions (`tx/ty/tz`) are no longer editable in Settings — marker geometry is now asset/code-fixed. Intentional architecture change; if field-correctable geometry is a clinical requirement, re-add offsets to `RegistrationParameters`. Otherwise add a print-template/QA check so physical markers match the baked layout. |

## Refuted (no action)
REG-04 (streak reset/advance semantics) and REG-05 (drift guard) are **equivalent** to
v1. S3-03 (incision ray direction) and S3-04 (line-clamp t-convention) are **equivalent**.
S4-D3 (language switch is now a system-language note), S4-D4 (reproj px tunable removed),
S4-D5 (`canConfirm` adds an "at least one side active" precondition) are real but do **not**
affect surgical correctness.

## Bottom line
The single highest registration risk remains **S1-01 (the marker rotation)**, which —
together with REG-01 — is the gating item for clinical trust and can only be resolved on
a LiDAR iPhone Pro with the physical jig. See also `StAR/docs/WP8-accuracy.md`.
