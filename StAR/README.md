# StAR (Swift) — native iOS rewrite

Fully Apple-native rewrite of the StAR surgical-AR app for Deep Brain
Stimulation. **No third-party dependencies.**

- **Stack:** Swift 6 · SwiftUI · SwiftData · ARKit · RealityKit · Vision · simd
- **Target:** iOS 26+, iPhone 15/16 **Pro** (LiDAR), portrait-locked
- **Plan & task breakdown:** see [`../WORKPLAN.md`](../WORKPLAN.md)
- **Reference implementation:** the legacy C++/Qt app under `../app`, `../core`,
  `../platform`, `../resources` on this branch.

## Layout
- `StARApp/App` — entry point, navigation, root view.
- `StARApp/Core` — **shared contracts** (domain types, geometry conventions,
  protocols, design tokens). Authored in WP0; every package depends on these and
  on nothing else cross-package. Don't change signatures without telling WP7.
- `StARApp/Features` — SwiftUI screens (Start, Scan, Confirm, Settings, AR).
- `StARApp/Services` — engine implementations (Registration, ARSession,
  Rendering, OCR, Persistence) behind the `Core/Contracts` protocols.
- `StARApp/Resources` — assets, fonts, String Catalog, AR reference images.
- `Tests` — unit tests (geometry & OCR are mandatory).

## Ground rules
1. Geometry layer is metres/radians; UI is mm/degrees — convert only at edges.
2. Cite the legacy file you ported in a file-header comment.
3. Registration & geometry are safety-critical: port exactly, validate (WP1/WP8).
4. `@MainActor` annotations in `Contracts.swift` are intentional — keep them.
