//  ARViewContainer.swift
//  WP7 — AR screen, HUD & integration (the integrator package).
//
//  Hosts the single, full-screen RealityKit ARView and wires WP2 (the surgical
//  session) and WP3 (the overlay renderer) to it.
//
//  Ported (intent only) from the legacy C++/Qt AR wiring:
//    - app/MainWindow.cpp  — the AR page setup, ARKitSession handoff, the camera
//                            feed widget and its lifecycle.
//    - app/MainWindow.h    — AR page members.
//  The legacy 480×640 boxed `ARWidget` and the rotated `ARRotatedStrip` are GONE:
//  the ARView fills the whole screen edge-to-edge (`.ignoresSafeArea()` is applied
//  by ARScreen) and the controls float over it as Liquid Glass (see HUD.swift).
//
//  Single module "StAR": all Core/* / Services/* types are visible.

import SwiftUI
import RealityKit
import ARKit

/// A `UIViewRepresentable` that hosts the one and only `ARView`.
///
/// The `SurgicalSessionImpl` and `RealityKitOverlay` are constructed and owned by
/// `ARScreen` (as `@State`) so the screen can observe `session.registration`; they
/// are injected here. This guarantees exactly ONE `ARSession` (the ARView's), which
/// WP2 attaches to and WP3 reaches via the coordinator's `arView`.
struct ARViewContainer: UIViewRepresentable {
    let session: SurgicalSessionImpl
    let overlay: RealityKitOverlay

    /// The coordinator owns the `ARView` and bridges WP3's `ARViewProviding` so the
    /// overlay can reach the live `ARView`. (`ARViewProviding` is declared in
    /// `RealityKitOverlay.swift` and refines `ARViewRepresentableHandle`.)
    @MainActor
    final class Coordinator: ARViewProviding {
        let arView: ARView

        init() {
            // Full-screen; the surrounding SwiftUI layout (.ignoresSafeArea) sizes it.
            arView = ARView(frame: .zero)
            // We drive registration ourselves; suppress RealityKit's default coaching
            // / origin debug chrome so the surgeon sees only the feed + overlay.
            arView.automaticallyConfigureSession = false
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeUIView(context: Context) -> ARView {
        let arView = context.coordinator.arView

        // Hand the ARView's single ARSession to WP2, attach WP3 to the same view,
        // then start tracking. Order matters: attach session before start().
        session.attach(session: arView.session)
        overlay.attach(to: context.coordinator)
        session.start()

        return arView
    }

    func updateUIView(_ uiView: ARView, context: Context) {
        // Plan/style are pushed into the session/overlay by ARScreen via setPlan /
        // overlay.update(...) on .onChange; nothing to forward per layout pass.
    }

    /// Tear down the single ARSession and the overlay when the representable is
    /// removed. ARScreen's `.onDisappear` also calls these defensively; both paths
    /// are idempotent (pause() / clear() are safe to call twice).
    static func dismantleUIView(_ uiView: ARView, coordinator: Coordinator) {
        uiView.session.pause()
    }
}
