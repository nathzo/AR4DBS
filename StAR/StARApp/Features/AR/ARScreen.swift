//  ARScreen.swift
//  WP7 — AR screen, HUD & integration (the integrator package).
//
//  Composes the full-screen ARView (ARViewContainer) with the floating Liquid
//  Glass HUD and drives the WP3 overlay from the WP2 registration state.
//
//  Ported (intent only) from the legacy C++/Qt AR page:
//    - app/MainWindow.cpp — the AR page composition, the registration→overlay
//                           handoff, the recalibrate / edit flows, camera handoff.
//    - app/MainWindow.h   — AR page members.
//
//  CAMERA HANDOFF: ScanView owns an AVCaptureSession that it stops in onDisappear,
//  and it routes via .confirm(.scan) before .ar, so by the time this screen appears
//  ScanView is already gone. The AR session is created/started only on this screen's
//  appear (ARViewContainer.makeUIView → session.start()), so the two camera sessions
//  never run simultaneously. The "Mode test AR" path goes straight Start → .ar with
//  no camera in flight. Exactly ONE ARSession exists (the ARView's).
//
//  Single module "StAR": all Core/* / Services/* types are visible.

import SwiftUI

/// The live AR experience: registration HUD + RealityKit overlay over a
/// full-screen, edge-to-edge ARView. Portrait, dark, navigation bar hidden.
struct ARScreen: View {
    @Environment(AppModel.self) private var model

    // Owned here so the view can observe `session.registration`. WP2/WP3 are both
    // @MainActor; constructing them on the main actor (this view) is safe.
    @State private var session = SurgicalSessionImpl()
    @State private var overlay = RealityKitOverlay()

    var body: some View {
        ZStack {
            ARViewContainer(session: session, overlay: overlay)
                .ignoresSafeArea()

            HUD(session: session)
        }
        .background(Brand.background)
        .toolbar(.hidden, for: .navigationBar)
        .navigationBarBackButtonHidden(true)
        .statusBarHidden(true)
        .onAppear {
            // Apply current settings + plan before the session starts tracking.
            session.updateParameters(model.registrationParameters)
            session.setPlan(PlanGeometry.from(model.currentPlan))
        }
        .onChange(of: session.registration) { _, state in
            // Drive the overlay purely from registration: render under the locked
            // Leksell frame, otherwise clear it (awaiting / calibrating).
            if case .locked(let worldToLeksell) = state {
                overlay.update(worldToLeksell: worldToLeksell,
                               geometry: PlanGeometry.from(model.currentPlan),
                               style: model.renderStyle)
            } else {
                overlay.clear()
            }
        }
        .onDisappear {
            // Release the single ARSession + overlay on navigate-away (Menu / edit),
            // avoiding a second camera session and the thermal/battery cost.
            session.stop()
            overlay.clear()
        }
    }
}
