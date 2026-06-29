//  HUD.swift
//  WP7 — AR screen, HUD & integration.
//
//  The Liquid Glass overlay that floats over the full-screen ARView: a top status
//  pill colour-coded by the registration state, and a bottom button row
//  (Recalibrer / back-to-Menu and Modifier le plan). A success haptic fires when
//  registration transitions into `.locked`.
//
//  Ported (intent only) from the legacy C++/Qt AR UI:
//    - app/MainWindow.cpp — the lock-state status label + colour, the "Recalibrer"
//                           and "Modifier le plan" actions, "← Menu" back nav.
//    - app/MainWindow.h   — AR page button members.
//  The legacy ARRotatedStrip (a Qt rotation hack) is GONE; this is native portrait
//  Liquid Glass anchored with safe-area-aware padding.
//
//  Single module "StAR": Brand / GlassActionButton / RegistrationState all visible.

import SwiftUI

/// SwiftUI overlay drawn on top of `ARViewContainer` (via a ZStack in `ARScreen`).
struct HUD: View {
    @Environment(AppModel.self) private var model

    /// The live WP2 session. `@Bindable`-free read access is enough — we only read
    /// `registration` (an `@Observable` property, so the view re-renders on change).
    let session: SurgicalSessionImpl

    var body: some View {
        VStack {
            statusPill
                .padding(.top, 12)
            Spacer()
            controls
                .padding(.horizontal, 20)
                .padding(.bottom, 32)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        // Haptic on the awaiting/calibrating → locked transition.
        .sensoryFeedback(.success, trigger: isLocked) { _, locked in locked }
    }

    // MARK: - Status pill (top)

    private var statusPill: some View {
        HStack(spacing: 8) {
            Image(systemName: statusSymbol)
            statusLabel
                .font(Brand.display(15))
        }
        .foregroundStyle(.white)
        .padding(.horizontal, 18)
        .padding(.vertical, 12)
        .glassEffect(.regular.tint(statusTint), in: .capsule)
        .animation(.easeInOut(duration: 0.25), value: statusSymbol)
    }

    /// The localized status text. For the calibrating phase the verbatim catalog
    /// key ("Calibration en cours…") is rendered, then a locale-neutral n/total
    /// progress badge is appended outside the catalog.
    @ViewBuilder
    private var statusLabel: some View {
        switch session.registration {
        case .awaitingMarkers:
            Text("Calibration requise : placez la caméra face aux repères")
        case .calibrating(let progress, let total):
            // HStack (not Text + Text, deprecated in iOS 26; not interpolation, which
            // would forge a new catalog key) keeps the exact "Calibration en cours…"
            // key and appends a locale-neutral progress badge.
            HStack(spacing: 6) {
                Text("Calibration en cours…")
                Text(verbatim: "\(progress)/\(total)")
            }
        case .locked:
            Text("Calibration réussie, repères verrouillés")
        }
    }

    // MARK: - Controls (bottom)

    private var controls: some View {
        GlassEffectContainer(spacing: 14) {
            HStack(spacing: 14) {
                // Locked → "Recalibrer" (drop the lock). Unlocked → "← Menu" (home).
                if isLocked {
                    GlassActionButton(title: "Recalibrer",
                                      systemImage: "arrow.counterclockwise",
                                      tint: Brand.neutralGrey) {
                        session.resetRegistration()
                    }
                } else {
                    GlassActionButton(title: "← Menu",
                                      systemImage: "chevron.left",
                                      tint: Brand.neutralGrey) {
                        model.path = []
                    }
                }

                GlassActionButton(title: "Modifier le plan",
                                  systemImage: "slider.horizontal.3",
                                  prominent: true) {
                    model.path.append(.confirm(mode: .edit))
                }
            }
        }
    }

    // MARK: - Derived state

    private var isLocked: Bool {
        if case .locked = session.registration { return true }
        return false
    }

    private var statusTint: Color {
        switch session.registration {
        case .awaitingMarkers: return Brand.impulseRed
        case .calibrating:     return .orange
        case .locked:          return Brand.arcBlue
        }
    }

    private var statusSymbol: String {
        switch session.registration {
        case .awaitingMarkers: return "viewfinder"
        case .calibrating:     return "dot.radiowaves.left.and.right"
        case .locked:          return "checkmark.seal.fill"
        }
    }
}
