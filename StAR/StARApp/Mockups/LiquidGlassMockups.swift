import SwiftUI

// ─────────────────────────────────────────────────────────────────────────────
// Liquid Glass mockups (iOS 26).
//
// Visual design study for the StAR rewrite using Apple's Liquid Glass design
// language: `GlassEffectContainer`, `.glassEffect(_:in:)`, `.buttonStyle(.glass)`
// / `.glassProminent`. These are NON-FUNCTIONAL previews to agree the look &
// feel — they reuse the real domain types (`SurgicalPlanDTO`, `LeksellTarget`,
// `Brand`) so the styling transfers directly into WP5/WP6/WP7.
//
// The app targets iOS 26, so Liquid Glass is used directly (no availability
// gating). WP5–WP7 build their real screens on this styling.
// ─────────────────────────────────────────────────────────────────────────────

// MARK: - Reusable glass pieces
//
// `GlassActionButton` now lives in `Features/Shared/GlassControls.swift` (WP5)
// and is reused here so the mockup and the real screens stay in sync.

/// Faux camera feed so the AR HUD mockup reads as a live view.
private struct FauxCameraBackdrop: View {
    var body: some View {
        ZStack {
            LinearGradient(colors: [Color(hex: "#1A1F1E"), Color(hex: "#0A0C0C")],
                           startPoint: .top, endPoint: .bottom)
            // Hint of a head / LiDAR mesh.
            RadialGradient(colors: [Color(hex: "#2B3A38").opacity(0.9), .clear],
                           center: .center, startRadius: 20, endRadius: 260)
            GeometryReader { geo in
                Path { p in
                    let step: CGFloat = 34
                    for x in stride(from: 0, through: geo.size.width, by: step) {
                        p.move(to: .init(x: x, y: 0)); p.addLine(to: .init(x: x, y: geo.size.height))
                    }
                    for y in stride(from: 0, through: geo.size.height, by: step) {
                        p.move(to: .init(x: 0, y: y)); p.addLine(to: .init(x: geo.size.width, y: y))
                    }
                }
                .stroke(Color.white.opacity(0.04), lineWidth: 0.5)
            }
        }
        .ignoresSafeArea()
    }
}

// MARK: - 1. Start screen

struct StartMockup: View {
    var body: some View {
        ZStack {
            Brand.background.ignoresSafeArea()
            // subtle brand glow behind the glass
            RadialGradient(colors: [Brand.arcBlue.opacity(0.18), .clear],
                           center: .top, startRadius: 10, endRadius: 420)
                .ignoresSafeArea()

            VStack(spacing: 0) {
                Spacer()
                VStack(spacing: 8) {
                    Text("StAR").font(Brand.display(64)).foregroundStyle(.white)
                    Text("NeuroRestore · Stereotactic AR")
                        .font(.subheadline).foregroundStyle(.white.opacity(0.6))
                }
                Spacer()

                GlassEffectContainer(spacing: 16) {
                    VStack(spacing: 16) {
                        GlassActionButton(title: "Nouvelle chirurgie",
                                          systemImage: "scope",
                                          prominent: true)
                        GlassActionButton(title: "Mode test AR",
                                          systemImage: "arkit",
                                          tint: Brand.voltYellow)
                        GlassActionButton(title: "Réglages",
                                          systemImage: "gearshape",
                                          tint: Brand.neutralGrey)
                    }
                }
                .padding(.horizontal, 28)
                .padding(.bottom, 48)
            }
        }
    }
}

// MARK: - 2. AR HUD

struct ARHUDMockup: View {
    enum Calibration { case awaiting, calibrating(Int), locked }
    @State private var state: Calibration = .locked

    var body: some View {
        ZStack {
            FauxCameraBackdrop()
            TrajectoryOverlay()           // the RealityKit overlay, faked in 2D for the mockup
                .padding(.horizontal, 40)
                .opacity(isLocked ? 1 : 0.25)

            VStack {
                statusPill.padding(.top, 12)
                Spacer()
                controls.padding(.bottom, 24)
            }
            .padding(.horizontal, 16)
        }
        // preview-only state switcher
        .overlay(alignment: .topTrailing) {
            Menu("state") {
                Button("Awaiting") { state = .awaiting }
                Button("Calibrating") { state = .calibrating(6) }
                Button("Locked") { state = .locked }
            }
            .font(.caption2).tint(.white.opacity(0.5)).padding(20)
        }
    }

    private var isLocked: Bool { if case .locked = state { return true }; return false }

    @ViewBuilder private var statusPill: some View {
        let (text, tint, icon) = statusInfo
        HStack(spacing: 10) {
            Image(systemName: icon)
            Text(text).font(Brand.display(15))
        }
        .foregroundStyle(.white)
        .padding(.horizontal, 18).padding(.vertical, 12)
        .glassEffect(.regular.tint(tint.opacity(0.45)), in: .capsule)
    }

    private var statusInfo: (String, Color, String) {
        switch state {
        case .awaiting:           return ("Calibration requise : caméra face aux repères", Brand.impulseRed, "viewfinder")
        case .calibrating(let n): return ("Calibration en cours… \(n)/10", Brand.voltYellow, "dot.radiowaves.left.and.right")
        case .locked:             return ("Repères verrouillés", Brand.arcBlue, "lock.fill")
        }
    }

    @ViewBuilder private var controls: some View {
        GlassEffectContainer(spacing: 14) {
            HStack(spacing: 14) {
                GlassActionButton(title: isLocked ? "Recalibrer" : "← Menu",
                                  systemImage: isLocked ? "arrow.clockwise" : "chevron.left",
                                  tint: Brand.neutralGrey)
                GlassActionButton(title: "Modifier le plan",
                                  systemImage: "slider.horizontal.3",
                                  prominent: true)
            }
        }
    }
}

/// 2-D stand-in for the RealityKit trajectory/target/incision overlay.
private struct TrajectoryOverlay: View {
    var body: some View {
        GeometryReader { geo in
            let top = CGPoint(x: geo.size.width * 0.46, y: geo.size.height * 0.18)
            let incision = CGPoint(x: geo.size.width * 0.5, y: geo.size.height * 0.42)
            let target = CGPoint(x: geo.size.width * 0.55, y: geo.size.height * 0.82)
            ZStack {
                Path { p in p.move(to: top); p.addLine(to: target) }
                    .stroke(Brand.impulseRed, style: .init(lineWidth: 5, lineCap: .round))
                    .shadow(color: Brand.impulseRed.opacity(0.6), radius: 8)
                Circle().fill(Brand.voltYellow).frame(width: 18, height: 18)
                    .position(incision).shadow(color: Brand.voltYellow, radius: 6)
                Circle().stroke(Brand.arcBlue, lineWidth: 4).frame(width: 26, height: 26)
                    .position(target).shadow(color: Brand.arcBlue, radius: 6)
            }
        }
    }
}

// MARK: - 3. Confirm / edit plan

struct ConfirmPlanMockup: View {
    private let plan = SurgicalPlanDTO.defaultTest

    var body: some View {
        ZStack {
            Brand.background.ignoresSafeArea()
            RadialGradient(colors: [Brand.arcBlue.opacity(0.12), .clear],
                           center: .topLeading, startRadius: 10, endRadius: 500).ignoresSafeArea()

            VStack(spacing: 20) {
                Text("Confirmer le plan").font(Brand.display(26)).foregroundStyle(.white)
                    .frame(maxWidth: .infinity, alignment: .leading)

                GlassEffectContainer(spacing: 16) {
                    HStack(spacing: 16) {
                        sideCard("Gauche", plan.left, tint: Brand.arcBlue)
                        sideCard("Droite", plan.right, tint: Brand.impulseRed)
                    }
                }
                Spacer()
                GlassActionButton(title: "Confirmer", systemImage: "checkmark", prominent: true)
            }
            .padding(24)
        }
    }

    private func sideCard(_ title: String, _ t: LeksellTarget, tint: Color) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title).font(Brand.display(19)).foregroundStyle(tint)
            field("X", t.xMM, "mm"); field("Y", t.yMM, "mm"); field("Z", t.zMM, "mm")
            field("Ring", t.ringDeg, "°"); field("Arc", t.arcDeg, "°")
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(.regular, in: .rect(cornerRadius: 24))
    }

    private func field(_ label: String, _ value: Double, _ unit: String) -> some View {
        HStack {
            Text(label).font(.subheadline).foregroundStyle(.white.opacity(0.6))
            Spacer()
            Text("\(value, specifier: "%.1f") \(unit)")
                .font(Brand.display(16)).foregroundStyle(.white)
                .monospacedDigit()
        }
        .padding(.vertical, 10).padding(.horizontal, 14)
        .glassEffect(.regular, in: .rect(cornerRadius: 12))
    }
}

// MARK: - Previews

#Preview("Start") { StartMockup() }
#Preview("AR HUD") { ARHUDMockup() }
#Preview("Confirm Plan") { ConfirmPlanMockup() }
