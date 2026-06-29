import SwiftUI

/// Start screen — the app's root. Three actions, mirroring the legacy
/// `StartScreen` (Qt): *Nouvelle chirurgie*, *Mode test AR*, *Réglages*.
/// Built on the Liquid Glass styling agreed in `StartMockup`.
///
/// Reference: `app/StartScreen.cpp`, `MainWindow.cpp` (navigation wiring,
/// default-test path).
struct StartView: View {
    @Environment(AppModel.self) private var model
#if DEBUG
    // Long-press the wordmark to open the dev-only on-device registration
    // validation harness (Features/Dev). DEBUG-only; excluded from release.
    @State private var showValidation = false
#endif

    var body: some View {
        ZStack {
            Brand.background.ignoresSafeArea()
            // Subtle brand glow behind the glass.
            RadialGradient(colors: [Brand.arcBlue.opacity(0.18), .clear],
                           center: .top, startRadius: 10, endRadius: 420)
                .ignoresSafeArea()

            VStack(spacing: 0) {
                Spacer()
                VStack(spacing: 8) {
                    // Brand wordmark — not localized. (Logo/neurorestore imagery
                    // lands with WP9 asset conversion.)
                    Text(verbatim: "StAR")
                        .font(Brand.display(64))
                        .foregroundStyle(.white)
#if DEBUG
                        .onLongPressGesture(minimumDuration: 1.0) { showValidation = true }
                        .fullScreenCover(isPresented: $showValidation) { ValidationView() }
#endif
                    Text("NeuroRestore · Stereotactic AR")
                        .font(.subheadline)
                        .foregroundStyle(.white.opacity(0.6))
                }
                Spacer()

                GlassEffectContainer(spacing: 16) {
                    VStack(spacing: 16) {
                        GlassActionButton(title: "Nouvelle chirurgie",
                                          systemImage: "scope",
                                          prominent: true) {
                            model.path.append(.scan)
                        }
                        GlassActionButton(title: "Mode test AR",
                                          systemImage: "arkit",
                                          tint: Brand.voltYellow) {
                            model.currentPlan = .defaultTest
                            model.path.append(.ar)
                        }
                        GlassActionButton(title: "Réglages",
                                          systemImage: "gearshape",
                                          tint: Brand.neutralGrey) {
                            model.path.append(.settings)
                        }
                    }
                }
                .padding(.horizontal, 28)
                .padding(.bottom, 48)
            }
        }
        .toolbar(.hidden, for: .navigationBar)
    }
}

#Preview {
    StartView()
        .environment(AppModel())
        .preferredColorScheme(.dark)
}
