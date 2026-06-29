import SwiftUI

/// Reusable Liquid Glass controls shared across the SwiftUI screens (WP5–WP7).
/// Promotes the styling agreed in `Mockups/LiquidGlassMockups.swift` into real,
/// reusable components so Start / Scan / AR / Confirm stay visually consistent.

/// A full-width Liquid Glass action button. Use inside a `GlassEffectContainer`
/// so multiple buttons share the glass morph.
struct GlassActionButton: View {
    let title: LocalizedStringKey
    var systemImage: String? = nil
    var prominent = false
    var tint: Color = Brand.arcBlue
    var busy = false
    var action: () -> Void = {}

    var body: some View {
        Button(action: action) {
            HStack(spacing: 10) {
                if busy {
                    ProgressView().controlSize(.small).tint(.white)
                } else if let systemImage {
                    Image(systemName: systemImage)
                }
                Text(title).font(Brand.display(17))
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 16)
        }
        .tint(tint)
        .disabled(busy)
        .modifier(GlassButtonStyle(prominent: prominent))
    }
}

/// Picks `.glassProminent` for the primary action and `.glass` otherwise.
private struct GlassButtonStyle: ViewModifier {
    let prominent: Bool
    func body(content: Content) -> some View {
        if prominent { content.buttonStyle(.glassProminent) }
        else { content.buttonStyle(.glass) }
    }
}
