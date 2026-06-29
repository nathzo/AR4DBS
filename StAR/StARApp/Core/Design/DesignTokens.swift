import SwiftUI
#if canImport(UIKit)
import UIKit
#endif

/// Brand palette and typography, ported from `OverlayRenderer::Style` and the
/// Qt stylesheets. Owned by WP0/WP9; used by all UI packages.
///
/// Colors resolve from the asset catalog (`Assets.xcassets`) when present and
/// fall back to the literal hex so previews and unit targets work without the
/// catalog. Keep the asset names and hex values in sync.
public enum Brand {
    public static let impulseRed  = Color("ImpulseRed",  fallbackHex: "#DE5F5E") // trajectory line / error
    public static let arcBlue     = Color("ArcBlue",     fallbackHex: "#75D0C5") // target / success / primary action
    public static let voltYellow  = Color("VoltYellow",  fallbackHex: "#E9DF4D") // incision marker
    public static let neutralGrey = Color("NeutralGrey", fallbackHex: "#8A8C8F") // secondary action
    public static let background  = Color.black

    /// Bundled display font (Diagramm-Bold.ttf). Register in Info.plist (UIAppFonts).
    public static func display(_ size: CGFloat) -> Font { .custom("Diagramm-Bold", size: size) }
}

public extension Color {
    /// Asset-catalog color with a hex fallback (so previews/tests work without
    /// the catalog bundle). If the named color is missing, the asset system
    /// resolves to clear, so we detect the main bundle's catalog explicitly.
    init(_ name: String, fallbackHex hex: String) {
        #if canImport(UIKit)
        if UIColor(named: name) != nil { self = Color(name); return }
        #endif
        self = Color(hex: hex)
    }

    /// Hex → Color. Accepts "#RRGGBB" / "RRGGBB".
    init(hex: String) {
        let s = hex.hasPrefix("#") ? String(hex.dropFirst()) : hex
        var v: UInt64 = 0
        Scanner(string: s).scanHexInt64(&v)
        self = Color(
            red: Double((v >> 16) & 0xFF) / 255,
            green: Double((v >> 8) & 0xFF) / 255,
            blue: Double(v & 0xFF) / 255
        )
    }
}
