import SwiftUI

/// Top-level navigation, replacing the legacy `QStackedWidget` flow in MainWindow.
/// Owned by WP0; driven by WP5/WP7.
///
///   start → scan → confirm(.scan) → ar
///   start → ar (Mode test AR, uses SurgicalPlanDTO.defaultTest)
///   ar → confirm(.edit) → ar ("Modifier le plan")
public enum AppRoute: Hashable {
    case start
    case scan
    case confirm(mode: ConfirmMode)
    case settings
    case ar
}

public enum ConfirmMode: Hashable {
    /// Right after OCR: every field flagged, confirmed in sequence.
    case scan
    /// From "Modifier le plan": pre-filled, only low-confidence fields flagged.
    case edit
}

/// App-wide navigation + active-plan state. WP0 defines; WP5/WP6/WP7 mutate.
@Observable
@MainActor
public final class AppModel {
    public var path: [AppRoute] = []
    public var currentPlan: SurgicalPlanDTO = .init()
    public var renderStyle: RenderStyle = .init()
    public var registrationParameters: RegistrationParameters = .init()

    public init() {}
}
