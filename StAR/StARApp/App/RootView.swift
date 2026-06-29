import SwiftUI

/// Root navigation host. WP0 ships this with placeholder destinations; WP5/WP6/WP7
/// replace the placeholders with the real screens (keep the `switch` shape).
struct RootView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        @Bindable var model = model
        NavigationStack(path: $model.path) {
            StartView()
                .navigationDestination(for: AppRoute.self) { route in
                    destination(for: route)
                }
        }
        .tint(Brand.arcBlue)
    }

    @ViewBuilder
    private func destination(for route: AppRoute) -> some View {
        switch route {
        case .start:                StartView()                   // WP5
        case .scan:                 ScanView()                    // WP5
        case .confirm(let mode):    ConfirmPlanView(mode: mode)    // WP6
        case .settings:             SettingsView()                // WP6
        case .ar:                   ARScreen()                    // WP7
        }
    }
}
