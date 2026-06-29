import SwiftUI
import SwiftData

/// App entry point. Owned by WP0.
/// Injects the app-wide `AppModel` and the SwiftData container.
@main
struct StARApp: App {
    @State private var model = AppModel()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(model)
                .preferredColorScheme(.dark)
                // Hydrate persisted render style + registration thresholds (WP6).
                .task { SettingsStore().hydrate(model) }
        }
        .modelContainer(for: SurgicalPlanRecord.self)
    }
}
