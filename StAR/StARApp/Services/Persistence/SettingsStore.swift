// SettingsStore.swift — UserDefaults-backed persistence for render style and
// registration thresholds.
//
// Ported from: app/SettingsDialog.cpp + app/MainWindow.cpp (QSettings load/save).
// The legacy app persisted the OverlayRenderer::Style colours, the movement
// thresholds and the AR test-depth-overlay flag via QSettings. WORKPLAN §2 maps
// QSettings → @AppStorage for these tunables (plan history goes to SwiftData).
//
// RenderStyle and RegistrationParameters are Codable; each is stored as a single
// JSON string under a stable key so the schema can evolve without migration.

import Foundation

/// Stable UserDefaults keys. Kept here so the catalog of settings keys is in one
/// place (mirrors the QSettings key strings in the legacy code).
enum SettingsKeys {
    static let renderStyle  = "star.renderStyle.json"
    static let registration = "star.registrationParameters.json"
    static let testDepthOverlay = "star.arTestDepthOverlay"
    /// Mirrors the legacy QSettings "language" key ("fr" / "en").
    static let language = "star.language"
}

/// Plain `UserDefaults` persistence for the two Codable settings structs.
///
/// This is a stateless value type whose methods touch `UserDefaults.standard`
/// (thread-safe). Views typically bind directly with `@AppStorage`; this helper
/// exists so the app can hydrate `AppModel` on launch and so non-view code can
/// load/save without an `@AppStorage` wrapper.
struct SettingsStore: Sendable {
    // UserDefaults is thread-safe but not marked Sendable; this storage is only
    // ever read, so the unchecked annotation is sound.
    nonisolated(unsafe) private let defaults: UserDefaults

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
    }

    // MARK: - Load

    /// Hydrate the two settings structs from disk, falling back to defaults when
    /// a key is absent or undecodable.
    func load() -> (style: RenderStyle, registration: RegistrationParameters) {
        let style = decode(RenderStyle.self, key: SettingsKeys.renderStyle) ?? RenderStyle()
        let reg = decode(RegistrationParameters.self, key: SettingsKeys.registration)
            ?? RegistrationParameters()
        return (style, reg)
    }

    /// Convenience for the app entry point: pull persisted settings into the
    /// observable `AppModel` once on launch. Must run on the main actor because
    /// `AppModel` is `@MainActor`.
    @MainActor
    func hydrate(_ model: AppModel) {
        let loaded = load()
        model.renderStyle = loaded.style
        model.registrationParameters = loaded.registration
    }

    // MARK: - Save

    func save(style: RenderStyle) {
        encode(style, key: SettingsKeys.renderStyle)
    }

    func save(registration: RegistrationParameters) {
        encode(registration, key: SettingsKeys.registration)
    }

    func save(style: RenderStyle, registration: RegistrationParameters) {
        save(style: style)
        save(registration: registration)
    }

    // MARK: - Codable <-> JSON-string helpers

    private func encode<T: Encodable>(_ value: T, key: String) {
        guard let data = try? JSONEncoder().encode(value),
              let json = String(data: data, encoding: .utf8) else { return }
        defaults.set(json, forKey: key)
    }

    private func decode<T: Decodable>(_ type: T.Type, key: String) -> T? {
        guard let json = defaults.string(forKey: key),
              let data = json.data(using: .utf8) else { return nil }
        return try? JSONDecoder().decode(T.self, from: data)
    }
}
