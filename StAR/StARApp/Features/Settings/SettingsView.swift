// SettingsView.swift — render-style colours, registration thresholds, and the
// AR test-depth-overlay toggle.
//
// Ported from: app/SettingsDialog.cpp (GraphicsSettingsDialog colour pickers +
// AR test-depth toggle; CalibrationSettingsDialog movement thresholds) and
// app/MainWindow.cpp (QSettings load/save). Persisted with @AppStorage via
// SettingsStore; changes are pushed back into AppModel so the AR overlay (WP3)
// picks them up live.
//
// Units: RegistrationParameters is metres/degrees internally; this UI presents
// the translation thresholds in MILLIMETRES and converts at the binding edge.
//
// Notes vs. legacy:
//  • The legacy colour picker was a fixed 3-swatch palette (IMPULSE_RED / ARC_BLUE
//    / VOLT_YELLOW). The native contract stores free hex strings, so this uses
//    SwiftUI ColorPickers (Color <-> hex) — a superset of the legacy behaviour.
//  • The "reprojection threshold" (px) from CalibrationSettingsDialog has no ARKit
//    analogue (image anchors expose no corner pixels); `lockStreakFrames` (native)
//    replaces it as the lock-quality tunable.
//  • Per-tag marker positions ARE restored (audit S4-D2): the "Position des repères"
//    section edits RegistrationParameters.marker{Left,Right}OffsetM in mm, which
//    MarkerFusion applies as the translation of leksell_T_marker.
//  • Language switching in the legacy app quit the process to re-apply the Qt
//    translation; iOS apps localise per system language, so this only shows an
//    informational note rather than relaunching.

import SwiftUI
#if canImport(UIKit)
import UIKit
#endif

struct SettingsView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss

    /// AR test-depth overlay flag — persisted directly via @AppStorage (mirrors
    /// the legacy QSettings "arTestDepthOverlay").
    @AppStorage(SettingsKeys.testDepthOverlay) private var testDepthOverlay = true

    private let store = SettingsStore()

    var body: some View {
        @Bindable var model = model

        ZStack {
            Brand.background.ignoresSafeArea()
            RadialGradient(colors: [Brand.arcBlue.opacity(0.12), .clear],
                           center: .topLeading, startRadius: 10, endRadius: 500)
                .ignoresSafeArea()

            Form {
                graphicsSection(model: model)
                calibrationSection(model: model)
                markersSection(model: model)
                testSection
                languageSection
            }
            .scrollContentBackground(.hidden)
        }
        .navigationTitle("Réglages")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarLeading) {
                Button("Fermer") { close() }
                    .tint(Brand.arcBlue)
            }
        }
        // Persist whenever the model's settings change.
        .onChange(of: model.renderStyle) { _, new in store.save(style: new) }
        .onChange(of: model.registrationParameters) { _, new in store.save(registration: new) }
    }

    // MARK: - Graphics (colours)

    @ViewBuilder
    private func graphicsSection(model: AppModel) -> some View {
        Section {
            ColorPicker("Trajectoire", selection: hexBinding(\.lineColorHex, model: model),
                        supportsOpacity: false)
            ColorPicker("Cible", selection: hexBinding(\.targetColorHex, model: model),
                        supportsOpacity: false)
            ColorPicker("Marqueur d'incision", selection: hexBinding(\.incisionColorHex, model: model),
                        supportsOpacity: false)
        } header: {
            Text("Paramètres graphiques")
        }
        .listRowBackground(glassRowBackground)
    }

    // MARK: - Calibration (registration thresholds)

    @ViewBuilder
    private func calibrationSection(model: AppModel) -> some View {
        Section {
            // Consecutive qualifying frames before locking (replaces the legacy
            // reprojection-threshold tunable).
            Stepper(value: lockStreakBinding(model: model), in: 1...60) {
                LabeledContent("Images de verrouillage",
                               value: "\(model.registrationParameters.lockStreakFrames)")
            }

            // Max two-marker disagreement, shown in mm (stored in m).
            sliderRow(title: "Désaccord max. des repères",
                      valueMM: marker(\.maxMarkerDisagreementM, model: model),
                      range: 0.5...10, step: 0.5, unit: "mm")

            // Movement translation threshold, shown in mm (stored in m).
            sliderRow(title: "Seuil de translation",
                      valueMM: marker(\.moveTransThreshM, model: model),
                      range: 1...50, step: 1, unit: "mm")

            // Movement rotation threshold, degrees (stored in degrees already).
            sliderRow(title: "Seuil de rotation",
                      valueMM: rotBinding(model: model),
                      range: 0.5...10, step: 0.5, unit: "°")
        } header: {
            Text("Paramètres de calibration")
        }
        .listRowBackground(glassRowBackground)
    }

    // MARK: - Marker positions (field-correctable Leksell-frame geometry, mm)

    @ViewBuilder
    private func markersSection(model: AppModel) -> some View {
        Section {
            markerRow("Gauche", \.markerLeftOffsetM, model: model)
            markerRow("Droite", \.markerRightOffsetM, model: model)
        } header: {
            Text("Position des repères")
        } footer: {
            Text("Position physique des repères dans le cadre de Leksell (mm). À corriger uniquement si l'impression diffère de la disposition de référence.")
        }
        .listRowBackground(glassRowBackground)
    }

    @ViewBuilder
    private func markerRow(_ title: LocalizedStringKey,
                           _ keyPath: WritableKeyPath<RegistrationParameters, SIMD3<Float>>,
                           model: AppModel) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title).font(.subheadline).foregroundStyle(.secondary)
            HStack(spacing: 12) {
                axisField("X", markerComp(keyPath, 0, model: model))
                axisField("Y", markerComp(keyPath, 1, model: model))
                axisField("Z", markerComp(keyPath, 2, model: model))
            }
        }
    }

    private func axisField(_ label: String, _ value: Binding<Double>) -> some View {
        HStack(spacing: 4) {
            Text(verbatim: label).font(.caption).foregroundStyle(.secondary)
            TextField(label, value: value, format: .number.precision(.fractionLength(1)))
                .keyboardType(.numbersAndPunctuation)   // allows the leading minus
                .multilineTextAlignment(.trailing)
                .monospacedDigit()
                .frame(maxWidth: .infinity)
            Text(verbatim: "mm").font(.caption2).foregroundStyle(.secondary)
        }
    }

    // MARK: - AR test depth overlay

    private var testSection: some View {
        Section {
            Toggle("Visualisation profondeur test AR", isOn: $testDepthOverlay)
                .tint(Brand.arcBlue)
        } header: {
            Text("Test AR")
        }
        .listRowBackground(glassRowBackground)
    }

    // MARK: - Language note

    private var languageSection: some View {
        Section {
            LabeledContent("Langue", value: Locale.current.language.languageCode?.identifier == "en"
                           ? "English" : "Français")
            Text("La langue suit les réglages du système iOS.")
                .font(.footnote)
                .foregroundStyle(.secondary)
        } header: {
            Text("Langue")
        }
        .listRowBackground(glassRowBackground)
    }

    // MARK: - Shared row styling

    private var glassRowBackground: some View {
        RoundedRectangle(cornerRadius: 12)
            .fill(.clear)
            .glassEffect(.regular, in: .rect(cornerRadius: 12))
    }

    /// A labelled slider whose backing value is presented and edited in display
    /// units (mm or degrees). `valueMM` already converts to/from the stored unit.
    private func sliderRow(title: LocalizedStringKey,
                           valueMM: Binding<Double>,
                           range: ClosedRange<Double>,
                           step: Double,
                           unit: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            LabeledContent(title) {
                Text("\(valueMM.wrappedValue, specifier: "%.1f") \(unit)")
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }
            Slider(value: valueMM, in: range, step: step)
                .tint(Brand.arcBlue)
        }
    }

    // MARK: - Bindings (unit conversion at the edge)

    /// Color <-> hex binding over a `RenderStyle` hex field. Writes back into
    /// `model.renderStyle`, which `.onChange` then persists.
    private func hexBinding(_ keyPath: WritableKeyPath<RenderStyle, String>,
                            model: AppModel) -> Binding<Color> {
        Binding(
            get: { Color(hex: model.renderStyle[keyPath: keyPath]) },
            set: { model.renderStyle[keyPath: keyPath] = $0.toHex() }
        )
    }

    private func lockStreakBinding(model: AppModel) -> Binding<Int> {
        Binding(
            get: { model.registrationParameters.lockStreakFrames },
            set: { model.registrationParameters.lockStreakFrames = $0 }
        )
    }

    /// Metre-stored Float field exposed as a millimetre `Double` for the UI.
    private func marker(_ keyPath: WritableKeyPath<RegistrationParameters, Float>,
                        model: AppModel) -> Binding<Double> {
        Binding(
            get: { Double(model.registrationParameters[keyPath: keyPath]) * 1000.0 },
            set: { model.registrationParameters[keyPath: keyPath] = Float($0 / 1000.0) }
        )
    }

    /// One axis (metre-stored) of a marker offset, exposed as a millimetre `Double`.
    private func markerComp(_ keyPath: WritableKeyPath<RegistrationParameters, SIMD3<Float>>,
                            _ axis: Int, model: AppModel) -> Binding<Double> {
        Binding(
            get: { Double(model.registrationParameters[keyPath: keyPath][axis]) * 1000.0 },
            set: { model.registrationParameters[keyPath: keyPath][axis] = Float($0 / 1000.0) }
        )
    }

    /// Degree-stored Float field exposed as a `Double` (no unit conversion).
    private func rotBinding(model: AppModel) -> Binding<Double> {
        Binding(
            get: { Double(model.registrationParameters.moveRotThreshDeg) },
            set: { model.registrationParameters.moveRotThreshDeg = Float($0) }
        )
    }

    // MARK: - Actions

    private func close() {
        if !model.path.isEmpty { model.path.removeLast() } else { dismiss() }
    }
}

// MARK: - Color → hex

private extension Color {
    /// Resolve to "#RRGGBB" for storage in `RenderStyle`. Round-trips with
    /// `Color(hex:)`. Falls back to black if the platform can't resolve it.
    func toHex() -> String {
        #if canImport(UIKit)
        let ui = UIColor(self)
        var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
        guard ui.getRed(&r, green: &g, blue: &b, alpha: &a) else { return "#000000" }
        let clamp: (CGFloat) -> Int = { Int((min(max($0, 0), 1) * 255).rounded()) }
        return String(format: "#%02X%02X%02X", clamp(r), clamp(g), clamp(b))
        #else
        return "#000000"
        #endif
    }
}

#Preview {
    NavigationStack { SettingsView() }
        .environment(AppModel())
        .preferredColorScheme(.dark)
}
