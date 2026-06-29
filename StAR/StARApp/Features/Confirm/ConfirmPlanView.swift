// ConfirmPlanView.swift — confirm / edit the surgical plan before AR overlay.
//
// Ported from: app/ConfirmPlanDialog.{h,cpp} (the whole confirm/edit dialog:
// Scan vs Edit modes, per-field flagging, the sequential-confirm focus chain,
// the Confirmer gate, and readWidgets read-back).
//
// Flow (see App/AppRoute.swift):
//   • Scan mode  (.confirm(.scan)): reached after OCR. Every field is flagged
//     (red) and must be touched/cleared; "Confirmer" is disabled until no flagged
//     field remains on any active side. On confirm → model.path = [.ar].
//   • Edit mode  (.confirm(.edit)): reached from the AR screen's "Modifier le
//     plan". Pre-filled from model.currentPlan; only low-confidence fields are
//     flagged. On confirm → model.path.removeLast() (back to the existing .ar).
//
// Units: the form works in mm / degrees (LeksellTarget's own units); no
// metre/radian conversion happens here.

import SwiftUI

struct ConfirmPlanView: View {
    let mode: ConfirmMode

    @Environment(AppModel.self) private var model
    @Environment(\.modelContext) private var context
    @Environment(\.dismiss) private var dismiss

    @State private var left: SideFormModel
    @State private var right: SideFormModel
    @FocusState private var focusedField: ConfirmFieldFocus?

    init(mode: ConfirmMode) {
        self.mode = mode
        // Seeded from `.defaultTest` so previews/inits are valid; `onAppear`
        // rebuilds from the live `model.currentPlan` (Environment isn't available
        // in `init`).
        let seed = SurgicalPlanDTO.defaultTest
        _left = State(initialValue: SideFormModel(title: "Gauche", tint: Brand.arcBlue,
                                                  target: seed.left, mode: mode))
        _right = State(initialValue: SideFormModel(title: "Droite", tint: Brand.impulseRed,
                                                   target: seed.right, mode: mode))
    }

    var body: some View {
        ZStack {
            Brand.background.ignoresSafeArea()
            RadialGradient(colors: [Brand.arcBlue.opacity(0.12), .clear],
                           center: .topLeading, startRadius: 10, endRadius: 500)
                .ignoresSafeArea()

            VStack(spacing: 18) {
                header
                banner
                ScrollView {
                    GlassEffectContainer(spacing: 16) {
                        VStack(spacing: 16) {
                            // Show the per-side "Activer" toggle in BOTH modes so a
                            // single-electrode case can deactivate the unused side
                            // (matches v1 ConfirmPlanDialog; audit S4-D1). Scan mode
                            // still defaults both sides active.
                            TargetForm(side: left,
                                       showsEnableToggle: true,
                                       focusedField: $focusedField,
                                       sideKind: .left)
                            TargetForm(side: right,
                                       showsEnableToggle: true,
                                       focusedField: $focusedField,
                                       sideKind: .right)
                        }
                    }
                }
                controls
            }
            .padding(20)
        }
        .toolbar(.hidden, for: .navigationBar)
        .navigationBarBackButtonHidden(true)
        .onAppear(perform: rebuildFromModel)
        .onSubmit(advanceFocus)
        .toolbar {
            // The decimal pad has no Return key, so the sequential-confirm chain
            // (legacy AutoSelectSpinBox::chainTo) is driven by a keyboard toolbar
            // button: clears the current field's flag and advances focus.
            ToolbarItemGroup(placement: .keyboard) {
                Spacer()
                Button(isLastField ? "Terminé" : "Suivant", action: advanceFocus)
                    .font(Brand.display(16))
                    .tint(Brand.arcBlue)
            }
        }
    }

    /// True when the focused field is the very last in the chain (right/arc).
    private var isLastField: Bool {
        guard let f = focusedField else { return false }
        return next(after: f) == nil
    }

    // MARK: - Pieces

    private var header: some View {
        Text(mode == .scan ? "Confirmer le plan" : "Modifier le plan")
            .font(Brand.display(26))
            .foregroundStyle(.white)
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder
    private var banner: some View {
        // Mirrors the legacy OCR status banner. In Scan mode the message reflects
        // whether OCR found anything; in Edit mode it nudges to verify.
        let detected = model.currentPlan.hasAny
        let (text, tint): (LocalizedStringKey, Color) = {
            if mode == .edit {
                return ("Vérifiez les valeurs avant de confirmer.", Brand.arcBlue)
            }
            return detected
                ? ("Coordonnées détectées. Vérifiez chaque champ avant de confirmer.", Brand.arcBlue)
                : ("Coordonnées non détectées. Saisissez les valeurs manuellement.", Brand.impulseRed)
        }()
        Text(text)
            .font(.subheadline)
            .foregroundStyle(.white)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .glassEffect(.regular.tint(tint.opacity(0.30)), in: .rect(cornerRadius: 12))
    }

    private var controls: some View {
        GlassEffectContainer(spacing: 14) {
            HStack(spacing: 14) {
                GlassActionButton(title: "Annuler",
                                  systemImage: "xmark",
                                  tint: Brand.neutralGrey) {
                    cancel()
                }
                GlassActionButton(title: "Confirmer",
                                  systemImage: "checkmark",
                                  prominent: true) {
                    confirm()
                }
                .disabled(!canConfirm)
                .opacity(canConfirm ? 1 : 0.5)
            }
        }
    }

    // MARK: - Gating (ConfirmPlanDialog::updateConfirmButton)

    /// Confirmer enabled only when no flagged field remains on any active side,
    /// and at least one side is active (can't confirm an empty plan).
    private var canConfirm: Bool {
        let anyActive = left.isEnabled || right.isEnabled
        return anyActive && (left.flaggedCount + right.flaggedCount == 0)
    }

    // MARK: - Actions

    private func rebuildFromModel() {
        let plan = model.currentPlan
        left = SideFormModel(title: "Gauche", tint: Brand.arcBlue, target: plan.left, mode: mode)
        right = SideFormModel(title: "Droite", tint: Brand.impulseRed, target: plan.right, mode: mode)
        // Scan mode auto-focuses the first field (legacy showEvent focuses m_left.x).
        if mode == .scan {
            focusedField = ConfirmFieldFocus(side: .left, field: .x)
        }
    }

    /// Return / “suivant” on a field: clear its flag (even when the value is
    /// unchanged, like AutoSelectSpinBox::chainTo) and advance to the next field.
    /// Order: left x→y→z→ring→arc → right x→y→z→ring→arc → dismiss keyboard.
    private func advanceFocus() {
        guard let current = focusedField else { return }
        // Commit (clamp + canonicalise) then clear the field's flag if it now
        // holds a value — mirrors AutoSelectSpinBox::chainTo's onConfirm.
        if let entry = currentEntry(for: current) {
            entry.commit()
            entry.clearFlagIfFilled()
        }
        focusedField = next(after: current)
    }

    private func currentEntry(for focus: ConfirmFieldFocus) -> FieldEntry? {
        let side = focus.side == .left ? left : right
        return side.fields.first { $0.field == focus.field }
    }

    private func next(after focus: ConfirmFieldFocus) -> ConfirmFieldFocus? {
        let fields = LeksellTarget.Field.allCases
        if let idx = fields.firstIndex(of: focus.field), idx + 1 < fields.count {
            return ConfirmFieldFocus(side: focus.side, field: fields[idx + 1])
        }
        // End of a side: hop to the right side's first field, else finish.
        if focus.side == .left {
            return ConfirmFieldFocus(side: .right, field: fields.first!)
        }
        return nil   // last field → dismiss keyboard
    }

    private func confirm() {
        // Clamp every field (covers fields edited but never advanced past).
        (left.fields + right.fields).forEach { $0.commit() }
        guard canConfirm else { return }
        var plan = model.currentPlan
        plan.left = left.readBack()
        plan.right = right.readBack()
        model.currentPlan = plan

        // Persist the confirmed plan to SwiftData history.
        PlanStore(context: context).save(plan, label: planLabel(plan))

        switch mode {
        case .scan:
            model.path = [.ar]            // fresh AR session after a scan.
        case .edit:
            if !model.path.isEmpty { model.path.removeLast() }  // back to existing .ar.
        }
    }

    private func cancel() {
        if !model.path.isEmpty { model.path.removeLast() } else { dismiss() }
    }

    private func planLabel(_ plan: SurgicalPlanDTO) -> String {
        var parts: [String] = []
        if plan.hasLeft { parts.append("G") }
        if plan.hasRight { parts.append("D") }
        return parts.joined(separator: "+")
    }
}

#Preview("Scan") {
    NavigationStack { ConfirmPlanView(mode: .scan) }
        .environment(AppModel())
        .preferredColorScheme(.dark)
}

#Preview("Edit") {
    let m = AppModel(); m.currentPlan = .defaultTest
    return NavigationStack { ConfirmPlanView(mode: .edit) }
        .environment(m)
        .preferredColorScheme(.dark)
}
