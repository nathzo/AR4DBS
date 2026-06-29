// TargetForm.swift — reusable per-side Leksell-target field form.
//
// Ported from: app/ConfirmPlanDialog.cpp (buildSide / makeSpinBox / clearFlag /
// updateConfirmButton / readWidgets). One TargetForm renders the 5 numeric
// fields (X/Y/Z mm, Ring/Arc deg) for one side ("Gauche" / "Droite") with:
//   • per-field red flagging (low confidence in Edit mode, all fields in Scan mode),
//   • a focus chain so Return/“suivant” advances to the next field and clears the
//     current field's flag (mirrors AutoSelectSpinBox::chainTo),
//   • sane mm/deg range clamping so a fat-fingered entry can't produce a wild
//     trajectory (mirrors makeSpinBox setRange),
//   • a per-side "Activer" toggle (LeksellTarget.isValid).
//
// Units stay mm / degrees here — the same units LeksellTarget exposes — so no
// metre/radian conversion happens at this layer.

import SwiftUI

/// One editable Leksell field. `value == nil` is the "—" sentinel (not yet
/// entered), matching the legacy spinbox minimum (-1) special value.
@Observable
final class FieldEntry: Identifiable {
    let field: LeksellTarget.Field
    var id: LeksellTarget.Field { field }
    /// Editable raw text — the source of truth so partial input like "140." can
    /// be typed. Empty string = "—" sentinel (not yet entered).
    var text: String
    /// Whether this field is currently flagged (must be confirmed/cleared).
    var flagged: Bool
    /// Original OCR confidence (for coloring the cleared state), if any.
    let confidence: Float?

    init(field: LeksellTarget.Field, value: Double?, flagged: Bool, confidence: Float?) {
        self.field = field
        self.flagged = flagged
        self.confidence = confidence
        if let v = value {
            self.text = (v == v.rounded()) ? String(Int(v)) : String(format: "%.1f", v)
        } else {
            self.text = ""
        }
    }

    /// Parsed numeric value (locale-tolerant), or `nil` if empty / unparsable.
    var value: Double? {
        let t = text.trimmingCharacters(in: .whitespaces).replacingOccurrences(of: ",", with: ".")
        return t.isEmpty ? nil : Double(t)
    }

    /// Inclusive sane range for this field (mirrors makeSpinBox setRange maxima).
    var range: ClosedRange<Double> {
        switch field {
        case .x:    return 0...300   // mm
        case .y:    return 0...300   // mm
        case .z:    return 0...200   // mm
        case .ring: return 0...360   // deg
        case .arc:  return 0...180   // deg
        }
    }

    var label: String {
        switch field {
        case .x:    return "X"
        case .y:    return "Y"
        case .z:    return "Z"
        case .ring: return String(localized: "Ring")
        case .arc:  return String(localized: "Arc")
        }
    }

    var unit: String {
        switch field {
        case .x, .y, .z: return "mm"
        case .ring, .arc: return "°"
        }
    }

    /// True once the field holds a real (parsable) value — the condition the
    /// legacy `clearFlag` required before lowering a flag.
    var hasValue: Bool { value != nil }

    /// Commit-time normalisation, called when the field loses focus / the chain
    /// advances. Clamps the value into the sane range (so confirm can't pass a
    /// wild trajectory) and rewrites the text canonically. Mirrors makeSpinBox's
    /// setRange clamping.
    func commit() {
        guard let v = value else { text = ""; return }
        let clamped = min(max(v, range.lowerBound), range.upperBound)
        text = (clamped == clamped.rounded()) ? String(Int(clamped)) : String(format: "%.1f", clamped)
    }

    /// Mirrors `ConfirmPlanDialog::clearFlag`: lower the flag once a real value
    /// is present. No-op while still at the "—" sentinel.
    func clearFlagIfFilled() {
        guard flagged, hasValue else { return }
        flagged = false
    }
}

/// The editable model for one side. Owns its 5 `FieldEntry`s and the enabled flag.
@Observable
final class SideFormModel: Identifiable {
    let id = UUID()
    let title: LocalizedStringKey
    let tint: Color
    var isEnabled: Bool
    var fields: [FieldEntry]

    init(title: LocalizedStringKey, tint: Color, target: LeksellTarget, mode: ConfirmMode) {
        self.title = title
        self.tint = tint
        // Legacy ConfirmPlanDialog defaults the "Activer" checkbox to checked for
        // BOTH sides in Scan mode (every field is then flagged and must be
        // confirmed). In Edit mode a side starts active iff it was already valid.
        switch mode {
        case .scan: self.isEnabled = true
        case .edit: self.isEnabled = target.isValid
        }
        self.fields = LeksellTarget.Field.allCases.map { field in
            let conf = target.confidence(for: field)
            let detected = conf != nil          // confidence == nil → not detected ("—")
            let value: Double? = detected ? Self.rawValue(of: field, in: target) : nil
            let flagged = Self.shouldFlag(mode: mode, confidence: conf)
            return FieldEntry(field: field, value: value, flagged: flagged, confidence: conf)
        }
    }

    private static func rawValue(of field: LeksellTarget.Field, in t: LeksellTarget) -> Double {
        switch field {
        case .x:    return t.xMM
        case .y:    return t.yMM
        case .z:    return t.zMM
        case .ring: return t.ringDeg
        case .arc:  return t.arcDeg
        }
    }

    /// Flag rule ported from `ConfirmPlanDialog::buildSide`:
    ///  • Scan mode: flag EVERY field (surgeon confirms each one).
    ///  • Edit mode: flag only fields below `kConfidenceThreshold` (incl. nil).
    static func shouldFlag(mode: ConfirmMode, confidence: Float?) -> Bool {
        switch mode {
        case .scan:
            return true
        case .edit:
            guard let c = confidence else { return true }  // not detected → flag
            return c < ConfirmThresholds.confidence
        }
    }

    /// Flagged-field count, but only when the side is active — matching
    /// `updateConfirmButton`'s `countFlagged` which ignores disabled sides.
    var flaggedCount: Int {
        guard isEnabled else { return 0 }
        return fields.reduce(0) { $0 + ($1.flagged ? 1 : 0) }
    }

    /// Read the edited values back into a `LeksellTarget`.
    /// Mirrors `ConfirmPlanDialog::readWidgets`: filled fields get confidence 1.0;
    /// sentinel fields keep their default (0). `isValid = isEnabled`.
    func readBack() -> LeksellTarget {
        var t = LeksellTarget()
        t.isValid = isEnabled
        for entry in fields {
            guard let v = entry.value else { continue }
            switch entry.field {
            case .x:    t.xMM = v
            case .y:    t.yMM = v
            case .z:    t.zMM = v
            case .ring: t.ringDeg = v
            case .arc:  t.arcDeg = v
            }
            t.confidence[entry.field.rawValue] = 1.0
        }
        return t
    }
}

/// Shared thresholds ported from `ConfirmPlanDialog.cpp`.
enum ConfirmThresholds {
    /// `kConfidenceThreshold = 0.99f` in ConfirmPlanDialog.cpp — Edit mode flags
    /// any field whose OCR confidence is below this.
    static let confidence: Float = 0.99
}

// MARK: - View

/// Renders one side. Pure subview: all state lives in the injected `SideFormModel`.
/// `focusedField` is hoisted to the parent so the focus chain can hop between the
/// two sides if needed and so the parent can drive initial focus in Scan mode.
struct TargetForm: View {
    @Bindable var side: SideFormModel
    /// Whether to show the per-side "Activer" toggle (Edit mode shows it so a
    /// surgeon can drop a side; Scan mode keeps both sides as scanned).
    var showsEnableToggle: Bool = true
    @FocusState.Binding var focusedField: ConfirmFieldFocus?
    let sideKind: ConfirmSideKind

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text(side.title)
                    .font(Brand.display(19))
                    .foregroundStyle(side.tint)
                Spacer()
                if showsEnableToggle {
                    Toggle("Activer", isOn: $side.isEnabled)
                        .labelsHidden()
                        .tint(Brand.arcBlue)
                }
            }

            ForEach(side.fields) { entry in
                fieldRow(entry)
                    .opacity(side.isEnabled ? 1 : 0.35)
                    .allowsHitTesting(side.isEnabled)
            }
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(.regular, in: .rect(cornerRadius: 24))
    }

    @ViewBuilder
    private func fieldRow(_ entry: FieldEntry) -> some View {
        let focus = ConfirmFieldFocus(side: sideKind, field: entry.field)
        HStack(spacing: 10) {
            Text(entry.label)
                .font(.subheadline)
                .foregroundStyle(.white.opacity(0.7))
                .frame(width: 44, alignment: .leading)

            TextField("—", text: Binding(
                get: { entry.text },
                set: { newText in
                    entry.text = newText
                    // Legacy ConfirmPlanDialog cleared a flag on any valueChanged
                    // once a real value was present — mirror that live.
                    entry.clearFlagIfFilled()
                }
            ))
                .keyboardType(.decimalPad)
                .multilineTextAlignment(.trailing)
                .monospacedDigit()
                .font(Brand.display(16))
                .foregroundStyle(.white)
                .focused($focusedField, equals: focus)
                .submitLabel(.next)

            Text(entry.unit)
                .font(.subheadline)
                .foregroundStyle(.white.opacity(0.5))
                .frame(width: 28, alignment: .leading)
        }
        .padding(.vertical, 10)
        .padding(.horizontal, 14)
        .glassEffect(.regular, in: .rect(cornerRadius: 12))
        .overlay {
            // Flagged fields show the IMPULSE_RED border (legacy kFlaggedStyle);
            // cleared fields show a subtle confirmation tint.
            RoundedRectangle(cornerRadius: 12)
                .strokeBorder(entry.flagged ? Brand.impulseRed : Brand.arcBlue.opacity(0.25),
                              lineWidth: entry.flagged ? 1.5 : 1)
        }
    }

}

/// Identifies which side a focus target belongs to.
enum ConfirmSideKind: Hashable { case left, right }

/// Hashable focus key spanning both sides and all 5 fields, so the parent can
/// build the sequential focus chain (x → y → z → ring → arc).
struct ConfirmFieldFocus: Hashable {
    let side: ConfirmSideKind
    let field: LeksellTarget.Field
}
