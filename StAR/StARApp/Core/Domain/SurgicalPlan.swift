import Foundation
import SwiftData

/// A complete surgical plan: up to two electrode trajectories (left + right).
///
/// `SurgicalPlanDTO` is the value type that flows through the live pipeline
/// (OCR → confirm → AR). `SurgicalPlanRecord` is the SwiftData-persisted form
/// kept for history. Keep the pipeline on the DTO; convert to/from the record
/// only at the persistence boundary (WP6 / Persistence service).
public struct SurgicalPlanDTO: Codable, Equatable, Sendable {
    public var left: LeksellTarget
    public var right: LeksellTarget

    public init(left: LeksellTarget = .init(), right: LeksellTarget = .init()) {
        self.left = left
        self.right = right
    }

    public var hasLeft: Bool  { left.isValid }
    public var hasRight: Bool { right.isValid }
    public var hasAny: Bool   { left.isValid || right.isValid }

    /// Default test targets (Leksell mm / degrees) — ported verbatim from
    /// `MainWindow.cpp defaultTestPlan()`. Used by "Mode test AR".
    public static let defaultTest = SurgicalPlanDTO(
        left: LeksellTarget(xMM: 140.4, yMM: 114.6, zMM: 80.0,
                            ringDeg: 74.2, arcDeg: 71.0, isValid: true,
                            confidence: Array(repeating: 1, count: 5)),
        right: LeksellTarget(xMM: 66.2, yMM: 118.2, zMM: 77.4,
                             ringDeg: 66.8, arcDeg: 111.1, isValid: true,
                             confidence: Array(repeating: 1, count: 5)))
}

/// SwiftData history record. Owned by WP6 (Persistence). Other packages should
/// depend on `SurgicalPlanDTO`, not this type.
@Model
public final class SurgicalPlanRecord {
    public var createdAt: Date
    public var label: String
    /// Encoded `SurgicalPlanDTO` (kept as data so the DTO stays the single source of truth).
    public var payload: Data

    public init(createdAt: Date = .now, label: String = "", payload: Data) {
        self.createdAt = createdAt
        self.label = label
        self.payload = payload
    }
}
