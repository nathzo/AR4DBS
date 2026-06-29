// PlanStore.swift — SwiftData persistence for surgical plans.
//
// Ported from: app/MainWindow.cpp (QSettings-based plan persistence) and the
// SurgicalPlan flow. The legacy app used QSettings; per WORKPLAN §2 the native
// rewrite stores plan history in SwiftData (settings move to @AppStorage).
//
// SurgicalPlanDTO is the single source of truth: it is JSON-encoded into the
// SurgicalPlanRecord.payload so the DTO stays authoritative and the record is a
// thin history wrapper.

import Foundation
import SwiftData

/// @MainActor helper over SwiftData for persisting / loading surgical plans.
///
/// `ModelContext` is not `Sendable`, so this type is pinned to the main actor —
/// SwiftUI views read `@Environment(\.modelContext)` (already main-actor) and
/// hand it here.
@MainActor
struct PlanStore {
    private let context: ModelContext

    init(context: ModelContext) {
        self.context = context
    }

    /// Encode `dto` and append a new history record. Failures to encode are
    /// swallowed (a plan that can't be archived must not block the surgical flow);
    /// the live `AppModel.currentPlan` remains the source of truth either way.
    func save(_ dto: SurgicalPlanDTO, label: String = "") {
        guard let payload = try? JSONEncoder().encode(dto) else { return }
        let record = SurgicalPlanRecord(createdAt: .now, label: label, payload: payload)
        context.insert(record)
        try? context.save()
    }

    /// Most-recent-first decoded plans, capped at `limit`. Undecodable records
    /// are skipped rather than throwing.
    func recentPlans(limit: Int = 20) -> [SurgicalPlanDTO] {
        var descriptor = PlanFetch.descriptor()
        descriptor.fetchLimit = max(0, limit)
        guard let records = try? context.fetch(descriptor) else { return [] }
        let decoder = JSONDecoder()
        return records.compactMap { try? decoder.decode(SurgicalPlanDTO.self, from: $0.payload) }
    }
}

/// Fetch-descriptor factory kept separate so the sort key stays in one place.
private enum PlanFetch {
    static func descriptor() -> FetchDescriptor<SurgicalPlanRecord> {
        FetchDescriptor<SurgicalPlanRecord>(
            sortBy: [SortDescriptor(\.createdAt, order: .reverse)]
        )
    }
}
