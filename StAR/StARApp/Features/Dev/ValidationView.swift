//  ValidationView.swift
//  DEV-ONLY registration-validation harness — the AR screen.
//
//  Full-screen RealityKit ARView + a Liquid-Glass diagnostics overlay that lets a
//  surgeon/engineer empirically resolve four open audit issues against the physical
//  Leksell jig (StAR/docs/v1-v2-equivalence-audit.md):
//
//    • S1-01  — cycle the candidate marker→Leksell rotations and SEE which one makes
//               the rendered XYZ triad + trajectory match the physical jig.
//    • REG-01 — each marker's face-on angle is shown live.
//    • REG-03 — range-vs-lateral placement error is decomposed in the readout.
//    • S3-02  — the LiDAR confidence at the incision raycast hit is shown live.
//
//  The ARView host follows Features/AR/ARViewContainer.swift: a UIViewRepresentable
//  whose Coordinator owns the ARView and conforms to the production `ARViewProviding`
//  handle. All heavy lifting lives in ValidationSession (live ARKit) and
//  RegistrationDiagnostics (pure math). On the simulator the ARView is replaced by a
//  graceful placeholder. Wrapped in #if DEBUG so it never ships.
//
//  Entry point: the orchestrator presents `ValidationView()` from a DEBUG-only hook
//  in StartView (e.g. a long-press on the wordmark → .fullScreenCover). `init()` is
//  public for that reason.
//
//  Single module "StAR": Core/* / Services/* types are visible.

import SwiftUI
import RealityKit
import ARKit
import simd
import Combine   // Cancellable (scene event subscription)

#if DEBUG
// MARK: - Screen

public struct ValidationView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var session = ValidationSession()
    @State private var note: String = "vue-1"
    @State private var shareURL: ShareURL?

    private let arSupported = ARWorldTrackingConfiguration.isSupported

    /// Leksell-frame skull-side endpoint of the default-test left trajectory — the
    /// analytic reference the LiDAR incision hit is compared against for REG-03.
    private let probeLineEnd = Trajectory.fromLeksell(SurgicalPlanDTO.defaultTest.left).lineEnd

    public init() {}

    public var body: some View {
        ZStack {
            Brand.background.ignoresSafeArea()

            if arSupported {
                ValidationARContainer(session: session)
                    .ignoresSafeArea()
            } else {
                simulatorPlaceholder
            }

            VStack(spacing: 0) {
                diagnosticsPanel
                Spacer(minLength: 0)
                controls
            }
            .padding()
        }
        .statusBarHidden(true)
        .sheet(item: $shareURL) { item in
            ShareSheet(url: item.url)
        }
    }

    // MARK: Simulator placeholder

    private var simulatorPlaceholder: some View {
        VStack(spacing: 12) {
            Image(systemName: "arkit")
                .font(.system(size: 48))
                .foregroundStyle(Brand.neutralGrey)
            Text("AR non disponible sur simulateur")
                .font(Brand.display(18))
                .foregroundStyle(.white)
            Text("Lancer sur un iPhone Pro (LiDAR) avec le gabarit Leksell physique.")
                .font(.footnote.monospaced())
                .foregroundStyle(Brand.neutralGrey)
                .multilineTextAlignment(.center)
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: Diagnostics overlay

    private var diagnosticsPanel: some View {
        let s = session.snapshot
        return ScrollView {
            VStack(alignment: .leading, spacing: 8) {
                row("Tracking", s.trackingStateLabel)
                row("Rotation", session.selectedRotationName, tint: Brand.arcBlue)
                row("Réf. images chargées", "\(s.referenceImageCount)",
                    tint: s.referenceImageCount == 2 ? Brand.arcBlue : Brand.impulseRed)
                row("ARImageAnchors (frame)", "\(s.rawImageAnchorCount)",
                    tint: s.rawImageAnchorCount > 0 ? Brand.arcBlue : Brand.neutralGrey)
                if s.referenceImageCount == 0 {
                    Text("⚠︎ Groupe 'LeksellMarkers' introuvable dans le bundle")
                        .font(.caption2.monospaced()).foregroundStyle(Brand.impulseRed)
                }
                Divider().overlay(Brand.neutralGrey)

                if s.anchors.isEmpty {
                    Text("Aucun marqueur détecté")
                        .font(.caption.monospaced())
                        .foregroundStyle(Brand.neutralGrey)
                }
                ForEach(s.anchors) { a in
                    anchorBlock(a)
                }

                Divider().overlay(Brand.neutralGrey)
                row("Désaccord origine", String(format: "%.2f mm", s.interMarkerOriginMM))
                row("Désaccord orient.", String(format: "%.2f°", s.interMarkerOrientationDeg))

                if let f = s.fusedWorldToLeksell?.translation {
                    row("Leksell origine (m)",
                        String(format: "%+.3f %+.3f %+.3f", f.x, f.y, f.z))
                }

                Divider().overlay(Brand.neutralGrey)
                row("Incision conf. (LiDAR)", s.incisionConfidence.rawValue,
                    tint: confidenceTint(s.incisionConfidence))
                if let inc = s.incisionHitWorld, let w2l = s.fusedWorldToLeksell {
                    // REG-03: the LiDAR-measured incision vs the ANALYTIC line-end the
                    // fused Leksell frame predicts, decomposed into range (depth, along
                    // the camera line of sight) vs lateral. A large range / small
                    // lateral split is the depth-error signature the dropped solvePnP
                    // depth constraint used to bound.
                    let analyticEnd = (w2l * SIMD4<Float>(probeLineEnd, 1)).xyz
                    let err = RegistrationDiagnostics.placementError(
                        estimatedWorld: inc,
                        groundTruthWorld: analyticEnd,
                        cameraPositionWorld: s.cameraPositionWorld)
                    row("Incision LiDAR (m)",
                        String(format: "%+.3f %+.3f %+.3f", inc.x, inc.y, inc.z))
                    row("LiDAR vs analytique (tot/range/lat) mm",
                        String(format: "%.1f / %.1f / %.1f",
                               err.total * 1000, err.rangeAlongCamera * 1000, err.lateral * 1000))
                }
            }
            .padding(14)
        }
        .frame(maxHeight: 320)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 18))
    }

    private func anchorBlock(_ a: DevAnchorInfo) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("\(a.name)  id=\(a.id)  \(a.isTracked ? "tracked" : "lost")")
                .font(.caption.monospaced().bold())
                .foregroundStyle(a.isTracked ? Brand.arcBlue : Brand.impulseRed)
            Text(String(format: "  face-on(+z): %.1f°   scale: %.3f",
                        a.faceOnAngleDeg, a.estimatedScaleFactor))
                .font(.caption2.monospaced()).foregroundStyle(.white)
            Text(String(format: "  axes vs cam  x:%.0f° y:%.0f° z:%.0f°",
                        a.axisAnglesDeg.x, a.axisAnglesDeg.y, a.axisAnglesDeg.z))
                .font(.caption2.monospaced()).foregroundStyle(Brand.voltYellow)
        }
    }

    private func row(_ label: String, _ value: String, tint: Color = .white) -> some View {
        HStack {
            Text(label).font(.caption.monospaced()).foregroundStyle(Brand.neutralGrey)
            Spacer()
            Text(value).font(.caption.monospaced().bold()).foregroundStyle(tint)
        }
    }

    private func confidenceTint(_ c: DevConfidence) -> Color {
        switch c {
        case .high: return Brand.arcBlue
        case .medium: return Brand.voltYellow
        case .low: return Brand.impulseRed
        case .unavailable: return Brand.neutralGrey
        }
    }

    // MARK: Controls

    private var controls: some View {
        GlassEffectContainer(spacing: 12) {
            VStack(spacing: 12) {
                // S1-01: the rotation-candidate cycler.
                Picker("Rotation candidate", selection: rotationBinding) {
                    ForEach(Array(RegistrationDiagnostics.candidateRotations.enumerated()),
                            id: \.offset) { idx, candidate in
                        Text(candidate.name).tag(idx)
                    }
                }
                .pickerStyle(.segmented)
                .disabled(!arSupported)

                TextField("Note / point de vue", text: $note)
                    .font(.callout.monospaced())
                    .textFieldStyle(.roundedBorder)

                HStack(spacing: 12) {
                    GlassActionButton(title: "Capturer échantillon",
                                      systemImage: "camera.metering.spot",
                                      tint: Brand.arcBlue) {
                        session.captureSample(note: note)
                    }
                    GlassActionButton(title: "Exporter CSV",
                                      systemImage: "square.and.arrow.up",
                                      tint: Brand.voltYellow) {
                        if let url = session.exportCSV() { shareURL = ShareURL(url: url) }
                    }
                }

                GlassActionButton(title: "Fermer",
                                  systemImage: "xmark",
                                  tint: Brand.neutralGrey) {
                    session.stop()
                    dismiss()
                }
            }
        }
    }

    private var rotationBinding: Binding<Int> {
        Binding(get: { session.selectedRotationIndex },
                set: { session.selectedRotationIndex = $0 })
    }
}

// MARK: - ARView host (mirrors Features/AR/ARViewContainer.swift)

private struct ValidationARContainer: UIViewRepresentable {
    let session: ValidationSession

    /// Owns the ARView and conforms to the production `ARViewProviding` handle.
    @MainActor
    final class Coordinator: ARViewProviding {
        let arView: ARView
        let session: ValidationSession
        var triadAnchor: AnchorEntity?
        var sceneSub: (any Cancellable)?
        var lastRotationIndex = -1

        init(session: ValidationSession) {
            self.session = session
            arView = ARView(frame: .zero)
            arView.automaticallyConfigureSession = false
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator(session: session) }

    func makeUIView(context: Context) -> ARView {
        let coordinator = context.coordinator
        let arView = coordinator.arView

        session.attach(to: arView)
        session.start()

        // Place a Leksell-frame anchor with an XYZ triad + the default-test
        // trajectory lines; re-pose it every frame from the fused world_T_leksell.
        let anchor = AnchorEntity(world: .zero)
        Self.buildTriad(into: anchor)
        Self.buildTrajectories(into: anchor)
        arView.scene.addAnchor(anchor)
        coordinator.triadAnchor = anchor

        coordinator.sceneSub = arView.scene.subscribe(to: SceneEvents.Update.self) { [weak coordinator] _ in
            MainActor.assumeIsolated {
                guard let coordinator else { return }
                if let w2l = coordinator.session.snapshot.fusedWorldToLeksell {
                    coordinator.triadAnchor?.isEnabled = true
                    coordinator.triadAnchor?.transform = Transform(matrix: w2l)
                } else {
                    coordinator.triadAnchor?.isEnabled = false
                }
            }
        }

        return arView
    }

    func updateUIView(_ uiView: ARView, context: Context) {}

    static func dismantleUIView(_ uiView: ARView, coordinator: Coordinator) {
        coordinator.sceneSub?.cancel()
        uiView.session.pause()
    }

    // MARK: Entity builders (Leksell metres, under the fused anchor)

    /// A 5 cm XYZ axis triad: +X red, +Y green, +Z blue. Thin boxes offset so each
    /// grows from the origin along its axis, letting the user eyeball whether the
    /// Leksell frame orientation matches the physical jig (S1-01).
    private static func buildTriad(into anchor: AnchorEntity) {
        let len: Float = 0.05
        let thick: Float = 0.003
        func axisBox(_ color: UIColor, size: SIMD3<Float>, pos: SIMD3<Float>) -> ModelEntity {
            let e = ModelEntity(mesh: .generateBox(size: size),
                                materials: [UnlitMaterial(color: color)])
            e.position = pos
            return e
        }
        anchor.addChild(axisBox(.systemRed,
                                size: SIMD3(len, thick, thick), pos: SIMD3(len / 2, 0, 0)))
        anchor.addChild(axisBox(.systemGreen,
                                size: SIMD3(thick, len, thick), pos: SIMD3(0, len / 2, 0)))
        anchor.addChild(axisBox(.systemBlue,
                                size: SIMD3(thick, thick, len), pos: SIMD3(0, 0, len / 2)))
    }

    /// The default-test trajectory lines (both sides) so the user sees if the plan
    /// aims sensibly under the selected rotation.
    private static func buildTrajectories(into anchor: AnchorEntity) {
        let plan = SurgicalPlanDTO.defaultTest
        let geo = PlanGeometry.from(plan)
        for traj in [geo.left, geo.right].compactMap({ $0 }) {
            anchor.addChild(lineEntity(from: traj.lineEnd, to: traj.target,
                                       color: UIColor(Brand.impulseRed)))
            // Target sphere.
            let dot = ModelEntity(mesh: .generateSphere(radius: 0.004),
                                  materials: [UnlitMaterial(color: UIColor(Brand.arcBlue))])
            dot.position = traj.target
            anchor.addChild(dot)
        }
    }

    /// A thin cylinder spanning `a → b` (Leksell metres).
    private static func lineEntity(from a: SIMD3<Float>, to b: SIMD3<Float>,
                                   color: UIColor) -> ModelEntity {
        let delta = b - a
        let height = simd_length(delta)
        let e = ModelEntity(mesh: .generateCylinder(height: max(height, 1e-4), radius: 0.0015),
                            materials: [UnlitMaterial(color: color)])
        if height > 1e-6 {
            let dir = delta / height
            e.orientation = safeRotation(from: SIMD3<Float>(0, 1, 0), to: dir)
        }
        e.position = (a + b) * 0.5
        return e
    }

    /// Safe `from→to` rotation. `simd_quatf(from:to:)` is undefined / produces NaN
    /// for (anti-)parallel inputs, and a NaN orientation crashes RealityKit's
    /// transform pipeline. Handle the degenerate cases explicitly.
    private static func safeRotation(from a: SIMD3<Float>, to b: SIMD3<Float>) -> simd_quatf {
        let d = simd_dot(a, b)
        if d >= 0.9999 { return simd_quatf(angle: 0, axis: SIMD3(0, 0, 1)) }   // parallel
        if d <= -0.9999 {                                                       // anti-parallel
            let axis = abs(a.x) < 0.9 ? simd_normalize(simd_cross(a, SIMD3(1, 0, 0)))
                                      : simd_normalize(simd_cross(a, SIMD3(0, 1, 0)))
            return simd_quatf(angle: .pi, axis: axis)
        }
        return simd_quatf(from: a, to: b)
    }
}

// MARK: - Share sheet

private struct ShareURL: Identifiable {
    let id = UUID()
    let url: URL
}

private struct ShareSheet: UIViewControllerRepresentable {
    let url: URL
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: [url], applicationActivities: nil)
    }
    func updateUIViewController(_ controller: UIActivityViewController, context: Context) {}
}
#endif
