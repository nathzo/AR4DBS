import SwiftUI

/// Scan screen — full-screen, edge-to-edge live viewfinder with floating Liquid
/// Glass controls. Aim at the Vantage planning monitor and tap *Capturer*; the
/// frame is OCR'd by the injected `PlanScanning` engine (WP4) and the result
/// flows to the confirm screen. *Annuler* returns to Start.
///
/// Reference: `app/ScanScreen.cpp` (live preview + capture → planDetected),
/// `MainWindow.cpp` (navigation wiring). The legacy 480×640 boxed feed and the
/// rotated control strip are intentionally dropped (see WP5 UI principles).
struct ScanView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.planScanner) private var scanner

    @State private var camera = CameraController()
    @State private var isScanning = false

    var body: some View {
        ZStack {
            CameraPreviewView(session: camera.session)
                .ignoresSafeArea()

            framingGuide

            VStack {
                Spacer()
                controls
                    .padding(.horizontal, 20)
                    .padding(.bottom, 32)
            }
        }
        .background(Brand.background)
        .toolbar(.hidden, for: .navigationBar)
        .navigationBarBackButtonHidden(true)
        .statusBarHidden(true)
        .onAppear { camera.start() }
        .onDisappear { camera.stop() }
    }

    // MARK: - Pieces

    /// Glass framing guide to help aim the camera at the monitor.
    private var framingGuide: some View {
        VStack {
            Spacer()
            RoundedRectangle(cornerRadius: 20)
                .strokeBorder(.white.opacity(0.5), style: StrokeStyle(lineWidth: 2, dash: [10, 8]))
                .aspectRatio(4.0 / 3.0, contentMode: .fit)
                .padding(.horizontal, 28)
            Text("Cadrez l'écran de planification")
                .font(Brand.display(15))
                .foregroundStyle(.white)
                .padding(.horizontal, 16)
                .padding(.vertical, 10)
                .glassEffect(.regular, in: .capsule)
                .padding(.top, 16)
            Spacer()
        }
        .allowsHitTesting(false)
    }

    private var controls: some View {
        GlassEffectContainer(spacing: 14) {
            HStack(spacing: 14) {
                GlassActionButton(title: "Annuler",
                                  systemImage: "xmark",
                                  tint: Brand.neutralGrey) {
                    cancel()
                }
                GlassActionButton(title: isScanning ? "Analyse en cours…" : "Capturer",
                                  systemImage: "camera.fill",
                                  prominent: true,
                                  busy: isScanning) {
                    capture()
                }
            }
        }
    }

    // MARK: - Actions

    private func capture() {
        guard !isScanning else { return }
        isScanning = true
        Task {
            let plan = await camera.scan(using: scanner)
            model.currentPlan = plan
            camera.stop()
            isScanning = false
            model.path.append(.confirm(mode: .scan))
        }
    }

    private func cancel() {
        camera.stop()
        if !model.path.isEmpty { model.path.removeLast() }
    }
}

// MARK: - Scanner injection

private struct PlanScannerKey: EnvironmentKey {
    static let defaultValue: any PlanScanning = VisionPlanScanner()
}

extension EnvironmentValues {
    /// The OCR engine used by `ScanView`. Defaults to WP4's `VisionPlanScanner`;
    /// inject a stub in previews/tests via `.environment(\.planScanner, …)`.
    var planScanner: any PlanScanning {
        get { self[PlanScannerKey.self] }
        set { self[PlanScannerKey.self] = newValue }
    }
}
