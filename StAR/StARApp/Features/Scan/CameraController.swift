import AVFoundation
import CoreVideo

/// Owns the `AVCaptureSession` behind the Scan viewfinder and vends the most
/// recent frame for OCR. Ported intent from the legacy `ScanScreen` camera
/// handling (`app/ScanScreen.cpp`): live preview + capture-the-current-frame.
///
/// Deliberately *not* `@MainActor`: the sample-buffer delegate fires on a
/// background queue and the Vision scan must run off the main actor, so the
/// latest `CVPixelBuffer` is held under a lock and never crosses an actor
/// boundary. Only the resulting (Sendable) `SurgicalPlanDTO` returns to the UI.
final class CameraController: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {

    let session = AVCaptureSession()

    private let sessionQueue = DispatchQueue(label: "ch.epfl.neurorestore.star.camera.session")
    private let outputQueue  = DispatchQueue(label: "ch.epfl.neurorestore.star.camera.output")
    private let output = AVCaptureVideoDataOutput()

    private let lock = NSLock()
    private var latest: CVPixelBuffer?
    private var configured = false

    // MARK: - Lifecycle

    /// Request authorization (if needed) and start streaming. Safe to call from
    /// `.onAppear`; configuration happens off the main thread.
    func start() {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            switch AVCaptureDevice.authorizationStatus(for: .video) {
            case .authorized:
                self.configureAndRun()
            case .notDetermined:
                AVCaptureDevice.requestAccess(for: .video) { granted in
                    guard granted else { return }
                    self.sessionQueue.async { self.configureAndRun() }
                }
            default:
                break   // denied / restricted → black viewfinder; capture yields an empty plan.
            }
        }
    }

    func stop() {
        sessionQueue.async { [weak self] in
            guard let self, self.session.isRunning else { return }
            self.session.stopRunning()
        }
    }

    private func configureAndRun() {
        if !configured {
            session.beginConfiguration()
            session.sessionPreset = .high

            if let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
               let input = try? AVCaptureDeviceInput(device: device),
               session.canAddInput(input) {
                session.addInput(input)
            }

            output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            output.alwaysDiscardsLateVideoFrames = true
            output.setSampleBufferDelegate(self, queue: outputQueue)
            if session.canAddOutput(output) { session.addOutput(output) }

            session.commitConfiguration()
            configured = true
        }
        if !session.isRunning { session.startRunning() }
    }

    // MARK: - Frame capture

    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        lock.lock(); latest = pixelBuffer; lock.unlock()
    }

    private func snapshot() -> CVPixelBuffer? {
        lock.lock(); defer { lock.unlock() }
        return latest
    }

    /// Grab the latest frame and OCR it (runs off the main actor). If no frame is
    /// available (simulator / denied permission), returns an empty plan so the
    /// confirm screen opens for manual entry — mirrors the legacy
    /// `emit planDetected({})` on capture failure.
    func scan(using scanner: any PlanScanning) async -> SurgicalPlanDTO {
        guard let pixelBuffer = snapshot() else { return SurgicalPlanDTO() }
        return await scanner.scan(pixelBuffer)
    }
}
