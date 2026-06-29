import SwiftUI
import AVFoundation

/// Full-screen, edge-to-edge live camera preview backed by an
/// `AVCaptureVideoPreviewLayer`. Portrait-locked (the app never rotates during a
/// procedure), `.resizeAspectFill` so the feed fills the screen with no
/// letterboxing — per the WP5 UI principle.
struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        if let connection = view.previewLayer.connection,
           connection.isVideoRotationAngleSupported(90) {
            connection.videoRotationAngle = 90   // portrait
        }
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {
        uiView.previewLayer.session = session
    }

    /// Backing UIView whose layer *is* the preview layer (no manual sizing).
    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }
}
