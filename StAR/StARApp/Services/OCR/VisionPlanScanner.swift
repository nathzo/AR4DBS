import Foundation
import Vision
import CoreImage

/// `PlanScanning` implementation using Apple Vision (no OpenCV). Ported from
/// `IOSOCREngine.mm` (Vision request config) + `PlanScanner.cpp` (screen
/// extraction → OCR → parse). Pipeline:
///   1. Detect the bright monitor quad (`VNDetectRectanglesRequest`) and rectify
///      it with `CIPerspectiveCorrection` — replaces the OpenCV contour pass.
///   2. Recognise text (`VNRecognizeTextRequest`, accurate, fr-FR/en-US).
///   3. Parse with `PlanTextParser` (pure, unit-tested separately).
public struct VisionPlanScanner: PlanScanning {

    private let ciContext = CIContext()

    public init() {}

    public func scan(_ image: CVPixelBuffer) async -> SurgicalPlanDTO {
        let full = CIImage(cvPixelBuffer: image)
        let screen = rectifiedScreen(from: full) ?? full
        let lines = await recognizeText(in: screen)
        return PlanTextParser.parse(lines)
    }

    // MARK: - Screen extraction

    /// Finds the largest bright quadrilateral and perspective-corrects it.
    /// Returns nil to fall back to the full frame (legacy behaviour).
    private func rectifiedScreen(from image: CIImage) -> CIImage? {
        let request = VNDetectRectanglesRequest()
        request.minimumAspectRatio = 0.3
        request.maximumAspectRatio = 1.0
        request.minimumSize = 0.2          // ≥ ~20% of the frame (legacy minArea 15%)
        request.maximumObservations = 1
        request.quadratureTolerance = 25

        let handler = VNImageRequestHandler(ciImage: image, options: [:])
        guard (try? handler.perform([request])) != nil,
              let quad = request.results?.first else { return nil }

        // VNRectangleObservation corners are normalized (0…1, origin bottom-left).
        let extent = image.extent
        func denorm(_ p: CGPoint) -> CGPoint {
            CGPoint(x: extent.origin.x + p.x * extent.width,
                    y: extent.origin.y + p.y * extent.height)
        }

        let filter = CIFilter(name: "CIPerspectiveCorrection")!
        filter.setValue(image, forKey: kCIInputImageKey)
        filter.setValue(CIVector(cgPoint: denorm(quad.topLeft)),     forKey: "inputTopLeft")
        filter.setValue(CIVector(cgPoint: denorm(quad.topRight)),    forKey: "inputTopRight")
        filter.setValue(CIVector(cgPoint: denorm(quad.bottomLeft)),  forKey: "inputBottomLeft")
        filter.setValue(CIVector(cgPoint: denorm(quad.bottomRight)), forKey: "inputBottomRight")
        return filter.outputImage
    }

    // MARK: - OCR

    private func recognizeText(in image: CIImage) async -> [OCRLine] {
        await withCheckedContinuation { continuation in
            let request = VNRecognizeTextRequest { request, _ in
                let lines: [OCRLine] = (request.results as? [VNRecognizedTextObservation] ?? [])
                    .compactMap { observation in
                        guard let top = observation.topCandidates(1).first,
                              !top.string.isEmpty else { return nil }
                        return OCRLine(text: top.string, confidence: top.confidence)
                    }
                continuation.resume(returning: lines)
            }
            request.recognitionLevel = .accurate
            request.recognitionLanguages = ["fr-FR", "en-US"]
            request.usesLanguageCorrection = true

            let handler = VNImageRequestHandler(ciImage: image, options: [:])
            do { try handler.perform([request]) }
            catch { continuation.resume(returning: []) }
        }
    }
}
