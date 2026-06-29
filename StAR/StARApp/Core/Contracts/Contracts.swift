import simd
import SwiftUI
import CoreVideo

/// ─────────────────────────────────────────────────────────────────────────────
/// Cross-package contracts. WP0 owns this file. Every other work package codes
/// against these protocols/types so packages can be built and tested in parallel
/// without depending on each other's concrete implementations.
/// Do NOT add concrete logic here.
/// ─────────────────────────────────────────────────────────────────────────────

// MARK: - OCR (WP4)

public struct OCRLine: Sendable {
    public let text: String
    public let confidence: Float   // 0...1 (Vision recognition confidence)
    public init(text: String, confidence: Float) { self.text = text; self.confidence = confidence }
}

/// Extracts a `SurgicalPlanDTO` from a captured camera image of the Vantage
/// planning monitor. Implemented by WP4 using Vision (no OpenCV).
public protocol PlanScanning: Sendable {
    /// `image` is a captured frame. Returns a plan whose targets may be invalid
    /// (then the UI opens the confirm dialog for manual entry).
    func scan(_ image: CVPixelBuffer) async -> SurgicalPlanDTO
}

// MARK: - Registration (WP1 fusion + WP2 runtime)

/// Pure fusion: turn the currently-tracked ARKit image anchors into a candidate
/// `world_T_leksell` and a per-frame quality verdict. No ARKit imports here —
/// inputs are plain transforms so this is unit-testable (WP1).
public protocol MarkerFusing: Sendable {
    /// `anchors` maps a marker to its current `world_T_marker` (ARKit anchor transform).
    /// Returns the fused `world_T_leksell` plus whether the frame qualifies for the
    /// lock streak (both markers agree within tolerance, tracked, scale ≈ 1).
    func fuse(anchors: [CoordinateConventions.MarkerID: simd_float4x4],
              parameters: RegistrationParameters) -> (worldToLeksell: simd_float4x4, qualifies: Bool)?
}

// MARK: - AR session (WP2)

/// Observable surgical AR session. WP2 implements with ARWorldTrackingConfiguration
/// (detectionImages + sceneReconstruction(.mesh) + frameSemantics(.sceneDepth)).
/// The AR screen (WP7) and renderer (WP3) observe this.
@MainActor
public protocol SurgicalSession: AnyObject, Observable {
    var registration: RegistrationState { get }
    var trackingQuality: TrackingQuality { get }

    func start()
    func stop()
    /// Drop the current lock and re-enter the calibration phase (the "Recalibrer" action).
    func resetRegistration()
    /// Inject / update the active plan geometry to overlay.
    func setPlan(_ geometry: PlanGeometry)
}

// MARK: - Overlay rendering (WP3)

/// Builds and updates the RealityKit overlay (trajectory line, target sphere,
/// incision marker) under the locked Leksell anchor. WP3 implements; WP7 hosts.
@MainActor
public protocol OverlayRendering: AnyObject {
    func attach(to arView: ARViewRepresentableHandle)
    func update(worldToLeksell: simd_float4x4, geometry: PlanGeometry, style: RenderStyle)
    func clear()
}

/// Opaque handle so WP3 doesn't hard-depend on WP7's view type.
public protocol ARViewRepresentableHandle: AnyObject {}

// MARK: - Style (WP3 + WP6)

public struct RenderStyle: Codable, Equatable, Sendable {
    public var lineColorHex: String     = "#DE5F5E" // IMPULSE_RED
    public var targetColorHex: String   = "#75D0C5" // ARC_BLUE
    public var incisionColorHex: String = "#E9DF4D" // VOLT_YELLOW
    public var lineWidthMM: Float        = 4
    public init() {}
}
