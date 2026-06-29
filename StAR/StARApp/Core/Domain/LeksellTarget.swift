import Foundation

/// One electrode target expressed in Leksell stereotactic frame coordinates.
///
/// Mirrors the fields shown on the Medtronic Vantage planning screen and the
/// legacy C++ `LeksellTarget` struct. Distances are millimetres, angles degrees.
///
/// Coordinate conventions are documented in `CoordinateConventions.swift`.
public struct LeksellTarget: Codable, Equatable, Hashable, Sendable {

    /// Field identity — used for per-field OCR confidence and UI flagging.
    public enum Field: Int, CaseIterable, Sendable {
        case x = 0, y, z, ring, arc
    }

    public var xMM: Double      // left-right (mm)
    public var yMM: Double      // anterior-posterior (mm)
    public var zMM: Double      // superior-inferior (mm)
    public var ringDeg: Double  // arc-carrier rotation (degrees)
    public var arcDeg: Double   // electrode tilt from vertical (degrees)
    public var isValid: Bool

    /// Per-field OCR confidence.
    /// `nil`  = field was not detected (cell should appear empty / flagged).
    /// `0...1` = Vision recognition confidence for that field.
    /// Indexed by `Field.rawValue`.
    public var confidence: [Float?]

    public init(xMM: Double = 0, yMM: Double = 0, zMM: Double = 0,
                ringDeg: Double = 0, arcDeg: Double = 0,
                isValid: Bool = false,
                confidence: [Float?] = Array(repeating: nil, count: Field.allCases.count)) {
        self.xMM = xMM; self.yMM = yMM; self.zMM = zMM
        self.ringDeg = ringDeg; self.arcDeg = arcDeg
        self.isValid = isValid
        self.confidence = confidence
    }

    public func confidence(for field: Field) -> Float? { confidence[field.rawValue] }
}
