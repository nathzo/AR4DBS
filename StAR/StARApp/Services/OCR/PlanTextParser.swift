import Foundation

/// Pure text → `SurgicalPlanDTO` parsing. Ported VERBATIM (logic-wise) from
/// `PlanScanner.cpp` (`parseLinesIOS`, `parseTargetFromLines`, `extractField`):
/// split OCR lines into "Gauche"/"Droite" groups (no mirroring if only one
/// header is present), then regex-extract X/Y/Z (mm), Ring, Arc with per-field
/// confidence taken from the source line. No Vision/UIKit deps → unit-testable.
public enum PlanTextParser {

    public static func parse(_ lines: [OCRLine]) -> SurgicalPlanDTO {
        func lower(_ s: String) -> String { s.lowercased() }

        // Locate the column headers.
        var idxGauche = -1, idxDroite = -1
        for (i, line) in lines.enumerated() {
            let low = lower(line.text)
            if idxGauche < 0, low.contains("gauche") { idxGauche = i }
            if idxDroite < 0, low.contains("droite") { idxDroite = i }
        }

        // Split into per-side groups. Only the side(s) whose header was found are
        // parsed; the other stays empty (no mirroring — matches legacy).
        var leftLines: [OCRLine] = []
        var rightLines: [OCRLine] = []

        if idxGauche >= 0 && idxDroite >= 0 {
            if idxGauche < idxDroite {
                leftLines  = Array(lines[idxGauche..<idxDroite])
                rightLines = Array(lines[idxDroite...])
            } else {
                rightLines = Array(lines[idxDroite..<idxGauche])
                leftLines  = Array(lines[idxGauche...])
            }
        } else if idxGauche >= 0 {
            leftLines = Array(lines[idxGauche...])
        } else if idxDroite >= 0 {
            rightLines = Array(lines[idxDroite...])
        }

        var plan = SurgicalPlanDTO()
        if !leftLines.isEmpty  { plan.left  = parseTarget(leftLines) }
        if !rightLines.isEmpty { plan.right = parseTarget(rightLines) }
        return plan
    }

    // MARK: - Field extraction

    // Patterns match the legacy regexes exactly (case-insensitive).
    private static let patterns: [(LeksellTarget.Field, String)] = [
        (.x,    #"X\s*\(mm\)[^0-9]*([0-9]+\.?[0-9]*)"#),
        (.y,    #"Y\s*\(mm\)[^0-9]*([0-9]+\.?[0-9]*)"#),
        (.z,    #"Z\s*\(mm\)[^0-9]*([0-9]+\.?[0-9]*)"#),
        (.ring, #"Ring[^0-9]*([0-9]+\.?[0-9]*)"#),
        (.arc,  #"Arc[^0-9]*([0-9]+\.?[0-9]*)"#),
    ]

    static func parseTarget(_ lines: [OCRLine]) -> LeksellTarget {
        // Flatten lines into one NSString; record each line's UTF-16 start offset
        // so a match position can be mapped back to its source line's confidence.
        let flat = NSMutableString()
        var lineStarts: [Int] = []
        for line in lines {
            lineStarts.append(flat.length)
            flat.append(line.text)
            flat.append("\n")
        }

        var target = LeksellTarget()
        var found = [LeksellTarget.Field: Bool]()

        for (field, pattern) in patterns {
            guard let (value, confidence) =
                    extractField(flat: flat, lineStarts: lineStarts,
                                 lines: lines, pattern: pattern) else { continue }
            found[field] = true
            target.confidence[field.rawValue] = confidence
            switch field {
            case .x:    target.xMM = value
            case .y:    target.yMM = value
            case .z:    target.zMM = value
            case .ring: target.ringDeg = value
            case .arc:  target.arcDeg = value
            }
        }

        // Valid only when every field was detected (legacy semantics).
        target.isValid = LeksellTarget.Field.allCases.allSatisfy { found[$0] == true }
        return target
    }

    private static func extractField(flat: NSString,
                                     lineStarts: [Int],
                                     lines: [OCRLine],
                                     pattern: String) -> (value: Double, confidence: Float)? {
        guard let re = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]),
              let match = re.firstMatch(in: flat as String,
                                        range: NSRange(location: 0, length: flat.length)),
              match.numberOfRanges >= 2 else { return nil }

        let group = match.range(at: 1)
        guard group.location != NSNotFound else { return nil }
        guard let value = Double(flat.substring(with: group)) else { return nil }

        // Line index = the last line whose start offset is ≤ the match position.
        var lineIndex = 0
        for (i, start) in lineStarts.enumerated() {
            if start <= group.location { lineIndex = i } else { break }
        }
        let confidence = (lineIndex < lines.count) ? lines[lineIndex].confidence : 0
        return (value, confidence)
    }
}
