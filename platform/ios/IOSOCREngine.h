#pragma once
#include <string>
#include <vector>
#include <opencv2/core.hpp>

// One line of text recognised by Apple Vision, with its per-candidate confidence.
struct OcrLine {
    std::string text;
    float       confidence; // 0.0–1.0 from VNRecognizedText.confidence
};

// Thin C++ wrapper around Apple's Vision text-recognition API.
// VNRecognizeTextRequest is available on iOS 13+ and runs on the Neural Engine.
// recognize() is synchronous — it blocks until Vision finishes.
class IOSOCREngine
{
public:
    static std::vector<OcrLine> recognize(const cv::Mat &bgr);
};
