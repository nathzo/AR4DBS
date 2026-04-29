#pragma once
#include <string>
#include <opencv2/core.hpp>

// Thin C++ wrapper around Apple's Vision text-recognition API.
// VNRecognizeTextRequest is available on iOS 13+ and runs on the Neural Engine.
// recognize() is synchronous — it blocks until Vision finishes, then returns the
// extracted text as UTF-8.  Caller is responsible for parsing (PlanScanner::parseText).
class IOSOCREngine
{
public:
    static std::string recognize(const cv::Mat &bgr);
};
