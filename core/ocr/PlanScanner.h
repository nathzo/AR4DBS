#pragma once
#include "core/math/SurgicalPlan.h"
#include <opencv2/core.hpp>
#include <string>

// Extracts Leksell coordinates from a camera frame showing the
// Medtronic Vantage planning screen.
//
// Requires Tesseract: vcpkg install tesseract:x64-windows
// Without it, scan() always returns an empty plan and the user
// fills in the values manually in ConfirmPlanDialog.
class PlanScanner
{
public:
    // Returns true when compiled with Tesseract support.
    static bool isAvailable();

    // Capture a frame and extract coordinates.
    // Both targets may be invalid if OCR failed — the caller should
    // still open ConfirmPlanDialog so the user can enter values manually.
    static SurgicalPlan scan(const cv::Mat &frame);

    // Parse a plain-text string (exposed for unit tests).
    static SurgicalPlan parseText(const std::string &text);

private:
    // Attempt to find and perspective-correct the bright monitor rectangle.
    // Falls back to the full frame if no clear rectangle is found.
    static cv::Mat extractScreen(const cv::Mat &frame);
};
