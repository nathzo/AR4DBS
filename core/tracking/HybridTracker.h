#pragma once
#include <opencv2/core.hpp>

// Combines AprilTag (init / recovery) with ViSP (continuous tracking)
class HybridTracker
{
public:
    HybridTracker();

    // Call once per frame; returns false if tracking is lost
    bool update(const cv::Mat &frame, cv::Mat &rvec, cv::Mat &tvec);

private:
    // Forward-declared to avoid heavy headers in this file
    struct Impl;
    Impl *m_impl;
};
