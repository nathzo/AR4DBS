#pragma once
#include <opencv2/core.hpp>

// Edge-based model tracker using ViSP (runs after AprilTag initialisation)
class VispTracker
{
public:
    VispTracker();

    void init(const cv::Mat &frame, const cv::Mat &rvec, const cv::Mat &tvec);
    bool track(const cv::Mat &frame, cv::Mat &rvec, cv::Mat &tvec);
    void reset();

    bool isInitialised() const { return m_initialised; }

private:
    bool m_initialised = false;
};
