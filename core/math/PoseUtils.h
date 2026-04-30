#pragma once
#include <opencv2/core.hpp>

// Utility functions for pose representation conversions
namespace PoseUtils
{
    // Rodrigues rvec + tvec → 4x4 homogeneous transform (CV_64F)
    cv::Mat toTransform(const cv::Mat &rvec, const cv::Mat &tvec);

    // 4x4 homogeneous transform → rvec + tvec
    void fromTransform(const cv::Mat &T, cv::Mat &rvec, cv::Mat &tvec);

    // Project a 3-D world point to image pixel using K, rvec, tvec, dist
    cv::Point2f project(const cv::Point3d &pt,
                        const cv::Mat &K,
                        const cv::Mat &rvec,
                        const cv::Mat &tvec,
                        const cv::Mat &dist = cv::Mat());
}
