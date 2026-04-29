#pragma once
#include <opencv2/core.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <vector>

struct TagPose {
    int     id;
    cv::Mat rvec; // Rodrigues rotation (tag → camera)
    cv::Mat tvec; // translation in metres (tag → camera)
};

class AprilTagTracker
{
public:
    // K: 3x3 camera matrix; distCoeffs: 1x4/5 distortion; markerSizeM: physical side length
    AprilTagTracker(const cv::Mat &K, const cv::Mat &distCoeffs, float markerSizeM);

    std::vector<TagPose> detect(const cv::Mat &frame);

    // Draws coordinate axes on frame for each detected tag (debug)
    void drawAxes(cv::Mat &frame, const std::vector<TagPose> &poses) const;

private:
    cv::Mat m_K;
    cv::Mat m_dist;
    float   m_markerSize;
    std::vector<cv::Point3f> m_objPts;
    cv::aruco::ArucoDetector m_detector;

    // --- Optimisation additions ---
    static constexpr float kDetectScale = 0.4f;  // downscale factor for detection

    cv::Mat m_grey;   // reused grayscale buffer
    cv::Mat m_small;  // reused downscaled buffer

    std::vector<std::vector<cv::Point2f>> m_corners;   // reused detection output
    std::vector<std::vector<cv::Point2f>> m_rejected;  // reused rejection output
    std::vector<int>                      m_ids;       // reused ID output
};
