#pragma once
#include <opencv2/core.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <vector>
#include <nlohmann/json.hpp>

struct TagPose {
    int     id;
    cv::Mat rvec; // Rodrigues rotation (tag → camera)
    cv::Mat tvec; // translation in metres (tag → camera)
};

struct TagConfig {
    int     id;
    cv::Mat R_tag_frame;   // 3×3 double
    cv::Mat t_tag_frame;   // 3×1 double
};

class AprilTagTracker
{
public:
    // K: 3x3 camera matrix; distCoeffs: 1x4/5 distortion; markerSizeM: physical side length
    AprilTagTracker(const cv::Mat &K, const cv::Mat &distCoeffs, float markerSizeM);

    void loadTagConfig(const std::string &path);

    bool estimateFramePose(const std::vector<TagPose> &poses,   // ← add this
                           cv::Mat &R_out, cv::Mat &t_out) const;

    // predictedR: optional 3×3 rotation (camera←frame, CV_64F) used to
    // disambiguate the two IPPE solutions. When empty the lower-reprojection-
    // error solution is returned (original behaviour).
    std::vector<TagPose> detect(const cv::Mat &frame,
                                const cv::Mat &predictedR = cv::Mat());

    // Draws coordinate axes on frame for each detected tag (debug)
    void drawAxes(cv::Mat &frame, const std::vector<TagPose> &poses) const;

private:
    cv::Mat m_K;
    cv::Mat m_dist;
    float   m_markerSize;
    std::vector<cv::Point3f> m_objPts;
    cv::aruco::ArucoDetector m_detector;

    static constexpr float kDetectScale = 0.4f;  // downscale factor for detection
    // ROI padding in scaled-image pixels around the last known tag centre.
    // 80 px at 0.4× ≈ 200 px in full-res — enough for moderate motion at 30 fps.
    static constexpr int   kRoiPad      = 80;

    cv::Mat m_grey;   // reused grayscale buffer
    cv::Mat m_small;  // reused downscaled buffer

    std::vector<std::vector<cv::Point2f>> m_corners;
    std::vector<std::vector<cv::Point2f>> m_rejected;
    std::vector<int>                      m_ids;

    // ROI state — updated each frame
    bool    m_roiActive = false;
    cv::Rect m_roi;                 // in m_small coordinates
    cv::Point2f m_roiOffset;        // top-left of the ROI in m_small coordinates

    // Run detection inside a sub-image; returns false on a miss (caller does full scan)
    bool detectInRoi(std::vector<std::vector<cv::Point2f>> &corners,
                     std::vector<int>                      &ids);
    // Update m_roi from the detected corners (in m_small coordinates)
    void updateRoi(const std::vector<std::vector<cv::Point2f>> &corners,
                   cv::Size smallSize);

    std::vector<TagConfig> m_tagConfigs;
};
