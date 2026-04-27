#include "AprilTagTracker.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

AprilTagTracker::AprilTagTracker(const cv::Mat &K,
                                 const cv::Mat &distCoeffs,
                                 float markerSizeM)
    : m_K(K.clone())
    , m_dist(distCoeffs.clone())
    , m_markerSize(markerSizeM)
    , m_detector(cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50),
                 cv::aruco::DetectorParameters())
{
    const float h = markerSizeM / 2.f;
    m_objPts = {
        { -h,  h, 0.f },
        {  h,  h, 0.f },
        {  h, -h, 0.f },
        { -h, -h, 0.f }
    };
}

std::vector<TagPose> AprilTagTracker::detect(const cv::Mat &frame)
{
    cv::Mat grey;
    cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);

    std::vector<std::vector<cv::Point2f>> corners, rejected;
    std::vector<int> ids;
    m_detector.detectMarkers(grey, corners, ids, rejected);

    std::vector<TagPose> result;
    result.reserve(ids.size());

    for (size_t i = 0; i < ids.size(); ++i) {
        TagPose tp;
        tp.id = ids[i];
        cv::solvePnP(m_objPts, corners[i], m_K, m_dist,
                     tp.rvec, tp.tvec, false, cv::SOLVEPNP_IPPE_SQUARE);
        result.push_back(std::move(tp));
    }
    return result;
}

void AprilTagTracker::drawAxes(cv::Mat &frame, const std::vector<TagPose> &poses) const
{
    for (const auto &p : poses)
        cv::drawFrameAxes(frame, m_K, m_dist, p.rvec, p.tvec, m_markerSize * 0.5f);
}
