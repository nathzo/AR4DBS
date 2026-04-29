#include "AprilTagTracker.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

AprilTagTracker::AprilTagTracker(const cv::Mat &K,
                                 const cv::Mat &distCoeffs,
                                 float markerSizeM)
    : m_K(K.clone())
    , m_dist(distCoeffs.clone())
    , m_markerSize(markerSizeM)
    , m_detector(
          cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50),
          [](){
              auto p = cv::aruco::DetectorParameters();
              p.cornerRefinementMethod    = cv::aruco::CORNER_REFINE_NONE;
              p.adaptiveThreshWinSizeMin  = 3;
              p.adaptiveThreshWinSizeMax  = 23;
              p.adaptiveThreshWinSizeStep = 10;
              return p;
          }())
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
    // 1. Grayscale into reused buffer
    cv::cvtColor(frame, m_grey, cv::COLOR_BGR2GRAY);

    // 2. Downsample into reused buffer
    cv::resize(m_grey, m_small, cv::Size(), kDetectScale, kDetectScale, cv::INTER_AREA);

    // 3. Detect on the small image using reused vectors
    m_corners.clear();
    m_rejected.clear();
    m_ids.clear();
    m_detector.detectMarkers(m_small, m_corners, m_ids, m_rejected);

    // 4. Scale corners back up to full-res before solvePnP
    const float invScale = 1.f / kDetectScale;
    for (auto &quad : m_corners)
        for (auto &pt : quad)
            pt *= invScale;

    std::vector<TagPose> result;
    result.reserve(m_ids.size());

    for (size_t i = 0; i < m_ids.size(); ++i) {
        TagPose tp;
        tp.id = m_ids[i];
        cv::solvePnP(m_objPts, m_corners[i], m_K, m_dist,
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