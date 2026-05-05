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
              p.cornerRefinementMethod    = cv::aruco::CORNER_REFINE_SUBPIX;
              p.adaptiveThreshWinSizeMin  = 3;
              p.adaptiveThreshWinSizeMax  = 13;  // 2 scales (3,13) instead of 3 — ~40% faster thresholding
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

// ── ROI helpers ───────────────────────────────────────────────────────────────

bool AprilTagTracker::detectInRoi(std::vector<std::vector<cv::Point2f>> &corners,
                                   std::vector<int>                      &ids)
{
    std::vector<std::vector<cv::Point2f>> rejTmp;
    std::vector<std::vector<cv::Point2f>> cornersTmp;
    std::vector<int> idsTmp;

    cv::Mat crop = m_small(m_roi);
    m_detector.detectMarkers(crop, cornersTmp, idsTmp, rejTmp);
    if (idsTmp.empty()) return false;

    // Shift corner coordinates from crop-local back to m_small coordinates
    for (auto &quad : cornersTmp)
        for (auto &pt : quad)
            pt += m_roiOffset;

    corners = std::move(cornersTmp);
    ids     = std::move(idsTmp);
    return true;
}

void AprilTagTracker::updateRoi(const std::vector<std::vector<cv::Point2f>> &corners,
                                 cv::Size smallSize)
{
    if (corners.empty()) { m_roiActive = false; return; }

    // Compute bounding box of all detected tag corners in m_small coordinates
    float xMin = 1e9f, xMax = -1e9f, yMin = 1e9f, yMax = -1e9f;
    for (const auto &quad : corners) {
        for (const auto &pt : quad) {
            xMin = std::min(xMin, pt.x); xMax = std::max(xMax, pt.x);
            yMin = std::min(yMin, pt.y); yMax = std::max(yMax, pt.y);
        }
    }

    const int x0 = std::max(0,              static_cast<int>(xMin) - kRoiPad);
    const int y0 = std::max(0,              static_cast<int>(yMin) - kRoiPad);
    const int x1 = std::min(smallSize.width,  static_cast<int>(xMax) + kRoiPad);
    const int y1 = std::min(smallSize.height, static_cast<int>(yMax) + kRoiPad);

    m_roi       = cv::Rect(x0, y0, x1 - x0, y1 - y0);
    m_roiOffset = cv::Point2f(static_cast<float>(x0), static_cast<float>(y0));
    m_roiActive = (m_roi.width > 0 && m_roi.height > 0);
}

// ── Main detect ───────────────────────────────────────────────────────────────

std::vector<TagPose> AprilTagTracker::detect(const cv::Mat &frame, const cv::Mat &predictedR)
{
    // 1. Grayscale + downscale into reused buffers
    cv::cvtColor(frame, m_grey, cv::COLOR_BGR2GRAY);
    cv::resize(m_grey, m_small, cv::Size(), kDetectScale, kDetectScale, cv::INTER_AREA);

    // 2. Try ROI first; fall back to full scan on a miss or every kFullScanPeriod frames.
    //
    // Without the periodic full scan a two-tag scene breaks: detectInRoi short-
    // circuits once it finds *one* tag, the ROI then shrinks to that tag only, and
    // the second tag is never rediscovered.  The forced full scan every
    // kFullScanPeriod frames lets both tags be re-included in the ROI regularly.
    m_corners.clear();
    m_ids.clear();
    m_rejected.clear();

    bool found = false;
    if (m_roiActive && (++m_fullScanCounter < kFullScanPeriod))
        found = detectInRoi(m_corners, m_ids);
    else
        m_fullScanCounter = 0;   // reset; full scan runs below

    if (!found) {
        m_detector.detectMarkers(m_small, m_corners, m_ids, m_rejected);
    }

    // 3. Update ROI for the next frame
    updateRoi(m_corners, m_small.size());

    // 4. Scale corners back to full-res before solvePnP
    const float invScale = 1.f / kDetectScale;
    for (auto &quad : m_corners)
        for (auto &pt : quad)
            pt *= invScale;

    std::vector<TagPose> result;
    result.reserve(m_ids.size());
    for (size_t i = 0; i < m_ids.size(); ++i) {
        TagPose tp;
        tp.id = m_ids[i];

        std::vector<cv::Mat> rvecs, tvecs;
        std::vector<double>  reprojErrors;
        cv::solvePnPGeneric(m_objPts, m_corners[i], m_K, m_dist,
                            rvecs, tvecs, false,
                            cv::SOLVEPNP_IPPE_SQUARE,
                            cv::noArray(), cv::noArray(),
                            reprojErrors);

        // IPPE_SQUARE always produces exactly two solutions (a front/back ambiguity).
        // OpenCV sorts them by reprojection error (index 0 = lower error), which is
        // unreliable when the tag is viewed at a steep angle.
        //
        // Physical constraint: tags are always mounted facing the camera, so the tag's
        // outward normal must point toward the camera. In OpenCV camera space the normal
        // is R * [0,0,1] = third column of R. For a front-facing tag its Z component
        // must be negative (pointing back toward the camera, which looks along +Z).
        // We override the reprojection-error choice only when it violates this.
        int best = 0;
        if (rvecs.size() >= 2) {
            cv::Mat R0, R1;
            cv::Rodrigues(rvecs[0], R0);
            cv::Rodrigues(rvecs[1], R1);
            if (R0.at<double>(2, 2) >= 0 && R1.at<double>(2, 2) < 0)
                best = 1;
        }

        tp.rvec    = rvecs[best];
        tp.tvec    = tvecs[best];
        tp.corners = m_corners[i];   // full-res corners for unified PnP
        result.push_back(std::move(tp));
    }
    return result;
}

void AprilTagTracker::drawAxes(cv::Mat &frame, const std::vector<TagPose> &poses) const
{
    for (const auto &p : poses)
        cv::drawFrameAxes(frame, m_K, m_dist, p.rvec, p.tvec, m_markerSize * 0.5f);
}