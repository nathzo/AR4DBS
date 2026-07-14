#include "AppController.h"
#include "DebugLogger.h"
#include "core/tracking/AprilTagTracker.h"
#include "core/math/IncisionLine.h"
#include "core/math/PoseUtils.h"
#include "core/rendering/OverlayRenderer.h"
#include "core/depth/DepthEstimator.h"
#include <cmath>
#include <cstdio>
#include <string>

#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QDebug>
#include <QPainter>
#include <QFont>
#include <QFontDatabase>
#include <QImage>
#include <QThread>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

static constexpr int    RAY_SAMPLES     = 60;
static float sampleDepthAt(const cv::Mat &depthMap, cv::Point2f p); // defined below
static constexpr double DEPTH_TOLERANCE = 0.25;
static constexpr int    DEPTH_SAMPLE_R  = 5;

AppController::AppController(QObject *parent) : QObject(parent) {
    DebugLogger::logEvent("AppController",
        QString("constructor started, thread=0x%1")
        .arg(reinterpret_cast<quintptr>(QThread::currentThread()), 0, 16));
}
AppController::~AppController() {
    DebugLogger::logEvent("AppController",
        QString("destructor started, thread=0x%1")
        .arg(reinterpret_cast<quintptr>(QThread::currentThread()), 0, 16));
}

bool AppController::init(const QString &calibPath,
                         const QString &tagConfigPath,
                         const QString &planPath,
                         const QString &depthModelPath)
{
    DebugLogger::logEvent("AppController::init", "starting initialization");

    m_K = loadCalibration(calibPath); // also populates m_dist
    if (m_K.empty()) {
        DebugLogger::logEvent("AppController::init", "ERROR: calibration load failed");
        qWarning() << "AppController: could not load calibration from" << calibPath;
        return false;
    }
    DebugLogger::logEvent("AppController::init", "calibration loaded");

    m_tagConfigs = loadTagConfigs(tagConfigPath);
    if (m_tagConfigs.empty()) {
        DebugLogger::logEvent("AppController::init", "ERROR: tag config load failed");
        qWarning() << "AppController: could not load tag config from" << tagConfigPath;
        return false;
    }
    DebugLogger::logEvent("AppController::init", QString("tag configs loaded, count=%1").arg(m_tagConfigs.size()));

    m_tracker  = std::make_unique<AprilTagTracker>(m_K, m_dist, m_markerSize);
    m_renderer = std::make_unique<OverlayRenderer>();
    m_renderer->setDistortion(m_dist);
    DebugLogger::logEvent("AppController::init", "tracker and renderer created");

    if (!depthModelPath.isEmpty()) {
        m_depth = std::make_unique<DepthEstimator>(depthModelPath.toStdString());
        if (!m_depth->isLoaded()) {
            DebugLogger::logEvent("AppController::init", "depth model load failed (ignored)");
            qWarning() << "AppController: depth model not loaded from" << depthModelPath;
            m_depth.reset();
        } else {
            DebugLogger::logEvent("AppController::init", "depth estimation enabled");
            qDebug() << "AppController: depth estimation enabled";
        }
    }

    DebugLogger::logEvent("AppController::init", "initialization complete");
    return true;
}

void AppController::setCalibration(const cv::Mat &K)
{
    m_K = K.clone();
    m_tracker = std::make_unique<AprilTagTracker>(m_K, m_dist, m_markerSize);
    qDebug() << "AppController: calibration updated fx=" << K.at<double>(0,0)
             << "fy=" << K.at<double>(1,1);
}

void AppController::setShowDepthOverlay(bool show) { m_showDepthOverlay = show; }
void AppController::setShowDepthVisualization(bool show) { m_showDepthVisualization = show; }

void AppController::setRenderStyle(OverlayRenderer::Style style)
{
    if (m_renderer) m_renderer->setStyle(style);
}

void AppController::setReprojThreshold(double px)
{
    m_maxInitReprojPx = px;
}

void AppController::setMovementThresholds(double transMm, double rotDeg)
{
    m_moveTransThresh = transMm / 1000.0;
    m_moveRotThresh   = rotDeg;
}

void AppController::setTagPosition(int tagId, double tx_m, double ty_m, double tz_m)
{
    for (auto &cfg : m_tagConfigs) {
        if (cfg.id != tagId) continue;
        cfg.t_frame_tag.at<double>(0) = tx_m;
        cfg.t_frame_tag.at<double>(1) = ty_m;
        cfg.t_frame_tag.at<double>(2) = tz_m;
        cfg.T_frame_tag.at<double>(0, 3) = tx_m;
        cfg.T_frame_tag.at<double>(1, 3) = ty_m;
        cfg.T_frame_tag.at<double>(2, 3) = tz_m;
        break;
    }
}

void AppController::setTagRotation(int tagId, double rx_rad, double ry_rad, double rz_rad)
{
    for (auto &cfg : m_tagConfigs) {
        if (cfg.id != tagId) continue;
        // Convert Euler angles to rotation matrix
        cv::Mat rvec = (cv::Mat_<double>(3, 1) << rx_rad, ry_rad, rz_rad);
        // Ensure position is always (3, 1) independent matrix
        cv::Mat tvec = cfg.t_frame_tag.clone();
        if (tvec.rows == 1 && tvec.cols == 3) {
            tvec = tvec.t();  // Transpose (1, 3) to (3, 1) if needed
        }
        cfg.t_frame_tag = tvec;  // Keep position as independent (3, 1)
        // Reconstruct T_frame_tag from independent components
        cfg.T_frame_tag = PoseUtils::toTransform(rvec, cfg.t_frame_tag);
        // Update rotation matrix directly from rvec
        cv::Rodrigues(rvec, cfg.R_frame_tag);
        break;
    }
}

void AppController::setSurgicalPlan(const SurgicalPlan &plan)
{
    DebugLogger::logEvent("AppController::setSurgicalPlan",
        QString("called with left=%1, right=%2")
        .arg(plan.hasLeft() ? 1 : 0)
        .arg(plan.hasRight() ? 1 : 0));
    m_lines[0] = plan.hasLeft()
        ? std::make_unique<IncisionLine>(IncisionLine::fromLeksell(plan.left))
        : nullptr;
    m_lines[1] = plan.hasRight()
        ? std::make_unique<IncisionLine>(IncisionLine::fromLeksell(plan.right))
        : nullptr;
#ifdef Q_OS_IOS
    {
        std::lock_guard<std::mutex> lk(m_depthMutex);
        for (auto &lock : m_lockedIncision) lock = LockedIncision{};
    }
    DebugLogger::logEvent("AppController::setSurgicalPlan", "incision locks cleared");
#endif
    DebugLogger::logEvent("AppController::setSurgicalPlan", "plan set successfully");
}

void AppController::onNewFrame(const cv::Mat &frame)
{
    m_frameTimer.restart();

    const auto detections = m_tracker->detect(frame);
    const cv::Mat T_cam_frame = fusePoses(detections);

    const bool anyLine = m_lines[0] || m_lines[1];
    if (T_cam_frame.empty() || !anyLine) {
        m_lastFrameMs = m_frameTimer.elapsed();
        emit frameReady(frame);
        return;
    }

    cv::Mat out = frame.clone();

#ifndef NDEBUG
    m_tracker->drawAxes(out, detections);
#endif

    cv::Mat rvec, tvec;
    PoseUtils::fromTransform(T_cam_frame, rvec, tvec);

    // Compute depth map and anchor (metric_depth_tag × rel_depth_tag).
    // Skipped when the previous frame already took > 50 ms.
    cv::Mat depthMap;
    double  depthAnchor = 0.0;
    if (m_depth && !detections.empty() && m_lastFrameMs < 50) {
        depthMap = m_depth->estimate(frame);
        if (!depthMap.empty()) {
            const double    tagMetricDepth = cv::norm(detections[0].tvec);
            const cv::Point2f tagPx = PoseUtils::project(
                cv::Point3d(0,0,0), m_K, detections[0].rvec, detections[0].tvec, m_dist);
            const float relTag = sampleDepthAt(depthMap, tagPx);
            if (relTag > 1e-4f)
                depthAnchor = tagMetricDepth * relTag;
        }
    }

    m_renderer->beginFrame(out);
    cv::drawFrameAxes(out, m_K, m_dist, rvec, tvec, 0.03f); //  3 cm frame origin axes

    if (depthAnchor < 1e-9) {
        renderOverlayOnto(out, rvec, tvec);
    } else {
        renderWithOcclusion(out, rvec, tvec, depthMap, depthAnchor);
    }

    m_renderer->endFrame();
    m_lastFrameMs = m_frameTimer.elapsed();
    emit frameReady(out);
}

// ── Depth helpers ─────────────────────────────────────────────────────────────

static float sampleDepthAt(const cv::Mat &depthMap, cv::Point2f p)
{
    const int x0 = std::max(0, static_cast<int>(p.x) - DEPTH_SAMPLE_R);
    const int y0 = std::max(0, static_cast<int>(p.y) - DEPTH_SAMPLE_R);
    const int x1 = std::min(depthMap.cols-1, static_cast<int>(p.x) + DEPTH_SAMPLE_R);
    const int y1 = std::min(depthMap.rows-1, static_cast<int>(p.y) + DEPTH_SAMPLE_R);
    if (x1 <= x0 || y1 <= y0) return 0.f;
    return static_cast<float>(cv::mean(depthMap(cv::Rect(x0, y0, x1-x0, y1-y0)))[0]);
}

// ── Ray–surface intersection ─────────────────────────────────────────────────

std::optional<cv::Point3d> AppController::findIncisionPoint(
    const cv::Mat     &depthMap,
    const cv::Mat     &rvec,
    const cv::Mat     &tvec,
    double             depthAnchor,
    const IncisionLine &line) const
{
    if (depthMap.empty() || depthAnchor < 1e-9) return std::nullopt;

    cv::Mat R;
    cv::Rodrigues(rvec, R);

    const cv::Point3d tgt = applyTagFrameRotation(line.target());
    const cv::Point3d end = applyTagFrameRotation(line.lineEnd());

    for (int i = 0; i <= RAY_SAMPLES; ++i) {
        const double t = static_cast<double>(i) / RAY_SAMPLES;
        const cv::Point3d pt = {
            end.x + t*(tgt.x-end.x),
            end.y + t*(tgt.y-end.y),
            end.z + t*(tgt.z-end.z)
        };
        const cv::Mat ptVec = (cv::Mat_<double>(3,1) << pt.x, pt.y, pt.z);
        const cv::Mat ptCam = R * ptVec + tvec.reshape(1, 3);
        const double expectedDepth = ptCam.at<double>(2);
        if (expectedDepth <= 0) continue;

        const cv::Point2f px = PoseUtils::project(pt, m_K, rvec, tvec, m_dist);
        const float relPt = sampleDepthAt(depthMap, px);
        if (relPt < 1e-4f) continue;

#ifdef Q_OS_IOS
        // Threshold crossing: first ray point where the measured surface is
        // closer than the trajectory point — the surface now blocks the path.
        // Depth convention (LiDAR metric): larger = farther.
        const double estimatedDepth = relPt * depthAnchor;
        if (estimatedDepth < expectedDepth)
            return pt;
#else
        const double estimatedDepth = depthAnchor / relPt; // disparity convention (higher=closer)
        if (std::abs(estimatedDepth - expectedDepth) < DEPTH_TOLERANCE * expectedDepth)
            return pt;
#endif
    }
    return std::nullopt;
}

// ── Incision quality gate ─────────────────────────────────────────────────────

#ifdef Q_OS_IOS
// Returns true when:
//   1. The measured depth at the candidate pixel agrees with the expected camera-space
//      Z within kDepthMatchTol (rules out bad LiDAR pixels and depth holes).
//   2. On LiDAR: the ARConfidenceLevel at that pixel is kMinLidarConf (= High).
//      This specifically rejects hair pixels, which produce low-confidence readings.
bool AppController::checkIncisionQuality(const cv::Point3d &pt,
                                          const cv::Mat     &rvec,
                                          const cv::Mat     &tvec,
                                          double             depthAnchor,
                                          const cv::Mat     &depthMap,
                                          const cv::Mat     &confidenceMap) const
{
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    const cv::Mat v = (cv::Mat_<double>(3,1) << pt.x, pt.y, pt.z);
    const cv::Mat ptCam = R * v + tvec.reshape(1, 3);
    const double expectedDepth = ptCam.at<double>(2);
    if (expectedDepth <= 0) return false;

    const cv::Point2f px = PoseUtils::project(pt, m_K, rvec, tvec, m_dist);
    const float rel = sampleDepthAt(depthMap, px);
    if (rel < 1e-4f) return false;

    const double measured = static_cast<double>(rel) * depthAnchor;
    if (std::abs(measured - expectedDepth) > kDepthMatchTol * expectedDepth) return false;

    if (m_usingLiDAR && !confidenceMap.empty()) {
        const int cx = std::max(0, std::min(confidenceMap.cols - 1, static_cast<int>(px.x)));
        const int cy = std::max(0, std::min(confidenceMap.rows - 1, static_cast<int>(px.y)));
        if (static_cast<int>(confidenceMap.at<uint8_t>(cy, cx)) < kMinLidarConf) return false;
    }

    return true;
}
#endif

// ── Pose fusion ───────────────────────────────────────────────────────────────

cv::Mat AppController::fusePoses(const std::vector<TagPose> &detections) const
{
    // ── 1. Per-tag poses + unified PnP point collection ───────────────────────
    // Each tag has a fixed absolute Leksell position stored in tag_config.json,
    // so a single visible tag is sufficient to establish the full frame pose.
    std::vector<cv::Mat>     framePoses;
    std::vector<cv::Point3f> objPts;
    std::vector<cv::Point2f> imgPts;

    for (const auto &det : detections) {
        const TagConfig *cfg = nullptr;
        for (const auto &c : m_tagConfigs)
            if (c.id == det.id) { cfg = &c; break; }
        if (!cfg) continue;

        // Per-tag frame pose (IPPE_SQUARE result)
        const cv::Mat T_cam_tag = PoseUtils::toTransform(det.rvec, det.tvec);
        framePoses.push_back(T_cam_tag * cfg->T_frame_tag.inv());

        // Leksell-frame 3-D positions of this tag's 4 corners.
        // cfg->T_frame_tag transforms tag-local points to Leksell frame:
        //   p_leksell = R_cfg * p_local + t_cfg
        if (det.corners.size() == 4) {
            const cv::Mat &R_cfg = cfg->R_frame_tag;
            const cv::Mat &t_cfg = cfg->t_frame_tag;
            const float h = m_markerSize / 2.f;
            const cv::Point3f local[4] = {
                {-h,  h, 0.f}, { h,  h, 0.f},
                { h, -h, 0.f}, {-h, -h, 0.f}
            };
            for (int c = 0; c < 4; ++c) {
                const cv::Mat p = (cv::Mat_<double>(3,1)
                    << local[c].x, local[c].y, local[c].z);
                const cv::Mat pf = R_cfg * p + t_cfg.reshape(1, 3);
                objPts.emplace_back(
                    static_cast<float>(pf.at<double>(0)),
                    static_cast<float>(pf.at<double>(1)),
                    static_cast<float>(pf.at<double>(2)));
                imgPts.push_back(det.corners[c]);
            }
        }
    }

    if (framePoses.empty()) return {};

    // ── 2. Fuse per-tag estimates in SO(3) → initial guess ───────────────────
    cv::Mat T_initial;
    if (framePoses.size() == 1) {
        T_initial = framePoses[0];
    } else {
        cv::Mat RSum    = cv::Mat::zeros(3, 3, CV_64F);
        cv::Mat tvecSum = cv::Mat::zeros(3, 1, CV_64F);
        for (const auto &T : framePoses) {
            cv::Mat r, t, R;
            PoseUtils::fromTransform(T, r, t);
            cv::Rodrigues(r, R);
            RSum    += R;
            tvecSum += t.reshape(1, 3);
        }
        const double n = static_cast<double>(framePoses.size());
        cv::Mat U, S, Vt;
        cv::SVD::compute(RSum, S, U, Vt);
        if (cv::determinant(U * Vt) < 0) U.col(2) *= -1;
        cv::Mat rFused;
        cv::Rodrigues(U * Vt, rFused);
        T_initial = PoseUtils::toTransform(rFused, tvecSum / n);
    }

    // ── 3. Unified multi-point PnP refinement ────────────────────────────────
    // When both tags are fully visible (8 correspondences), run one solvePnP
    // over all corners together. The wide inter-tag baseline makes the depth
    // axis far better constrained than any single-tag solve.
    if (objPts.size() >= 8) {
        cv::Mat rvec, tvec;
        PoseUtils::fromTransform(T_initial, rvec, tvec);
        const bool ok = cv::solvePnP(
            objPts, imgPts, m_K, m_dist,
            rvec, tvec, /*useExtrinsicGuess=*/true,
            cv::SOLVEPNP_ITERATIVE);
        if (ok)
            return PoseUtils::toTransform(rvec, tvec);
    }

    // SO(3)-fused per-tag estimate (single tag, or if unified PnP failed)
    return T_initial;
}

// ── Simple overlay renderer (shared by both AprilTag and ARKit paths) ─────────

// Draws both trajectory lines onto `out`. Caller must wrap with beginFrame/endFrame.
void AppController::renderOverlayOnto(cv::Mat &out,
                                      const cv::Mat &rvec,
                                      const cv::Mat &tvec)
{
    cv::Mat R;
    cv::Rodrigues(rvec, R);

    // Skip any sub-segment whose endpoint is behind the camera plane (Z ≤ 0).
    // Without this check, cv::projectPoints reflects behind-camera points through
    // the principal point, making the line appear to flip to the opposite side of
    // the image when the viewpoint is near-parallel to the trajectory.
    auto inFront = [&](const cv::Point3d &pt) -> bool {
        const cv::Mat v = (cv::Mat_<double>(3,1) << pt.x, pt.y, pt.z);
        const cv::Mat result = R * v + tvec.reshape(1, 3);
        return result.at<double>(2) > 0;
    };

    for (int i = 0; i < 2; ++i) {
        if (!m_lines[i]) continue;
        const auto &line = *m_lines[i];
        const cv::Point3d tgt = applyTagFrameRotation(line.target());
        const cv::Point3d end = applyTagFrameRotation(line.lineEnd());

        const cv::Point3d diff = { tgt.x-end.x, tgt.y-end.y, tgt.z-end.z };
        std::array<cv::Point3d, RAY_SAMPLES + 1> pts;
        std::array<bool,        RAY_SAMPLES + 1> ptVis;
        for (int s = 0; s <= RAY_SAMPLES; ++s) {
            const double t = static_cast<double>(s) / RAY_SAMPLES;
            pts[s]   = { end.x + t*diff.x, end.y + t*diff.y, end.z + t*diff.z };
            ptVis[s] = inFront(pts[s]);
        }
        for (int s = 0; s < RAY_SAMPLES; ++s) {
            if (ptVis[s] && ptVis[s+1])
                m_renderer->drawSegment(pts[s], pts[s+1], m_K, rvec, tvec);
        }
        if (ptVis[RAY_SAMPLES])
            m_renderer->drawTargetMarker(tgt, m_K, rvec, tvec);
    }
}

// ── Occlusion-aware overlay (Step 4) ─────────────────────────────────────────
// For each trajectory, walks 60 equal segments from skull entry to brain target.
// Each segment is projected to screen; its 3-D depth in camera space is compared
// to the MiDaS surface depth at that pixel (scaled to metres via depthAnchor).
// Only segments where the surface is farther than the point are drawn.

void AppController::renderWithOcclusion(cv::Mat       &out,
                                        const cv::Mat &rvec,
                                        const cv::Mat &tvec,
                                        const cv::Mat &depthMap,
                                        double         depthAnchor,
                                        const cv::Mat &confidenceMap)
{
    cv::Mat R;
    cv::Rodrigues(rvec, R);

    // Returns the estimated metric surface depth at the pixel corresponding to pt.
    // iOS: LiDAR metric depth convention — larger = farther; surfaceDepth = rel * anchor (= 1.0).
    // Desktop MiDaS: normalised disparity — larger = closer; surfaceDepth = anchor / rel.
    auto surfaceDepth = [&](const cv::Point3d &pt) -> double {
        const cv::Point2f px = PoseUtils::project(pt, m_K, rvec, tvec, m_dist);
        const float rel = sampleDepthAt(depthMap, px);
        if (rel < 1e-4f) return 1e9; // OOB or no data → assume no occlusion
#ifdef Q_OS_IOS
        return rel * depthAnchor;
#else
        return depthAnchor / rel;
#endif
    };

    // Returns the actual Z-depth of a frame-space point in camera space.
    auto cameraDepth = [&](const cv::Point3d &pt) -> double {
        const cv::Mat v = (cv::Mat_<double>(3,1) << pt.x, pt.y, pt.z);
        cv::Mat result = R * v + tvec.reshape(1, 3);
        return result.at<double>(2);
    };

    // A point is visible when the head surface (measured by MiDaS) is at least
    // as far as the point itself. The 0.95 factor gives a 5% depth tolerance so
    // points right at the surface don't flicker due to MiDaS noise.
    auto visible = [&](const cv::Point3d &pt) -> bool {
        const double cd = cameraDepth(pt);
        if (cd <= 0) return false;
        return surfaceDepth(pt) >= cd * 0.95;
    };

    for (int i = 0; i < 2; ++i) {
        if (!m_lines[i]) continue;
        const auto &line = *m_lines[i];
        const cv::Point3d tgt = applyTagFrameRotation(line.target());
        const cv::Point3d end = applyTagFrameRotation(line.lineEnd());

        // Pre-compute all RAY_SAMPLES+1 points and test visibility once each.
        // Adjacent segments share endpoints, so the naive per-segment approach
        // calls visible() twice for every interior point. Caching halves the
        // number of project()+sampleDepthAt() calls (120 → 61 per trajectory).
        const cv::Point3d diff = { tgt.x-end.x, tgt.y-end.y, tgt.z-end.z };
        std::array<cv::Point3d, RAY_SAMPLES + 1> pts;
        std::array<bool,        RAY_SAMPLES + 1> ptVis;

        for (int s = 0; s <= RAY_SAMPLES; ++s) {
            const double t = static_cast<double>(s) / RAY_SAMPLES;
            pts[s] = { end.x + t*diff.x, end.y + t*diff.y, end.z + t*diff.z };
        }

#ifdef Q_OS_IOS
        double t_inc = 0.0; // needed by both visibility computation and rendering below
        {
            // Lock the depth mutex to safely access m_lockedIncision.
            // Race condition was occurring on LiDAR phones where multiple threads
            // accessed this structure simultaneously (onARFrame vs renderWithOcclusion).
            std::lock_guard<std::mutex> lk(m_depthMutex);
            LockedIncision &lock = m_lockedIncision[i];
            if (lock.locked) {
                // Incision locked: derive visibility geometrically from the frozen
                // position — no depth sampling needed until recalibration resets the lock.
                const double diff_len_sq = diff.x*diff.x + diff.y*diff.y + diff.z*diff.z;
                t_inc = 1.0;
                if (diff_len_sq > 1e-12) {
                    const cv::Point3d off = {
                        lock.leksellPt.x - end.x,
                        lock.leksellPt.y - end.y,
                        lock.leksellPt.z - end.z
                    };
                    t_inc = (off.x*diff.x + off.y*diff.y + off.z*diff.z) / diff_len_sq;
                }
                for (int s = 0; s <= RAY_SAMPLES; ++s)
                    ptVis[s] = (static_cast<double>(s) / RAY_SAMPLES <= t_inc) && (cameraDepth(pts[s]) > 0);
            } else {
                for (int s = 0; s <= RAY_SAMPLES; ++s)
                    ptVis[s] = visible(pts[s]);
            }
        }
#else
        for (int s = 0; s <= RAY_SAMPLES; ++s)
            ptVis[s] = visible(pts[s]);
#endif

        for (int s = 0; s < RAY_SAMPLES; ++s) {
            if (ptVis[s] && ptVis[s+1])
                m_renderer->drawSegment(pts[s], pts[s+1], m_K, rvec, tvec);
        }

        if (ptVis[RAY_SAMPLES]) // tgt == pts[RAY_SAMPLES], already tested above
            m_renderer->drawTargetMarker(tgt, m_K, rvec, tvec);

#ifdef Q_OS_IOS
        {
            std::lock_guard<std::mutex> lk(m_depthMutex);
            LockedIncision &lock = m_lockedIncision[i];
            if (lock.locked) {
                // Fill the gap from the last full sample to the exact incision point.
                const int s_last = std::max(0, std::min(RAY_SAMPLES, static_cast<int>(t_inc * RAY_SAMPLES)));
                m_renderer->drawSegment(pts[s_last], lock.leksellPt, m_K, rvec, tvec);
                m_renderer->drawIncisionMarker(lock.leksellPt, m_K, rvec, tvec);
            } else {
                auto hit = findIncisionPoint(depthMap, rvec, tvec, depthAnchor, line);
                if (hit.has_value()) {
                    if (checkIncisionQuality(*hit, rvec, tvec, depthAnchor, depthMap, confidenceMap)) {
                        if (lock.streakCount == 0) {
                            lock.streakSum   = *hit;
                            lock.streakCount = 1;
                        } else {
                            const cv::Point3d avg = {
                                lock.streakSum.x / lock.streakCount,
                                lock.streakSum.y / lock.streakCount,
                                lock.streakSum.z / lock.streakCount
                            };
                            const cv::Point3d diff = {
                                hit->x - avg.x, hit->y - avg.y, hit->z - avg.z
                            };
                            const double dist = std::sqrt(
                                diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
                            if (dist < kIncisionLockRadius) {
                                lock.streakSum  += *hit;
                                lock.streakCount++;
                            } else {
                                lock.streakSum   = *hit;
                                lock.streakCount = 1;
                            }
                        }
                        if (lock.streakCount >= kIncisionLockFrames) {
                            lock.leksellPt = {
                                lock.streakSum.x / lock.streakCount,
                                lock.streakSum.y / lock.streakCount,
                                lock.streakSum.z / lock.streakCount
                            };
                            lock.locked = true;
                        }
                    } else {
                        lock.streakCount = 0;
                    }
                    m_renderer->drawIncisionMarker(*hit, m_K, rvec, tvec);
                } else {
                    lock.streakCount = 0;
                }
            }
        }
#else
        auto hit = findIncisionPoint(depthMap, rvec, tvec, depthAnchor, line);
        if (hit.has_value())
            m_renderer->drawIncisionMarker(hit.value(), m_K, rvec, tvec);
#endif
    }
}

// ── ARKit path ────────────────────────────────────────────────────────────────

#ifdef Q_OS_IOS

// ARKit camera (Y-up, Z-toward-viewer) → OpenCV camera (Y-down, Z-into-scene):
// flip Y and Z axes.
static const cv::Mat kARKitFlip = (cv::Mat_<double>(4, 4)
    << 1,  0,  0, 0,
       0, -1,  0, 0,
       0,  0, -1, 0,
       0,  0,  0, 1);

void AppController::resetARRegistration()
{
    const bool wasLocked = !m_T_world_leksell.empty();
    const int streakSize = static_cast<int>(m_streakPoses.size());
    DebugLogger::logEvent("AppController::resetARRegistration",
        QString("CALLED: wasLocked=%1, streakSize=%2/10, thread=0x%3")
        .arg(wasLocked ? 1 : 0).arg(streakSize)
        .arg(reinterpret_cast<quintptr>(QThread::currentThread()), 0, 16));

    m_T_world_leksell     = cv::Mat();
    m_anchorTrackingState = 2;
    m_depthAnchor         = m_usingLiDAR ? 1.0 : 0.0;
    m_prevWorldTCamera    = cv::Mat();
    m_lowFeatFrames       = 0;
    {
        std::lock_guard<std::mutex> lk(m_depthMutex);
        for (auto &lock : m_lockedIncision) lock = LockedIncision{};
    }
    if (!m_streakPoses.empty()) {
        m_streakPoses.clear();
        DebugLogger::logEvent("AppController::resetARRegistration", "STREAK CLEARED: emitting calibrationProgressChanged(false)");
        emit calibrationProgressChanged(false);
    }
    DebugLogger::logEvent("AppController::resetARRegistration", "RESET COMPLETE: emitting lockStateChanged(false)");
    emit lockStateChanged(false);
    DebugLogger::logEvent("AppController::resetARRegistration", "ALL SIGNALS EMITTED");
}

void AppController::onLidarAvailable(bool available)
{
    DebugLogger::logEvent("AppController::onLidarAvailable",
        QString("Depth source changed: %1").arg(available ? "LiDAR" : "none"));
    m_usingLiDAR  = available;
    m_depthAnchor = available ? 1.0 : 0.0;
    qDebug() << "AppController: depth source ="
             << (available ? "LiDAR" : "none (no LiDAR)");
}

void AppController::onTrackingQualityChanged(int state)
{
    const QString stateNames[] = {"Unavailable", "Limited", "Normal"};
    const QString stateName = (state >= 0 && state <= 2) ? stateNames[state] : "Unknown";
    const QString anchorName = (m_anchorTrackingState >= 0 && m_anchorTrackingState <= 2)
        ? stateNames[m_anchorTrackingState] : "Unknown";

    DebugLogger::logEvent("AppController::onTrackingQualityChanged",
        QString("state=%1 (%2), locked=%3, anchorState=%4 (%5)")
        .arg(state).arg(stateName)
        .arg(m_T_world_leksell.empty() ? 0 : 1)
        .arg(m_anchorTrackingState).arg(anchorName));

    m_trackingState = state;
    // Trigger re-calibration only once locked, and only when quality drops below
    // the level recorded at anchor time (so a lock done at "limited" quality won't
    // immediately reset on the next frame).
    if (!m_T_world_leksell.empty() && state < m_anchorTrackingState) {
        DebugLogger::logEvent("AppController::onTrackingQualityChanged",
            QString("TRIGGERING RESET: quality dropped from %1 to %2").arg(anchorName).arg(stateName));
        resetARRegistration();
    }
}

void AppController::onLidarDepth(const cv::Mat &depthMetric, const cv::Mat &confidence)
{
    std::lock_guard<std::mutex> lk(m_depthMutex);
    m_depthMapReady = depthMetric;
    m_confidenceMap = confidence;
}

void AppController::onARFrame(const cv::Mat &frame,
                               const cv::Mat &world_T_camera,
                               int featurePoints)
{
    // Log AR frame state at entry (sample every 30 frames to avoid spam)
    static int s_frameCounter = 0;
    if (++s_frameCounter % 30 == 0) {
        DebugLogger::logEvent("AppController::onARFrame",
            QString("frame #%1: locked=%2, streak=%3/%4, features=%5")
            .arg(s_frameCounter)
            .arg(m_T_world_leksell.empty() ? 0 : 1)
            .arg(m_streakPoses.size())
            .arg(10)
            .arg(featurePoints));
    }

    m_featurePointCount = featurePoints;
    m_frameTimer.restart();
    const bool anyLine = m_lines[0] || m_lines[1];

    // ARKit → OpenCV camera convention: flip Y and Z.
    const cv::Mat world_T_camera_cv = kARKitFlip * world_T_camera * kARKitFlip;

    // ── Tracking ──────────────────────────────────────────────────────────────
    cv::Mat T_cam_leksell;

    // Debug metrics populated during init phase; displayed in depth-overlay panel.
    int    dbgCorners  = 0;
    double dbgAngleDeg = -1.0; // negative = not computable (no pose)
    double dbgReprojPx = -1.0;

    if (m_T_world_leksell.empty()) {
        // Init phase: detect tags and test quality conditions for locking.
        const auto    detections  = m_tracker->detect(frame);
        const cv::Mat T_from_tags = fusePoses(detections);

        // Collect metrics for the debug overlay.
        for (const auto &d : detections)
            dbgCorners += static_cast<int>(d.corners.size());
        if (!T_from_tags.empty()) {
            cv::Mat rvec, tvec, R;
            PoseUtils::fromTransform(T_from_tags, rvec, tvec);
            cv::Rodrigues(rvec, R);

            // Compute reproj error without frame rotation: same as computeReprojError but uses
            // only position (t_frame_tag), ignoring rotation (R_frame_tag). This verifies that
            // tags are at correct positions, but isn't affected by rotation calibration errors.
            dbgReprojPx = computeReprojErrorNoFrameRotation(detections, rvec, tvec);

            // Compute angle to the tag plane normal (completely configuration-independent)
            // Use individual camera-to-tag-marker poses and fuse rotations via SO(3) averaging
            std::vector<cv::Mat> rotations_marker;
            for (const auto &det : detections) {
                if (det.corners.size() == 4) {
                    cv::Mat R_det;
                    cv::Rodrigues(det.rvec, R_det);
                    rotations_marker.push_back(R_det);
                }
            }

            if (!rotations_marker.empty()) {
                cv::Mat R_fused;
                if (rotations_marker.size() == 1) {
                    R_fused = rotations_marker[0].clone();
                } else {
                    // SO(3) averaging: sum rotation matrices and compute mean via SVD
                    cv::Mat RSum = cv::Mat::zeros(3, 3, CV_64F);
                    for (const auto &R : rotations_marker)
                        RSum += R;
                    cv::Mat U, S, Vt;
                    cv::SVD::compute(RSum, S, U, Vt);
                    if (cv::determinant(U * Vt) < 0) U.col(2) *= -1;
                    R_fused = U * Vt;
                }

                // Compute angle from fused camera-to-tag-marker rotation
                cv::Mat tag_normal = (cv::Mat_<double>(3, 1) << 0, 0, -1);
                cv::Mat tag_normal_cam = R_fused.t() * tag_normal;
                const double cosA = std::max(-1.0, std::min(1.0, -tag_normal_cam.at<double>(2, 0)));
                dbgAngleDeg = std::acos(cosA) * 180.0 / M_PI;
            }
        }

        if (meetsInitConditions(detections, T_from_tags)) {
            // Accumulate qualifying frames; lock only after 10 consecutive successes.
            if (m_streakPoses.empty()) {
                DebugLogger::logEvent("AppController::onARFrame", "STREAK STARTED: first qualifying frame");
                emit calibrationProgressChanged(true); // streak just started
            }
            m_streakPoses.push_back(world_T_camera_cv * T_from_tags);
            DebugLogger::logEvent("AppController::onARFrame",
                QString("STREAK PROGRESS: %1/10 frames (corners=%2, angle=%.1f°, reproj=%.2fpx)")
                .arg(m_streakPoses.size()).arg(dbgCorners).arg(dbgAngleDeg).arg(dbgReprojPx));

            if (static_cast<int>(m_streakPoses.size()) >= 10) {
                // SO(3)-average the 10 world_T_leksell candidates.
                cv::Mat RSum = cv::Mat::zeros(3, 3, CV_64F);
                cv::Mat tSum = cv::Mat::zeros(3, 1, CV_64F);
                for (const auto &T : m_streakPoses) {
                    RSum += T.rowRange(0, 3).colRange(0, 3);
                    tSum += T.rowRange(0, 3).col(3);
                }
                cv::Mat U, S, Vt;
                cv::SVD::compute(RSum, S, U, Vt);
                if (cv::determinant(U * Vt) < 0) U.col(2) *= -1;

                const cv::Mat R_avg = U * Vt;
                const cv::Mat t_avg = tSum / static_cast<double>(m_streakPoses.size());

                // Build the world-to-Leksell transformation (without tag-frame tilt;
                // tilt is applied per-object using tag coordinates from tag_config.json)
                m_T_world_leksell = cv::Mat::eye(4, 4, CV_64F);
                R_avg.copyTo(m_T_world_leksell.rowRange(0, 3).colRange(0, 3));
                t_avg.copyTo(m_T_world_leksell.rowRange(0, 3).col(3));

                m_anchorTrackingState = m_trackingState;
                m_streakPoses.clear();
                DebugLogger::logEvent("AppController::onARFrame", "LOCKED: emitting calibrationProgressChanged(false) and lockStateChanged(true)");
                emit calibrationProgressChanged(false);
                emit lockStateChanged(true);
                DebugLogger::logEvent("AppController::onARFrame", "LOCKED: signals emitted successfully");
            }
        } else {
            // Streak broken — reset accumulator.
            if (!m_streakPoses.empty()) {
                DebugLogger::logEvent("AppController::onARFrame",
                    QString("STREAK BROKEN at frame %1/10: conditions not met (corners=%2, angle=%.1f°, reproj=%.2fpx)")
                    .arg(m_streakPoses.size()).arg(dbgCorners).arg(dbgAngleDeg).arg(dbgReprojPx));
                m_streakPoses.clear();
                emit calibrationProgressChanged(false);
            }
        }
        // T_cam_leksell stays empty → overlay hidden until locked.
    } else {
        // Locked phase: derive pose purely from ARKit world tracking.
        T_cam_leksell = world_T_camera_cv.inv() * m_T_world_leksell;

        // Re-calibrate if features stay low while the camera is actively moving.
        // Disabled on LiDAR devices: ARKit uses scene geometry instead of visual
        // features, so rawFeaturePoints.count is naturally low even during healthy tracking.
        if (!m_usingLiDAR && !m_prevWorldTCamera.empty()) {
            const cv::Mat t_now  = world_T_camera_cv.col(3).rowRange(0, 3);
            const cv::Mat t_prev = m_prevWorldTCamera.col(3).rowRange(0, 3);
            const double transDelta = cv::norm(t_now - t_prev);

            const cv::Mat R_now   = world_T_camera_cv.rowRange(0, 3).colRange(0, 3);
            const cv::Mat R_prev  = m_prevWorldTCamera.rowRange(0, 3).colRange(0, 3);
            const cv::Mat R_delta = R_now * R_prev.t();
            const double  tr      = R_delta.at<double>(0,0) + R_delta.at<double>(1,1)
                                  + R_delta.at<double>(2,2);
            const double rotDelta = std::acos(std::max(-1.0, std::min(1.0, (tr - 1.0) / 2.0)));

            const bool moving = transDelta > m_moveTransThresh || rotDelta > (m_moveRotThresh * M_PI / 180.0);
            if (moving && featurePoints < 30)
                ++m_lowFeatFrames;
            else
                m_lowFeatFrames = 0;

            if (m_lowFeatFrames >= 10)
                resetARRegistration();
        }
    }
    m_prevWorldTCamera = world_T_camera_cv.clone();

    // ── Depth map ─────────────────────────────────────────────────────────────
    cv::Mat depthMap, confidenceMap;
    {
        std::lock_guard<std::mutex> lk(m_depthMutex);
        depthMap      = m_depthMapReady;
        confidenceMap = m_confidenceMap;
    }

    // ── Render ─────────────────────────────────────────────────────────────────
    cv::Mat out = frame.clone();

    // Draw angle indicator during calibration phase (same gate as debug diagnostics)
    if (!m_showDepthOverlay && m_T_world_leksell.empty() && dbgAngleDeg >= 0.0) {
        // Color: red if below 175, blue if 175 or above
        const bool belowThreshold = dbgAngleDeg < 175.0;
        const QColor color = belowThreshold
            ? QColor(222, 95, 94, 255)      // IMPULSE_RED #DE5F5E
            : QColor(117, 208, 197, 255);   // ARC_BLUE #75D0C5

        // Format the text with translated message and angle value (no degree symbol due to encoding)
        QString angleMsg = tr("Angle : %1 (> 175 requis)").arg(dbgAngleDeg, 0, 'f', 1);

        // Load Diagramm-Bold font once
        static int fontId = QFontDatabase::addApplicationFont(":/resources/Diagramm-Bold.ttf");
        static QString fontFamily = fontId != -1 ? QFontDatabase::applicationFontFamilies(fontId).first() : "Arial";

        QColor shadowColor(0, 0, 0, 255);
        QFont font(fontFamily, 54);
        font.setBold(true);
        QFontMetrics fm(font);

        // Calculate text dimensions
        int textW = fm.horizontalAdvance(angleMsg) + 8;
        int textH = fm.height() + 8;

        // Create image for text with explicit RGBA format
        QImage textImg(textW, textH, QImage::Format_ARGB32);
        textImg.fill(Qt::transparent);

        QPainter p(&textImg);
        p.setFont(font);
        p.setRenderHint(QPainter::Antialiasing, true);
        p.setRenderHint(QPainter::SmoothPixmapTransform, true);

        // Draw shadow (black, offset)
        p.setPen(shadowColor);
        p.drawText(3, fm.ascent() + 3, angleMsg);

        // Draw text with full opacity
        p.setPen(color);
        p.drawText(1, fm.ascent() + 1, angleMsg);
        p.end();

        // Composite onto frame at position (20, 70), adjusting for larger font size
        int startX = 20;
        int startY = 105 - fm.ascent();

        // Only render if text fits within frame bounds
        if (startY + textImg.height() <= out.rows && startY >= 0) {
            for (int dy = 0; dy < textImg.height() && startY + dy < out.rows; ++dy) {
                const QRgb *src = reinterpret_cast<const QRgb *>(textImg.scanLine(dy));
                uchar *dst = out.ptr<uchar>(startY + dy, startX);

                for (int dx = 0; dx < textImg.width() && startX + dx < out.cols; ++dx) {
                    const quint32 px = src[dx];
                    const int a = qAlpha(px);
                    if (a == 0) {
                        dst += 3;
                        continue;
                    }

                    const int ia = 255 - a;
                    const int r = qRed(px);
                    const int g = qGreen(px);
                    const int b = qBlue(px);

                    dst[0] = static_cast<uchar>(((dst[0] * ia) >> 8) + b);
                    dst[1] = static_cast<uchar>(((dst[1] * ia) >> 8) + g);
                    dst[2] = static_cast<uchar>(((dst[2] * ia) >> 8) + r);
                    dst += 3;
                }
            }
        }
    }

    if (!m_showDepthOverlay && (T_cam_leksell.empty() || !anyLine)) {
        m_lastFrameMs = m_frameTimer.elapsed();
        emit frameReady(out);
        return;
    }

    // Debug diagnostics — test AR mode only
    if (m_showDepthOverlay) {
        int y = 100;
        const int lineH = 88;

        // Load Diagramm-Bold font once
        static int fontId = QFontDatabase::addApplicationFont(":/resources/Diagramm-Bold.ttf");
        static QString fontFamily = fontId != -1 ? QFontDatabase::applicationFontFamilies(fontId).first() : "Arial";

        // Brand colors for angle indication
        const QColor IMPULSE_RED(222, 95, 94, 255);    // #DE5F5E
        const QColor ARC_BLUE(117, 208, 197, 255);     // #75D0C5

        auto dbg = [&](const std::string &msg, bool ok, bool isAngle = false) {
            // Render using Qt with Diagramm-Bold font
            // Use brand colors for angle indication, standard colors for others
            QColor textColor = isAngle ? (ok ? ARC_BLUE : IMPULSE_RED)
                                       : (ok ? QColor(50, 220, 50, 255) : QColor(255, 50, 50, 255));
            QColor shadowColor(0, 0, 0, 255);
            QFont font(fontFamily, 45);
            QFontMetrics fm(font);
            QString qmsg = QString::fromStdString(msg);

            // Calculate text dimensions
            int textW = fm.horizontalAdvance(qmsg) + 8;
            int textH = fm.height() + 8;

            // Create image for text with explicit RGBA format
            QImage textImg(textW, textH, QImage::Format_ARGB32);
            textImg.fill(Qt::transparent);

            QPainter p(&textImg);
            p.setFont(font);
            p.setRenderHint(QPainter::Antialiasing, true);
            p.setRenderHint(QPainter::SmoothPixmapTransform, true);

            // Draw shadow (black, offset)
            p.setPen(shadowColor);
            p.drawText(3, fm.ascent() + 3, qmsg);

            // Draw text with full opacity
            p.setPen(textColor);
            p.drawText(1, fm.ascent() + 1, qmsg);
            p.end();

            // Composite onto frame at position (20, y), but clamp to frame bounds
            int startX = 20;
            int startY = y - fm.ascent();

            // Only render if text fits within frame bounds
            if (startY + textImg.height() <= out.rows && startY >= 0) {
                for (int dy = 0; dy < textImg.height() && startY + dy < out.rows; ++dy) {
                    const QRgb *src = reinterpret_cast<const QRgb *>(textImg.scanLine(dy));
                    uchar *dst = out.ptr<uchar>(startY + dy, startX);

                    for (int dx = 0; dx < textImg.width() && startX + dx < out.cols; ++dx) {
                        const quint32 px = src[dx];
                        const int a = qAlpha(px);
                        if (a == 0) {
                            dst += 3;
                            continue;
                        }

                        const int ia = 255 - a;
                        const int r = qRed(px);
                        const int g = qGreen(px);
                        const int b = qBlue(px);

                        dst[0] = static_cast<uchar>(((dst[0] * ia) >> 8) + b);
                        dst[1] = static_cast<uchar>(((dst[1] * ia) >> 8) + g);
                        dst[2] = static_cast<uchar>(((dst[2] * ia) >> 8) + r);
                        dst += 3;
                    }
                }
            }

            if (startY + textImg.height() <= out.rows) {
                y += lineH;
            }
        };
        dbg(m_showDepthOverlay ? "overlay flag: ON" : "overlay flag: OFF",
            m_showDepthOverlay);
        dbg(m_usingLiDAR ? "depth source: LiDAR" : "depth source: None", m_usingLiDAR);
        {
            char buf[64];
            if (!m_T_world_leksell.empty()) {
                dbg("anchor: ESTABLISHED", true);
            } else if (!m_streakPoses.empty()) {
                std::snprintf(buf, sizeof(buf), "anchor: streak %d / 10",
                              static_cast<int>(m_streakPoses.size()));
                dbg(buf, false);
            } else {
                dbg("anchor: calibrating...", false);
            }
        }
        dbg(m_trackingState == 2 ? "ARKit: normal"
            : m_trackingState == 1 ? "ARKit: LIMITED"
            : "ARKit: UNAVAILABLE",
            m_trackingState == 2);
        {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "features: %d", m_featurePointCount);
            dbg(buf, m_featurePointCount >= 50);
        }
        if (!m_T_world_leksell.empty()) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "drift guard: %d / 10", m_lowFeatFrames);
            dbg(buf, m_lowFeatFrames == 0);
        }
        if (m_usingLiDAR && !m_T_world_leksell.empty()) {
            std::lock_guard<std::mutex> lk(m_depthMutex);
            const char *const sides[2] = {"L", "R"};
            for (int i = 0; i < 2; ++i) {
                if (!m_lines[i]) continue;
                char buf[64];
                const LockedIncision &lock = m_lockedIncision[i];
                if (lock.locked) {
                    std::snprintf(buf, sizeof(buf), "incision %s: LOCKED", sides[i]);
                    dbg(buf, true);
                } else if (lock.streakCount > 0) {
                    std::snprintf(buf, sizeof(buf), "incision %s: streak %d / %d",
                                  sides[i], lock.streakCount, kIncisionLockFrames);
                    dbg(buf, false);
                } else {
                    std::snprintf(buf, sizeof(buf), "incision %s: searching", sides[i]);
                    dbg(buf, false);
                }
            }
        }
        {
            char buf[64];
            if (m_T_world_leksell.empty()) {
                std::snprintf(buf, sizeof(buf), "corners: %d / 8", dbgCorners);
                dbg(buf, dbgCorners >= 8);
                if (dbgAngleDeg >= 0.0) {
                    std::snprintf(buf, sizeof(buf), "angle:  %.1f deg  (need >=175)", dbgAngleDeg);
                    dbg(buf, dbgAngleDeg >= 175.0);
                    std::snprintf(buf, sizeof(buf), "reproj: %.2f px  (need <=%.2f)", dbgReprojPx, m_maxInitReprojPx);
                    dbg(buf, dbgReprojPx >= 0.0 && dbgReprojPx <= m_maxInitReprojPx);
                } else {
                    dbg("angle:  -- (no pose)", false);
                    dbg("reproj: -- (no pose)", false);
                }
            }
        }
        dbg(depthMap.empty() ? "depthMap: EMPTY" :
            "depthMap: " + std::to_string(depthMap.cols) + "x" + std::to_string(depthMap.rows),
            !depthMap.empty());
        if (m_usingLiDAR)
            dbg(confidenceMap.empty() ? "conf map: EMPTY" :
                "conf map: " + std::to_string(confidenceMap.cols) + "x"
                              + std::to_string(confidenceMap.rows),
                !confidenceMap.empty());
    }


    // Depth visualization overlay: red = close, blue = far.
    if (m_showDepthOverlay && m_showDepthVisualization && !depthMap.empty()) {
        cv::Mat vizDepth;
        cv::normalize(depthMap, vizDepth, 0.0, 1.0, cv::NORM_MINMAX);
        cv::Mat invDepth = 1.0f - vizDepth;
        cv::Mat depth8u, colored;
        invDepth.convertTo(depth8u, CV_8U, 255.0);
        cv::applyColorMap(depth8u, colored, cv::COLORMAP_JET);
        cv::addWeighted(out, 0.7, colored, 0.3, 0, out);
    }

    if (T_cam_leksell.empty() || !anyLine) {
        m_lastFrameMs = m_frameTimer.elapsed();
        emit frameReady(out);
        return;
    }

    cv::Mat rvec, tvec;
    PoseUtils::fromTransform(T_cam_leksell, rvec, tvec);

    m_renderer->beginFrame(out);
    if (m_showDepthOverlay)
        cv::drawFrameAxes(out, m_K, m_dist, rvec, tvec, 0.05f);

    if (!depthMap.empty() && m_depthAnchor > 1e-9) {
        renderWithOcclusion(out, rvec, tvec, depthMap, m_depthAnchor, confidenceMap);
    } else {
        renderOverlayOnto(out, rvec, tvec);
    }

    m_renderer->endFrame();

    // ── Angle indicator during calibration phase ───────────────────────────────
    // Display angle on regular AR screen using same rendering as debug lines
    if (m_T_world_leksell.empty() && dbgAngleDeg >= 0.0) {
        static int fontId = QFontDatabase::addApplicationFont(":/resources/Diagramm-Regular.ttf");
        static QString fontFamily = fontId != -1 ? QFontDatabase::applicationFontFamilies(fontId).first() : "Arial";

        const QColor IMPULSE_RED(222, 95, 94, 255);    // #DE5F5E
        const QColor ARC_BLUE(117, 208, 197, 255);     // #75D0C5

        // Determine color based on angle threshold
        QColor angleColor = (dbgAngleDeg >= 175.0) ? ARC_BLUE : IMPULSE_RED;

        // Format angle message
        char angleBuf[64];
        std::snprintf(angleBuf, sizeof(angleBuf), "Angle : %.1f", dbgAngleDeg);
        QString angleMsg = QString::fromUtf8(angleBuf);

        // Render angle using same process as debug lines
        QFont font(fontFamily, 36);
        QFontMetrics fm(font);

        int textW = fm.horizontalAdvance(angleMsg) + 8;
        int textH = fm.height() + 8;

        QImage textImg(textW, textH, QImage::Format_ARGB32);
        textImg.fill(Qt::transparent);

        QPainter p(&textImg);
        p.setFont(font);
        p.setRenderHint(QPainter::Antialiasing, true);

        // Draw shadow (black, offset)
        p.setPen(QColor(0, 0, 0, 255));
        p.drawText(3, fm.ascent() + 3, angleMsg);

        // Draw angle text
        p.setPen(angleColor);
        p.drawText(1, fm.ascent() + 1, angleMsg);
        p.end();

        // Composite onto frame at position (20, 70)
        int startX = 20;
        int startY = 70 - fm.ascent();

        if (startY + textImg.height() <= out.rows && startY >= 0) {
            for (int dy = 0; dy < textImg.height() && startY + dy < out.rows; ++dy) {
                const QRgb *src = reinterpret_cast<const QRgb *>(textImg.scanLine(dy));
                uchar *dst = out.ptr<uchar>(startY + dy, startX);

                for (int dx = 0; dx < textImg.width() && startX + dx < out.cols; ++dx) {
                    const quint32 px = src[dx];
                    const int a = qAlpha(px);
                    if (a == 0) {
                        dst += 3;
                        continue;
                    }

                    const int ia = 255 - a;
                    const int r = qRed(px);
                    const int g = qGreen(px);
                    const int b = qBlue(px);

                    dst[0] = static_cast<uchar>(((dst[0] * ia) >> 8) + b);
                    dst[1] = static_cast<uchar>(((dst[1] * ia) >> 8) + g);
                    dst[2] = static_cast<uchar>(((dst[2] * ia) >> 8) + r);
                    dst += 3;
                }
            }
        }
    }

    m_lastFrameMs = m_frameTimer.elapsed();
    emit frameReady(out);
}

bool AppController::meetsInitConditions(const std::vector<TagPose> &detections,
                                        const cv::Mat              &T_from_tags) const
{
    if (T_from_tags.empty() || detections.size() < 2) return false;

    int corners = 0;
    for (const auto &d : detections)
        corners += static_cast<int>(d.corners.size());
    if (corners < 8) return false;

    cv::Mat rvec, tvec, R;
    PoseUtils::fromTransform(T_from_tags, rvec, tvec);
    cv::Rodrigues(rvec, R);

    // Compute angle to the tag plane normal (completely configuration-independent)
    // Use individual camera-to-tag-marker poses and fuse rotations via SO(3) averaging
    std::vector<cv::Mat> rotations_marker;
    for (const auto &det : detections) {
        if (det.corners.size() == 4) {
            cv::Mat R_det;
            cv::Rodrigues(det.rvec, R_det);
            rotations_marker.push_back(R_det);
        }
    }

    double angleDeg = 0.0;
    if (!rotations_marker.empty()) {
        cv::Mat R_fused;
        if (rotations_marker.size() == 1) {
            R_fused = rotations_marker[0].clone();
        } else {
            // SO(3) averaging: sum rotation matrices and compute mean via SVD
            cv::Mat RSum = cv::Mat::zeros(3, 3, CV_64F);
            for (const auto &R : rotations_marker)
                RSum += R;
            cv::Mat U, S, Vt;
            cv::SVD::compute(RSum, S, U, Vt);
            if (cv::determinant(U * Vt) < 0) U.col(2) *= -1;
            R_fused = U * Vt;
        }

        // Compute angle from fused camera-to-tag-marker rotation
        cv::Mat tag_normal = (cv::Mat_<double>(3, 1) << 0, 0, -1);
        cv::Mat tag_normal_cam = R_fused.t() * tag_normal;
        const double cosA = std::max(-1.0, std::min(1.0, -tag_normal_cam.at<double>(2, 0)));
        angleDeg = std::acos(cosA) * 180.0 / M_PI;
    }

    if (angleDeg < 175.0) return false;

    // Compute reproj error without frame rotation: same as computeReprojError but uses
    // only position (t_frame_tag), ignoring rotation (R_frame_tag). This verifies that
    // tags are at correct positions, but isn't affected by rotation calibration errors.
    return computeReprojErrorNoFrameRotation(detections, rvec, tvec) <= m_maxInitReprojPx;
}

double AppController::computeReprojError(const std::vector<TagPose> &detections,
                                         const cv::Mat              &rvec,
                                         const cv::Mat              &tvec) const
{
    std::vector<cv::Point3f> objPts;
    std::vector<cv::Point2f> imgPts;

    for (const auto &det : detections) {
        const TagConfig *cfg = nullptr;
        for (const auto &c : m_tagConfigs)
            if (c.id == det.id) { cfg = &c; break; }
        if (!cfg || det.corners.size() != 4) continue;

        const float h = m_markerSize / 2.f;
        const cv::Point3f local[4] = {
            {-h,  h, 0.f}, { h,  h, 0.f},
            { h, -h, 0.f}, {-h, -h, 0.f}
        };
        for (int c = 0; c < 4; ++c) {
            const cv::Mat p = (cv::Mat_<double>(3,1)
                << local[c].x, local[c].y, local[c].z);
            // Ensure proper shape for matrix addition
            cv::Mat t_col = cfg->t_frame_tag.clone();
            if (t_col.rows == 1 && t_col.cols == 3) {
                t_col = t_col.t();  // Transpose (1, 3) to (3, 1)
            }
            const cv::Mat pf = cfg->R_frame_tag * p + t_col;
            objPts.emplace_back(
                static_cast<float>(pf.at<double>(0)),
                static_cast<float>(pf.at<double>(1)),
                static_cast<float>(pf.at<double>(2)));
            imgPts.push_back(det.corners[c]);
        }
    }

    if (objPts.empty()) return 1e9;

    std::vector<cv::Point2f> projected;
    cv::projectPoints(objPts, rvec, tvec, m_K, m_dist, projected);

    double err = 0.0;
    for (size_t i = 0; i < projected.size(); ++i) {
        const double dx = projected[i].x - imgPts[i].x;
        const double dy = projected[i].y - imgPts[i].y;
        err += dx * dx + dy * dy;
    }
    return std::sqrt(err / projected.size());
}

double AppController::computeReprojErrorNoFrameRotation(const std::vector<TagPose> &detections,
                                                         const cv::Mat              &rvec,
                                                         const cv::Mat              &tvec) const
{
    std::vector<cv::Point3f> objPts;
    std::vector<cv::Point2f> imgPts;

    // CRITICAL TRANSFORMATION CHAIN (verify this is correct):
    // - rvec, tvec: decomposed from T_from_tags via PoseUtils::fromTransform
    // - T_from_tags: camera-to-Leksell transform (from fusePoses)
    // - cv::projectPoints(pts, rvec, tvec): expects object->camera transform
    //   So pts should be in Leksell frame, and [R|t] should transform Leksell->camera
    //   But we have camera->Leksell from T_from_tags, so cv::projectPoints may need inversion!
    // Extract fused camera-to-Leksell rotation
    cv::Mat R_fused;
    cv::Rodrigues(rvec, R_fused);

    for (const auto &det : detections) {
        const TagConfig *cfg = nullptr;
        for (const auto &c : m_tagConfigs)
            if (c.id == det.id) { cfg = &c; break; }
        if (!cfg || det.corners.size() != 4) continue;

        // Extract detected camera-to-marker rotation and position
        cv::Mat R_det;
        cv::Rodrigues(det.rvec, R_det);

        // CRITICAL FIX: Use DETECTED rotation with CONFIGURED position
        // But: configured position must be interpreted relative to DETECTED rotation, not configured rotation!
        //
        // The configured position (cfg->t_frame_tag) is the marker's origin position in Leksell.
        // It's defined for the configured rotation (cfg->R_frame_tag).
        // But the marker's ACTUAL rotation is different (R_det from detection).
        //
        // The marker corners in Leksell space are:
        //   p_leksell = R_actual * p_marker + t_actual
        //
        // where R_actual is the detected rotation (via R_marker_leksell)
        // and t_actual is cfg->t_frame_tag (the configured position is absolute, not rotation-dependent)
        //
        // This way:
        // - Reproj error STILL verifies that tags are at configured positions ✓
        // - Reproj error is NOT affected by rotation calibration (uses detected rotation) ✓

        // Derive marker-to-Leksell rotation: R_fused * R_det
        // Chain: marker → camera (R_det) → Leksell (R_fused)
        cv::Mat R_marker_leksell = R_fused * R_det;

        // Use configured position directly (position is absolute, independent of rotation)
        cv::Mat t_col = cfg->t_frame_tag.clone();
        if (t_col.rows == 1 && t_col.cols == 3) {
            t_col = t_col.t();
        }

        const float h = m_markerSize / 2.f;
        const cv::Point3f local[4] = {
            {-h,  h, 0.f}, { h,  h, 0.f},
            { h, -h, 0.f}, {-h, -h, 0.f}
        };
        for (int c = 0; c < 4; ++c) {
            const cv::Mat p = (cv::Mat_<double>(3,1)
                << local[c].x, local[c].y, local[c].z);
            // Transform using DETECTED rotation + CONFIGURED position
            // This verifies position while being independent of rotation calibration errors
            const cv::Mat pf = R_marker_leksell * p + t_col;
            objPts.emplace_back(
                static_cast<float>(pf.at<double>(0)),
                static_cast<float>(pf.at<double>(1)),
                static_cast<float>(pf.at<double>(2)));
            imgPts.push_back(det.corners[c]);
        }
    }

    if (objPts.empty()) return 1e9;

    std::vector<cv::Point2f> projected;
    cv::projectPoints(objPts, rvec, tvec, m_K, m_dist, projected);

    double err = 0.0;
    for (size_t i = 0; i < projected.size(); ++i) {
        const double dx = projected[i].x - imgPts[i].x;
        const double dy = projected[i].y - imgPts[i].y;
        err += dx * dx + dy * dy;
    }
    return std::sqrt(err / projected.size());
}

#endif // Q_OS_IOS

// ── Loaders ───────────────────────────────────────────────────────────────────

cv::Mat AppController::loadCalibration(const QString &path)
{
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) return {};
    const auto obj = QJsonDocument::fromJson(f.readAll()).object();
    m_dist = (cv::Mat_<double>(1, 4)
        << obj["k1"].toDouble(0), obj["k2"].toDouble(0),
           obj["p1"].toDouble(0), obj["p2"].toDouble(0));
    return (cv::Mat_<double>(3, 3) <<
        obj["fx"].toDouble(900), 0,                       obj["cx"].toDouble(640),
        0,                       obj["fy"].toDouble(900),  obj["cy"].toDouble(360),
        0,                       0,                        1);
}

std::vector<TagConfig> AppController::loadTagConfigs(const QString &path)
{
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) return {};
    const auto root = QJsonDocument::fromJson(f.readAll()).object();
    m_markerSize = static_cast<float>(root["marker_size_m"].toDouble(0.05));
    const auto arr = root["tags"].toArray();
    std::vector<TagConfig> configs;
    for (const auto &entry : arr) {
        const auto obj = entry.toObject();
        TagConfig cfg;
        cfg.id = obj["id"].toInt();
        cv::Mat rvec = (cv::Mat_<double>(3,1)
            << obj["rx_rad"].toDouble(0), obj["ry_rad"].toDouble(0), obj["rz_rad"].toDouble(0));
        // Load position as independent (3, 1) directly from JSON — NOT derived from T_frame_tag
        cfg.t_frame_tag = (cv::Mat_<double>(3,1)
            << obj["tx_m"].toDouble(0), obj["ty_m"].toDouble(0), obj["tz_m"].toDouble(0));
        // Create T_frame_tag from independent components
        cfg.T_frame_tag = PoseUtils::toTransform(rvec, cfg.t_frame_tag);
        // Rotation matrix directly from rvec — NOT derived from T_frame_tag decomposition
        cv::Rodrigues(rvec, cfg.R_frame_tag);
        configs.push_back(std::move(cfg));
    }
    return configs;
}

cv::Point3d AppController::applyTagFrameRotation(const cv::Point3d &point_leksell) const
{
    if (m_tagConfigs.empty()) return point_leksell;

    const TagConfig &tag = m_tagConfigs[0]; // use first tag as reference frame

    // T_frame_tag transforms from tag frame to Leksell frame
    // We need the inverse: from Leksell to tag frame
    cv::Mat T_inv = cv::Mat::zeros(4, 4, CV_64F);
    T_inv.at<double>(3, 3) = 1.0;

    // Invert: R_inv = R^T, t_inv = -R^T * t
    cv::Mat R = tag.T_frame_tag.rowRange(0, 3).colRange(0, 3);
    cv::Mat t = tag.T_frame_tag.rowRange(0, 3).col(3);
    cv::Mat R_inv = R.t();
    cv::Mat t_inv = -R_inv * t;

    R_inv.copyTo(T_inv.rowRange(0, 3).colRange(0, 3));
    t_inv.copyTo(T_inv.rowRange(0, 3).col(3));

    // Transform to tag frame
    cv::Mat p_leksell = (cv::Mat_<double>(4, 1) << point_leksell.x, point_leksell.y, point_leksell.z, 1.0);
    cv::Mat p_tag = T_inv * p_leksell;

    // Transform back to Leksell frame using T_frame_tag which includes all rotation/translation
    // from the calibrated tag configuration (rx, ry, rz, tx, ty, tz)
    cv::Mat p_final = tag.T_frame_tag * p_tag;

    return cv::Point3d(p_final.at<double>(0, 0),
                       p_final.at<double>(1, 0),
                       p_final.at<double>(2, 0));
}
