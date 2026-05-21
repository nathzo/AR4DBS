#include "AppController.h"
#include "core/tracking/AprilTagTracker.h"
#include "core/math/IncisionLine.h"
#include "core/math/PoseUtils.h"
#include "core/rendering/OverlayRenderer.h"
#include "core/depth/DepthEstimator.h"
#ifdef Q_OS_IOS
#include "platform/ios/CoreMLDepthEstimator.h"
#endif

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>

#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QDebug>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

static constexpr int    RAY_SAMPLES     = 60;
static float sampleDepthAt(const cv::Mat &depthMap, cv::Point2f p); // defined below
static constexpr double DEPTH_TOLERANCE = 0.25;
static constexpr int    DEPTH_SAMPLE_R  = 5;

AppController::AppController(QObject *parent) : QObject(parent) {}
AppController::~AppController()
{
#ifdef Q_OS_IOS
    // Wait for the model-load thread and any in-flight inference before destroying members.
    while (m_depthModelLoading.load() || m_depthInFlight.load())
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
#endif
}

bool AppController::init(const QString &calibPath,
                         const QString &tagConfigPath,
                         const QString &planPath,
                         const QString &depthModelPath)
{
    m_K = loadCalibration(calibPath); // also populates m_dist
    if (m_K.empty()) {
        qWarning() << "AppController: could not load calibration from" << calibPath;
        return false;
    }

    m_tagConfigs = loadTagConfigs(tagConfigPath);
    if (m_tagConfigs.empty()) {
        qWarning() << "AppController: could not load tag config from" << tagConfigPath;
        return false;
    }

    m_tracker  = std::make_unique<AprilTagTracker>(m_K, m_dist, m_markerSize);
    m_renderer = std::make_unique<OverlayRenderer>();
    m_renderer->setDistortion(m_dist);

    if (!depthModelPath.isEmpty()) {
#ifdef Q_OS_IOS
        // CoreML model loading triggers ANE compilation on first run, which can block
        // for >20 s and trips the iOS scene-create watchdog (0x8BADF00D).
        // Load on a background thread so the UI appears immediately.
        // All accesses to m_iosDepth on the camera thread are gated on
        // m_depthModelReady (acquire), which is set here with release ordering
        // only after m_iosDepth is fully constructed and safe to use.
        m_depthModelLoading.store(true);
        std::string modelPath = depthModelPath.toStdString();
        std::thread([this, modelPath]() {
            auto est = std::make_unique<CoreMLDepthEstimator>(modelPath);
            if (est->isLoaded()) {
                m_iosDepth = std::move(est);
                m_depthModelReady.store(true, std::memory_order_release);
                qDebug() << "AppController: iOS CoreML depth estimation enabled";
            } else {
                qWarning() << "AppController: CoreML depth model failed to load";
            }
            m_depthModelLoading.store(false);
        }).detach();
#else
        m_depth = std::make_unique<DepthEstimator>(depthModelPath.toStdString());
        if (!m_depth->isLoaded()) {
            qWarning() << "AppController: depth model not loaded from" << depthModelPath;
            m_depth.reset();
        } else {
            qDebug() << "AppController: depth estimation enabled";
        }
#endif
    }

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

void AppController::setSurgicalPlan(const SurgicalPlan &plan)
{
    m_lines[0] = plan.hasLeft()
        ? std::make_unique<IncisionLine>(IncisionLine::fromLeksell(plan.left))
        : nullptr;
    m_lines[1] = plan.hasRight()
        ? std::make_unique<IncisionLine>(IncisionLine::fromLeksell(plan.right))
        : nullptr;
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

    const cv::Point3d &tgt = line.target();
    const cv::Point3d &end = line.lineEnd();

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
        // Depth convention (LiDAR and Depth Anything v2 Metric): larger = farther.
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
        const cv::Point3d &tgt = line.target();
        const cv::Point3d &end = line.lineEnd();

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
                                        double         depthAnchor)
{
    cv::Mat R;
    cv::Rodrigues(rvec, R);

    // Returns the estimated metric surface depth at the pixel corresponding to pt.
    // iOS (LiDAR and Depth Anything v2 Metric): depth convention — larger = farther.
    //   surfaceDepth = rel * anchor  (anchor ≈ 1.0 for metric model, = 1.0 for LiDAR)
    // Desktop MiDaS (normalised disparity — larger = closer):
    //   surfaceDepth = anchor / rel
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
        const cv::Point3d &tgt = line.target();
        const cv::Point3d &end = line.lineEnd();

        // Pre-compute all RAY_SAMPLES+1 points and test visibility once each.
        // Adjacent segments share endpoints, so the naive per-segment approach
        // calls visible() twice for every interior point. Caching halves the
        // number of project()+sampleDepthAt() calls (120 → 61 per trajectory).
        const cv::Point3d diff = { tgt.x-end.x, tgt.y-end.y, tgt.z-end.z };
        std::array<cv::Point3d, RAY_SAMPLES + 1> pts;
        std::array<bool,        RAY_SAMPLES + 1> ptVis;
        for (int s = 0; s <= RAY_SAMPLES; ++s) {
            const double t = static_cast<double>(s) / RAY_SAMPLES;
            pts[s]   = { end.x + t*diff.x, end.y + t*diff.y, end.z + t*diff.z };
            ptVis[s] = visible(pts[s]);
        }
        for (int s = 0; s < RAY_SAMPLES; ++s) {
            if (ptVis[s] && ptVis[s+1])
                m_renderer->drawSegment(pts[s], pts[s+1], m_K, rvec, tvec);
        }

        if (ptVis[RAY_SAMPLES]) // tgt == pts[RAY_SAMPLES], already tested above
            m_renderer->drawTargetMarker(tgt, m_K, rvec, tvec);

        auto hit = findIncisionPoint(depthMap, rvec, tvec, depthAnchor, line);
        if (hit.has_value())
            m_renderer->drawIncisionMarker(hit.value(), m_K, rvec, tvec);
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
    m_T_world_leksell     = cv::Mat();
    m_anchorTrackingState = 2;
    m_depthAnchor         = m_usingLiDAR ? 1.0 : 0.0;
    m_prevWorldTCamera    = cv::Mat();
    m_lowFeatFrames       = 0;
    emit lockStateChanged(false);
}

void AppController::onLidarAvailable(bool available)
{
    m_usingLiDAR  = available;
    m_depthAnchor = available ? 1.0 : 0.0;
    qDebug() << "AppController: depth source ="
             << (available ? "LiDAR" : "Depth Anything v2 Metric");
}

void AppController::onTrackingQualityChanged(int state)
{
    m_trackingState = state;
    // Trigger re-calibration only once locked, and only when quality drops below
    // the level recorded at anchor time (so a lock done at "limited" quality won't
    // immediately reset on the next frame).
    if (!m_T_world_leksell.empty() && state < m_anchorTrackingState)
        resetARRegistration();
}

void AppController::onLidarDepth(const cv::Mat &depthMetric)
{
    std::lock_guard<std::mutex> lk(m_depthMutex);
    m_depthMapReady = depthMetric;
}

void AppController::onARFrame(const cv::Mat &frame,
                               const cv::Mat &world_T_camera,
                               int featurePoints)
{
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
            const double cosA = std::max(-1.0, std::min(1.0, -R.at<double>(2, 2)));
            dbgAngleDeg = std::acos(cosA) * 180.0 / M_PI;
            dbgReprojPx = computeReprojError(detections, rvec, tvec);
        }

        // Update depth anchor from visible tags (Depth Anything v2 only).
        if (!m_usingLiDAR && !detections.empty()) {
            cv::Mat depthForAnchor;
            {
                std::lock_guard<std::mutex> lk(m_depthMutex);
                depthForAnchor = m_depthMapReady;
            }
            if (!depthForAnchor.empty()) {
                double anchorSum   = 0.0;
                int    anchorCount = 0;
                for (const auto &det : detections) {
                    const double      depth = det.tvec.at<double>(2);
                    const cv::Point2f px    = PoseUtils::project(
                        cv::Point3d(0,0,0), m_K, det.rvec, det.tvec, m_dist);
                    const float rel = sampleDepthAt(depthForAnchor, px);
                    if (rel > 1e-4f) {
                        anchorSum += depth / rel;
                        ++anchorCount;
                    }
                }
                if (anchorCount > 0) {
                    const double newAnchor = anchorSum / anchorCount;
                    m_depthAnchor = (m_depthAnchor < 1e-9)
                        ? newAnchor
                        : (1.0 - kAnchorAlpha) * m_depthAnchor + kAnchorAlpha * newAnchor;
                }
            }
        }

        if (meetsInitConditions(detections, T_from_tags)) {
            m_T_world_leksell     = world_T_camera_cv * T_from_tags; // world_T_leksell
            m_anchorTrackingState = m_trackingState; // record quality at lock time
            emit lockStateChanged(true);
        }
        // T_cam_leksell stays empty → overlay hidden until locked.
    } else {
        // Locked phase: derive pose purely from ARKit world tracking.
        T_cam_leksell = world_T_camera_cv.inv() * m_T_world_leksell;

        // Re-calibrate if features stay low while the camera is actively moving.
        // Low count at rest is normal (IMU takes over); only flag it during motion.
        if (!m_prevWorldTCamera.empty()) {
            const cv::Mat t_now  = world_T_camera_cv.col(3).rowRange(0, 3);
            const cv::Mat t_prev = m_prevWorldTCamera.col(3).rowRange(0, 3);
            const double transDelta = cv::norm(t_now - t_prev);

            const cv::Mat R_now   = world_T_camera_cv.rowRange(0, 3).colRange(0, 3);
            const cv::Mat R_prev  = m_prevWorldTCamera.rowRange(0, 3).colRange(0, 3);
            const cv::Mat R_delta = R_now * R_prev.t();
            const double  tr      = R_delta.at<double>(0,0) + R_delta.at<double>(1,1)
                                  + R_delta.at<double>(2,2);
            const double rotDelta = std::acos(std::max(-1.0, std::min(1.0, (tr - 1.0) / 2.0)));

            const bool moving = transDelta > 0.003 || rotDelta > (0.3 * M_PI / 180.0);
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
    cv::Mat depthMap;
    {
        std::lock_guard<std::mutex> lk(m_depthMutex);
        depthMap = m_depthMapReady;
    }

    if (m_mlDepthEnabled && !m_usingLiDAR
            && m_depthModelReady.load(std::memory_order_acquire)
            && !m_depthInFlight.exchange(true)) {
        std::thread([this, frame]() mutable {
            cv::Mat depth = m_iosDepth->estimate(frame);
            {
                std::lock_guard<std::mutex> lk(m_depthMutex);
                m_depthMapReady = depth;
            }
            m_depthInFlight.store(false);
        }).detach();
    }

    // ── Render ─────────────────────────────────────────────────────────────────
    if (!m_showDepthOverlay && (T_cam_leksell.empty() || !anyLine)) {
        m_lastFrameMs = m_frameTimer.elapsed();
        emit frameReady(frame);
        return;
    }

    cv::Mat out = frame.clone();

    // Debug diagnostics — test AR mode only
    if (m_showDepthOverlay) {
        int y = 100;
        const int lineH = 70;
        auto dbg = [&](const std::string &msg, bool ok) {
            const cv::Scalar shadow(0, 0, 0);
            const cv::Scalar color = ok ? cv::Scalar(50, 220, 50) : cv::Scalar(50, 50, 255);
            cv::putText(out, msg, {22, y+2}, cv::FONT_HERSHEY_SIMPLEX, 1.5, shadow, 5, cv::LINE_AA);
            cv::putText(out, msg, {20, y},   cv::FONT_HERSHEY_SIMPLEX, 1.5, color,  2, cv::LINE_AA);
            y += lineH;
        };
        dbg(m_showDepthOverlay ? "overlay flag: ON" : "overlay flag: OFF",
            m_showDepthOverlay);
        dbg(m_usingLiDAR ? "depth source: LiDAR" : "depth source: DA v2 Metric",
            true);
        dbg(m_usingLiDAR ? "iosDepth model: N/A (LiDAR)"
            : !m_mlDepthEnabled ? "iosDepth model: disabled"
            : m_depthModelReady.load() ? "iosDepth model: LOADED"
            : m_depthModelLoading.load() ? "iosDepth model: loading..."
            : "iosDepth model: NULL",
            m_usingLiDAR || !m_mlDepthEnabled || m_depthModelReady.load());
        dbg(!m_T_world_leksell.empty() ? "anchor: ESTABLISHED" : "anchor: calibrating...",
            !m_T_world_leksell.empty());
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
        {
            char buf[64];
            if (m_T_world_leksell.empty()) {
                std::snprintf(buf, sizeof(buf), "corners: %d / 8", dbgCorners);
                dbg(buf, dbgCorners >= 8);
                if (dbgAngleDeg >= 0.0) {
                    std::snprintf(buf, sizeof(buf), "angle:  %.1f deg  (need >=175)", dbgAngleDeg);
                    dbg(buf, dbgAngleDeg >= 175.0);
                    std::snprintf(buf, sizeof(buf), "reproj: %.2f px  (need <=0.8)", dbgReprojPx);
                    dbg(buf, dbgReprojPx >= 0.0 && dbgReprojPx <= kMaxInitReprojPx);
                } else {
                    dbg("angle:  -- (no pose)", false);
                    dbg("reproj: -- (no pose)", false);
                }
            }
        }
        if (!m_usingLiDAR && m_mlDepthEnabled)
            dbg(m_depthInFlight.load() ? "depth: IN FLIGHT" : "depth: idle",
                !m_depthInFlight.load());
        dbg(depthMap.empty() ? "depthMap: EMPTY" :
            "depthMap: " + std::to_string(depthMap.cols) + "x" + std::to_string(depthMap.rows),
            !depthMap.empty());
    }

    // Depth visualization overlay: red = close, blue = far.
    if (m_showDepthOverlay && !depthMap.empty()) {
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
        renderWithOcclusion(out, rvec, tvec, depthMap, m_depthAnchor);
    } else {
        renderOverlayOnto(out, rvec, tvec);
    }

    m_renderer->endFrame();

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

    cv::Mat rvec, tvec;
    PoseUtils::fromTransform(T_from_tags, rvec, tvec);

    cv::Mat R;
    cv::Rodrigues(rvec, R);
    // cosA = -R(2,2). Face-on gives cosA ≈ -1 (angle ≈ 180°).
    // Valid when angle ≥ 175°, i.e. cosA ≤ -kMaxInitAngleCos ≈ -0.9962.
    if (-R.at<double>(2, 2) > -kMaxInitAngleCos) return false;

    return computeReprojError(detections, rvec, tvec) <= kMaxInitReprojPx;
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
            const cv::Mat pf = cfg->R_frame_tag * p + cfg->t_frame_tag.reshape(1, 3);
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
        cv::Mat tvec = (cv::Mat_<double>(3,1)
            << obj["tx_m"].toDouble(0), obj["ty_m"].toDouble(0), obj["tz_m"].toDouble(0));
        cfg.T_frame_tag = PoseUtils::toTransform(rvec, tvec);
        cv::Mat r_tmp;
        PoseUtils::fromTransform(cfg.T_frame_tag, r_tmp, cfg.t_frame_tag);
        cv::Rodrigues(r_tmp, cfg.R_frame_tag);
        configs.push_back(std::move(cfg));
    }
    return configs;
}
