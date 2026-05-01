#include "AppController.h"
#include "core/tracking/AprilTagTracker.h"
#include "core/math/IncisionLine.h"
#include "core/math/PoseUtils.h"
#include "core/rendering/OverlayRenderer.h"
#include "core/depth/DepthEstimator.h"
#ifdef Q_OS_IOS
#include "platform/ios/CoreMLDepthEstimator.h"
#endif

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
static constexpr int kDepthThrottleFrames = 30;

AppController::AppController(QObject *parent) : QObject(parent) {}
AppController::~AppController() = default;

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
        // Step 3: on iOS use the CoreML estimator (Neural Engine, no LiDAR).
        m_iosDepth = std::make_unique<CoreMLDepthEstimator>(depthModelPath.toStdString());
        if (!m_iosDepth->isLoaded()) {
            qWarning() << "AppController: CoreML depth model not loaded from" << depthModelPath;
            m_iosDepth.reset();
        } else {
            qDebug() << "AppController: iOS CoreML depth estimation enabled";
        }
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

        const double estimatedDepth = depthAnchor / relPt;
        if (std::abs(estimatedDepth - expectedDepth) < DEPTH_TOLERANCE * expectedDepth)
            return pt;
    }
    return std::nullopt;
}

// ── Pose fusion ───────────────────────────────────────────────────────────────

cv::Mat AppController::fusePoses(const std::vector<TagPose> &detections) const
{
    std::vector<cv::Mat> framePoses;
    for (const auto &det : detections) {
        const TagConfig *cfg = nullptr;
        for (const auto &c : m_tagConfigs)
            if (c.id == det.id) { cfg = &c; break; }
        if (!cfg) continue;
        const cv::Mat T_cam_tag   = PoseUtils::toTransform(det.rvec, det.tvec);
        const cv::Mat T_cam_frame = T_cam_tag * cfg->T_frame_tag.inv();
        framePoses.push_back(T_cam_frame);
    }
    if (framePoses.empty()) return {};
    if (framePoses.size() == 1) return framePoses[0];

    cv::Mat tvecSum = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat rvecSum = cv::Mat::zeros(3, 1, CV_64F);
    for (const auto &T : framePoses) {
        cv::Mat r, t;
        PoseUtils::fromTransform(T, r, t);
        tvecSum += t.reshape(1, 3);
        rvecSum += r.reshape(1, 3);
    }
    const double n = static_cast<double>(framePoses.size());
    return PoseUtils::toTransform(rvecSum / n, tvecSum / n);
}

// ── Simple overlay renderer (shared by both AprilTag and ARKit paths) ─────────

// Draws both trajectory lines onto `out`. Caller must wrap with beginFrame/endFrame.
void AppController::renderOverlayOnto(cv::Mat &out,
                                      const cv::Mat &rvec,
                                      const cv::Mat &tvec)
{
    for (int i = 0; i < 2; ++i) {
        if (!m_lines[i]) continue;
        const auto &line = *m_lines[i];
        m_renderer->drawSegment(line.lineEnd(), line.target(), m_K, rvec, tvec);
        m_renderer->drawTargetMarker(line.target(), m_K, rvec, tvec);
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
    // depthAnchor / rel(px) converts the unitless MiDaS value to metres.
    auto surfaceDepth = [&](const cv::Point3d &pt) -> double {
        const cv::Point2f px = PoseUtils::project(pt, m_K, rvec, tvec, m_dist);
        const float rel = sampleDepthAt(depthMap, px);
        if (rel < 1e-4f) return 1e9;
        return depthAnchor / rel;
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

        for (int s = 0; s < RAY_SAMPLES; ++s) {
            const double t0 = static_cast<double>(s)   / RAY_SAMPLES;
            const double t1 = static_cast<double>(s+1) / RAY_SAMPLES;
            const cv::Point3d p0 = {end.x + t0*(tgt.x-end.x),
                                    end.y + t0*(tgt.y-end.y),
                                    end.z + t0*(tgt.z-end.z)};
            const cv::Point3d p1 = {end.x + t1*(tgt.x-end.x),
                                    end.y + t1*(tgt.y-end.y),
                                    end.z + t1*(tgt.z-end.z)};
            if (visible(p0) && visible(p1))
                m_renderer->drawSegment(p0, p1, m_K, rvec, tvec);
        }

        if (visible(tgt))
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
    m_T_cam_frame_filt    = cv::Mat();
    m_world_T_camera_prev = cv::Mat();
    m_depthAnchor         = 0.0;
    m_depthFrameCount     = 0;
}

void AppController::onARFrame(const cv::Mat &frame,
                               const cv::Mat &world_T_camera)
{
    m_frameTimer.restart();
    const bool anyLine = m_lines[0] || m_lines[1];

    // ── 1. Prediction: propagate last filtered pose using ARKit's Δpose ──────
    // ARKit gives accurate relative motion between frames even when its absolute
    // world position drifts. We use it to predict where the frame is now.
    cv::Mat T_cam_frame;
    if (!m_T_cam_frame_filt.empty() && !m_world_T_camera_prev.empty()) {
        // cam_new_T_cam_old in ARKit space, converted to OpenCV camera space.
        // kARKitFlip is its own inverse (it is a reflection), so:
        //   T_opencv = kARKitFlip * T_arkit * kARKitFlip
        const cv::Mat delta_arkit = world_T_camera.inv() * m_world_T_camera_prev;
        const cv::Mat delta_cam   = kARKitFlip * delta_arkit * kARKitFlip;
        T_cam_frame = delta_cam * m_T_cam_frame_filt;
    }

    // ── 2. Measurement: detect tags and compute absolute pose ────────────────
    const auto    detections  = m_tracker->detect(frame);
    const cv::Mat T_from_tags = fusePoses(detections);

    // ── 3. Update: blend prediction with measurement ─────────────────────────
    if (!T_from_tags.empty()) {
        if (T_cam_frame.empty()) {
            // No prediction yet (first frame) — initialise directly from tags.
            T_cam_frame = T_from_tags;
        } else {
            // Complementary filter — blend in SO(3), not in Rodrigues space.
            // Linear rvec blending fails when |rvec| ≈ π because the same
            // rotation has two representations (rvec and -rvec); blending across
            // that sign boundary produces a nonsense rotation (~180° flip).
            // Blending rotation matrices and re-orthogonalizing via SVD is safe.
            cv::Mat r_pred, t_pred, r_meas, t_meas;
            PoseUtils::fromTransform(T_cam_frame, r_pred, t_pred);
            PoseUtils::fromTransform(T_from_tags, r_meas, t_meas);

            cv::Mat R_pred, R_meas;
            cv::Rodrigues(r_pred, R_pred);
            cv::Rodrigues(r_meas, R_meas);
            cv::Mat R_raw = (1.0 - kAlpha) * R_pred + kAlpha * R_meas;

            // SVD projects the blended matrix back onto SO(3).
            // The det check flips a reflection (det=-1) to a proper rotation.
            cv::Mat U, S, Vt;
            cv::SVD::compute(R_raw, S, U, Vt);
            if (cv::determinant(U * Vt) < 0) U.col(2) *= -1;
            cv::Mat r_fused;
            cv::Rodrigues(U * Vt, r_fused);

            const cv::Mat t_fused = (1.0 - kAlpha) * t_pred + kAlpha * t_meas;
            T_cam_frame = PoseUtils::toTransform(r_fused, t_fused);
        }
    }
    // If no prediction and no tags: T_cam_frame stays empty → no overlay.

    // ── 4. Store state for next frame ────────────────────────────────────────
    m_world_T_camera_prev = world_T_camera.clone();
    m_T_cam_frame_filt    = T_cam_frame.clone(); // empty clone is still empty

    // ── 5. Depth estimation and anchor update ────────────────────────────────
    // Tags visible → run depth every frame so the metric anchor is always fresh.
    // Tags absent  → run depth every kDepthThrottleFrames frames to spare CPU
    //                (anchor persists from last tag sighting for occlusion rendering).
    const bool tagsVisible = !detections.empty();
    ++m_depthFrameCount;
    cv::Mat depthMap;
    if (m_iosDepth && (tagsVisible || m_depthFrameCount >= kDepthThrottleFrames)) {
        if (tagsVisible) m_depthFrameCount = 0;
        depthMap = m_iosDepth->estimate(frame);
        if (!depthMap.empty() && tagsVisible) {
            const double tagMetricDepth = cv::norm(detections[0].tvec);
            const cv::Point2f tagPx = PoseUtils::project(
                cv::Point3d(0,0,0), m_K,
                detections[0].rvec, detections[0].tvec, m_dist);
            const float relTag = sampleDepthAt(depthMap, tagPx);
            if (relTag > 1e-4f)
                m_depthAnchor = tagMetricDepth * relTag;
        }
    }

    // ── 6. Render ─────────────────────────────────────────────────────────────
    cv::Mat out = frame.clone();
#ifndef NDEBUG
    m_tracker->drawAxes(out, detections);
#endif

    if (T_cam_frame.empty() || !anyLine) {
        m_lastFrameMs = m_frameTimer.elapsed();
        emit frameReady(out);
        return;
    }

    cv::Mat rvec, tvec;
    PoseUtils::fromTransform(T_cam_frame, rvec, tvec);
    m_renderer->beginFrame(out);

    if (!depthMap.empty() && m_depthAnchor > 1e-9) {
        renderWithOcclusion(out, rvec, tvec, depthMap, m_depthAnchor);
    } else {
        renderOverlayOnto(out, rvec, tvec);
    }

    m_renderer->endFrame();

    m_lastFrameMs = m_frameTimer.elapsed();
    emit frameReady(out);
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
        configs.push_back(std::move(cfg));
    }
    return configs;
}
