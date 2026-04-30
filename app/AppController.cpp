#include "AppController.h"
#include "core/tracking/AprilTagTracker.h"
#include "core/math/IncisionLine.h"
#include "core/math/PoseUtils.h"
#include "core/rendering/OverlayRenderer.h"
#include "core/depth/DepthEstimator.h"

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
AppController::~AppController() = default;

bool AppController::init(const QString &calibPath,
                         const QString &tagConfigPath,
                         const QString &planPath,
                         const QString &depthModelPath)
{
    m_K    = loadCalibration(calibPath);
    m_dist = cv::Mat::zeros(1, 4, CV_64F);
    if (m_K.empty()) {
        qWarning() << "AppController: could not load calibration from" << calibPath;
        return false;
    }

    m_tagConfigs = loadTagConfigs(tagConfigPath);
    if (m_tagConfigs.empty()) {
        qWarning() << "AppController: could not load tag config from" << tagConfigPath;
        return false;
    }

    m_tracker  = std::make_unique<AprilTagTracker>(m_K, m_dist, 0.05f);
    m_renderer = std::make_unique<OverlayRenderer>();

    if (!depthModelPath.isEmpty()) {
        m_depth = std::make_unique<DepthEstimator>(depthModelPath.toStdString());
        if (!m_depth->isLoaded()) {
            qWarning() << "AppController: depth model not loaded from" << depthModelPath;
            m_depth.reset();
        } else {
            qDebug() << "AppController: depth estimation enabled";
        }
    }

    return true;
}

void AppController::setCalibration(const cv::Mat &K)
{
    m_K = K.clone();
    m_tracker = std::make_unique<AprilTagTracker>(m_K, m_dist, 0.05f);
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

    // Depth map computed once and shared across both trajectories.
    // Skipped when the previous frame already took > 50 ms.
    cv::Mat depthMap;
    cv::Point2f tagPx;
    double tagMetricDepth = 0;
    float  relTag = 0.f;
    if (m_depth && !detections.empty() && m_lastFrameMs < 50) {
        depthMap       = m_depth->estimate(frame);
        tagMetricDepth = cv::norm(detections[0].tvec);
        tagPx          = PoseUtils::project(
            cv::Point3d(0,0,0), m_K, detections[0].rvec, detections[0].tvec);
        relTag         = sampleDepthAt(depthMap, tagPx);
    }

    const bool hasDepth = !depthMap.empty() && relTag > 1e-4f;

    m_renderer->beginFrame(out);

    if (!hasDepth) {
        // Simple mode: draw both trajectories fully
        for (int i = 0; i < 2; ++i) {
            if (!m_lines[i]) continue;
            const auto &line = *m_lines[i];
            m_renderer->drawSegment(line.lineEnd(), line.target(), m_K, rvec, tvec);
            m_renderer->drawTargetMarker(line.target(), m_K, rvec, tvec);
        }
    } else {
        // Occlusion-aware mode — R, lambdas computed once, reused for both lines
        cv::Mat R;
        cv::Rodrigues(rvec, R);

        auto surfaceDepth = [&](const cv::Point3d &pt) -> double {
            const cv::Point2f px = PoseUtils::project(pt, m_K, rvec, tvec);
            const float rel = sampleDepthAt(depthMap, px);
            if (rel < 1e-4f) return 1e9;
            return tagMetricDepth * relTag / rel;
        };

        auto cameraDepth = [&](const cv::Point3d &pt) -> double {
            const cv::Mat v = (cv::Mat_<double>(3,1) << pt.x, pt.y, pt.z);
            const cv::Mat ptCam = R * v + tvec.reshape(1,3);
            return ptCam.at<double>(2);
        };

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
                const cv::Point3d p0 = {
                    end.x + t0*(tgt.x-end.x),
                    end.y + t0*(tgt.y-end.y),
                    end.z + t0*(tgt.z-end.z)
                };
                const cv::Point3d p1 = {
                    end.x + t1*(tgt.x-end.x),
                    end.y + t1*(tgt.y-end.y),
                    end.z + t1*(tgt.z-end.z)
                };
                if (visible(p0) && visible(p1))
                    m_renderer->drawSegment(p0, p1, m_K, rvec, tvec);
            }

            if (visible(tgt))
                m_renderer->drawTargetMarker(tgt, m_K, rvec, tvec);

            auto hit = findIncisionPoint(depthMap, rvec, tvec, tagPx, tagMetricDepth, line);
            if (hit.has_value())
                m_renderer->drawIncisionMarker(hit.value(), m_K, rvec, tvec);
        }
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
    const cv::Point2f &tagPx,
    double             tagMetricDepth,
    const IncisionLine &line) const
{
    if (depthMap.empty()) return std::nullopt;

    const float relTag = sampleDepthAt(depthMap, tagPx);
    if (relTag < 1e-4f) return std::nullopt;

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

        const cv::Point2f px = PoseUtils::project(pt, m_K, rvec, tvec);
        const float relPt = sampleDepthAt(depthMap, px);
        if (relPt < 1e-4f) continue;

        const double estimatedDepth = tagMetricDepth * relTag / relPt;
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

// ── Loaders ───────────────────────────────────────────────────────────────────

cv::Mat AppController::loadCalibration(const QString &path)
{
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) return {};
    const auto obj = QJsonDocument::fromJson(f.readAll()).object();
    return (cv::Mat_<double>(3, 3) <<
        obj["fx"].toDouble(900), 0,                       obj["cx"].toDouble(640),
        0,                       obj["fy"].toDouble(900),  obj["cy"].toDouble(360),
        0,                       0,                        1);
}

std::vector<TagConfig> AppController::loadTagConfigs(const QString &path)
{
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) return {};
    const auto arr = QJsonDocument::fromJson(f.readAll()).object()["tags"].toArray();
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
