#pragma once
#include <QObject>
#include <QElapsedTimer>
#include <opencv2/core.hpp>
#include <memory>
#include <optional>
#include <vector>

#include "core/math/SurgicalPlan.h"

// Register cv::Mat with Qt's meta-type system so it can be passed through
// queued (cross-thread) signal/slot connections.
Q_DECLARE_METATYPE(cv::Mat)

class AprilTagTracker;
class IncisionLine;
class OverlayRenderer;
class DepthEstimator;
struct TagPose;

struct TagConfig {
    int     id;
    cv::Mat T_frame_tag; // 4x4 CV_64F homogeneous transform
};

class AppController : public QObject
{
    Q_OBJECT
public:
    explicit AppController(QObject *parent = nullptr);
    ~AppController();

    // planPath is optional: if empty, no trajectory is drawn until
    // setSurgicalPlan() is called.
    bool init(const QString &calibrationPath,
              const QString &tagConfigPath,
              const QString &planPath      = QString(),
              const QString &depthModelPath = QString());

public slots:
    void onNewFrame(const cv::Mat &frame);
    void setCalibration(const cv::Mat &K);
    void setSurgicalPlan(const SurgicalPlan &plan);

#ifdef Q_OS_IOS
    // ARKit path: pose is provided by ARKit instead of solvePnP every frame.
    // onARFrame is called from MainWindow's busy-guard lambda on the worker thread.
    void onARFrame(const cv::Mat &frame, const cv::Mat &world_T_camera);
    void resetARRegistration(); // call before each AR session start
#endif

signals:
    void frameReady(const cv::Mat &annotated);

private:
    cv::Mat                        loadCalibration(const QString &path);
    std::vector<TagConfig>         loadTagConfigs(const QString &path);

    cv::Mat fusePoses(const std::vector<TagPose> &detections) const;

    // Draws trajectory overlay onto `out` using the given pose.
    // Shared between the AprilTag path (onNewFrame) and the ARKit path (onARFrame).
    void renderOverlayOnto(cv::Mat &out,
                           const cv::Mat &rvec,
                           const cv::Mat &tvec);

    std::optional<cv::Point3d> findIncisionPoint(
        const cv::Mat     &depthMap,
        const cv::Mat     &rvec,
        const cv::Mat     &tvec,
        const cv::Point2f &tagPx,
        double             tagMetricDepth,
        const IncisionLine &line) const;

    cv::Mat m_K;
    cv::Mat m_dist;
    float   m_markerSize = 0.05f; // overwritten from tag_config.json marker_size_m
    std::vector<TagConfig>           m_tagConfigs;
    std::unique_ptr<AprilTagTracker> m_tracker;
    std::unique_ptr<OverlayRenderer> m_renderer;
    std::unique_ptr<DepthEstimator>  m_depth;

    // Up to two active trajectories: [0]=left, [1]=right
    // nullptr means that side is inactive
    std::unique_ptr<IncisionLine> m_lines[2];

    QElapsedTimer m_frameTimer;
    qint64        m_lastFrameMs = 0;

#ifdef Q_OS_IOS
    // Empty until at least one tag detection succeeds.
    // Refreshed every frame tags are visible; ARKit falls back to it when not.
    cv::Mat m_world_T_frame;
#endif
};
