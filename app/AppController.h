#pragma once
#include <QObject>
#include <QElapsedTimer>
#include <opencv2/core.hpp>
#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
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

#ifdef Q_OS_IOS
class CoreMLDepthEstimator;
#endif

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
    void setShowDepthOverlay(bool show);

#ifdef Q_OS_IOS
    // ARKit path: pose is provided by ARKit instead of solvePnP every frame.
    // onARFrame is called from MainWindow's busy-guard lambda on the worker thread.
    void onARFrame(const cv::Mat &frame, const cv::Mat &world_T_camera);
    void resetARRegistration(); // call before each AR session start

    // LiDAR integration — connected to ARKitSession signals.
    void onLidarAvailable(bool available);
    void onLidarDepth(const cv::Mat &depthMetric);
#endif

signals:
    void frameReady(const cv::Mat &annotated);

private:
    cv::Mat                        loadCalibration(const QString &path);
    std::vector<TagConfig>         loadTagConfigs(const QString &path);

    cv::Mat fusePoses(const std::vector<TagPose> &detections) const;

    // Full overlay without occlusion (fallback when no depth is available).
    void renderOverlayOnto(cv::Mat &out,
                           const cv::Mat &rvec,
                           const cv::Mat &tvec);

    // Occlusion-aware overlay. depthAnchor = metric_depth_tag × rel_depth_tag;
    // used by both onNewFrame (desktop) and onARFrame (iOS).
    void renderWithOcclusion(cv::Mat       &out,
                             const cv::Mat &rvec,
                             const cv::Mat &tvec,
                             const cv::Mat &depthMap,
                             double         depthAnchor);

    // Walks the trajectory and returns the first point where the line enters
    // the head surface. depthAnchor = metric_depth_tag × rel_depth_tag.
    std::optional<cv::Point3d> findIncisionPoint(
        const cv::Mat     &depthMap,
        const cv::Mat     &rvec,
        const cv::Mat     &tvec,
        double             depthAnchor,
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

    bool m_showDepthOverlay = false;

#ifdef Q_OS_IOS
    cv::Mat m_T_cam_frame_filt;    // filtered pose state; empty until first tag seen
    cv::Mat m_world_T_camera_prev; // previous ARKit pose, for computing Δ

    // CoreML monocular depth estimator (Neural Engine, no LiDAR).
    std::unique_ptr<CoreMLDepthEstimator> m_iosDepth;

    // true when the device has a LiDAR scanner and ARKit is providing scene depth.
    // When true, CoreML inference is skipped and m_depthAnchor is fixed at 1.0.
    bool m_usingLiDAR = false;

    // Scale anchor: converts the depth map to metric metres.
    //   LiDAR path:          always 1.0 (LiDAR values are already in metres).
    //   Depth Anything v2:   tagMetricDepth * relTag (disparity convention), EMA-smoothed.
    double m_depthAnchor = 0.0;

    // Tag measurement blend weight. Small = smooth/slow correction; large = fast/noisy.
    static constexpr double kAlpha = 0.07;
    // Depth anchor EMA weight. Slow convergence keeps the incision marker stable
    // across frames even when per-frame relTag sampling is noisy.
    static constexpr double kAnchorAlpha = 0.05;

    // Async depth inference — background thread, never blocks the camera loop.
    // m_depthInFlight: true while a background estimate() call is running.
    // m_depthMapReady: last completed depth map (portrait orientation), guarded by mutex.
    std::atomic<bool> m_depthInFlight{false};
    std::mutex        m_depthMutex;
    cv::Mat           m_depthMapReady;
#endif
};
