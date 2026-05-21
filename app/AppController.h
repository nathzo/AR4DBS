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
    cv::Mat R_frame_tag; // 3x3 rotation  (precomputed from T_frame_tag)
    cv::Mat t_frame_tag; // 3x1 translation (precomputed from T_frame_tag)
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

    // ARTrackingState mirror for the debug overlay: 0=normal, 1=limited, 2=unavailable.
    void onTrackingQualityChanged(int state);
#endif

signals:
    void frameReady(const cv::Mat &annotated);
#ifdef Q_OS_IOS
    void lockStateChanged(bool locked);
#endif

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

#ifdef Q_OS_IOS
    bool   meetsInitConditions(const std::vector<TagPose> &detections,
                               const cv::Mat              &T_from_tags) const;
    double computeReprojError(const std::vector<TagPose>  &detections,
                              const cv::Mat               &rvec,
                              const cv::Mat               &tvec) const;
#endif

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
    // ── ARKit tracking quality ───────────────────────────────────────────────
    // 0 = normal, 1 = limited, 2 = not available. Updated via onTrackingQualityChanged.
    int m_trackingState = 0;

    // ── Anchor-locking state ─────────────────────────────────────────────────
    // Empty until both tags are seen face-on with low reprojection error.
    // Once set, the overlay runs purely from ARKit world tracking.
    // Cleared by resetARRegistration() to re-enter the init phase.
    cv::Mat m_T_world_leksell;

    // Strict thresholds: both conditions must hold simultaneously for the lock
    // to be accepted, ensuring the established coordinates are reliable.
    // Face-on reads as ~180° in our convention (R(2,2) ≈ +1 → cosA = -R(2,2) ≈ -1).
    // Valid range: [175°, 180°], so cosA ≤ -cos(5°) ≈ -0.9962.
    static constexpr double kMaxInitAngleCos = 0.9962; // cos(5°) tolerance around 180°
    static constexpr double kMaxInitReprojPx = 0.8;   // RMS reprojection error cap

    static constexpr double kAnchorAlpha = 0.05;

    // ── CoreML / LiDAR depth ─────────────────────────────────────────────────
    std::unique_ptr<CoreMLDepthEstimator> m_iosDepth;
    bool   m_usingLiDAR     = false;
    bool   m_mlDepthEnabled = false;
    double m_depthAnchor    = 0.0;

    std::atomic<bool> m_depthInFlight{false};
    std::mutex        m_depthMutex;
    cv::Mat           m_depthMapReady;

    std::atomic<bool> m_depthModelLoading{false};
    std::atomic<bool> m_depthModelReady{false};
#endif
};
