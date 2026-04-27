#pragma once
#include <QObject>
#include <opencv2/core.hpp>
#include <memory>
#include <optional>
#include <vector>

#include "core/math/SurgicalPlan.h"

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
    // Advance to the next active target (cycles when at the last one).
    void nextTarget();

signals:
    void frameReady(const cv::Mat &annotated);
    // Emitted when the plan changes or nextTarget() is called.
    // total = number of active targets (0, 1, or 2).
    void targetChanged(int activeIndex, int total);

private:
    cv::Mat                        loadCalibration(const QString &path);
    std::vector<TagConfig>         loadTagConfigs(const QString &path);

    cv::Mat fusePoses(const std::vector<TagPose> &detections) const;

    std::optional<cv::Point3d> findIncisionPoint(
        const cv::Mat     &depthMap,
        const cv::Mat     &rvec,
        const cv::Mat     &tvec,
        const cv::Point2f &tagPx,
        double             tagMetricDepth) const;

    cv::Mat m_K;
    cv::Mat m_dist;
    std::vector<TagConfig>           m_tagConfigs;
    std::unique_ptr<AprilTagTracker> m_tracker;
    std::unique_ptr<OverlayRenderer> m_renderer;
    std::unique_ptr<DepthEstimator>  m_depth;

    // Up to two active trajectories: [0]=left, [1]=right
    // nullptr means that side is inactive
    std::unique_ptr<IncisionLine> m_lines[2];
    int                           m_activeIndex = 0;
};
