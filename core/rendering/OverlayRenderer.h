#pragma once
#include <opencv2/core.hpp>
#include <QColor>
#include <QPointF>
#include <QImage>
#include <memory>

class QPainter;

class OverlayRenderer
{
public:
    struct Style {
        QColor lineColor     {117, 208, 197};  // ARC_BLUE    #75D0C5
        QColor targetColor   {233, 223,  77};  // VOLT_YELLOW #E9DF4D
        QColor incisionColor {196,  82,  85};  // IMPULSE_RED #c45255

        float lineWidth    = 2.f;
        float targetRadius = 12.f;
        float crossArm     = 10.f;

        float glowWidthMul = 5.f;
        float glowAlpha    = 0.30f;
    };

    explicit OverlayRenderer();
    explicit OverlayRenderer(const Style &style);
    ~OverlayRenderer();

    // Full draw (no occlusion).
    void draw(cv::Mat           &frame,
              const cv::Point3d &target,
              const cv::Point3d &lineEnd,
              const cv::Point3d *incisionPt,
              const cv::Mat     &K,
              const cv::Mat     &rvec,
              const cv::Mat     &tvec);

    // Granular primitives for occlusion-aware rendering.
    // Call beginFrame() first, then any combination of these, then endFrame().
    void beginFrame(cv::Mat &frame);
    void endFrame();

    void drawSegment(const cv::Point3d &p0,
                     const cv::Point3d &p1,
                     const cv::Mat     &K,
                     const cv::Mat     &rvec,
                     const cv::Mat     &tvec);

    void drawTargetMarker(const cv::Point3d &target,
                          const cv::Mat     &K,
                          const cv::Mat     &rvec,
                          const cv::Mat     &tvec);

    void drawIncisionMarker(const cv::Point3d &pt,
                            const cv::Mat     &K,
                            const cv::Mat     &rvec,
                            const cv::Mat     &tvec);

private:
    Style               m_style;
    cv::Mat            *m_frame   = nullptr;
    QImage              m_overlay;
    std::unique_ptr<QPainter> m_painter;

    void ensureOverlay(int w, int h);
    void blendToFrame();

    void paintLine    (QPointF a, QPointF b) const;
    void paintTarget  (QPointF centre)       const;
    void paintIncision(QPointF centre)       const;
};
