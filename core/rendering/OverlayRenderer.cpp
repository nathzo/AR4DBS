#include "OverlayRenderer.h"
#include "core/math/PoseUtils.h"

#include <QPainter>
#include <QtGlobal>
#include <cmath>

OverlayRenderer::OverlayRenderer()               : m_style()    {}
OverlayRenderer::OverlayRenderer(const Style &s) : m_style(s)   {}
OverlayRenderer::~OverlayRenderer() = default;

void OverlayRenderer::setDistortion(const cv::Mat &dist) { m_dist = dist.clone(); }

// ── Overlay management ────────────────────────────────────────────────────────

void OverlayRenderer::ensureOverlay(int w, int h)
{
    if (m_overlay.width() != w || m_overlay.height() != h)
        m_overlay = QImage(w, h, QImage::Format_ARGB32_Premultiplied);
    m_overlay.fill(Qt::transparent);
}

// Alpha-composite the ARGB overlay onto the BGR cv::Mat frame.
void OverlayRenderer::blendToFrame()
{
    if (!m_frame || m_overlay.isNull()) return;
    cv::Mat &bgr = *m_frame;
    const int h = bgr.rows, w = bgr.cols;

    for (int y = 0; y < h; ++y) {
        const QRgb *src = reinterpret_cast<const QRgb *>(m_overlay.scanLine(y));
        uchar      *dst = bgr.ptr<uchar>(y);
        for (int x = 0; x < w; ++x, dst += 3) {
            // Format_ARGB32_Premultiplied: stored as premul BGRA on little-endian
            const quint32 px = src[x];
            const int a = (px >> 24) & 0xff;
            if (a == 0) continue;
            const int ia = 255 - a;
            // premul channels: R=(px>>16)&0xff, G=(px>>8)&0xff, B=px&0xff
            dst[0] = static_cast<uchar>(((dst[0] * ia) >> 8) + ( px        & 0xff));
            dst[1] = static_cast<uchar>(((dst[1] * ia) >> 8) + ((px >>  8) & 0xff));
            dst[2] = static_cast<uchar>(((dst[2] * ia) >> 8) + ((px >> 16) & 0xff));
        }
    }
}

// ── Frame-level API ───────────────────────────────────────────────────────────

void OverlayRenderer::beginFrame(cv::Mat &frame)
{
    m_frame = &frame;
    ensureOverlay(frame.cols, frame.rows);
    m_painter = std::make_unique<QPainter>(&m_overlay);
    m_painter->setRenderHint(QPainter::Antialiasing);
    m_painter->setRenderHint(QPainter::SmoothPixmapTransform);
}

void OverlayRenderer::endFrame()
{
    if (m_painter) { m_painter->end(); m_painter.reset(); }
    blendToFrame();
    m_frame = nullptr;
}

// ── Private paint helpers (painter must be open) ─────────────────────────────

// Returns true only when both coordinates are finite numbers within a generous
// screen-space range.  QPainter (Qt 6 raster engine) asserts / calls qFatal on
// NaN or Inf coordinates, which triggers abort().
static bool isValidPoint(QPointF p)
{
    return std::isfinite(p.x()) && std::isfinite(p.y())
        && p.x() > -1e5f && p.x() < 1e5f
        && p.y() > -1e5f && p.y() < 1e5f;
}

void OverlayRenderer::paintLine(QPointF a, QPointF b) const
{
    if (!isValidPoint(a) || !isValidPoint(b)) return;

    QColor g = m_style.lineColor;
    g.setAlphaF(m_style.glowAlpha);
    m_painter->setPen(QPen(g, m_style.lineWidth * m_style.glowWidthMul,
                           Qt::SolidLine, Qt::RoundCap));
    m_painter->drawLine(a, b);

    m_painter->setPen(QPen(m_style.lineColor, m_style.lineWidth,
                           Qt::SolidLine, Qt::RoundCap));
    m_painter->drawLine(a, b);
}

void OverlayRenderer::paintTarget(QPointF c) const
{
    if (!isValidPoint(c)) return;

    QColor g = m_style.targetColor;
    g.setAlphaF(m_style.glowAlpha);
    m_painter->setPen(Qt::NoPen);
    m_painter->setBrush(g);
    m_painter->drawEllipse(c, 10.0, 10.0);

    m_painter->setBrush(m_style.targetColor);
    m_painter->drawEllipse(c, 5.0, 5.0);
}

void OverlayRenderer::paintIncision(QPointF c) const
{
    if (!isValidPoint(c)) return;

    const float arm = m_style.crossArm;

    QColor g = m_style.incisionColor;
    g.setAlphaF(m_style.glowAlpha + 0.10f);
    m_painter->setPen(QPen(g, arm * 0.6f, Qt::SolidLine, Qt::RoundCap));
    m_painter->drawLine(QPointF(c.x()-arm*1.4f, c.y()), QPointF(c.x()+arm*1.4f, c.y()));
    m_painter->drawLine(QPointF(c.x(), c.y()-arm*1.4f), QPointF(c.x(), c.y()+arm*1.4f));

    m_painter->setPen(QPen(m_style.incisionColor, 2.f, Qt::SolidLine, Qt::RoundCap));
    m_painter->drawLine(QPointF(c.x()-arm, c.y()), QPointF(c.x()+arm, c.y()));
    m_painter->drawLine(QPointF(c.x(), c.y()-arm), QPointF(c.x(), c.y()+arm));

    m_painter->setPen(Qt::NoPen);
    m_painter->setBrush(m_style.incisionColor);
    m_painter->drawEllipse(c, 4.0, 4.0);
}

// ── Public granular API ───────────────────────────────────────────────────────

void OverlayRenderer::drawSegment(const cv::Point3d &p0, const cv::Point3d &p1,
                                   const cv::Mat &K,
                                   const cv::Mat &rvec, const cv::Mat &tvec)
{
    if (!m_painter) return;
    const cv::Point2f a = PoseUtils::project(p0, K, rvec, tvec, m_dist);
    const cv::Point2f b = PoseUtils::project(p1, K, rvec, tvec, m_dist);
    paintLine(QPointF(a.x, a.y), QPointF(b.x, b.y));
}

void OverlayRenderer::drawTargetMarker(const cv::Point3d &target,
                                        const cv::Mat &K,
                                        const cv::Mat &rvec, const cv::Mat &tvec)
{
    if (!m_painter) return;
    const cv::Point2f p = PoseUtils::project(target, K, rvec, tvec, m_dist);
    paintTarget(QPointF(p.x, p.y));
}

void OverlayRenderer::drawIncisionMarker(const cv::Point3d &pt,
                                          const cv::Mat &K,
                                          const cv::Mat &rvec, const cv::Mat &tvec)
{
    if (!m_painter) return;
    const cv::Point2f p = PoseUtils::project(pt, K, rvec, tvec, m_dist);
    paintIncision(QPointF(p.x, p.y));
}

// ── Full draw (convenience) ───────────────────────────────────────────────────

void OverlayRenderer::draw(cv::Mat           &frame,
                            const cv::Point3d &target,
                            const cv::Point3d &lineEnd,
                            const cv::Point3d *incisionPt,
                            const cv::Mat     &K,
                            const cv::Mat     &rvec,
                            const cv::Mat     &tvec)
{
    beginFrame(frame);

    const cv::Point2f pT  = PoseUtils::project(target,  K, rvec, tvec, m_dist);
    const cv::Point2f pLE = PoseUtils::project(lineEnd, K, rvec, tvec, m_dist);
    paintLine(QPointF(pT.x, pT.y), QPointF(pLE.x, pLE.y));
    paintTarget(QPointF(pT.x, pT.y));
    if (incisionPt) {
        const cv::Point2f pI = PoseUtils::project(*incisionPt, K, rvec, tvec, m_dist);
        paintIncision(QPointF(pI.x, pI.y));
    }

    endFrame();
}
