#include "GLWidget.h"
#include <QPainter>
#include <QPaintEvent>
#include <opencv2/imgproc.hpp>

GLWidget::GLWidget(QWidget *parent)
    : QWidget(parent)
{
    // Prevent Qt from painting a background before paintEvent — avoids flicker
    setAttribute(Qt::WA_OpaquePaintEvent);
    setAttribute(Qt::WA_NoSystemBackground);
}

void GLWidget::setFrame(const cv::Mat &bgr)
{
    // Convert BGR → RGB into m_rgb (member), then wrap without deep copy.
    // m_rgb keeps the pixel data alive for the lifetime of m_image.
    cv::cvtColor(bgr, m_rgb, cv::COLOR_BGR2RGB);
    m_image = QImage(m_rgb.data, m_rgb.cols, m_rgb.rows,
                     static_cast<int>(m_rgb.step),
                     QImage::Format_RGB888);
    update();
}

void GLWidget::paintEvent(QPaintEvent *event)
{
    Q_UNUSED(event)

    QPainter painter(this);
    painter.fillRect(rect(), Qt::black);

    if (m_image.isNull())
        return;

    // The incoming frame is landscape (ARKit native: width > height).
    // The widget is portrait. Rotate 90° CW around the widget centre so the
    // frame fills portrait orientation, then scale with aspect-ratio preserved.
    const bool needsRotation = m_image.width() > m_image.height();

    QSize frameSize = needsRotation
        ? QSize(m_image.height(), m_image.width())  // swapped after 90° rotation
        : m_image.size();

    QSize scaled = frameSize.scaled(rect().size(), Qt::KeepAspectRatio);
    QRect target(
        (rect().width()  - scaled.width())  / 2,
        (rect().height() - scaled.height()) / 2,
        scaled.width(), scaled.height()
    );

    if (needsRotation) {
        painter.translate(target.center());
        painter.rotate(90.0);
        painter.translate(-target.center());
        // After the transform the drawing rect maps back to the image dimensions
        QRect srcRect(
            target.center().x() - scaled.height() / 2,
            target.center().y() - scaled.width()  / 2,
            scaled.height(), scaled.width()
        );
        painter.drawImage(srcRect, m_image);
    } else {
        painter.drawImage(target, m_image);
    }
}
