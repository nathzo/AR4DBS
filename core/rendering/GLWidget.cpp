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
    if (m_image.isNull())
        return;

    QPainter painter(this);
    // Scale the frame to fill the widget while preserving aspect ratio
    painter.drawImage(rect(), m_image);
}
