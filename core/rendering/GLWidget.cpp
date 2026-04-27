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
    // Convert BGR → RGB and take a deep copy so the cv::Mat can be freed
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    m_image = QImage(rgb.data, rgb.cols, rgb.rows,
                     static_cast<int>(rgb.step),
                     QImage::Format_RGB888).copy();
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
