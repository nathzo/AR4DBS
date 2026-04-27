#pragma once
#include <QWidget>
#include <QImage>
#include <opencv2/core.hpp>

// Plain QWidget that blits a BGR cv::Mat to the screen each frame.
// Using QWidget instead of QOpenGLWidget avoids the OpenGL texture-cache
// assertion (qopengltexturecache.cpp line 173) that fires when a QImage is
// uploaded as a GL texture while the GL context is not fully ready.
class GLWidget : public QWidget
{
    Q_OBJECT
public:
    explicit GLWidget(QWidget *parent = nullptr);

public slots:
    void setFrame(const cv::Mat &bgr);

protected:
    void paintEvent(QPaintEvent *event) override;

private:
    QImage m_image;
};
