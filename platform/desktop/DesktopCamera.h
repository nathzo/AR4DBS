#pragma once
#include <QObject>
#include <opencv2/core.hpp>

// Grabs frames from the system webcam using OpenCV (desktop testing only)
class DesktopCamera : public QObject
{
    Q_OBJECT
public:
    explicit DesktopCamera(int deviceIndex = 0, QObject *parent = nullptr);
    ~DesktopCamera();

    void start();
    void stop();

signals:
    void frameReady(const cv::Mat &bgr);

private:
    struct Impl;
    Impl *m_impl;
};
