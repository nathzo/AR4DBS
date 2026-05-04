typedef struct opaqueCMSampleBuffer *CMSampleBufferRef;
#pragma once
#include <QObject>
#include <opencv2/core.hpp>

// Delivers frames from the iPhone back camera via Qt Multimedia.
// Emits calibrationReady once after the session starts with intrinsics
// derived from the active capture format's field of view.
class IOSCamera : public QObject
{
    Q_OBJECT
public:
    // captureWidth/captureHeight: preferred resolution (default 1280x720)
    explicit IOSCamera(int captureWidth  = 1280,
                       int captureHeight = 720,
                       QObject *parent   = nullptr);
    ~IOSCamera();

    void start();
    void stop();

    void handleSampleBuffer(CMSampleBufferRef sampleBuffer);  // called by native delegate

signals:
    void frameReady(const cv::Mat &bgr);
    void calibrationReady(const cv::Mat &K);   // emitted once after start()

private:
    struct Impl;
    Impl *m_impl;
};
