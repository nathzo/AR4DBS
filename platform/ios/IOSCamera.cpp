#include "IOSCamera.h"
#include "CameraIntrinsics.h"

#include <QCamera>
#include <QMediaCaptureSession>
#include <QMediaDevices>
#include <QVideoSink>
#include <QVideoFrame>
#include <QImage>
#include <QTimer>

#include <opencv2/imgproc.hpp>

struct IOSCamera::Impl {
    QCamera              *camera        = nullptr;
    QVideoSink           *sink          = nullptr;
    QMediaCaptureSession  session;
    int                   captureWidth  = 1280;
    int                   captureHeight = 720;
    bool                  calibEmitted  = false;
};

IOSCamera::IOSCamera(int captureWidth, int captureHeight, QObject *parent)
    : QObject(parent)
    , m_impl(new Impl)
{
    m_impl->captureWidth  = captureWidth;
    m_impl->captureHeight = captureHeight;

    // Pick the back (wide) camera
    QCameraDevice backDevice;
    for (const QCameraDevice &dev : QMediaDevices::videoInputs()) {
        if (dev.position() == QCameraDevice::BackFace) {
            backDevice = dev;
            break;
        }
    }

    m_impl->camera = backDevice.isNull()
        ? new QCamera(this)
        : new QCamera(backDevice, this);

    m_impl->sink = new QVideoSink(this);
    m_impl->session.setCamera(m_impl->camera);
    m_impl->session.setVideoSink(m_impl->sink);

    connect(m_impl->sink, &QVideoSink::videoFrameChanged,
            this, [this](const QVideoFrame &frame) {

        // Read intrinsics from AVFoundation on the first frame — by then
        // the session is running and activeFormat.videoFieldOfView is valid
        if (!m_impl->calibEmitted) {
            m_impl->calibEmitted = true;
            const auto d = readCameraIntrinsics(m_impl->captureWidth,
                                                m_impl->captureHeight);
            cv::Mat K = (cv::Mat_<double>(3, 3)
                << d.fx, 0,    d.cx,
                   0,    d.fy, d.cy,
                   0,    0,    1);
            emit calibrationReady(K);
        }

        if (!frame.isValid()) return;

        QVideoFrame mapped = frame;
        if (!mapped.map(QVideoFrame::ReadOnly)) return;

        QImage img = mapped.toImage().convertToFormat(QImage::Format_RGB888);
        mapped.unmap();
        if (img.isNull()) return;

        cv::Mat rgb(img.height(), img.width(), CV_8UC3,
                    const_cast<uchar *>(img.bits()),
                    static_cast<size_t>(img.bytesPerLine()));
        cv::Mat bgr;
        cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
        emit frameReady(bgr.clone());
    });
}

IOSCamera::~IOSCamera()
{
    stop();
    delete m_impl;
}

void IOSCamera::start()
{
    m_impl->calibEmitted = false; // allow re-read if restarted

    // On iOS 14+ the camera session delivers no frames and never appears in
    // Settings unless we explicitly request permission before starting.
    // The Qt FFmpeg multimedia plugin does not do this automatically.
    requestCameraAccess([this](bool granted) {
        if (granted) {
            m_impl->camera->start();
        } else {
            qWarning() << "IOSCamera: camera permission denied — no frames will be delivered";
        }
    });
}

void IOSCamera::stop()
{
    if (m_impl->camera)
        m_impl->camera->stop();
}
