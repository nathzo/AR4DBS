#include "DesktopCamera.h"
#include <QTimer>
#include <opencv2/videoio.hpp>

struct DesktopCamera::Impl {
    cv::VideoCapture cap;
    QTimer          *timer = nullptr;
    int              deviceIndex;
};

DesktopCamera::DesktopCamera(int deviceIndex, QObject *parent)
    : QObject(parent)
    , m_impl(new Impl)
{
    m_impl->deviceIndex = deviceIndex;
    m_impl->timer = new QTimer(this);
    connect(m_impl->timer, &QTimer::timeout, this, [this]() {
        cv::Mat frame;
        if (m_impl->cap.read(frame) && !frame.empty())
            emit frameReady(frame);
    });
}

DesktopCamera::~DesktopCamera()
{
    stop();
    delete m_impl;
}

void DesktopCamera::start()
{
    if (!m_impl->cap.open(m_impl->deviceIndex))
        return;
    m_impl->timer->start(33); // ~30 fps
}

void DesktopCamera::stop()
{
    m_impl->timer->stop();
    m_impl->cap.release();
}
