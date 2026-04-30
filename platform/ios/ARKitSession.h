#pragma once
#ifdef Q_OS_IOS

#include <QObject>
#include <opencv2/core.hpp>

// Wraps ARKit's ARSession and delivers frames to AppController.
// Emits frameReady with the BGR image AND the ARKit world_T_camera transform
// (4×4 CV_64F, column-major simd_float4x4 converted to row-major cv::Mat).
// Emits calibrationReady once, on the first frame, with the camera intrinsics
// scaled to the actual capture resolution.
class ARKitSession : public QObject
{
    Q_OBJECT
public:
    explicit ARKitSession(QObject *parent = nullptr);
    ~ARKitSession();

    void start();
    void stop();

signals:
    void frameReady(const cv::Mat &bgr, const cv::Mat &world_T_camera);
    void calibrationReady(const cv::Mat &K);

private:
    struct Impl;
    Impl *m_impl;
};

#endif // Q_OS_IOS
