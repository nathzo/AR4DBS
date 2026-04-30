#include "ARKitSession.h"

#import <ARKit/ARKit.h>
#import <CoreVideo/CoreVideo.h>
#include <opencv2/imgproc.hpp>
#include <QDebug>
#include <QElapsedTimer>
#include <cstring>

static constexpr qint64 MIN_FRAME_MS = 33; // ~30 fps cap

// ── Forward-declare Impl so the delegate can reference it ────────────────────
struct ARKitSession::Impl {
    ARSession    *session      = nil;
    id            delegate     = nil; // ARDelegate (Obj-C type hidden from header)
    bool          calibEmitted = false;
    QElapsedTimer frameTimer;
    ARKitSession *q            = nullptr; // back-pointer for signal emission
};

// ── Objective-C delegate ──────────────────────────────────────────────────────
@interface ARDelegate : NSObject <ARSessionDelegate>
@property (nonatomic, assign) ARKitSession::Impl *impl;
@end

@implementation ARDelegate

- (void)session:(ARSession *)session didUpdateFrame:(ARFrame *)frame
{
    ARKitSession::Impl *impl = self.impl;
    if (!impl || !impl->q) return;

    // Throttle to MIN_FRAME_MS
    if (impl->frameTimer.isValid() &&
        impl->frameTimer.elapsed() < MIN_FRAME_MS)
        return;
    impl->frameTimer.restart();

    CVPixelBufferRef pb = frame.capturedImage;
    CVPixelBufferLockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);

    const size_t w        = CVPixelBufferGetWidth(pb);
    const size_t h        = CVPixelBufferGetHeight(pb);
    const size_t yStride  = CVPixelBufferGetBytesPerRowOfPlane(pb, 0);
    const size_t uvStride = CVPixelBufferGetBytesPerRowOfPlane(pb, 1);
    const uint8_t *yData  = static_cast<const uint8_t *>(
                                CVPixelBufferGetBaseAddressOfPlane(pb, 0));
    const uint8_t *uvData = static_cast<const uint8_t *>(
                                CVPixelBufferGetBaseAddressOfPlane(pb, 1));

    cv::Mat yMat(static_cast<int>(h),   static_cast<int>(w), CV_8UC1, (void*)yData,  yStride);
    cv::Mat uvMat(static_cast<int>(h/2), static_cast<int>(w/2), CV_8UC2, (void*)uvData, uvStride);

    cv::Mat bgr;
    cv::cvtColorTwoPlane(yMat, uvMat, bgr, cv::COLOR_YUV2BGR_NV12);

    CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);

    // ARKit frames are always landscape-right — rotate to portrait
    cv::Mat bgrPortrait;
    cv::rotate(bgr, bgrPortrait, cv::ROTATE_90_CLOCKWISE);

    // Adjust intrinsics for the 90° rotation:
    // after rotation: new_w = old_h, new_fy = old_fx, new_fx = old_fy
    // cx and cy swap and reflect
    if (!impl->calibEmitted) {
        impl->calibEmitted = true;
        matrix_float3x3 intr = frame.camera.intrinsics;

        double fx = intr.columns[0][0];
        double fy = intr.columns[1][1];
        double cx = intr.columns[2][0];
        double cy = intr.columns[2][1];

        // After ROTATE_90_CLOCKWISE: x_new = (h-1) - y_old, y_new = x_old
        double new_fx = fy;
        double new_fy = fx;
        double new_cx = (h - 1) - cy;   // h = landscape height = portrait width
        double new_cy = cx;

        cv::Mat K = (cv::Mat_<double>(3, 3)
            << new_fx, 0,      new_cx,
               0,      new_fy, new_cy,
               0,      0,      1);
        emit impl->q->calibrationReady(K);
    }

    // Convert simd_float4x4 (column-major) → 4×4 CV_64F row-major cv::Mat.
    // ARKit camera.transform = world_T_camera  (camera pose in world space).
    simd_float4x4 T = frame.camera.transform;
    cv::Mat world_T_camera(4, 4, CV_64F);
    for (int col = 0; col < 4; ++col)
        for (int row = 0; row < 4; ++row)
            world_T_camera.at<double>(row, col) =
                static_cast<double>(T.columns[col][row]);

    emit impl->q->frameReady(bgrPortrait, world_T_camera);
}

- (void)session:(ARSession *)session didFailWithError:(NSError *)error
{
    qWarning() << "ARKitSession: session failed:"
               << QString::fromNSString(error.localizedDescription);
}

- (void)sessionWasInterrupted:(ARSession *)session
{
    qWarning() << "ARKitSession: session interrupted";
}

- (void)sessionInterruptionEnded:(ARSession *)session
{
    qDebug() << "ARKitSession: interruption ended — resuming";
}

@end

// ── ARKitSession C++ implementation ──────────────────────────────────────────

ARKitSession::ARKitSession(QObject *parent)
    : QObject(parent)
    , m_impl(new Impl)
{
    m_impl->q        = this;
    m_impl->session  = [[ARSession alloc] init];
    auto *del        = [[ARDelegate alloc] init];
    del.impl         = m_impl;
    m_impl->delegate = del;
    m_impl->session.delegate = del;
}

ARKitSession::~ARKitSession()
{
    stop();
    // Nil out back-pointer before delegate might receive any stray callbacks.
    static_cast<ARDelegate *>(m_impl->delegate).impl = nullptr;
    delete m_impl;
}

void ARKitSession::start()
{
    m_impl->calibEmitted = false;

    ARWorldTrackingConfiguration *config =
        [[ARWorldTrackingConfiguration alloc] init];
    config.planeDetection = ARPlaneDetectionNone; // we only need camera tracking
    [m_impl->session runWithConfiguration:config];
}

void ARKitSession::stop()
{
    if (m_impl->session)
        [m_impl->session pause];
}
