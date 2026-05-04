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
    ARSession    *session       = nil;
    id            delegate      = nil; // ARDelegate (Obj-C type hidden from header)
    bool          calibEmitted  = false;
    bool          lidarActive   = false; // true when LiDAR frameSemantics is enabled
    int           cameraWidth   = 0;    // stored after first frame for LiDAR resize
    int           cameraHeight  = 0;
    QElapsedTimer frameTimer;
    ARKitSession *q             = nullptr; // back-pointer for signal emission
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

    // Cache camera dimensions on the first frame (used for LiDAR depth resize).
    if (impl->cameraWidth == 0) {
        impl->cameraWidth  = static_cast<int>(w);
        impl->cameraHeight = static_cast<int>(h);
    }

    // Emit calibration once using ARKit's native landscape intrinsics directly.
    if (!impl->calibEmitted) {
        impl->calibEmitted = true;
        matrix_float3x3 intr = frame.camera.intrinsics;

        cv::Mat K = (cv::Mat_<double>(3, 3)
            << intr.columns[0][0], 0,                   intr.columns[2][0],
               0,                   intr.columns[1][1],  intr.columns[2][1],
               0,                   0,                   1);
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

    emit impl->q->frameReady(bgr, world_T_camera);

    // Extract LiDAR depth when active.
    // sceneDepth.depthMap is kCVPixelFormatType_DepthFloat32 — metric metres,
    // higher = farther — same convention as the CoreML estimator output.
    if (impl->lidarActive && frame.sceneDepth != nil) {
        CVPixelBufferRef depthBuf = frame.sceneDepth.depthMap;
        CVPixelBufferLockBaseAddress(depthBuf, kCVPixelBufferLock_ReadOnly);

        const size_t dw         = CVPixelBufferGetWidth(depthBuf);
        const size_t dh         = CVPixelBufferGetHeight(depthBuf);
        const size_t bytesPerRow = CVPixelBufferGetBytesPerRow(depthBuf);
        const uint8_t *base     = static_cast<const uint8_t *>(
                                      CVPixelBufferGetBaseAddress(depthBuf));

        cv::Mat depthRaw(static_cast<int>(dh), static_cast<int>(dw), CV_32F);
        for (size_t r = 0; r < dh; ++r)
            memcpy(depthRaw.ptr<float>(static_cast<int>(r)),
                   base + r * bytesPerRow,
                   dw * sizeof(float));

        CVPixelBufferUnlockBaseAddress(depthBuf, kCVPixelBufferLock_ReadOnly);

        // Replace NaN (unmeasured pixels) with 0 — treated as "no data" downstream.
        cv::patchNaNs(depthRaw, 0.0f);

        // Resize to match camera frame dimensions so depth pixels align with image pixels.
        cv::Mat depthScaled;
        cv::resize(depthRaw, depthScaled,
                   cv::Size(impl->cameraWidth, impl->cameraHeight),
                   0, 0, cv::INTER_LINEAR);

        emit impl->q->lidarDepthReady(depthScaled);
    }
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
    m_impl->cameraWidth  = 0;
    m_impl->cameraHeight = 0;

    ARWorldTrackingConfiguration *config =
        [[ARWorldTrackingConfiguration alloc] init];
    config.planeDetection = ARPlaneDetectionNone; // we only need camera tracking

    // Enable LiDAR scene depth when the device supports it (iPhone 12 Pro and later).
    const bool hasLidar = [ARWorldTrackingConfiguration
                              supportsFrameSemantics:ARFrameSemanticSceneDepth];
    m_impl->lidarActive = hasLidar;
    if (hasLidar)
        config.frameSemantics |= ARFrameSemanticSceneDepth;

    emit lidarAvailable(hasLidar);
    [m_impl->session runWithConfiguration:config];
}

void ARKitSession::stop()
{
    if (m_impl->session)
        [m_impl->session pause];
}
