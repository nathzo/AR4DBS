#import <AVFoundation/AVFoundation.h>
#include "IOSCamera.h"
#include "CameraIntrinsics.h"
#include <QDebug>
#include <QElapsedTimer>
#include <opencv2/imgproc.hpp>

static constexpr qint64 MIN_FRAME_MS = 32;

// ─── Objective-C delegate that receives native frames ────────────────────────
@interface FrameDelegate : NSObject <AVCaptureVideoDataOutputSampleBufferDelegate>
@property (nonatomic, assign) IOSCamera *owner;  // back-pointer to Qt object
@end

@implementation FrameDelegate
- (void)captureOutput:(AVCaptureOutput *)output
didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
       fromConnection:(AVCaptureConnection *)connection
{
    // Forward to C++ side
    self.owner->handleSampleBuffer(sampleBuffer);
}
@end

// ─── Impl ────────────────────────────────────────────────────────────────────
struct IOSCamera::Impl {
    AVCaptureSession            *session    = nil;
    AVCaptureDevice             *device     = nil;
    AVCaptureDeviceInput        *input      = nil;
    AVCaptureVideoDataOutput    *output     = nil;
    FrameDelegate               *delegate   = nil;
    dispatch_queue_t             queue      = nil;
    int                          captureWidth  = 1280;
    int                          captureHeight = 720;
    bool                         calibEmitted  = false;
    QElapsedTimer                frameTimer;
};

// ─── Constructor ─────────────────────────────────────────────────────────────
IOSCamera::IOSCamera(int captureWidth, int captureHeight, QObject *parent)
    : QObject(parent)
    , m_impl(new Impl)
{
    m_impl->captureWidth  = captureWidth;
    m_impl->captureHeight = captureHeight;

    // 1. Pick the 1× wide-angle lens explicitly
    AVCaptureDevice *device = nil;
    if (@available(iOS 13.0, *)) {
        device = [AVCaptureDevice
            defaultDeviceWithDeviceType:AVCaptureDeviceTypeBuiltInWideAngleCamera
                              mediaType:AVMediaTypeVideo
                               position:AVCaptureDevicePositionBack];
    }
    if (!device)
        device = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];

    m_impl->device = device;

    // 2. Force zoom to 1.0 before the session starts
    NSError *err = nil;
    if ([device lockForConfiguration:&err]) {
        device.videoZoomFactor = 1.0;
        [device unlockForConfiguration];
    }

    // 3. Build the capture session
    m_impl->session = [[AVCaptureSession alloc] init];
    [m_impl->session beginConfiguration];
    m_impl->session.sessionPreset = AVCaptureSessionPreset1280x720;

    // Input
    m_impl->input = [AVCaptureDeviceInput deviceInputWithDevice:device error:&err];
    if (m_impl->input && [m_impl->session canAddInput:m_impl->input])
        [m_impl->session addInput:m_impl->input];

    // Output — BGRA so we can wrap it cheaply into a cv::Mat
    m_impl->output = [[AVCaptureVideoDataOutput alloc] init];
    m_impl->output.videoSettings = @{
        (NSString *)kCVPixelBufferPixelFormatTypeKey :
            @(kCVPixelFormatType_32BGRA)
    };
    m_impl->output.alwaysDiscardsLateVideoFrames = YES;

    m_impl->delegate = [[FrameDelegate alloc] init];
    m_impl->delegate.owner = this;
    m_impl->queue = dispatch_queue_create("iosCamera.frames", DISPATCH_QUEUE_SERIAL);
    [m_impl->output setSampleBufferDelegate:m_impl->delegate
                                      queue:m_impl->queue];

    if ([m_impl->session canAddOutput:m_impl->output])
        [m_impl->session addOutput:m_impl->output];

    // Lock orientation to landscape-right to match your preview
    AVCaptureConnection *conn =
        [m_impl->output connectionWithMediaType:AVMediaTypeVideo];
    if (@available(iOS 17.0, *)) {
        conn.videoRotationAngle = 0;
    } else {
        if (conn.isVideoOrientationSupported)
            conn.videoOrientation = AVCaptureVideoOrientationPortrait;
    }

    [m_impl->session commitConfiguration];
}

// ─── Destructor ───────────────────────────────────────────────────────────────
IOSCamera::~IOSCamera()
{
    stop();
    delete m_impl;
}

// ─── start / stop ────────────────────────────────────────────────────────────
void IOSCamera::start()
{
    m_impl->calibEmitted = false;
    requestCameraAccess([this](bool granted) {
        if (granted) {
            dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
                [m_impl->session startRunning];
            });
        } else {
            qWarning() << "IOSCamera: camera permission denied";
        }
    });
}

void IOSCamera::stop()
{
    if (m_impl->session && m_impl->session.isRunning)
        [m_impl->session stopRunning];
}

// ─── Frame handler (called on m_impl->queue, NOT the main thread) ────────────
void IOSCamera::handleSampleBuffer(CMSampleBufferRef sampleBuffer)
{
    // Throttle
    if (m_impl->frameTimer.isValid() &&
        m_impl->frameTimer.elapsed() < MIN_FRAME_MS)
        return;
    m_impl->frameTimer.restart();

    // Emit calibration once
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

    // Wrap pixel buffer as cv::Mat (zero-copy)
    CVImageBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);

    void  *base   = CVPixelBufferGetBaseAddress(pixelBuffer);
    size_t width  = CVPixelBufferGetWidth(pixelBuffer);
    size_t height = CVPixelBufferGetHeight(pixelBuffer);
    size_t stride = CVPixelBufferGetBytesPerRow(pixelBuffer);

    // BGRA → BGR
    cv::Mat bgra(static_cast<int>(height), static_cast<int>(width),
                 CV_8UC4, base, stride);
    cv::Mat bgr;
    cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);

    // Strip CVPixelBuffer row padding so the mat is exactly captureWidth wide
    if (bgr.cols != m_impl->captureWidth || !bgr.isContinuous()) {
        bgr = bgr(cv::Rect(0, 0, m_impl->captureWidth,
                                  m_impl->captureHeight)).clone();
    }

    CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);

    emit frameReady(bgr);
}