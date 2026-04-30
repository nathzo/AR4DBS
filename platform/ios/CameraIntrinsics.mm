#import <AVFoundation/AVFoundation.h>
#include "CameraIntrinsics.h"
#include <cmath>
#include <functional>

CameraIntrinsicsData readCameraIntrinsics(int captureWidth, int captureHeight)
{
    CameraIntrinsicsData result;
    result.width  = captureWidth;
    result.height = captureHeight;
    result.cx     = captureWidth  / 2.0;
    result.cy     = captureHeight / 2.0;

    // Locate the back wide-angle camera
    AVCaptureDevice *device = nil;
    if (@available(iOS 13.0, *)) {
        device = [AVCaptureDevice
            defaultDeviceWithDeviceType:AVCaptureDeviceTypeBuiltInWideAngleCamera
                              mediaType:AVMediaTypeVideo
                               position:AVCaptureDevicePositionBack];
    }
    if (!device)
        device = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];

    if (!device) {
        // Conservative fallback: iPhone wide-angle is ~69° HFOV
        double hFovRad = 69.0 * M_PI / 180.0;
        result.fx = result.fy = (captureWidth / 2.0) / std::tan(hFovRad / 2.0);
        result.valid = false;
        return result;
    }

    // videoFieldOfView is the horizontal FOV in degrees for the active format.
    // It is set once the capture session has started.
    float hFovDeg = device.activeFormat.videoFieldOfView;
    if (hFovDeg < 1.0f) {
        // Session not started yet or format not set — use fallback
        double hFovRad = 69.0 * M_PI / 180.0;
        result.fx = result.fy = (captureWidth / 2.0) / std::tan(hFovRad / 2.0);
        result.valid = false;
        return result;
    }

    double hFovRad = hFovDeg * M_PI / 180.0;
    result.fx    = (captureWidth / 2.0) / std::tan(hFovRad / 2.0);
    result.fy    = result.fx; // iPhone pixels are square
    result.valid = true;
    return result;
}

void requestCameraAccess(std::function<void(bool)> callback)
{
    AVAuthorizationStatus status =
        [AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeVideo];

    if (status == AVAuthorizationStatusAuthorized) {
        // Already granted — proceed immediately
        callback(true);
        return;
    }

    if (status == AVAuthorizationStatusNotDetermined) {
        // First time — show the system permission prompt.
        // The completion block may be called on a background thread; dispatch to
        // main so Qt camera start() runs on the same thread that owns the QObject.
        [AVCaptureDevice requestAccessForMediaType:AVMediaTypeVideo
                               completionHandler:^(BOOL granted) {
            dispatch_async(dispatch_get_main_queue(), ^{
                callback(granted);
            });
        }];
        return;
    }

    // Denied or restricted — inform caller so it can show a message
    callback(false);
}

void forceMainCameraZoom()
{
    AVCaptureDevice *device = nil;
    if (@available(iOS 13.0, *)) {
        device = [AVCaptureDevice
            defaultDeviceWithDeviceType:AVCaptureDeviceTypeBuiltInWideAngleCamera
                              mediaType:AVMediaTypeVideo
                               position:AVCaptureDevicePositionBack];
    }
    if (!device) return;

    NSError *error = nil;
    if ([device lockForConfiguration:&error]) {
        device.videoZoomFactor = 1.0;
        [device unlockForConfiguration];
    } else {
        NSLog(@"forceMainCameraZoom: could not lock device — %@", error);
    }
}