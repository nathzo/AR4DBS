#pragma once
#include <functional>

// Plain C++ struct — safe to include from both .cpp and .mm files
struct CameraIntrinsicsData {
    double fx = 0, fy = 0;  // focal lengths in pixels
    double cx = 0, cy = 0;  // principal point in pixels
    int    width  = 0;
    int    height = 0;
    bool   valid  = false;
};

// Reads horizontal FOV from the active back-camera format and computes
// pixel-level intrinsics for the given capture resolution.
// Must be called after AVCaptureSession has started (activeFormat is set).
// Implemented in CameraIntrinsics.mm (Objective-C++).
CameraIntrinsicsData readCameraIntrinsics(int captureWidth, int captureHeight);

// Requests camera access from the user if not already determined.
// - If already authorized   → calls callback(true) immediately on the calling thread.
// - If not yet determined   → shows the iOS permission prompt; calls callback(true/false)
//                             on the main thread once the user responds.
// - If denied / restricted  → calls callback(false) immediately on the calling thread.
// Implemented in CameraIntrinsics.mm (Objective-C++).
void requestCameraAccess(std::function<void(bool)> callback);

void forceMainCameraZoom();