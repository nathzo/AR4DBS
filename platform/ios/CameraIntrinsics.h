#pragma once

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
