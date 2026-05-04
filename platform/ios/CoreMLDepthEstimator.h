#pragma once
#include <QtGlobal>
#ifdef Q_OS_IOS
#include <opencv2/core.hpp>
#include <string>

// Monocular depth estimator backed by a CoreML model (e.g. MiDaS small .mlpackage).
// Output: CV_32F map, same size as input, values in [0,1] where 1 = closest.
// Runs on the Neural Engine on A12+ (iPhone XS and later, iOS 14+).
// No LiDAR sensor required — works on all current iPhones.
//
// Expected model: MiDaS v2.1 Small converted with coremltools.
// Input:  MultiArray float32 [1,3,H,W] or [3,H,W], ImageNet-normalised, OR image type.
// Output: MultiArray float32 [1,H,W] or [H,W], raw disparity (higher = closer).
class CoreMLDepthEstimator
{
public:
    explicit CoreMLDepthEstimator(const std::string &modelPath);
    ~CoreMLDepthEstimator();

    bool        isLoaded()  const;
    cv::Mat     estimate(const cv::Mat &bgr);
    std::string lastError() const;  // empty when last estimate() succeeded

private:
    struct Impl;
    Impl *m_impl;
};

#endif // Q_OS_IOS
