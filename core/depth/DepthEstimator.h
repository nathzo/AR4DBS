#pragma once
#include <opencv2/core.hpp>
#include <string>

// Runs MiDaS v2.1 Small monocular depth estimation.
// Output depth map: CV_32F, same size as input, values in [0,1] where 1 = closest.
// Model: https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx
class DepthEstimator
{
public:
    explicit DepthEstimator(const std::string &modelPath);
    ~DepthEstimator();

    // Returns empty Mat if model not loaded
    bool isLoaded() const;

    // Returns normalised depth map: 1 = closest surface, 0 = furthest
    cv::Mat estimate(const cv::Mat &bgr);

private:
    struct Impl;
    Impl *m_impl;
};
