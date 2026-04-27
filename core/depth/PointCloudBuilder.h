#pragma once
#include <opencv2/core.hpp>
#include <vector>

struct Point3f { float x, y, z; };

// Unprojects a depth map into a 3-D point cloud using camera intrinsics
class PointCloudBuilder
{
public:
    // K: 3x3 camera matrix (CV_64F)
    explicit PointCloudBuilder(const cv::Mat &K);

    std::vector<Point3f> build(const cv::Mat &depthMap);

private:
    cv::Mat m_K;
};
