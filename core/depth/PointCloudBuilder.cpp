#include "PointCloudBuilder.h"

PointCloudBuilder::PointCloudBuilder(const cv::Mat &K)
    : m_K(K.clone())
{}

std::vector<Point3f> PointCloudBuilder::build(const cv::Mat &depthMap)
{
    // TODO: iterate pixels, unproject (u,v,d) → (X,Y,Z) using K inverse
    return {};
}
