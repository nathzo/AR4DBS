#pragma once
#include <opencv2/core.hpp>
#include <optional>
#include <vector>

struct Triangle {
    cv::Point3f v0, v1, v2;
};

// Ray–mesh intersection via brute-force (BVH to be added later)
class RayCaster
{
public:
    explicit RayCaster(const std::vector<Triangle> &mesh);

    // Returns the closest hit point along the ray, or nullopt if no hit
    std::optional<cv::Point3f> cast(const cv::Point3f &origin,
                                    const cv::Point3f &direction) const;

private:
    std::vector<Triangle> m_mesh;
};
