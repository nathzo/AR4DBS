#include "RayCaster.h"

RayCaster::RayCaster(const std::vector<Triangle> &mesh)
    : m_mesh(mesh)
{}

std::optional<cv::Point3f> RayCaster::cast(const cv::Point3f &origin,
                                            const cv::Point3f &direction) const
{
    // TODO: Möller–Trumbore intersection test per triangle; return closest hit
    return std::nullopt;
}
