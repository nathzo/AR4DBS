#include "HybridTracker.h"
#include "AprilTagTracker.h"
#include "VispTracker.h"

struct HybridTracker::Impl {
    AprilTagTracker april;
    VispTracker     visp;
};

HybridTracker::HybridTracker()
    : m_impl(new Impl)
{}

bool HybridTracker::update(const cv::Mat &frame, cv::Mat &rvec, cv::Mat &tvec)
{
    // 1. Try ViSP if already initialised
    if (m_impl->visp.isInitialised()) {
        if (m_impl->visp.track(frame, rvec, tvec))
            return true;
        m_impl->visp.reset(); // drift / loss → fall through to re-init
    }

    // 2. Re-initialise from AprilTags
    auto tags = m_impl->april.detect(frame);
    if (tags.empty())
        return false;

    // TODO: fuse multi-tag poses; for now use the first tag
    rvec = tags[0].rvec;
    tvec = tags[0].tvec;
    m_impl->visp.init(frame, rvec, tvec);
    return true;
}
