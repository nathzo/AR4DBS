#include "VispTracker.h"

VispTracker::VispTracker()
{
    // TODO: load frame_model.obj and camera intrinsics; configure vpMbEdgeTracker
}

void VispTracker::init(const cv::Mat &frame, const cv::Mat &rvec, const cv::Mat &tvec)
{
    // TODO: set initial pose from AprilTag result and start ViSP tracking
    m_initialised = true;
}

bool VispTracker::track(const cv::Mat &frame, cv::Mat &rvec, cv::Mat &tvec)
{
    // TODO: vpMbEdgeTracker::track(); extract updated pose
    return m_initialised;
}

void VispTracker::reset()
{
    m_initialised = false;
}
