#include "PoseUtils.h"
#include <opencv2/calib3d.hpp>

namespace PoseUtils
{

cv::Mat toTransform(const cv::Mat &rvec, const cv::Mat &tvec)
{
    cv::Mat R;
    cv::Rodrigues(rvec, R);

    cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
    R.copyTo(T(cv::Rect(0, 0, 3, 3)));
    tvec.reshape(1, 3).copyTo(T(cv::Rect(3, 0, 1, 3)));
    return T;
}

void fromTransform(const cv::Mat &T, cv::Mat &rvec, cv::Mat &tvec)
{
    cv::Mat R = T(cv::Rect(0, 0, 3, 3));
    cv::Rodrigues(R, rvec);
    tvec = T(cv::Rect(3, 0, 1, 3)).clone();
}

cv::Point2f project(const cv::Point3d &pt,
                    const cv::Mat &K,
                    const cv::Mat &rvec,
                    const cv::Mat &tvec)
{
    // Use double throughout to avoid type mismatch in OpenCV 4.7+
    std::vector<cv::Point3d> in = {pt};
    std::vector<cv::Point2d> out;
    cv::Mat noDistortion = cv::Mat::zeros(1, 4, CV_64F);
    cv::projectPoints(in, rvec, tvec, K, noDistortion, out);
    return cv::Point2f(static_cast<float>(out[0].x),
                       static_cast<float>(out[0].y));
}

} // namespace PoseUtils
