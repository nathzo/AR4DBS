#pragma once
#include <opencv2/core.hpp>
#include "SurgicalPlan.h"

// Represents the DBS trajectory: target point + direction → skull-side endpoint
class IncisionLine
{
public:
    struct Plan {
        double x, y, z;   // DBS target in frame coordinates (metres)
        double theta;      // polar angle from frame z-axis (radians)
        double phi;        // azimuthal angle (radians)
        double length;     // line length from target toward skull (metres)
    };

    explicit IncisionLine(const Plan &plan);

    // Construct from Leksell stereotactic coordinates.
    // lengthM: visualised trajectory length from target toward skull (metres).
    static IncisionLine fromLeksell(const LeksellTarget &t, double lengthM = 0.30);

    cv::Point3d target()  const { return m_target;  } // DBS target (deep end)
    cv::Point3d lineEnd() const { return m_lineEnd; } // skull-side end of trajectory
    cv::Point3d direction() const { return m_dir;   } // unit vector target → skull

private:
    cv::Point3d m_target;
    cv::Point3d m_lineEnd;
    cv::Point3d m_dir;
};
