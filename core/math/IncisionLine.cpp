#include "IncisionLine.h"
#include <cmath>

// Leksell convention:
//   Arc  = tilt of the electrode from vertical (0° = pointing straight down)
//   Ring = rotation around the vertical axis
// We map: theta = Arc (polar from Z),  phi = Ring (azimuthal)
IncisionLine IncisionLine::fromLeksell(const LeksellTarget &t, double lengthM)
{
    Plan p;
    p.x      = t.x_mm / 1000.0;
    p.y      = t.y_mm / 1000.0;
    p.z      = t.z_mm / 1000.0;
    p.theta  = t.arc_deg  * M_PI / 180.0;
    p.phi    = t.ring_deg * M_PI / 180.0;
    p.length = lengthM;
    return IncisionLine(p);
}

IncisionLine::IncisionLine(const Plan &p)
{
    m_dir = {
        std::sin(p.theta) * std::cos(p.phi),
        std::sin(p.theta) * std::sin(p.phi),
        std::cos(p.theta)
    };

    m_target  = { p.x, p.y, p.z };
    m_lineEnd = { p.x + m_dir.x * p.length,
                  p.y + m_dir.y * p.length,
                  p.z + m_dir.z * p.length };
}
