#include "IncisionLine.h"
#include <cmath>

// Leksell frame convention (matches tag_config.json coordinate system):
//   x: 0 = right (tag 1), 200 mm = left (tag 0)
//   y: 0 = posterior,     200 mm = anterior
//   z: 0 = superior (out of tags / toward camera), increases inferior
//
// Arc  = position on the arc, sweeping right ear → top of skull → left ear:
//   Arc=0°   → right ear (−x), independent of Ring
//   Arc=90°  → top of skull (−z) at Ring=90°; nose (+y) at Ring=0°; etc.
//   Arc=180° → left ear (+x), independent of Ring
//
// Ring = orientation of the arc plane, sweeping nose → top of skull → back:
//   Ring=0°   → nose (+y)
//   Ring=90°  → top of skull (−z)
//   Ring=180° → back of head (−y)
//
// Resulting direction (target → skull entry):
//   d = (−cos(Arc),  sin(Arc)·cos(Ring),  −sin(Arc)·sin(Ring))
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
        -std::cos(p.theta),
         std::sin(p.theta) * std::cos(p.phi),
        -std::sin(p.theta) * std::sin(p.phi)
    };

    m_target  = { p.x, p.y, p.z };
    m_lineEnd = { p.x + m_dir.x * p.length,
                  p.y + m_dir.y * p.length,
                  p.z + m_dir.z * p.length };
}
