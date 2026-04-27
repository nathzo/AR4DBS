#pragma once

// One electrode target in Leksell stereotactic frame coordinates.
// Matches the fields shown on the Medtronic Vantage planning screen.
struct LeksellTarget {
    double x_mm     = 0;   // left-right (mm)
    double y_mm     = 0;   // anterior-posterior (mm)
    double z_mm     = 0;   // superior-inferior (mm)
    double ring_deg = 0;   // arc-carrier rotation (degrees)
    double arc_deg  = 0;   // electrode tilt from vertical (degrees)
    bool   valid    = false;
};

struct SurgicalPlan {
    LeksellTarget left;
    LeksellTarget right;

    bool hasLeft()  const { return left.valid;  }
    bool hasRight() const { return right.valid; }
    bool hasAny()   const { return left.valid || right.valid; }
};
