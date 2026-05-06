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

    // Per-field OCR confidence.
    // -1.0  = field was not detected (cell should appear empty).
    //  0–1  = Vision recognition confidence for that field.
    // Index: 0=x, 1=y, 2=z, 3=ring, 4=arc
    static constexpr int kFieldCount = 5;
    float confidence[kFieldCount] = {-1.f, -1.f, -1.f, -1.f, -1.f};
};

struct SurgicalPlan {
    LeksellTarget left;
    LeksellTarget right;

    bool hasLeft()  const { return left.valid;  }
    bool hasRight() const { return right.valid; }
    bool hasAny()   const { return left.valid || right.valid; }
};
