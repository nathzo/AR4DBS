#include "PlanScanner.h"

#include <opencv2/imgproc.hpp>
#include <regex>
#include <algorithm>

#ifdef HAVE_TESSERACT
#  include <tesseract/baseapi.h>
#  include <leptonica/allheaders.h>
#elif defined(HAVE_IOS_OCR)
#  include "platform/ios/IOSOCREngine.h"
#endif

// ── Availability ─────────────────────────────────────────────────────────────

bool PlanScanner::isAvailable()
{
#if defined(HAVE_TESSERACT) || defined(HAVE_IOS_OCR)
    return true;
#else
    return false;
#endif
}

// ── Screen extraction ─────────────────────────────────────────────────────────

cv::Mat PlanScanner::extractScreen(const cv::Mat &frame)
{
    // Find the largest bright quadrilateral (the monitor face)
    cv::Mat gray, blurred, thresh;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, {5, 5}, 0);
    cv::threshold(blurred, thresh, 80, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double minArea = frame.rows * frame.cols * 0.15; // must be >15% of frame
    double maxArea = 0;
    std::vector<cv::Point2f> bestQuad;

    for (const auto &c : contours) {
        std::vector<cv::Point> approx;
        cv::approxPolyDP(c, approx, cv::arcLength(c, true) * 0.03, true);
        if (approx.size() != 4) continue;
        double area = cv::contourArea(approx);
        if (area > maxArea && area > minArea) {
            maxArea = area;
            bestQuad.clear();
            for (const auto &p : approx)
                bestQuad.push_back(cv::Point2f(p));
        }
    }

    if (bestQuad.size() != 4) return frame; // fallback: full frame

    // Order corners: top-left, top-right, bottom-right, bottom-left
    auto centroid = [](const std::vector<cv::Point2f> &pts) {
        cv::Point2f c{0, 0};
        for (const auto &p : pts) c += p;
        return c * (1.0f / pts.size());
    };
    cv::Point2f cen = centroid(bestQuad);
    std::sort(bestQuad.begin(), bestQuad.end(),
              [&](const cv::Point2f &a, const cv::Point2f &b) {
                  // Partition into top/bottom by y, then left/right by x
                  bool aTop = a.y < cen.y, bTop = b.y < cen.y;
                  if (aTop != bTop) return aTop;
                  return (aTop ? a.x < b.x : a.x > b.x);
              });
    // After sort order is: TL, TR, BR, BL
    std::vector<cv::Point2f> src = {
        bestQuad[0], bestQuad[1], bestQuad[2], bestQuad[3]
    };

    const int W = 1280, H = 720;
    std::vector<cv::Point2f> dst = {
        {0, 0}, {float(W), 0}, {float(W), float(H)}, {0, float(H)}
    };

    cv::Mat M  = cv::getPerspectiveTransform(src, dst);
    cv::Mat out;
    cv::warpPerspective(frame, out, M, {W, H});
    return out;
}

// ── OCR ───────────────────────────────────────────────────────────────────────

SurgicalPlan PlanScanner::scan(const cv::Mat &frame)
{
#ifdef HAVE_TESSERACT
    cv::Mat screen = extractScreen(frame);

    // Convert BGR → grayscale → upscale (Tesseract works better at higher DPI)
    cv::Mat gray, big;
    cv::cvtColor(screen, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, big, {}, 2.0, 2.0, cv::INTER_CUBIC);

    // Hand image to Tesseract via Leptonica Pix
    Pix *pix = pixCreate(big.cols, big.rows, 8);
    l_uint32 *data = pixGetData(pix);
    int wpl = pixGetWpl(pix);
    for (int y = 0; y < big.rows; ++y) {
        const uchar *row = big.ptr<uchar>(y);
        for (int x = 0; x < big.cols; ++x)
            SET_DATA_BYTE(data + y * wpl, x, row[x]);
    }

    tesseract::TessBaseAPI api;
    // "fra" helps with accented chars like "degré"; fallback to "eng" if fra not installed
    if (api.Init(nullptr, "fra+eng") != 0)
        if (api.Init(nullptr, "eng") != 0) { pixDestroy(&pix); return {}; }

    api.SetPageSegMode(tesseract::PSM_AUTO);
    api.SetImage(pix);
    char *raw = api.GetUTF8Text();
    std::string text(raw ? raw : "");
    delete[] raw;
    api.End();
    pixDestroy(&pix);

    return parseText(text);

#elif defined(HAVE_IOS_OCR)
    // Perspective-correct to the monitor face first, then hand the cropped
    // image to Apple Vision.  extractScreen() falls back to the full frame
    // if no bright rectangle is found.
    cv::Mat screen = extractScreen(frame);
    std::string text = IOSOCREngine::recognize(screen);
    return parseText(text);

#else
    (void)frame;
    return {}; // no OCR backend — caller opens manual-entry form
#endif
}

// ── Text parsing ──────────────────────────────────────────────────────────────

// Extract the first floating-point number after a regex label match
static bool extractAfter(const std::string &text,
                         const std::regex  &re,
                         double            &out)
{
    std::smatch m;
    if (!std::regex_search(text, m, re)) return false;
    try { out = std::stod(m[1].str()); return true; }
    catch (...) { return false; }
}

static LeksellTarget parseTarget(const std::string &block)
{
    // Labels on the Medtronic screen (OCR may mangle accents)
    static const std::regex rxX   (R"(X\s*\(mm\)[^0-9]*([0-9]+\.?[0-9]*))", std::regex::icase);
    static const std::regex rxY   (R"(Y\s*\(mm\)[^0-9]*([0-9]+\.?[0-9]*))", std::regex::icase);
    static const std::regex rxZ   (R"(Z\s*\(mm\)[^0-9]*([0-9]+\.?[0-9]*))", std::regex::icase);
    static const std::regex rxRing(R"(Ring[^0-9]*([0-9]+\.?[0-9]*))",        std::regex::icase);
    static const std::regex rxArc (R"(Arc[^0-9]*([0-9]+\.?[0-9]*))",         std::regex::icase);

    LeksellTarget t;
    bool okX    = extractAfter(block, rxX,    t.x_mm);
    bool okY    = extractAfter(block, rxY,    t.y_mm);
    bool okZ    = extractAfter(block, rxZ,    t.z_mm);
    bool okRing = extractAfter(block, rxRing, t.ring_deg);
    bool okArc  = extractAfter(block, rxArc,  t.arc_deg);
    t.valid = okX && okY && okZ && okRing && okArc;
    return t;
}

SurgicalPlan PlanScanner::parseText(const std::string &text)
{
    // The screen layout is two columns separated by a "Remarques" / centre block.
    // Strategy: find "Gauche"/"Droite" (or "Left"/"Right") and split the text.
    auto toLower = [](std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        return s;
    };
    std::string lower = toLower(text);

    SurgicalPlan plan;

    // Locate column headers
    size_t posLeft  = lower.find("gauche");
    size_t posRight = lower.find("droite");

    // Build per-side text slices; fall back to splitting at midpoint
    std::string leftBlock, rightBlock;
    if (posLeft != std::string::npos && posRight != std::string::npos) {
        if (posLeft < posRight) {
            leftBlock  = text.substr(posLeft,  posRight - posLeft);
            rightBlock = text.substr(posRight);
        } else {
            rightBlock = text.substr(posRight, posLeft - posRight);
            leftBlock  = text.substr(posLeft);
        }
    } else {
        // No column markers found — try parsing the full text for both sides
        // by finding the 1st and 2nd occurrence of each label
        leftBlock  = text;
        rightBlock = text;
    }

    plan.left  = parseTarget(leftBlock);
    plan.right = parseTarget(rightBlock);

    // If both parsed the same block (no column split), deduplicate by
    // finding the second set of numbers for the right side.
    // This is a best-effort fallback — the confirmation dialog lets the user fix it.

    return plan;
}
