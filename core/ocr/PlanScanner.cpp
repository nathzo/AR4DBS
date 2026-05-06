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
    cv::Mat gray, blurred, thresh;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, {5, 5}, 0);
    cv::threshold(blurred, thresh, 80, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double minArea = frame.rows * frame.cols * 0.15;
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

    if (bestQuad.size() != 4) return frame;

    auto centroid = [](const std::vector<cv::Point2f> &pts) {
        cv::Point2f c{0, 0};
        for (const auto &p : pts) c += p;
        return c * (1.0f / pts.size());
    };
    cv::Point2f cen = centroid(bestQuad);
    std::sort(bestQuad.begin(), bestQuad.end(),
              [&](const cv::Point2f &a, const cv::Point2f &b) {
                  bool aTop = a.y < cen.y, bTop = b.y < cen.y;
                  if (aTop != bTop) return aTop;
                  return (aTop ? a.x < b.x : a.x > b.x);
              });

    const int W = 1280, H = 720;
    std::vector<cv::Point2f> src = { bestQuad[0], bestQuad[1], bestQuad[2], bestQuad[3] };
    std::vector<cv::Point2f> dst = { {0,0}, {float(W),0}, {float(W),float(H)}, {0,float(H)} };

    cv::Mat M = cv::getPerspectiveTransform(src, dst);
    cv::Mat out;
    cv::warpPerspective(frame, out, M, {W, H});
    return out;
}

// ── iOS: line-aware parsing with per-field confidence ─────────────────────────

#ifdef HAVE_IOS_OCR

// Search the flattened text of a line group for one field pattern.
// Returns true on success; sets val and conf (confidence of the matching line).
static bool extractField(const std::vector<OcrLine> &lines,
                         const std::string           &flat,
                         const std::vector<size_t>   &lineStarts,
                         const std::regex            &re,
                         double &val, float &conf)
{
    std::smatch m;
    if (!std::regex_search(flat, m, re)) return false;
    try {
        val = std::stod(m[1].str());
        // Map the capture-group start position back to a line index.
        size_t pos = (size_t)m.position(1);
        auto it = std::upper_bound(lineStarts.begin(), lineStarts.end(), pos);
        int li = (int)std::distance(lineStarts.begin(), it) - 1;
        conf = (li >= 0 && li < (int)lines.size()) ? lines[li].confidence : 0.f;
        return true;
    } catch (...) { return false; }
}

static LeksellTarget parseTargetFromLines(const std::vector<OcrLine> &lines)
{
    static const std::regex rxX   (R"(X\s*\(mm\)[^0-9]*([0-9]+\.?[0-9]*))", std::regex::icase);
    static const std::regex rxY   (R"(Y\s*\(mm\)[^0-9]*([0-9]+\.?[0-9]*))", std::regex::icase);
    static const std::regex rxZ   (R"(Z\s*\(mm\)[^0-9]*([0-9]+\.?[0-9]*))", std::regex::icase);
    static const std::regex rxRing(R"(Ring[^0-9]*([0-9]+\.?[0-9]*))",        std::regex::icase);
    static const std::regex rxArc (R"(Arc[^0-9]*([0-9]+\.?[0-9]*))",         std::regex::icase);

    // Flatten lines into one string; record the start offset of each line.
    std::string flat;
    std::vector<size_t> lineStarts;
    for (const auto &l : lines) {
        lineStarts.push_back(flat.size());
        flat += l.text + '\n';
    }

    LeksellTarget t;
    bool okX    = extractField(lines, flat, lineStarts, rxX,    t.x_mm,     t.confidence[0]);
    bool okY    = extractField(lines, flat, lineStarts, rxY,    t.y_mm,     t.confidence[1]);
    bool okZ    = extractField(lines, flat, lineStarts, rxZ,    t.z_mm,     t.confidence[2]);
    bool okRing = extractField(lines, flat, lineStarts, rxRing, t.ring_deg, t.confidence[3]);
    bool okArc  = extractField(lines, flat, lineStarts, rxArc,  t.arc_deg,  t.confidence[4]);
    // confidence[] defaults to -1 (not detected); only overwritten when okX etc. is true.
    t.valid = okX && okY && okZ && okRing && okArc;
    return t;
}

static SurgicalPlan parseLinesIOS(const std::vector<OcrLine> &lines)
{
    auto toLower = [](std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        return s;
    };

    // Find the line indices of the "Gauche" and "Droite" column headers.
    int idxGauche = -1, idxDroite = -1;
    for (int i = 0; i < (int)lines.size(); ++i) {
        std::string low = toLower(lines[i].text);
        if (idxGauche < 0 && low.find("gauche") != std::string::npos) idxGauche = i;
        if (idxDroite < 0 && low.find("droite") != std::string::npos) idxDroite = i;
    }

    // Split into per-side line groups.
    // If only one header is found, only that side is parsed — the other stays empty.
    // The old "parse both sides from the same block" fallback is intentionally removed.
    std::vector<OcrLine> leftLines, rightLines;

    if (idxGauche >= 0 && idxDroite >= 0) {
        if (idxGauche < idxDroite) {
            leftLines  = { lines.begin() + idxGauche, lines.begin() + idxDroite };
            rightLines = { lines.begin() + idxDroite, lines.end() };
        } else {
            rightLines = { lines.begin() + idxDroite, lines.begin() + idxGauche };
            leftLines  = { lines.begin() + idxGauche, lines.end() };
        }
    } else if (idxGauche >= 0) {
        leftLines  = { lines.begin() + idxGauche, lines.end() };
    } else if (idxDroite >= 0) {
        rightLines = { lines.begin() + idxDroite, lines.end() };
    }

    SurgicalPlan plan;
    if (!leftLines.empty())  plan.left  = parseTargetFromLines(leftLines);
    if (!rightLines.empty()) plan.right = parseTargetFromLines(rightLines);
    return plan;
}

#endif // HAVE_IOS_OCR

// ── OCR dispatch ──────────────────────────────────────────────────────────────

SurgicalPlan PlanScanner::scan(const cv::Mat &frame)
{
#ifdef HAVE_TESSERACT
    cv::Mat screen = extractScreen(frame);

    cv::Mat gray, big;
    cv::cvtColor(screen, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, big, {}, 2.0, 2.0, cv::INTER_CUBIC);

    Pix *pix = pixCreate(big.cols, big.rows, 8);
    l_uint32 *data = pixGetData(pix);
    int wpl = pixGetWpl(pix);
    for (int y = 0; y < big.rows; ++y) {
        const uchar *row = big.ptr<uchar>(y);
        for (int x = 0; x < big.cols; ++x)
            SET_DATA_BYTE(data + y * wpl, x, row[x]);
    }

    tesseract::TessBaseAPI api;
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
    cv::Mat screen = extractScreen(frame);
    std::vector<OcrLine> lines = IOSOCREngine::recognize(screen);
    return parseLinesIOS(lines);

#else
    (void)frame;
    return {};
#endif
}

// ── Text parsing (Tesseract / unit tests) ─────────────────────────────────────

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
    static const std::regex rxX   (R"(X\s*\(mm\)[^0-9]*([0-9]+\.?[0-9]*))", std::regex::icase);
    static const std::regex rxY   (R"(Y\s*\(mm\)[^0-9]*([0-9]+\.?[0-9]*))", std::regex::icase);
    static const std::regex rxZ   (R"(Z\s*\(mm\)[^0-9]*([0-9]+\.?[0-9]*))", std::regex::icase);
    static const std::regex rxRing(R"(Ring[^0-9]*([0-9]+\.?[0-9]*))",        std::regex::icase);
    static const std::regex rxArc (R"(Arc[^0-9]*([0-9]+\.?[0-9]*))",         std::regex::icase);

    LeksellTarget t;
    bool okX    = extractAfter(block, rxX,    t.x_mm);     if (okX)    t.confidence[0] = 1.f;
    bool okY    = extractAfter(block, rxY,    t.y_mm);     if (okY)    t.confidence[1] = 1.f;
    bool okZ    = extractAfter(block, rxZ,    t.z_mm);     if (okZ)    t.confidence[2] = 1.f;
    bool okRing = extractAfter(block, rxRing, t.ring_deg); if (okRing) t.confidence[3] = 1.f;
    bool okArc  = extractAfter(block, rxArc,  t.arc_deg);  if (okArc)  t.confidence[4] = 1.f;
    t.valid = okX && okY && okZ && okRing && okArc;
    return t;
}

SurgicalPlan PlanScanner::parseText(const std::string &text)
{
    auto toLower = [](std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        return s;
    };
    std::string lower = toLower(text);

    size_t posLeft  = lower.find("gauche");
    size_t posRight = lower.find("droite");

    std::string leftBlock, rightBlock;
    if (posLeft != std::string::npos && posRight != std::string::npos) {
        if (posLeft < posRight) {
            leftBlock  = text.substr(posLeft,  posRight - posLeft);
            rightBlock = text.substr(posRight);
        } else {
            rightBlock = text.substr(posRight, posLeft - posRight);
            leftBlock  = text.substr(posLeft);
        }
    } else if (posLeft != std::string::npos) {
        leftBlock  = text.substr(posLeft);
        // rightBlock stays empty — no mirroring
    } else if (posRight != std::string::npos) {
        rightBlock = text.substr(posRight);
        // leftBlock stays empty — no mirroring
    }
    // else: no headers found → both sides empty

    SurgicalPlan plan;
    if (!leftBlock.empty())  plan.left  = parseTarget(leftBlock);
    if (!rightBlock.empty()) plan.right = parseTarget(rightBlock);
    return plan;
}
