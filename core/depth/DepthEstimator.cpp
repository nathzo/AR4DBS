#include "DepthEstimator.h"
#include <opencv2/imgproc.hpp>
#include <cstring>

#ifdef HAVE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>

static constexpr int MIDAS_SIZE = 384;
static const float   MEAN[3]   = {0.485f, 0.456f, 0.406f};
static const float   STD[3]    = {0.229f, 0.224f, 0.225f};

struct DepthEstimator::Impl {
    Ort::Env            env{ORT_LOGGING_LEVEL_WARNING, "DepthEstimator"};
    Ort::SessionOptions opts;
    Ort::Session        session{nullptr};
    bool                loaded = false;
};

DepthEstimator::DepthEstimator(const std::string &modelPath)
    : m_impl(new Impl)
{
    try {
        m_impl->opts.SetIntraOpNumThreads(2);
        m_impl->opts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
#ifdef _WIN32
        // ONNX Runtime on Windows requires a wide-character path
        std::wstring widePath(modelPath.begin(), modelPath.end());
        m_impl->session = Ort::Session(m_impl->env, widePath.c_str(), m_impl->opts);
#else
        m_impl->session = Ort::Session(m_impl->env, modelPath.c_str(), m_impl->opts);
#endif
        m_impl->loaded = true;
    } catch (const Ort::Exception &e) {
        fprintf(stderr, "[DepthEstimator] load failed: %s\n", e.what());
        m_impl->loaded = false;
    }
}

DepthEstimator::~DepthEstimator() { delete m_impl; }
bool DepthEstimator::isLoaded() const { return m_impl->loaded; }

cv::Mat DepthEstimator::estimate(const cv::Mat &bgr)
{
    if (!m_impl->loaded) return {};

    cv::Mat rgb, resized, fp;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, resized, {MIDAS_SIZE, MIDAS_SIZE});
    resized.convertTo(fp, CV_32FC3, 1.f / 255.f);

    // Split HWC → three HW planes, normalize each, then pack into CHW tensor.
    // cv::Mat arithmetic uses NEON on arm64; much faster than the triple loop.
    static const float INV_STD[3] = {1.f/STD[0], 1.f/STD[1], 1.f/STD[2]};
    std::vector<cv::Mat> ch(3);
    cv::split(fp, ch);
    std::vector<float> tensor(3 * MIDAS_SIZE * MIDAS_SIZE);
    for (int c = 0; c < 3; ++c) {
        ch[c] = (ch[c] - MEAN[c]) * INV_STD[c];
        std::memcpy(tensor.data() + c * MIDAS_SIZE * MIDAS_SIZE,
                    ch[c].data,
                    MIDAS_SIZE * MIDAS_SIZE * sizeof(float));
    }

    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<int64_t, 4> inputShape = {1, 3, MIDAS_SIZE, MIDAS_SIZE};
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo, tensor.data(), tensor.size(), inputShape.data(), inputShape.size());

    // Read node names dynamically — MiDaS models vary between exports
    Ort::AllocatorWithDefaultOptions alloc;
    auto inputNamePtr  = m_impl->session.GetInputNameAllocated(0, alloc);
    auto outputNamePtr = m_impl->session.GetOutputNameAllocated(0, alloc);
    const char *inputName  = inputNamePtr.get();
    const char *outputName = outputNamePtr.get();
    auto outputs = m_impl->session.Run(
        Ort::RunOptions{nullptr}, &inputName, &inputTensor, 1, &outputName, 1);

    const float *data = outputs[0].GetTensorData<float>();
    cv::Mat disp(MIDAS_SIZE, MIDAS_SIZE, CV_32F, const_cast<float *>(data));
    cv::Mat norm;
    cv::normalize(disp, norm, 0.0, 1.0, cv::NORM_MINMAX, CV_32F);
    cv::Mat result;
    cv::resize(norm, result, bgr.size(), 0, 0, cv::INTER_LINEAR);
    return result;
}

#else
// ── Stub when ONNX Runtime is not installed ───────────────────────────────────
struct DepthEstimator::Impl { bool loaded = false; };
DepthEstimator::DepthEstimator(const std::string &) : m_impl(new Impl) {}
DepthEstimator::~DepthEstimator() { delete m_impl; }
bool    DepthEstimator::isLoaded() const { return false; }
cv::Mat DepthEstimator::estimate(const cv::Mat &) { return {}; }
#endif
