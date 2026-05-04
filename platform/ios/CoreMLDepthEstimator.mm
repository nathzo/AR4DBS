#include "CoreMLDepthEstimator.h"
#ifdef Q_OS_IOS

#import <CoreML/CoreML.h>
#include <opencv2/imgproc.hpp>
#include <cstring>

// ImageNet mean and 1/std — used for MultiArray inputs that expect ImageNet normalisation.
// Apple's Depth Anything v2 CoreML model uses image-type input so these are not applied.
static const float kMean[3]   = {0.485f, 0.456f, 0.406f};
static const float kInvStd[3] = {1.f/0.229f, 1.f/0.224f, 1.f/0.225f};

// ─────────────────────────────────────────────────────────────────────────────

struct CoreMLDepthEstimator::Impl {
    // __strong is required: ObjC pointers in C++ structs are __unsafe_unretained
    // by default under ARC and do not retain the object. Without __strong the
    // MLModel is released when the constructor's local variable goes out of scope
    // and the pointer becomes dangling, causing a crash on the next inference call.
    __strong MLModel  *model      = nil;
    __strong NSString *inputName  = nil;
    __strong NSString *outputName = nil;
    bool      isImage    = false;  // true → model takes a CVPixelBuffer input
    int       inputW     = 256;
    int       inputH     = 256;
    bool      loaded     = false;
};

// ─────────────────────────────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────────────────────────────

CoreMLDepthEstimator::CoreMLDepthEstimator(const std::string &modelPath)
    : m_impl(new Impl)
{
    NSString *path = [NSString stringWithUTF8String:modelPath.c_str()];
    NSURL    *url  = [NSURL fileURLWithPath:path];

    // Prefer the Neural Engine; fall back to GPU then CPU automatically.
    MLModelConfiguration *cfg = [[MLModelConfiguration alloc] init];
    cfg.computeUnits = MLComputeUnitsAll;

    NSError *err  = nil;
    MLModel *mdl  = [MLModel modelWithContentsOfURL:url configuration:cfg error:&err];
    if (!mdl || err) {
        NSLog(@"[CoreMLDepth] load failed: %@", err.localizedDescription);
        return;
    }
    m_impl->model = mdl;

    // Discover input/output names and the spatial size the model expects.
    MLModelDescription *desc = mdl.modelDescription;
    m_impl->inputName  = desc.inputDescriptionsByName.allKeys.firstObject;
    m_impl->outputName = desc.outputDescriptionsByName.allKeys.firstObject;

    MLFeatureDescription *inDesc = desc.inputDescriptionsByName[m_impl->inputName];
    if (inDesc.type == MLFeatureTypeImage) {
        m_impl->isImage = true;
        m_impl->inputW  = (int)inDesc.imageConstraint.pixelsWide;
        m_impl->inputH  = (int)inDesc.imageConstraint.pixelsHigh;
    } else if (inDesc.type == MLFeatureTypeMultiArray) {
        NSArray<NSNumber*> *shape = inDesc.multiArrayConstraint.shape;
        NSUInteger n = shape.count;
        if (n >= 2) {
            m_impl->inputH = shape[n-2].intValue;
            m_impl->inputW = shape[n-1].intValue;
        }
    }

    m_impl->loaded = true;
    NSLog(@"[CoreMLDepth] ready — input %dx%d (image-type=%d)",
          m_impl->inputW, m_impl->inputH, (int)m_impl->isImage);
}

CoreMLDepthEstimator::~CoreMLDepthEstimator() { delete m_impl; }
bool CoreMLDepthEstimator::isLoaded() const    { return m_impl->loaded; }

// ─────────────────────────────────────────────────────────────────────────────
// estimate()
// ─────────────────────────────────────────────────────────────────────────────

cv::Mat CoreMLDepthEstimator::estimate(const cv::Mat &bgr)
{
    if (!m_impl->loaded || bgr.empty()) return {};

    const int W = m_impl->inputW;
    const int H = m_impl->inputH;

    NSError *err              = nil;
    id<MLFeatureProvider> inp = nil;

    if (m_impl->isImage) {
        // ── Image-type input: wrap as BGRA CVPixelBuffer ──────────────────────
        // CoreML handles normalisation internally for image inputs.
        cv::Mat resized;
        cv::resize(bgr, resized, {W, H});

        NSDictionary *attrs = @{
            (NSString*)kCVPixelBufferCGImageCompatibilityKey:            @YES,
            (NSString*)kCVPixelBufferCGBitmapContextCompatibilityKey:    @YES
        };
        CVPixelBufferRef pb = nullptr;
        CVPixelBufferCreate(kCFAllocatorDefault, W, H,
                            kCVPixelFormatType_32BGRA,
                            (__bridge CFDictionaryRef)attrs, &pb);
        if (!pb) return {};

        CVPixelBufferLockBaseAddress(pb, 0);
        uint8_t    *dst    = static_cast<uint8_t*>(CVPixelBufferGetBaseAddress(pb));
        const size_t stride = CVPixelBufferGetBytesPerRow(pb);
        for (int y = 0; y < H; ++y) {
            const uchar *src = resized.ptr<uchar>(y);
            uint8_t     *row = dst + y * stride;
            for (int x = 0; x < W; ++x) {
                row[x*4+0] = src[x*3+0];  // B
                row[x*4+1] = src[x*3+1];  // G
                row[x*4+2] = src[x*3+2];  // R
                row[x*4+3] = 255;
            }
        }
        CVPixelBufferUnlockBaseAddress(pb, 0);

        MLFeatureValue *fv = [MLFeatureValue featureValueWithPixelBuffer:pb];
        CFRelease(pb);
        inp = [[MLDictionaryFeatureProvider alloc]
               initWithDictionary:@{m_impl->inputName: fv} error:&err];

    } else {
        // ── MultiArray input: CHW float32 with ImageNet normalisation ─────────
        cv::Mat rgb, resized, fp;
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
        cv::resize(rgb, resized, {W, H});
        resized.convertTo(fp, CV_32FC3, 1.f / 255.f);

        std::vector<float> tensor(3 * H * W);
        std::vector<cv::Mat> ch(3);
        cv::split(fp, ch);
        for (int c = 0; c < 3; ++c) {
            ch[c] = (ch[c] - kMean[c]) * kInvStd[c];
            std::memcpy(tensor.data() + c * H * W,
                        ch[c].data, H * W * sizeof(float));
        }

        // Support models exported with a batch dimension ([1,3,H,W]) or without ([3,H,W]).
        MLModelDescription *desc = m_impl->model.modelDescription;
        NSArray<NSNumber*> *modelShape =
            desc.inputDescriptionsByName[m_impl->inputName]
                .multiArrayConstraint.shape;
        NSArray<NSNumber*> *mlShape, *strides;
        if (modelShape.count == 4) {
            mlShape = @[@1, @3, @(H), @(W)];
            strides = @[@(3*H*W), @(H*W), @(W), @1];
        } else {
            mlShape = @[@3, @(H), @(W)];
            strides = @[@(H*W), @(W), @1];
        }

        MLMultiArray *arr = [[MLMultiArray alloc]
            initWithDataPointer:tensor.data()
                          shape:mlShape
                      dataType:MLMultiArrayDataTypeFloat32
                        strides:strides
                    deallocator:nil
                          error:&err];
        if (err || !arr) return {};

        inp = [[MLDictionaryFeatureProvider alloc]
               initWithDictionary:@{m_impl->inputName:
                   [MLFeatureValue featureValueWithMultiArray:arr]}
               error:&err];
    }

    if (err || !inp) return {};

    id<MLFeatureProvider> out =
        [m_impl->model predictionFromFeatures:inp error:&err];
    if (err || !out) return {};

    MLMultiArray *outArr =
        [out featureValueForName:m_impl->outputName].multiArrayValue;
    if (!outArr) return {};

    // Output shape: [1,H,W] or [H,W] — take the last two dimensions.
    NSUInteger n      = outArr.shape.count;
    const int  outH   = outArr.shape[n-2].intValue;
    const int  outW   = outArr.shape[n-1].intValue;

    cv::Mat disp(outH, outW, CV_32F);
    std::memcpy(disp.data, outArr.dataPointer,
                (size_t)outH * outW * sizeof(float));

    // Normalise raw output to [0,1], then invert so that 1 = closest.
    // Depth Anything v2 outputs metric-like depth (larger = farther), which is the
    // opposite of the disparity convention expected by AppController's anchor formula.
    cv::Mat norm;
    cv::normalize(disp, norm, 0.0, 1.0, cv::NORM_MINMAX, CV_32F);
    norm = 1.0f - norm;   // invert: 0=far, 1=close
    cv::Mat result;
    cv::resize(norm, result, bgr.size(), 0, 0, cv::INTER_LINEAR);
    return result;
}

#endif // Q_OS_IOS
