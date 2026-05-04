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
    // ARC + C++ pImpl via raw new/delete is unreliable: __strong on ObjC pointers
    // inside a C++ struct is not guaranteed to be honoured by the ARC optimizer when
    // the struct lifetime is managed outside ARC (raw new/delete through an opaque
    // pointer).  Bypass ARC entirely by storing the three ObjC objects as opaque
    // void* retained with CFBridgingRetain and released with CFRelease in ~Impl().
    void        *model      = nullptr;   // retains MLModel*
    void        *inputName  = nullptr;   // retains NSString*
    void        *outputName = nullptr;   // retains NSString*
    bool         isImage    = false;     // true → model takes a CVPixelBuffer input
    int          inputW     = 256;
    int          inputH     = 256;
    bool         loaded     = false;
    std::string  lastError;

    ~Impl() {
        if (model)      { CFRelease(model);      model      = nullptr; }
        if (inputName)  { CFRelease(inputName);  inputName  = nullptr; }
        if (outputName) { CFRelease(outputName); outputName = nullptr; }
    }
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
    m_impl->model = (void*)CFBridgingRetain(mdl);

    // Discover input/output names and the spatial size the model expects.
    MLModelDescription *desc = mdl.modelDescription;
    m_impl->inputName  = (void*)CFBridgingRetain(desc.inputDescriptionsByName.allKeys.firstObject);
    m_impl->outputName = (void*)CFBridgingRetain(desc.outputDescriptionsByName.allKeys.firstObject);

    MLFeatureDescription *inDesc = desc.inputDescriptionsByName[(__bridge NSString*)m_impl->inputName];
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
bool        CoreMLDepthEstimator::isLoaded()  const { return m_impl->loaded; }
std::string CoreMLDepthEstimator::lastError() const { return m_impl->lastError; }

// ─────────────────────────────────────────────────────────────────────────────
// estimate()
// ─────────────────────────────────────────────────────────────────────────────

cv::Mat CoreMLDepthEstimator::estimate(const cv::Mat &bgr)
{
    m_impl->lastError.clear();
    if (!m_impl->loaded || bgr.empty()) {
        m_impl->lastError = bgr.empty() ? "input frame empty" : "model not loaded";
        return {};
    }

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
        if (!pb) {
            m_impl->lastError = "CVPixelBufferCreate failed";
            return {};
        }

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
               initWithDictionary:@{(__bridge NSString*)m_impl->inputName: fv} error:&err];

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
        MLModelDescription *desc = ((__bridge MLModel*)m_impl->model).modelDescription;
        NSArray<NSNumber*> *modelShape =
            desc.inputDescriptionsByName[(__bridge NSString*)m_impl->inputName]
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
        if (err || !arr) {
            m_impl->lastError = err ? err.localizedDescription.UTF8String : "MLMultiArray alloc nil";
            return {};
        }

        inp = [[MLDictionaryFeatureProvider alloc]
               initWithDictionary:@{(__bridge NSString*)m_impl->inputName:
                   [MLFeatureValue featureValueWithMultiArray:arr]}
               error:&err];
    }

    if (err || !inp) {
        m_impl->lastError = std::string("inp: ") +
            (err ? err.localizedDescription.UTF8String : "nil provider");
        return {};
    }

    id<MLFeatureProvider> out =
        [(__bridge MLModel*)m_impl->model predictionFromFeatures:inp error:&err];
    if (err || !out) {
        m_impl->lastError = std::string("predict: ") +
            (err ? err.localizedDescription.UTF8String : "nil output");
        return {};
    }

    // Try stored output key first; fall back to any MultiArray in the output.
    NSString *outKey = (__bridge NSString*)m_impl->outputName;
    MLFeatureValue *outFV = [out featureValueForName:outKey];
    if (!outFV) {
        // Stored key not in prediction output — try every available feature name.
        for (NSString *k in out.featureNames) {
            MLFeatureValue *v = [out featureValueForName:k];
            if (v.multiArrayValue) { outFV = v; outKey = k; break; }
        }
    }
    if (!outFV) {
        // Build a list of available keys to show on screen.
        NSString *keys = [out.featureNames.allObjects componentsJoinedByString:@","];
        m_impl->lastError = std::string("no arr in: ") + keys.UTF8String;
        return {};
    }

    cv::Mat disp;

    if (outFV.type == MLFeatureTypeImage) {
        // Apple's DepthAnythingV2 outputs a single-channel CVPixelBuffer.
        CVPixelBufferRef outPB = outFV.imageBufferValue;
        if (!outPB) {
            m_impl->lastError = "image output: nil pixel buffer";
            return {};
        }
        CVPixelBufferLockBaseAddress(outPB, kCVPixelBufferLock_ReadOnly);
        const OSType fmt   = CVPixelBufferGetPixelFormatType(outPB);
        const int    outW  = (int)CVPixelBufferGetWidth(outPB);
        const int    outH  = (int)CVPixelBufferGetHeight(outPB);
        const size_t stride = CVPixelBufferGetBytesPerRow(outPB);
        const void  *base  = CVPixelBufferGetBaseAddress(outPB);

        disp = cv::Mat(outH, outW, CV_32F);
        if (fmt == kCVPixelFormatType_OneComponent32Float) {
            for (int y = 0; y < outH; ++y)
                std::memcpy(disp.ptr<float>(y),
                            static_cast<const uint8_t*>(base) + y * stride,
                            outW * sizeof(float));
        } else if (fmt == kCVPixelFormatType_OneComponent16Half) {
            for (int y = 0; y < outH; ++y) {
                const uint16_t *src = reinterpret_cast<const uint16_t*>(
                    static_cast<const uint8_t*>(base) + y * stride);
                float *dst = disp.ptr<float>(y);
                for (int x = 0; x < outW; ++x) {
                    uint32_t bits = ((uint32_t)(src[x] & 0x8000u) << 16)
                                  | ((uint32_t)((src[x] & 0x7C00u) + 0x1C000u) << 13)
                                  | ((uint32_t)(src[x] & 0x03FFu) << 13);
                    std::memcpy(&dst[x], &bits, sizeof(float));
                }
            }
        } else {
            CVPixelBufferUnlockBaseAddress(outPB, kCVPixelBufferLock_ReadOnly);
            m_impl->lastError = std::string("unsupported pb fmt: ") +
                std::to_string((unsigned)fmt);
            return {};
        }
        CVPixelBufferUnlockBaseAddress(outPB, kCVPixelBufferLock_ReadOnly);

    } else {
        MLMultiArray *outArr = outFV.multiArrayValue;
        if (!outArr) {
            m_impl->lastError = std::string("out type=") +
                std::to_string((int)outFV.type) + " key=" + outKey.UTF8String;
            return {};
        }
        NSUInteger n     = outArr.shape.count;
        const int  outH  = outArr.shape[n-2].intValue;
        const int  outW  = outArr.shape[n-1].intValue;
        disp = cv::Mat(outH, outW, CV_32F);
        if (outArr.dataType == MLMultiArrayDataTypeFloat16) {
            const uint16_t *src = static_cast<const uint16_t*>(outArr.dataPointer);
            float          *dst = disp.ptr<float>();
            const size_t    N   = (size_t)outH * outW;
            for (size_t i = 0; i < N; ++i) {
                uint32_t bits = ((uint32_t)(src[i] & 0x8000u) << 16)
                              | ((uint32_t)((src[i] & 0x7C00u) + 0x1C000u) << 13)
                              | ((uint32_t)(src[i] & 0x03FFu) << 13);
                std::memcpy(&dst[i], &bits, sizeof(float));
            }
        } else {
            std::memcpy(disp.data, outArr.dataPointer,
                        (size_t)outH * outW * sizeof(float));
        }
    }

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
