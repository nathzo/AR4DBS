#include "IOSOCREngine.h"

#import <Vision/Vision.h>
#import <CoreImage/CoreImage.h>
#include <opencv2/imgproc.hpp>

std::string IOSOCREngine::recognize(const cv::Mat &bgr)
{
    // ── Convert BGR cv::Mat → CGImage ────────────────────────────────────────
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    // Make the mat continuous so CGDataProvider can read it as a flat buffer.
    cv::Mat cont = rgb.isContinuous() ? rgb : rgb.clone();

    NSData *pixelData = [NSData dataWithBytes:cont.data
                                       length:cont.total() * cont.elemSize()];
    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    CGDataProviderRef dp = CGDataProviderCreateWithCFData((__bridge CFDataRef)pixelData);
    CGImageRef cg = CGImageCreate(
        cont.cols, cont.rows,
        8,                                      // bits per component
        24,                                     // bits per pixel (RGB)
        cont.step,                              // bytes per row
        cs,
        kCGBitmapByteOrderDefault | kCGImageAlphaNone,
        dp, nullptr, false, kCGRenderingIntentDefault);
    CGDataProviderRelease(dp);
    CGColorSpaceRelease(cs);

    // ── Run VNRecognizeTextRequest synchronously ──────────────────────────────
    __block std::string result;

    VNRecognizeTextRequest *req = [[VNRecognizeTextRequest alloc]
        initWithCompletionHandler:^(VNRequest *r, NSError *err) {
            if (err) return;
            for (VNRecognizedTextObservation *obs in r.results) {
                VNRecognizedText *top = [obs topCandidates:1].firstObject;
                if (top && top.string.length > 0) {
                    result += top.string.UTF8String;
                    result += '\n';
                }
            }
        }];

    // Accurate mode uses the Neural Engine; Fast mode is for real-time scanning.
    req.recognitionLevel = VNRequestTextRecognitionLevelAccurate;
    // French first (accented coords), English fallback.
    req.recognitionLanguages = @[@"fr-FR", @"en-US"];
    req.usesLanguageCorrection = YES;

    // performRequests:error: is synchronous — the completion handler fires
    // before this call returns.
    VNImageRequestHandler *handler =
        [[VNImageRequestHandler alloc] initWithCGImage:cg options:@{}];
    CGImageRelease(cg);

    NSError *err = nil;
    [handler performRequests:@[req] error:&err];

    return result;
}
