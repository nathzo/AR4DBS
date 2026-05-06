#include "IOSOCREngine.h"

#import <Vision/Vision.h>
#import <CoreImage/CoreImage.h>
#include <opencv2/imgproc.hpp>

std::vector<OcrLine> IOSOCREngine::recognize(const cv::Mat &bgr)
{
    // ── Convert BGR cv::Mat → CGImage ────────────────────────────────────────
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    cv::Mat cont = rgb.isContinuous() ? rgb : rgb.clone();

    NSData *pixelData = [NSData dataWithBytes:cont.data
                                       length:cont.total() * cont.elemSize()];
    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    CGDataProviderRef dp = CGDataProviderCreateWithCFData((__bridge CFDataRef)pixelData);
    CGImageRef cg = CGImageCreate(
        cont.cols, cont.rows,
        8, 24, cont.step,
        cs,
        kCGBitmapByteOrderDefault | kCGImageAlphaNone,
        dp, nullptr, false, kCGRenderingIntentDefault);
    CGDataProviderRelease(dp);
    CGColorSpaceRelease(cs);

    // ── Run VNRecognizeTextRequest synchronously ──────────────────────────────
    __block std::vector<OcrLine> result;

    VNRecognizeTextRequest *req = [[VNRecognizeTextRequest alloc]
        initWithCompletionHandler:^(VNRequest *r, NSError *err) {
            if (err) return;
            for (VNRecognizedTextObservation *obs in r.results) {
                // topCandidates:1 gives the best string + its recognition confidence.
                VNRecognizedText *top = [obs topCandidates:1].firstObject;
                if (top && top.string.length > 0) {
                    OcrLine line;
                    line.text       = top.string.UTF8String;
                    line.confidence = top.confidence; // 0.0–1.0
                    result.push_back(line);
                }
            }
        }];

    req.recognitionLevel    = VNRequestTextRecognitionLevelAccurate;
    req.recognitionLanguages = @[@"fr-FR", @"en-US"];
    req.usesLanguageCorrection = YES;

    VNImageRequestHandler *handler =
        [[VNImageRequestHandler alloc] initWithCGImage:cg options:@{}];
    CGImageRelease(cg);

    NSError *err = nil;
    [handler performRequests:@[req] error:&err];

    return result;
}
