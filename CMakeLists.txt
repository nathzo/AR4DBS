cmake_minimum_required(VERSION 3.21)

# vcpkg is only used on Windows — skip it on macOS / iOS
if(WIN32 AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "C:/dev/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()

# iOS deployment target must be set before project() on Apple platforms
if(IOS)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "16.0" CACHE STRING "" FORCE)
endif()

project(DBSAR VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MSVC does not define M_PI unless this is set
if(MSVC)
    add_compile_definitions(_USE_MATH_DEFINES)
endif()

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# ─── Qt ───────────────────────────────────────────────────────────────────────
find_package(Qt6 REQUIRED COMPONENTS
    Core
    Gui
    Widgets
    Multimedia
    MultimediaWidgets
    Concurrent
)

# ─── OpenCV (ArUco / AprilTag detection) ─────────────────────────────────────
# On iOS we use the official pre-built opencv2.framework; on desktop find_package works.
if(IOS)
    # Set OPENCV_IOS_FRAMEWORK to the *directory that contains* opencv2.framework, e.g.:
    #   ~/dev/opencv-ios     (so the full path is ~/dev/opencv-ios/opencv2.framework)
    set(OPENCV_IOS_FRAMEWORK "" CACHE PATH "Directory containing opencv2.framework for iOS")
    if(NOT OPENCV_IOS_FRAMEWORK)
        message(FATAL_ERROR "OPENCV_IOS_FRAMEWORK must be set when building for iOS.\n"
            "Download opencv-4.x.x-ios-framework.zip from github.com/opencv/opencv/releases\n"
            "and set OPENCV_IOS_FRAMEWORK to the folder that contains opencv2.framework.")
    endif()
    # Expose headers and remember search path for Xcode
    set(OpenCV_INCLUDE_DIRS "${OPENCV_IOS_FRAMEWORK}/opencv2.framework/Headers")
    set(OpenCV_LIBS          "-framework opencv2")
    message(STATUS "OpenCV iOS framework: ${OPENCV_IOS_FRAMEWORK}/opencv2.framework")
else()
    find_package(OpenCV REQUIRED COMPONENTS core imgproc objdetect calib3d aruco videoio)
endif()

# ─── ViSP (edge-based model tracker) — optional for now ──────────────────────
find_package(VISP QUIET)
if(VISP_FOUND)
    message(STATUS "ViSP found — edge tracking enabled")
else()
    message(STATUS "ViSP not found — edge tracking disabled (AprilTag-only mode)")
endif()

# ─── ONNX Runtime (monocular depth inference) ────────────────────────────────
# ONNXRUNTIME_ROOT takes priority over vcpkg — use it to pin a specific
# pre-built release (recommended: v1.18.1 from github.com/microsoft/onnxruntime/releases)
set(ONNXRUNTIME_ROOT "" CACHE PATH "Root of a pre-built ONNX Runtime (include/ and lib/)")

if(ONNXRUNTIME_ROOT)
    if(IOS)
        # iOS: static library extracted from the XCFramework arm64-iphoneos slice
        add_library(_onnxruntime_manual STATIC IMPORTED)
        set_target_properties(_onnxruntime_manual PROPERTIES
            IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.a")
    elseif(WIN32)
        add_library(_onnxruntime_manual SHARED IMPORTED)
        set_target_properties(_onnxruntime_manual PROPERTIES
            IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
            IMPORTED_IMPLIB   "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
    else()
        # macOS host build
        add_library(_onnxruntime_manual SHARED IMPORTED)
        set_target_properties(_onnxruntime_manual PROPERTIES
            IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.dylib")
    endif()
    target_include_directories(_onnxruntime_manual INTERFACE "${ONNXRUNTIME_ROOT}/include")
    set(ONNXRUNTIME_TARGET _onnxruntime_manual)
    set(onnxruntime_FOUND TRUE)
    message(STATUS "ONNX Runtime: using pre-built from ${ONNXRUNTIME_ROOT}")
    add_compile_definitions(HAVE_ONNXRUNTIME)
else()
    # Fall back to vcpkg only on Windows
    if(WIN32)
        find_package(onnxruntime CONFIG QUIET)
    endif()
    if(onnxruntime_FOUND)
        set(ONNXRUNTIME_TARGET $<IF:$<TARGET_EXISTS:onnxruntime::onnxruntime>,onnxruntime::onnxruntime,onnxruntime>)
        message(STATUS "ONNX Runtime: using vcpkg build (may have schema issues on v1.19)")
        add_compile_definitions(HAVE_ONNXRUNTIME)
    else()
        set(ONNXRUNTIME_TARGET "")
        message(STATUS "ONNX Runtime not found — depth estimation disabled")
    endif()
endif()

# ─── Plan scanner (OCR wizard) ───────────────────────────────────────────────
# Set to OFF to skip the scan/confirm wizard and go straight to AR on launch.
# Useful for rapid iteration during desktop testing.
option(FEATURE_PLAN_SCANNER "Enable surgical plan scanning and confirmation wizard" ON)

set(SCANNER_SOURCES "")
if(FEATURE_PLAN_SCANNER)
    message(STATUS "Plan scanner: enabled")
    add_compile_definitions(FEATURE_PLAN_SCANNER)

    list(APPEND SCANNER_SOURCES
        app/StartScreen.h       app/StartScreen.cpp
        app/ScanScreen.h        app/ScanScreen.cpp
        app/ConfirmPlanDialog.h app/ConfirmPlanDialog.cpp
        core/ocr/PlanScanner.h  core/ocr/PlanScanner.cpp
        core/math/SurgicalPlan.h
    )

    # Optional OCR backend — desktop only (no Tesseract for iOS)
    if(NOT IOS)
        find_package(Tesseract QUIET)
        if(Tesseract_FOUND)
            message(STATUS "Tesseract found — OCR enabled")
            add_compile_definitions(HAVE_TESSERACT)
        else()
            message(STATUS "Tesseract not found — manual coordinate entry only")
            message(STATUS "  To enable OCR: vcpkg install tesseract:x64-windows")
        endif()
    else()
        message(STATUS "Tesseract: skipped on iOS — manual coordinate entry only")
    endif()
else()
    message(STATUS "Plan scanner: disabled (FEATURE_PLAN_SCANNER=OFF)")
    # SurgicalPlan.h is still needed by AppController even without the wizard
    list(APPEND SCANNER_SOURCES core/math/SurgicalPlan.h)
endif()

# ─── Sources ──────────────────────────────────────────────────────────────────
set(APP_SOURCES
    app/main.cpp
    app/AppController.h
    app/AppController.cpp
    app/MainWindow.h
    app/MainWindow.cpp
)

set(CORE_SOURCES
    # tracking — AprilTag only for now; ViSP/Hybrid added when ViSP is installed
    core/tracking/AprilTagTracker.h
    core/tracking/AprilTagTracker.cpp

    core/depth/DepthEstimator.h
    core/depth/DepthEstimator.cpp

    core/math/IncisionLine.h
    core/math/IncisionLine.cpp
    core/math/RayCaster.h
    core/math/RayCaster.cpp
    core/math/PoseUtils.h
    core/math/PoseUtils.cpp
    core/rendering/OverlayRenderer.h
    core/rendering/OverlayRenderer.cpp
    core/rendering/GLWidget.h
    core/rendering/GLWidget.cpp
)

# Platform layer: only the camera differs between desktop and iPhone
if(IOS)
    set(PLATFORM_SOURCES
        platform/ios/IOSCamera.h
        platform/ios/IOSCamera.cpp
        platform/ios/CameraIntrinsics.h
        platform/ios/CameraIntrinsics.mm   # Objective-C++ — reads AVFoundation FOV
    )
else()
    set(PLATFORM_SOURCES
        platform/desktop/DesktopCamera.h
        platform/desktop/DesktopCamera.cpp
    )
endif()

# ─── Executable ───────────────────────────────────────────────────────────────
qt_add_executable(DBSAR
    ${APP_SOURCES}
    ${CORE_SOURCES}
    ${PLATFORM_SOURCES}
    ${SCANNER_SOURCES}
)

# ─── Resources ────────────────────────────────────────────────────────────────
qt_add_resources(DBSAR "assets"
    PREFIX "/resources"
    BASE "resources"
    FILES
        resources/tag_config.json
        resources/calibration.json
        resources/surgical_plan.json
)

# ─── Include paths ────────────────────────────────────────────────────────────
target_include_directories(DBSAR PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    $<$<BOOL:${VISP_FOUND}>:${VISP_INCLUDE_DIRS}>
)

# ─── Link libraries ───────────────────────────────────────────────────────────
target_link_libraries(DBSAR PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::Multimedia
    Qt6::MultimediaWidgets
    Qt6::Concurrent
    ${OpenCV_LIBS}
    $<$<BOOL:${VISP_FOUND}>:${VISP_LIBRARIES}>
    $<$<BOOL:${onnxruntime_FOUND}>:${ONNXRUNTIME_TARGET}>
    $<$<BOOL:${Tesseract_FOUND}>:Tesseract::libtesseract>
)

# ─── iOS bundle settings ──────────────────────────────────────────────────────
if(IOS)
    set_target_properties(DBSAR PROPERTIES
        MACOSX_BUNDLE                   TRUE
        MACOSX_BUNDLE_BUNDLE_NAME       "DBS AR"
        MACOSX_BUNDLE_BUNDLE_VERSION    "${PROJECT_VERSION}"
        MACOSX_BUNDLE_INFO_PLIST        "${CMAKE_SOURCE_DIR}/platform/ios/Info.plist.in"

        XCODE_ATTRIBUTE_TARGETED_DEVICE_FAMILY          "1"    # 1 = iPhone, 1,2 = iPhone+iPad
        XCODE_ATTRIBUTE_IPHONEOS_DEPLOYMENT_TARGET      "16.0"
        XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER       "ch.epfl.neurorestore.dbsar"
        XCODE_ATTRIBUTE_DEVELOPMENT_TEAM                ""     # ← paste your 10-char Team ID here
        XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY              "iPhone Developer"
        XCODE_ATTRIBUTE_ENABLE_BITCODE                  "NO"   # ONNX Runtime has no bitcode

        # Camera permission — required for App Store submission
        XCODE_ATTRIBUTE_INFOPLIST_KEY_NSCameraUsageDescription
            "Camera access is required to overlay the surgical plan on the patient."

        # Tell Xcode where to find opencv2.framework
        XCODE_ATTRIBUTE_FRAMEWORK_SEARCH_PATHS
            "${OPENCV_IOS_FRAMEWORK} $(inherited)"
    )
endif()
