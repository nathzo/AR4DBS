cmake_minimum_required(VERSION 3.21)

# vcpkg is only used on Windows — skip it on macOS / iOS
if(WIN32 AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "C:/dev/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()

# iOS deployment target must be set before project() on Apple platforms
if(IOS)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "17.0" CACHE STRING "" FORCE)
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
)
# QtConcurrent is part of qtbase on desktop; on iOS the kit may omit it
if(NOT IOS)
    find_package(Qt6 REQUIRED COMPONENTS Concurrent)
endif()

# ─── OpenCV (ArUco / AprilTag detection) ─────────────────────────────────────
# On iOS and macOS we can use the pre-built opencv2.framework directly.
# Set OPENCV_FRAMEWORK to the directory that *contains* opencv2.framework.
# Leave it empty on desktop to fall back to a system install (e.g. Homebrew).
set(OPENCV_FRAMEWORK "" CACHE PATH "Directory containing opencv2.framework (iOS and macOS)")

if(OPENCV_FRAMEWORK)
    if(NOT EXISTS "${OPENCV_FRAMEWORK}/opencv2.framework")
        message(FATAL_ERROR "OPENCV_FRAMEWORK is set but opencv2.framework was not found in:\n"
            "  ${OPENCV_FRAMEWORK}\n"
            "Make sure the path points to the folder that *contains* opencv2.framework.")
    endif()
    # iOS frameworks have flat headers: Headers/core.hpp (not Headers/opencv2/core.hpp).
    # Use -F (framework search path) so <opencv2/foo.hpp> resolves via the framework mechanism.
    set(OpenCV_INCLUDE_DIRS "")
    set(OpenCV_LIBS          "-framework opencv2")
    message(STATUS "OpenCV framework: ${OPENCV_FRAMEWORK}/opencv2.framework")
elseif(IOS)
    message(FATAL_ERROR "OPENCV_FRAMEWORK must be set when building for iOS.\n"
        "Download opencv-4.x.x-ios-framework.zip from github.com/opencv/opencv/releases\n"
        "and set OPENCV_FRAMEWORK to the folder that contains opencv2.framework.")
else()
    find_package(OpenCV REQUIRED COMPONENTS core imgproc objdetect calib3d aruco videoio)
endif()

# Keep the old iOS-specific variable working as an alias
if(IOS AND NOT OPENCV_FRAMEWORK AND OPENCV_IOS_FRAMEWORK)
    set(OPENCV_FRAMEWORK "${OPENCV_IOS_FRAMEWORK}" CACHE PATH "" FORCE)
endif()

# ─── ViSP (edge-based model tracker) — optional for now ──────────────────────
find_package(VISP QUIET)
if(VISP_FOUND)
    message(STATUS "ViSP found — edge tracking enabled")
else()
    message(STATUS "ViSP not found — edge tracking disabled (AprilTag-only mode)")
endif()

# ─── ONNX Runtime (monocular depth inference) ────────────────────────────────
# iOS: depth is disabled — ONNX Runtime is only available via CocoaPods on iOS,
# Desktop: set ONNXRUNTIME_ROOT to a pre-built folder to enable depth.
# iOS:     set ONNXRUNTIME_ROOT to the extracted pod-archive-onnxruntime-mobile-c-x.x.x folder.
#          Expected layout (matches the CocoaPods pod archive):
#            <ONNXRUNTIME_ROOT>/Headers/                      ← C API headers
#            <ONNXRUNTIME_ROOT>/onnxruntime.xcframework/
#              ios-arm64/onnxruntime.framework/onnxruntime    ← arm64 device binary

set(ONNXRUNTIME_ROOT "" CACHE PATH "Root of a pre-built ONNX Runtime (include/ and lib/)")

if(ONNXRUNTIME_ROOT)
    if(IOS)
        # iOS CocoaPod layout: framework binary + top-level Headers/
        add_library(_onnxruntime_manual STATIC IMPORTED)
        set_target_properties(_onnxruntime_manual PROPERTIES
            IMPORTED_LOCATION
                "${ONNXRUNTIME_ROOT}/onnxruntime.xcframework/ios-arm64/onnxruntime.framework/onnxruntime")
        target_include_directories(_onnxruntime_manual INTERFACE "${ONNXRUNTIME_ROOT}/Headers")
    elseif(WIN32)
        add_library(_onnxruntime_manual SHARED IMPORTED)
        set_target_properties(_onnxruntime_manual PROPERTIES
            IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
            IMPORTED_IMPLIB   "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
        target_include_directories(_onnxruntime_manual INTERFACE "${ONNXRUNTIME_ROOT}/include")
    else()
        # macOS host build
        add_library(_onnxruntime_manual SHARED IMPORTED)
        set_target_properties(_onnxruntime_manual PROPERTIES
            IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.dylib")
    endif()
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

    # Optional OCR backend
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
        # Use Apple Vision framework (built into iOS 13+) instead of Tesseract
        find_library(FW_VISION Vision REQUIRED)
        add_compile_definitions(HAVE_IOS_OCR)
        list(APPEND SCANNER_SOURCES
            platform/ios/IOSOCREngine.h
            platform/ios/IOSOCREngine.mm
        )
        message(STATUS "OCR: using Apple Vision framework (HAVE_IOS_OCR)")
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
        platform/ios/ARKitSession.h
        platform/ios/ARKitSession.mm          # Objective-C++ — ARKit session + pose delivery
        platform/ios/CameraIntrinsics.h
        platform/ios/CameraIntrinsics.mm      # Objective-C++ — reads AVFoundation FOV (kept for reference)
        platform/ios/IOSCamera.h
        platform/ios/IOSCamera.cpp
        platform/ios/CoreMLDepthEstimator.h
        platform/ios/CoreMLDepthEstimator.mm  # Objective-C++ — Neural Engine monocular depth (no LiDAR)
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

# On iOS, Qt 6.4+ uses FFmpeg as the ONLY multimedia backend (AVFoundation is
# integrated inside it via VideoToolbox).  For a static iOS build we must:
#   1. Force-import the FFmpeg media plugin so the linker includes it.
#   2. Link Qt's bundled FFmpeg CMake target so sws_*/av_* symbols resolve.
if(IOS)
    # Force the static FFmpeg plugin to be linked (required for static Qt builds)
    qt_import_plugins(DBSAR
        INCLUDE Qt6::QFFmpegMediaPlugin
    )

    # Qt exposes its bundled FFmpeg as the CMake imported target Qt6::BundledFFmpeg.
    # Linking it directly is more reliable than find_library on the .a file.
    if(TARGET Qt6::BundledFFmpeg)
        target_link_libraries(DBSAR PRIVATE Qt6::BundledFFmpeg)
        message(STATUS "Qt bundled FFmpeg: Qt6::BundledFFmpeg target linked")
    else()
        # Qt 6.11.0 iOS ships FFmpeg as *dynamic* xcframeworks in <Qt>/lib/ffmpeg/.
        # Layout: lib<n>.xcframework/ios-arm64/lib<n>.framework/lib<n>  (Mach-O dylib)
        #
        # Three things must happen for the app to run on device:
        #   1. Link: -F <slice-dir> -framework lib<n>  (same pattern as OpenCV)
        #   2. Embed: copy lib<n>.framework into DBSAR.app/Frameworks/
        #   3. Rpath:  @executable_path/Frameworks so dyld finds them at runtime
        #
        # XCODE_EMBED_FRAMEWORKS triggers Xcode's "Embed Frameworks" build phase which
        # does step 2 and normally adds the rpath automatically. We also set
        # LD_RUNPATH_SEARCH_PATHS explicitly as a belt-and-suspenders measure.
        get_filename_component(_qt_lib_dir "${Qt6_DIR}/../.." ABSOLUTE)
        set(_ffmpeg_dir "${_qt_lib_dir}/ffmpeg")
        set(_ffmpeg_found 0)
        set(_ffmpeg_fws "")   # list of .framework dirs to embed

        foreach(_ff avcodec avformat avutil swresample swscale)
            set(_fw_slice "${_ffmpeg_dir}/lib${_ff}.xcframework/ios-arm64")
            set(_fw_dir   "${_fw_slice}/lib${_ff}.framework")
            if(EXISTS "${_fw_dir}/lib${_ff}")
                # Use the same two-step pattern that already works for OpenCV:
                #   target_link_options  → adds "-F/path/ios-arm64" as one token  ← proven
                #   target_link_libraries → adds "-framework lib<n>"               ← same as OpenCV_LIBS
                # find_library is NOT used because with the Xcode generator it adds
                # the path to FRAMEWORK_SEARCH_PATHS but that setting does NOT always
                # propagate to the linker's -F flags in practice.
                target_link_options(DBSAR PRIVATE "-F${_fw_slice}")
                target_link_libraries(DBSAR PRIVATE "-framework lib${_ff}")
                list(APPEND _ffmpeg_fws "${_fw_dir}")
                message(STATUS "Qt FFmpeg: ${_fw_dir}")
                math(EXPR _ffmpeg_found "${_ffmpeg_found}+1")
            else()
                message(WARNING "Qt FFmpeg: framework not found at ${_fw_dir}/lib${_ff}")
            endif()
        endforeach()

        if(_ffmpeg_found GREATER 0)
            # Embed every framework into DBSAR.app/Frameworks/ at build time.
            # Xcode's Embed Frameworks phase also adds @executable_path/Frameworks
            # to the binary's LC_RPATH automatically.
            set_target_properties(DBSAR PROPERTIES
                XCODE_EMBED_FRAMEWORKS                        "${_ffmpeg_fws}"
                XCODE_EMBED_FRAMEWORKS_CODE_SIGN_ON_COPY      YES
                XCODE_EMBED_FRAMEWORKS_REMOVE_HEADERS_ON_COPY YES)
            # Belt-and-suspenders: set the rpath explicitly so dyld can resolve
            # @rpath/lib<n>.framework/lib<n> to the embedded copy at runtime.
            set_target_properties(DBSAR PROPERTIES
                XCODE_ATTRIBUTE_LD_RUNPATH_SEARCH_PATHS
                    "@executable_path/Frameworks $(inherited)")
            message(STATUS "Qt FFmpeg: ${_ffmpeg_found}/5 frameworks will be embedded")
        else()
            message(WARNING
                "Qt6::BundledFFmpeg target not found and no FFmpeg frameworks found "
                "in ${_ffmpeg_dir} — multimedia will not work on device.")
        endif()
    endif()
endif()

# On iOS the ONNX model must be a real file in the bundle (not a Qt resource)
# because ONNX Runtime opens it with fopen().  On desktop it sits next to the exe.
if(IOS)
    set_source_files_properties(resources/model-small.onnx PROPERTIES
        MACOSX_PACKAGE_LOCATION Resources)
    target_sources(DBSAR PRIVATE resources/model-small.onnx)
endif()

# CoreML depth model for iOS monocular depth (CoreMLDepthEstimator).
# Set COREML_DEPTH_MODEL to the path of an .mlmodel or .mlpackage file.
# Xcode will compile it to .mlmodelc and place it in the app bundle Resources/.
# The path passed to AppController::init() at runtime should be:
#   [[NSBundle mainBundle] pathForResource:@"<model-name>" ofType:@"mlmodelc"]
# Leave empty to skip — depth occlusion will be disabled at runtime.
set(COREML_DEPTH_MODEL "" CACHE FILEPATH
    "Path to CoreML depth model (.mlmodel or .mlpackage) to bundle with the iOS app")
if(IOS AND COREML_DEPTH_MODEL)
    if(NOT EXISTS "${COREML_DEPTH_MODEL}")
        message(WARNING "COREML_DEPTH_MODEL is set but file not found: ${COREML_DEPTH_MODEL}")
    else()
        set_source_files_properties("${COREML_DEPTH_MODEL}" PROPERTIES
            MACOSX_PACKAGE_LOCATION Resources)
        target_sources(DBSAR PRIVATE "${COREML_DEPTH_MODEL}")
        message(STATUS "CoreML depth model: ${COREML_DEPTH_MODEL}")
    endif()
elseif(IOS)
    message(STATUS "CoreML depth model: not set — depth occlusion disabled (set COREML_DEPTH_MODEL to enable)")
endif()

# ─── Include paths ────────────────────────────────────────────────────────────
target_include_directories(DBSAR PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    $<$<BOOL:${VISP_FOUND}>:${VISP_INCLUDE_DIRS}>
)

# Tell both the compiler and linker where to find opencv2.framework
if(OPENCV_FRAMEWORK)
    target_compile_options(DBSAR PRIVATE "-F${OPENCV_FRAMEWORK}")
    target_link_options(DBSAR PRIVATE "-F${OPENCV_FRAMEWORK}")
endif()

# ─── Link libraries ───────────────────────────────────────────────────────────
target_link_libraries(DBSAR PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::Multimedia
    Qt6::MultimediaWidgets
    $<$<NOT:$<PLATFORM_ID:iOS>>:Qt6::Concurrent>
    ${OpenCV_LIBS}
    $<$<BOOL:${VISP_FOUND}>:${VISP_LIBRARIES}>
    $<$<BOOL:${onnxruntime_FOUND}>:${ONNXRUNTIME_TARGET}>
    $<$<BOOL:${Tesseract_FOUND}>:Tesseract::libtesseract>
)

# iOS system frameworks — use find_library so the linker gets a proper .framework path
if(IOS)
    find_library(FW_FOUNDATION   Foundation   REQUIRED)
    find_library(FW_AVFOUNDATION AVFoundation REQUIRED)
    find_library(FW_UIKIT        UIKit        REQUIRED)
    find_library(FW_ARKIT        ARKit        REQUIRED)
    find_library(FW_COREVIDEO    CoreVideo    REQUIRED)
    find_library(FW_COREML       CoreML       REQUIRED)  # always needed: CoreMLDepthEstimator
    target_link_libraries(DBSAR PRIVATE
        ${FW_FOUNDATION}
        ${FW_AVFOUNDATION}
        ${FW_UIKIT}
        ${FW_ARKIT}
        ${FW_COREVIDEO}
        ${FW_COREML}
        $<$<BOOL:${FW_VISION}>:${FW_VISION}>
    )
endif()

# ─── iOS bundle settings ──────────────────────────────────────────────────────
if(IOS)
    set_target_properties(DBSAR PROPERTIES
        MACOSX_BUNDLE                   TRUE
        MACOSX_BUNDLE_BUNDLE_NAME       "DBS AR"
        MACOSX_BUNDLE_BUNDLE_VERSION    "${PROJECT_VERSION}"
        MACOSX_BUNDLE_INFO_PLIST        "${CMAKE_SOURCE_DIR}/platform/ios/Info.plist.in"

        XCODE_ATTRIBUTE_TARGETED_DEVICE_FAMILY          "1"    # 1 = iPhone, 1,2 = iPhone+iPad
        XCODE_ATTRIBUTE_IPHONEOS_DEPLOYMENT_TARGET      "17.0"
        XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER       "ch.epfl.neurorestore.dbsar"
        XCODE_ATTRIBUTE_DEVELOPMENT_TEAM                "53M8KPV935"     # ← paste your 10-char Team ID here
        XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY              "iPhone Developer"
        XCODE_ATTRIBUTE_ENABLE_BITCODE                  "NO"   # ONNX Runtime has no bitcode

        # Tell Xcode where to find opencv2.framework
        XCODE_ATTRIBUTE_FRAMEWORK_SEARCH_PATHS
            "${OPENCV_FRAMEWORK} $(inherited)"
    )
endif()
