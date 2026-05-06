#include <QApplication>
#include <QMessageLogContext>
#include <QtGlobal>
#include <QMetaType>
#include <QFont>
#include "MainWindow.h"
#include "AppController.h" // pulls in Q_DECLARE_METATYPE(cv::Mat)

#include <opencv2/core/utils/logger.hpp>

// Suppress benign teardown noise (OpenGL context cleanup, timer stop warnings)
static void appMessageHandler(QtMsgType type,
                              const QMessageLogContext &,
                              const QString &msg)
{
    // Suppress all debug messages — they are development-only noise.
    if (type == QtDebugMsg) return;

    // Suppress known benign Qt/ONNX teardown warnings.
    static const QStringList noise = {
        "QOpenGLWidget", "QBasicTimer::stop",
        "Cannot make QOpenGLContext", "still alive", "orphan",
        "Schema error", "already registered",
        "QThreadStorage: entry",          // thread-local storage teardown order
    };
    for (const auto &p : noise)
        if (msg.contains(p, Qt::CaseInsensitive)) return;

    if (type == QtWarningMsg)  fprintf(stderr, "[W] %s\n", msg.toLocal8Bit().constData());
    else if (type == QtCriticalMsg) fprintf(stderr, "[E] %s\n", msg.toLocal8Bit().constData());
    else if (type == QtFatalMsg)  { fprintf(stderr, "[FATAL] %s\n", msg.toLocal8Bit().constData()); abort(); }
}

int main(int argc, char *argv[])
{
#ifndef Q_OS_IOS
    // Force the software/raster paint engine for all QWidget windows on desktop.
    // Must be set BEFORE QApplication is constructed.
    // Prevents Qt 6 routing QPainter through the OpenGL texture cache
    // (qopengltexturecache) which asserts when the GL context is not ready.
    QApplication::setAttribute(Qt::AA_UseSoftwareOpenGL);
#endif

    qInstallMessageHandler(appMessageHandler);

    // Register cv::Mat so it can be passed through queued (cross-thread) connections.
    qRegisterMetaType<cv::Mat>("cv::Mat");

    // Suppress OpenCV INFO spam (plugin load attempts, backend enumeration).
    // These go straight to stderr and bypass Qt's message handler.
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);

    QApplication app(argc, argv);
    app.setApplicationName("AR4DBS");
    app.setOrganizationName("NeuroRestore");
    app.setFont(QFont("Arial", 12));

    MainWindow window;
    window.setWindowTitle("AR4DBS");
    window.resize(1280, 720);
    window.show();

    return app.exec();
}
