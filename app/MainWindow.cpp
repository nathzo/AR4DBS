#include "MainWindow.h"
#include "AppController.h"
#include "core/rendering/GLWidget.h"

#include <QFile>
#include <QMessageBox>
#include <QCoreApplication>
#include <QPushButton>
#include <QVBoxLayout>
#include <QWidget>
#include <QCloseEvent>
#include <QAtomicInt>
#include <cmath>

// Default DBS targets in Leksell frame coordinates (mm / degrees).
static SurgicalPlan defaultTestPlan()
{
    SurgicalPlan p;

    p.left.x_mm     = 140.4;
    p.left.y_mm     = 114.6;
    p.left.z_mm     =  80.0;
    p.left.arc_deg  =  71.0;
    p.left.ring_deg =  74.2;
    p.left.valid    = true;

    p.right.x_mm     =  66.2;
    p.right.y_mm     = 118.2;
    p.right.z_mm     =  77.4;
    p.right.arc_deg  = 111.1;
    p.right.ring_deg =  66.8;
    p.right.valid    = true;

    return p;
}

#ifdef Q_OS_IOS
#  include "platform/ios/ARKitSession.h"
#else
#  include "platform/desktop/DesktopCamera.h"
#endif

#ifdef FEATURE_PLAN_SCANNER
#  include <QStackedWidget>
#  include "app/StartScreen.h"
#  include "app/ScanScreen.h"
#  include "app/ConfirmPlanDialog.h"
#endif

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    // Force dark background on the root window so nothing system-coloured shows
    // through during transitions or around safe-area insets on iOS.
    setStyleSheet("QMainWindow { background-color: #1a1b1d; }");

    // ── Controller — runs on a dedicated thread so detect+blend never block UI ─
    m_controller       = new AppController;   // no parent — will be moved to thread
    m_controllerThread = new QThread(this);
    m_controller->moveToThread(m_controllerThread);
    // Destroy the controller when the thread finishes, and quit the thread when
    // the window closes (see closeEvent).
    connect(m_controllerThread, &QThread::finished,
            m_controller,       &QObject::deleteLater);
    m_controllerThread->start();

#ifdef Q_OS_IOS
    // Compiled CoreML model: Xcode places it in the bundle root or Resources/ depending on build settings.
    QString depthModel = QCoreApplication::applicationDirPath() + "/model-small.mlmodelc";
    if (!QFile::exists(depthModel))
        depthModel = QCoreApplication::applicationDirPath() + "/Resources/model-small.mlmodelc";
#else
    const QString depthModel = QCoreApplication::applicationDirPath() + "/model-small.onnx";
#endif

#ifdef FEATURE_PLAN_SCANNER
    // With the scanner: init without a plan path (plan comes from the wizard)
    if (!m_controller->init(":/resources/calibration.json",
                            ":/resources/tag_config.json",
                            /*planPath=*/QString(),
                            depthModel)) {
        QMessageBox::critical(this, "Init failed",
            "Could not initialise. Check resources/ folder.");
        return;
    }
#else
    // Without the scanner: load the fallback JSON plan directly
    if (!m_controller->init(":/resources/calibration.json",
                            ":/resources/tag_config.json",
                            ":/resources/surgical_plan.json",
                            depthModel)) {
        QMessageBox::critical(this, "Init failed",
            "Could not initialise. Check resources/ folder.");
        return;
    }
#endif

    // ── AR camera (used in the AR phase) ─────────────────────────────────────
    //
    // Frame-drop guard: AppController lives on a worker thread, so the
    // camera→controller connection is QueuedConnection.  If processing takes
    // longer than the camera's emission interval, frames pile up in the worker's
    // event queue without bound — causing memory growth, 100 % CPU, and thermal
    // throttling within seconds.
    //
    // Fix: allow at most ONE frame pending in the queue at any time.  If the
    // worker is still processing, the new frame is silently dropped.
    // `busy` is shared between the lambda and the queued functor via a
    // ref-counted QAtomicInt.
    auto busy = std::make_shared<QAtomicInt>(0);

#ifdef Q_OS_IOS
    // ARKit path: frameReady carries both the BGR image and the ARKit
    // world_T_camera pose matrix, so no per-frame AprilTag detection is needed
    // once the surgical frame has been registered.
    m_arCamera = new ARKitSession(this);
    connect(m_arCamera, &ARKitSession::calibrationReady,
            m_controller, &AppController::setCalibration);
    connect(m_arCamera, &ARKitSession::lidarAvailable,
            m_controller, &AppController::onLidarAvailable);
    connect(m_arCamera, &ARKitSession::lidarDepthReady,
            m_controller, &AppController::onLidarDepth);
    connect(m_arCamera, &ARKitSession::frameReady, this,
            [this, busy](const cv::Mat &frame, const cv::Mat &world_T_camera) {
        if (!busy->testAndSetAcquire(0, 1)) return;
        QMetaObject::invokeMethod(m_controller,
            [this, frame, world_T_camera, busy]() {
                m_controller->onARFrame(frame, world_T_camera);
                busy->storeRelease(0);
            });
    });
#else
    m_arCamera = new DesktopCamera(0, this);
    connect(m_arCamera, &DesktopCamera::frameReady, this,
            [this, busy](const cv::Mat &frame) {
        if (!busy->testAndSetAcquire(0, 1)) return;
        QMetaObject::invokeMethod(m_controller,
            [this, frame, busy]() {
                m_controller->onNewFrame(frame);
                busy->storeRelease(0);
            });
    });
#endif

    // ── AR display widget + button row ───────────────────────────────────────
    auto *arContainer = new QWidget(this);
    auto *arLayout    = new QVBoxLayout(arContainer);
    arLayout->setContentsMargins(0, 0, 0, 0);
    arLayout->setSpacing(0);

    m_glWidget = new GLWidget(arContainer);
    arLayout->addWidget(m_glWidget, 1);

    // Button row at the bottom of the AR view — same shape/layout as ScanScreen
    auto *arBtnRow = new QWidget(arContainer);
    arBtnRow->setStyleSheet("background: #1a1b1d;");
    auto *arBtnLayout = new QHBoxLayout(arBtnRow);
    arBtnLayout->setContentsMargins(16, 8, 16, 16);
    arBtnLayout->setSpacing(0);

    m_btnBackToMenu = new QPushButton("← Menu", arBtnRow);
    m_btnBackToMenu->setStyleSheet(
        "QPushButton { background:#8A8C8F; color:#1a1b1d; border-radius:8px;"
        "              padding:12px 24px; font-size:13pt; font-weight:bold; }"
        "QPushButton:pressed { background:#6d6f72; }");

    m_btnEditPlan = new QPushButton("Modifier le plan", arBtnRow);
    m_btnEditPlan->setStyleSheet(
        "QPushButton { background:#75D0C5; color:#1a1b1d; border-radius:8px;"
        "              padding:12px 32px; font-size:14pt; font-weight:bold; }"
        "QPushButton:pressed { background:#5ab8ae; }");
    m_btnEditPlan->setVisible(false);

    arBtnLayout->addWidget(m_btnBackToMenu);
    arBtnLayout->addStretch(1);
    arBtnLayout->addWidget(m_btnEditPlan);
    arLayout->addWidget(arBtnRow);

    connect(m_controller, &AppController::frameReady,
            m_glWidget,   &GLWidget::setFrame);
    connect(m_btnEditPlan, &QPushButton::clicked,
            this, [this]() {
#ifdef FEATURE_PLAN_SCANNER
        editPlan();
#endif
    });

#ifdef FEATURE_PLAN_SCANNER
    // ── Back-to-menu button (AR → start screen) ───────────────────────────────
    connect(m_btnBackToMenu, &QPushButton::clicked, this, [this]() {
        m_arCamera->stop();
        m_btnEditPlan->setVisible(false);
#ifdef Q_OS_IOS
        m_controller->setShowDepthOverlay(false);
#endif
        m_stack->setCurrentIndex(0);
    });

    // ── Wizard flow ───────────────────────────────────────────────────────────
    m_stack       = new QStackedWidget(this);
    m_startScreen = new StartScreen(this);
    m_scanScreen  = new ScanScreen(this);

    m_stack->addWidget(m_startScreen);  // index 0
    m_stack->addWidget(m_scanScreen);   // index 1
    m_stack->addWidget(arContainer);    // index 2
    m_stack->setCurrentIndex(0);
    setCentralWidget(m_stack);

    // Start screen → scan
    connect(m_startScreen, &StartScreen::newSurgeryRequested, this, [this]() {
        m_stack->setCurrentIndex(1);
        m_scanScreen->startCamera();
    });

    // Start screen → direct AR (testing shortcut — loads default test targets)
    connect(m_startScreen, &StartScreen::directARRequested, this, [this]() {
        m_currentPlan = defaultTestPlan();
        m_controller->setSurgicalPlan(m_currentPlan);
        m_btnEditPlan->setVisible(true);
#ifdef Q_OS_IOS
        m_controller->setShowDepthOverlay(true);
#endif
        startAR();
    });

    // Scan screen → confirm dialog
    connect(m_scanScreen, &ScanScreen::planDetected,
            this, &MainWindow::onPlanDetected);

    // Scan screen → back to start
    connect(m_scanScreen, &ScanScreen::cancelled, this, [this]() {
        m_scanScreen->stopCamera();
        m_stack->setCurrentIndex(0);
    });

#else
    // No wizard: go straight to AR with default test targets
    m_controller->setSurgicalPlan(defaultTestPlan());
    setCentralWidget(arContainer);
    m_arCamera->start();
#endif
}

MainWindow::~MainWindow() {}

void MainWindow::closeEvent(QCloseEvent *event)
{
    // Stop cameras first so no more frames are queued at the controller.
    if (m_arCamera) {
        m_arCamera->disconnect();
        m_arCamera->stop();
    }
#ifdef FEATURE_PLAN_SCANNER
    if (m_scanScreen) m_scanScreen->stopCamera();
#endif

    // Disconnect controller signals before asking the thread to quit so no
    // cross-thread calls land on a half-destroyed controller.
    if (m_controller) m_controller->disconnect();

    // Shut down the worker thread; deleteLater on m_controller fires when finished.
    if (m_controllerThread) {
        m_controllerThread->quit();
        m_controllerThread->wait();
    }

    event->accept();
}

#ifdef FEATURE_PLAN_SCANNER
void MainWindow::startAR()
{
    m_stack->setCurrentIndex(2);
#ifdef Q_OS_IOS
    // Re-register the surgical frame on each new AR session start so that
    // returning from the menu and re-entering AR picks up a fresh anchor.
    m_controller->resetARRegistration();
#endif
    m_arCamera->start();
}

void MainWindow::onPlanDetected(const SurgicalPlan &detected)
{
    m_scanScreen->stopCamera();

    ConfirmPlanDialog dlg(detected, this);
    if (dlg.exec() != QDialog::Accepted) {
        // User cancelled — go back to scan screen
        m_stack->setCurrentIndex(1);
        m_scanScreen->startCamera();
        return;
    }

    m_currentPlan = dlg.plan();
    m_controller->setSurgicalPlan(m_currentPlan);
    m_btnEditPlan->setVisible(true);
    startAR();
}

void MainWindow::editPlan()
{
    m_arCamera->stop();

    ConfirmPlanDialog dlg(m_currentPlan, this);
    if (dlg.exec() == QDialog::Accepted) {
        m_currentPlan = dlg.plan();
        m_controller->setSurgicalPlan(m_currentPlan);
    }

#ifdef Q_OS_IOS
    m_controller->resetARRegistration();
#endif
    m_arCamera->start();
}
#endif
