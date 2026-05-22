#include "MainWindow.h"
#include "AppController.h"
#include "core/rendering/GLWidget.h"

#include <QFile>
#include <QLabel>
#include <QMessageBox>
#include <QCoreApplication>
#include <QGuiApplication>
#include <QScreen>
#include <QPushButton>
#include <QVBoxLayout>
#include <QWidget>
#include <QCloseEvent>
#include <QAtomicInt>
#include <QPainter>
#include <QMouseEvent>
#include <QResizeEvent>
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
    for (float &c : p.left.confidence)  c = 1.f;

    p.right.x_mm     =  66.2;
    p.right.y_mm     = 118.2;
    p.right.z_mm     =  77.4;
    p.right.arc_deg  = 111.1;
    p.right.ring_deg =  66.8;
    p.right.valid    = true;
    for (float &c : p.right.confidence) c = 1.f;

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

// ── ARRotatedStrip ────────────────────────────────────────────────────────────
// Same rotation mechanics as RotatedStrip in ScanScreen.cpp.
// Inner widget is rendered rotated 90° CW so controls read correctly when the
// phone is held in landscape.
class ARRotatedStrip : public QWidget
{
public:
    QLabel      *statusLabel = nullptr;
    QPushButton *btnMenu     = nullptr;
    QPushButton *btnEdit     = nullptr;

    explicit ARRotatedStrip(int stripH, QWidget *parent = nullptr)
        : QWidget(parent), m_stripH(stripH)
    {
        setFixedHeight(stripH);

        m_inner = new QWidget;
        m_inner->setStyleSheet("background: black;");

        auto *lay = new QVBoxLayout(m_inner);
        lay->setContentsMargins(8, 8, 8, 8);
        lay->setSpacing(8);

        // Status label — holds calibration state on iOS; transparent on desktop.
        statusLabel = new QLabel(m_inner);
        statusLabel->setAlignment(Qt::AlignCenter);
        statusLabel->setWordWrap(true);
        statusLabel->setFixedHeight(60);
        statusLabel->setStyleSheet(
            "color: transparent; background: transparent;"
            "padding: 6px; font-size: 12pt;");
        lay->addStretch(1);
        lay->addWidget(statusLabel);
        lay->addStretch(3);

        btnMenu = new QPushButton("← Menu", m_inner);
        btnMenu->setStyleSheet(
            "QPushButton { background:#8A8C8F; color:black; border-radius:8px;"
            "              padding:8px 20px; font-family:'Arial';"
            "              font-size:13pt; font-weight:bold; }"
            "QPushButton:pressed { background:#6d6f72; }");
        lay->addWidget(btnMenu, 0, Qt::AlignHCenter);
        lay->addStretch(3);

        btnEdit = new QPushButton(m_inner);
        btnEdit->setText("Modifier\nle plan");
        btnEdit->setStyleSheet(
            "QPushButton { background:#75D0C5; color:black; border-radius:8px;"
            "              padding:6px 16px; font-family:'Arial';"
            "              font-size:13pt; font-weight:bold; }"
            "QPushButton:pressed { background:#5ab8ae; }");
        btnEdit->setVisible(false);
        lay->addWidget(btnEdit, 0, Qt::AlignHCenter);
        lay->addStretch(1);
    }

    ~ARRotatedStrip() override { delete m_inner; }

protected:
    void resizeEvent(QResizeEvent *) override { syncInner(); }

    void paintEvent(QPaintEvent *) override
    {
        syncInner();
        QPainter p(this);
        p.translate(width(), 0);
        p.rotate(90);
        m_inner->render(&p);
    }

    void mousePressEvent(QMouseEvent *e) override
    {
        m_pressedBtn = findBtn(toInner(e->pos()));
    }

    void mouseReleaseEvent(QMouseEvent *e) override
    {
        QPushButton *released = findBtn(toInner(e->pos()));
        if (m_pressedBtn && m_pressedBtn == released)
            m_pressedBtn->click();
        m_pressedBtn = nullptr;
    }

private:
    void syncInner()
    {
        m_inner->resize(m_stripH, width());
        if (auto *l = m_inner->layout()) l->activate();
        m_inner->ensurePolished();
    }

    QPoint toInner(QPoint s) const { return { s.y(), width() - s.x() }; }

    QPushButton *findBtn(QPoint innerPt) const
    {
        QWidget *w = m_inner->childAt(innerPt);
        while (w && w != m_inner) {
            if (auto *b = qobject_cast<QPushButton *>(w)) return b;
            w = w->parentWidget();
        }
        return nullptr;
    }

    QWidget     *m_inner      = nullptr;
    QPushButton *m_pressedBtn = nullptr;
    int          m_stripH     = 0;
};

// ─────────────────────────────────────────────────────────────────────────────

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    // Force dark background on the root window so nothing system-coloured shows
    // through during transitions or around safe-area insets on iOS.
    setStyleSheet("QMainWindow { background-color: #000000; }");

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
    connect(m_controller, &AppController::lockStateChanged,
            this, [this](bool locked) { setArLocked(locked); },
            Qt::QueuedConnection);
    connect(m_controller, &AppController::calibrationProgressChanged,
            this, [this](bool inProgress) { setArCalibrating(inProgress); },
            Qt::QueuedConnection);
    // ARTrackingState: 0 = not available, 1 = limited, 2 = normal.
    // AppController::onTrackingQualityChanged handles both the debug overlay update
    // and the re-calibration trigger (quality drop below anchor-time level).
    connect(m_arCamera, &ARKitSession::trackingQualityChanged,
            m_controller, &AppController::onTrackingQualityChanged);
    connect(m_arCamera, &ARKitSession::frameReady, this,
            [this, busy](const cv::Mat &frame, const cv::Mat &world_T_camera, int featurePoints) {
        if (!busy->testAndSetAcquire(0, 1)) return;
        QMetaObject::invokeMethod(m_controller,
            [this, frame, world_T_camera, featurePoints, busy]() {
                m_controller->onARFrame(frame, world_T_camera, featurePoints);
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

    const int arW = QGuiApplication::primaryScreen()->availableGeometry().width();
    m_glWidget = new GLWidget(arContainer);
    m_glWidget->setFixedHeight(arW * 640 / 480);
    arLayout->addStretch(1);
    arLayout->addWidget(m_glWidget);
    arLayout->addSpacing(8);

    // ── Rotated control strip ─────────────────────────────────────────────────
    m_arStrip = new ARRotatedStrip(150, arContainer);
    m_btnBackToMenu = m_arStrip->btnMenu;
    m_btnEditPlan   = m_arStrip->btnEdit;
#ifdef Q_OS_IOS
    m_calibLabel    = m_arStrip->statusLabel;
#endif
    arLayout->addWidget(m_arStrip);
    arLayout->addSpacing(8);

#ifdef Q_OS_IOS
    setArLocked(false); // set initial label text and button text
#endif

    connect(m_controller, &AppController::frameReady,
            m_glWidget,   &GLWidget::setFrame);
    connect(m_btnEditPlan, &QPushButton::clicked,
            this, [this]() {
#ifdef FEATURE_PLAN_SCANNER
        editPlan();
#endif
    });

#ifdef FEATURE_PLAN_SCANNER
    // ── Back / Recalibrate button ─────────────────────────────────────────────
    connect(m_btnBackToMenu, &QPushButton::clicked, this, [this]() {
#ifdef Q_OS_IOS
        if (m_arLocked) {
            // Reset the anchor and re-enter the init phase without leaving AR.
            QMetaObject::invokeMethod(m_controller, &AppController::resetARRegistration,
                                      Qt::QueuedConnection);
            return;
        }
#endif
        m_arCamera->stop();
        m_btnEditPlan->setVisible(false);
#ifdef Q_OS_IOS
        QMetaObject::invokeMethod(m_controller,
            [this]() { m_controller->setShowDepthOverlay(false); },
            Qt::QueuedConnection);
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
        const SurgicalPlan plan = m_currentPlan;
        QMetaObject::invokeMethod(m_controller,
            [this, plan]() { m_controller->setSurgicalPlan(plan); },
            Qt::QueuedConnection);
        m_btnEditPlan->setVisible(true);
#ifdef Q_OS_IOS
        QMetaObject::invokeMethod(m_controller,
            [this]() { m_controller->setShowDepthOverlay(true); },
            Qt::QueuedConnection);
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
    // Reset UI immediately so the label shows "Calibration requise" as soon as
    // the AR view appears, before the queued resetARRegistration runs on the
    // controller thread.
    setArLocked(false);
    QMetaObject::invokeMethod(m_controller, &AppController::resetARRegistration,
                              Qt::QueuedConnection);
#endif
    m_arCamera->start();
}

void MainWindow::onPlanDetected(const SurgicalPlan &detected)
{
    m_scanScreen->stopCamera();

    ConfirmPlanDialog dlg(detected, ConfirmPlanDialog::Mode::Scan, this);
    if (dlg.exec() != QDialog::Accepted) {
        // User cancelled — go back to scan screen
        m_stack->setCurrentIndex(1);
        m_scanScreen->startCamera();
        return;
    }

    m_currentPlan = dlg.plan();
    const SurgicalPlan plan = m_currentPlan;
    QMetaObject::invokeMethod(m_controller,
        [this, plan]() { m_controller->setSurgicalPlan(plan); },
        Qt::QueuedConnection);
    m_btnEditPlan->setVisible(true);
    startAR();
}

void MainWindow::editPlan()
{
    m_arCamera->stop();

    ConfirmPlanDialog dlg(m_currentPlan, ConfirmPlanDialog::Mode::Edit, this);
    if (dlg.exec() == QDialog::Accepted) {
        m_currentPlan = dlg.plan();
        const SurgicalPlan plan = m_currentPlan;
        QMetaObject::invokeMethod(m_controller,
            [this, plan]() { m_controller->setSurgicalPlan(plan); },
            Qt::QueuedConnection);
    }

#ifdef Q_OS_IOS
    setArLocked(false);
    QMetaObject::invokeMethod(m_controller, &AppController::resetARRegistration,
                              Qt::QueuedConnection);
#endif
    m_arCamera->start();
}

#ifdef Q_OS_IOS
void MainWindow::setArLocked(bool locked)
{
    m_arLocked = locked;
    if (locked) {
        m_arCalibrating = false;
        m_calibLabel->setText("Calibration réussie, repères verrouillés");
        m_calibLabel->setStyleSheet(
            "color: #75D0C5; background: rgba(117,208,197,50);"
            "padding: 6px; font-size: 12pt;");
        m_btnBackToMenu->setText("Recalibrer");
    } else {
        m_arCalibrating = false;
        m_calibLabel->setText("Calibration requise : placez la caméra face aux repères");
        m_calibLabel->setStyleSheet(
            "color: #DE5F5E; background: rgba(222,95,94,50);"
            "padding: 6px; font-size: 12pt;");
        m_btnBackToMenu->setText("← Retour");
    }
    if (m_arStrip) m_arStrip->update();
}

void MainWindow::setArCalibrating(bool calibrating)
{
    if (m_arLocked) return; // locked state takes priority
    m_arCalibrating = calibrating;
    m_calibLabel->setText(calibrating
        ? "Calibration en cours..."
        : "Calibration requise : placez la caméra face aux repères");
    m_calibLabel->setStyleSheet(
        "color: #DE5F5E; background: rgba(222,95,94,50);"
        "padding: 6px; font-size: 12pt;");
    if (m_arStrip) m_arStrip->update();
}
#endif

#endif
