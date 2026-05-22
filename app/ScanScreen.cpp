#include "ScanScreen.h"
#include "core/ocr/PlanScanner.h"
#include "core/rendering/GLWidget.h"

#include <QCoreApplication>
#include <QVBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsProxyWidget>
#include <QGuiApplication>
#include <QScreen>

#ifdef Q_OS_IOS
#  include "platform/ios/IOSCamera.h"
#else
#  include "platform/desktop/DesktopCamera.h"
#endif

#include <opencv2/core.hpp>

struct ScanScreen::Impl {
#ifdef Q_OS_IOS
    IOSCamera     *camera  = nullptr;
#else
    DesktopCamera *camera  = nullptr;
#endif
    GLWidget      *preview = nullptr;
    QLabel        *status  = nullptr;
    cv::Mat        lastFrame;
};

ScanScreen::ScanScreen(QWidget *parent)
    : QWidget(parent)
    , m_impl(new Impl)
{
    setStyleSheet("background-color: black;");

    // Portrait screen dimensions — needed to size the rotated control widget.
    const QRect ag = QGuiApplication::primaryScreen()->availableGeometry();
    const int   W  = ag.width();   // portrait width  (= height of the landscape strip content)
    const int   SH = 200;          // strip height in portrait  = strip width in landscape

    // Root layout: camera fills the top; rotated control strip sits at the bottom.
    auto *root = new QVBoxLayout(this);
    root->setContentsMargins(0, 0, 0, 0);
    root->setSpacing(0);

    // ── Camera preview (fills all remaining space) ────────────────────────────
    m_impl->preview = new GLWidget(this);
    root->addWidget(m_impl->preview, 1);

    // ── Rotated control strip ─────────────────────────────────────────────────
    //
    // The controls widget is authored in its own "landscape" coordinate space
    // (width = SH, height = W).  It is then embedded in a QGraphicsScene and
    // rotated 90° CW so that it fills a W × SH scene rect.
    //
    // Rotation(+90) + setPos(0, SH) maps widget (x,y) → scene (y, SH−x):
    //   • Widget corner (0,  0) → scene (0,  SH)  ← bottom-left
    //   • Widget corner (SH, 0) → scene (0,   0)  ← top-left
    //   • Widget corner (0,  W) → scene (W,  SH)  ← bottom-right
    //   • Widget corner (SH, W) → scene (W,   0)  ← top-right
    //
    // This means:
    //   • VBox +y in widget  → +x in scene  → left-to-right in Qt portrait
    //     → top-to-bottom in landscape (phone CW)  ✓
    //   • Text flows +x in widget  → −y in scene (upward in Qt portrait)
    //     → left-to-right in landscape (phone CW)  ✓

    auto *ctrlWidget = new QWidget;          // no Qt parent — scene proxy will own it
    ctrlWidget->setFixedSize(SH, W);
    ctrlWidget->setStyleSheet("background-color: black;");

    auto *ctrlLayout = new QVBoxLayout(ctrlWidget);
    ctrlLayout->setContentsMargins(16, 16, 16, 16);
    ctrlLayout->setSpacing(12);

    // Status label — top of the landscape strip
    m_impl->status = new QLabel(
        PlanScanner::isAvailable()
            ? "Pointez l'écran Medtronic et appuyez sur Capturer"
            : "OCR non disponible — saisissez les coordonnées manuellement",
        ctrlWidget);
    m_impl->status->setAlignment(Qt::AlignCenter);
    m_impl->status->setWordWrap(true);
    m_impl->status->setStyleSheet(
        "color: #75D0C5; background: rgba(117,208,197,50); padding: 6px; font-size: 12pt;");
    ctrlLayout->addWidget(m_impl->status);

    // Capturer — vertically centred in the landscape strip
    auto *btnCapture = new QPushButton(
        PlanScanner::isAvailable() ? "Capturer" : "Saisir manuellement", ctrlWidget);
    btnCapture->setStyleSheet(
        "QPushButton { background:#DE5F5E; color:white; border-radius:8px;"
        "              padding:12px 32px; font-family:'Arial'; font-size:14pt; font-weight:bold; }"
        "QPushButton:pressed { background:#a33c3f; }");
    ctrlLayout->addStretch(1);
    ctrlLayout->addWidget(btnCapture, 0, Qt::AlignHCenter);
    ctrlLayout->addStretch(1);

    // Retour — bottom of the landscape strip
    auto *btnBack = new QPushButton("← Retour", ctrlWidget);
    btnBack->setStyleSheet(
        "QPushButton { background:#8A8C8F; color: black; border-radius:8px;"
        "              padding:12px 24px; font-family:'Arial'; font-size:13pt; font-weight:bold; }"
        "QPushButton:pressed { background:#6d6f72; }");
    ctrlLayout->addWidget(btnBack, 0, Qt::AlignHCenter);

    // Place the controls widget into a scene, rotated 90° CW.
    auto *scene = new QGraphicsScene(0, 0, W, SH, this);
    auto *proxy = scene->addWidget(ctrlWidget);
    proxy->setRotation(90);
    proxy->setPos(0, SH);

    auto *gv = new QGraphicsView(scene, this);
    gv->setFixedHeight(SH);
    gv->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    gv->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    gv->setFrameShape(QFrame::NoFrame);
    gv->setStyleSheet("background: black; border: none;");
    gv->setAlignment(Qt::AlignLeft | Qt::AlignTop);
    gv->setInteractive(true);
    root->addWidget(gv);

    connect(btnBack,    &QPushButton::clicked, this, &ScanScreen::cancelled);
    connect(btnCapture, &QPushButton::clicked, this, &ScanScreen::onCapture);

    // ── Camera ────────────────────────────────────────────────────────────────
#ifdef Q_OS_IOS
    m_impl->camera = new IOSCamera(640, 480, this);
#else
    m_impl->camera = new DesktopCamera(0, this);
#endif
    connect(m_impl->camera,
#ifdef Q_OS_IOS
            &IOSCamera::frameReady,
#else
            &DesktopCamera::frameReady,
#endif
            this, [this](const cv::Mat &frame) {
        m_impl->lastFrame = frame;
        m_impl->preview->setFrame(frame);
    });
}

ScanScreen::~ScanScreen()
{
    stopCamera();
    delete m_impl;
}

void ScanScreen::startCamera()
{
    // Reset the status label to its idle message whenever the camera is
    // (re)started — e.g. after the user cancels the confirmation dialog.
    m_impl->status->setText(
        PlanScanner::isAvailable()
            ? "Pointez l'écran Medtronic et appuyez sur Capturer"
            : "OCR non disponible — saisissez les coordonnées manuellement");
    m_impl->camera->start();
}

void ScanScreen::stopCamera() { if (m_impl->camera) m_impl->camera->stop(); }

void ScanScreen::onCapture()
{
    if (m_impl->lastFrame.empty()) {
        emit planDetected({});
        return;
    }

    m_impl->status->setText("Analyse en cours…");
    QCoreApplication::processEvents();   // let the label repaint before OCR blocks

    SurgicalPlan plan = PlanScanner::scan(m_impl->lastFrame);
    emit planDetected(plan);
}
