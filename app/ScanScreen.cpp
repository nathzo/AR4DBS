#include "ScanScreen.h"
#include "core/ocr/PlanScanner.h"
#include "core/rendering/GLWidget.h"

#include <QCoreApplication>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLabel>

#ifdef Q_OS_IOS
#  include "platform/ios/IOSCamera.h"
#else
#  include "platform/desktop/DesktopCamera.h"
#endif

#include <opencv2/core.hpp>

struct ScanScreen::Impl {
#ifdef Q_OS_IOS
    IOSCamera     *camera = nullptr;
#else
    DesktopCamera *camera = nullptr;
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

    auto *root = new QVBoxLayout(this);
    root->setContentsMargins(0, 0, 0, 0);
    root->setSpacing(0);

    // Camera preview
    m_impl->preview = new GLWidget(this);
    root->addWidget(m_impl->preview, 1);

    // Status bar
    m_impl->status = new QLabel(
        PlanScanner::isAvailable()
            ? "Pointez l'écran Medtronic et appuyez sur Capturer"
            : "OCR non disponible — saisissez les coordonnées manuellement",
        this);
    m_impl->status->setAlignment(Qt::AlignCenter);
    m_impl->status->setStyleSheet(
        "color: #75D0C5; background: rgba(26,27,29,200); padding: 6px; font-size: 12pt;");
    root->addWidget(m_impl->status);

    // Buttons row
    auto *btnRow = new QHBoxLayout;
    btnRow->setContentsMargins(16, 8, 16, 16);
    root->addLayout(btnRow);

    auto *btnBack = new QPushButton("← Retour", this);
    btnBack->setStyleSheet(
        "QPushButton { background:#75D0C5; color: black; border-radius:8px;"
        "              padding:12px 24px; font-size:13pt; font-weight:bold; }"
        "QPushButton:pressed { background:#5ab8ae; }");

    auto *btnCapture = new QPushButton(
        PlanScanner::isAvailable() ? "Capturer" : "Saisir manuellement", this);
    btnCapture->setStyleSheet(
        "QPushButton { background:#DE5F5E; color:white; border-radius:8px;"
        "              padding:12px 32px; font-size:14pt; font-weight:bold; }"
        "QPushButton:pressed { background:#a33c3f; }");

    btnRow->addWidget(btnBack);
    btnRow->addStretch();
    btnRow->addWidget(btnCapture);

    connect(btnBack,    &QPushButton::clicked, this, &ScanScreen::cancelled);
    connect(btnCapture, &QPushButton::clicked, this, &ScanScreen::onCapture);

    // Camera
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
        m_impl->lastFrame = frame;          // cv::Mat is ref-counted, no deep copy needed
        m_impl->preview->setFrame(frame);
    });
}

ScanScreen::~ScanScreen()
{
    stopCamera();
    delete m_impl;
}

void ScanScreen::startCamera() { m_impl->camera->start(); }
void ScanScreen::stopCamera()  { if (m_impl->camera) m_impl->camera->stop(); }

void ScanScreen::onCapture()
{
    if (m_impl->lastFrame.empty()) {
        // No frame yet — open dialog with empty fields
        emit planDetected({});
        return;
    }

    m_impl->status->setText("Analyse en cours…");
    QCoreApplication::processEvents(); // let the label update paint

    SurgicalPlan plan = PlanScanner::scan(m_impl->lastFrame);

    if (plan.hasAny())
        m_impl->status->setText("✓ Coordonnées détectées");
    else
        m_impl->status->setText("Coordonnées non détectées — saisie manuelle");

    emit planDetected(plan);
}
