#include "ScanScreen.h"
#include "core/ocr/PlanScanner.h"
#include "core/rendering/GLWidget.h"

#include <QCoreApplication>
#include <QGuiApplication>
#include <QScreen>
#include <QVBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QPainter>
#include <QMouseEvent>
#include <QResizeEvent>

#ifdef Q_OS_IOS
#  include "platform/ios/IOSCamera.h"
#else
#  include "platform/desktop/DesktopCamera.h"
#endif

#include <opencv2/core.hpp>

// ── RotatedStrip ──────────────────────────────────────────────────────────────
// A fixed-height strip (stripH px tall in portrait) whose inner content is
// rendered rotated 90° CW.  When the phone is held landscape (CW rotation):
//   • the strip appears as a vertical panel on the right of the camera view
//   • all text and buttons are readable left-to-right
//
// Rotation math — translate(0, stripH) + rotate(−90°) in QPainter maps
//   inner (x, y)  →  screen (y,  stripH − x)
// so:
//   VBox flow  (+y in inner)  →  screen +x  →  landscape top-to-bottom  ✓
//   text flow  (+x in inner)  →  screen −y  →  landscape left-to-right  ✓
// Inverse for hit-testing: screen (sx, sy)  →  inner (stripH − sy,  sx)

class RotatedStrip : public QWidget
{
public:
    QPushButton *captureBtn = nullptr;
    QPushButton *backBtn    = nullptr;

    explicit RotatedStrip(int stripH, QWidget *parent = nullptr)
        : QWidget(parent), m_stripH(stripH)
    {
        setFixedHeight(stripH);

        // Inner widget lives outside the normal widget hierarchy so it is
        // never painted by Qt's own traversal; RotatedStrip owns and deletes it.
        m_inner = new QWidget;
        m_inner->setStyleSheet("background: black;");

        auto *lay = new QVBoxLayout(m_inner);
        lay->setContentsMargins(8, 8, 8, 8);
        lay->setSpacing(8);

        m_status = new QLabel(m_inner);
        m_status->setAlignment(Qt::AlignCenter);
        m_status->setWordWrap(true);
        // Minimum height to ensure readability; adapts vertically based on text length.
        // Center position stays constant by using equal stretches above and below.
        m_status->setMinimumHeight(60);
        m_status->setStyleSheet(
            "color: #75D0C5; background: rgba(117,208,197,50);"
            "padding: 6px; font-size: 12pt;");
        lay->addStretch(1);             // gap from edge
        lay->addWidget(m_status);
        lay->addStretch(2);             // gap below status (increased for button spacing)

        captureBtn = new QPushButton(m_inner);
        captureBtn->setStyleSheet(
            "QPushButton { background:#DE5F5E; color:white; border-radius:8px;"
            "              padding:12px 28px;"
            "              font-size:16pt; font-weight:bold; }"
            "QPushButton:pressed { background:#a33c3f; }");
        lay->addWidget(captureBtn, 0, Qt::AlignHCenter);
        lay->addStretch(3);             // gap toward menu button

        backBtn = new QPushButton("← Menu", m_inner);
        backBtn->setStyleSheet(
            "QPushButton { background:#8A8C8F; color:black; border-radius:8px;"
            "              padding:8px 20px;"
            "              font-size:13pt; font-weight:bold; }"
            "QPushButton:pressed { background:#6d6f72; }");
        lay->addWidget(backBtn, 0, Qt::AlignHCenter);
        lay->addStretch(1);             // gap from bottom
    }

    ~RotatedStrip() override { delete m_inner; }

    void setStatusText(const QString &text)
    {
        m_status->setText(text);
        update();   // repaint the strip
    }

protected:
    void resizeEvent(QResizeEvent *) override { syncInner(); }

    void paintEvent(QPaintEvent *) override
    {
        syncInner();
        QPainter p(this);
        // translate(width(), 0) + rotate(+90°):
        //   painter's +x → screen downward  (landscape CCW: rightward)
        //   painter's +y → screen leftward  (landscape CCW: downward)
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
        // Inner widget: width = stripH (landscape panel height),
        //               height = outer width (landscape panel width).
        m_inner->resize(m_stripH, width());
        if (auto *l = m_inner->layout()) l->activate();
        m_inner->ensurePolished();
    }

    // Map a screen coordinate (in this widget) to inner-widget coordinates.
    // Inverse of inner(x,y) → screen(W−y, x):  inner = (sy, W−sx)
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
    QLabel      *m_status     = nullptr;
    QPushButton *m_pressedBtn = nullptr;
    int          m_stripH     = 0;
};

// ── ScanScreen ────────────────────────────────────────────────────────────────

struct ScanScreen::Impl {
#ifdef Q_OS_IOS
    IOSCamera     *camera = nullptr;
#else
    DesktopCamera *camera = nullptr;
#endif
    GLWidget      *preview = nullptr;
    RotatedStrip  *strip   = nullptr;
    cv::Mat        lastFrame;
};

ScanScreen::ScanScreen(QWidget *parent)
    : QWidget(parent)
    , m_impl(new Impl)
{
    setStyleSheet("background-color: black;");

    const int W = QGuiApplication::primaryScreen()->availableGeometry().width();

    // Root layout: camera fills the top; rotated control strip at the bottom.
    auto *root = new QVBoxLayout(this);
    root->setContentsMargins(0, 0, 0, 0);
    root->setSpacing(0);

    // ── Camera preview ────────────────────────────────────────────────────────
    // The camera delivers 640×480 landscape frames; GLWidget rotates them 90° so
    // the displayed height = W × (640/480).  Fixing the widget to exactly that
    // height leaves no room for black bars inside it.  A stretch above absorbs
    // the leftover screen height — in landscape that black space lands at the far
    // edge, not between the camera and the control panel.
    m_impl->preview = new GLWidget(this);
    m_impl->preview->setFixedHeight(W * 640 / 480);
    root->addStretch(1);
    root->addWidget(m_impl->preview);
    root->addSpacing(8);

    // ── Control strip ─────────────────────────────────────────────────────────
    m_impl->strip = new RotatedStrip(150, this);
    m_impl->strip->captureBtn->setText(
        PlanScanner::isAvailable() ? tr("Capturer") : tr("Saisir manuellement"));
    m_impl->strip->setStatusText(
        PlanScanner::isAvailable()
            ? tr("Pointez l'écran Medtronic et appuyez sur Capturer")
            : tr("OCR non disponible — saisissez les coordonnées manuellement"));
    root->addWidget(m_impl->strip);
    root->addSpacing(8);

    connect(m_impl->strip->backBtn,    &QPushButton::clicked,
            this, &ScanScreen::cancelled);
    connect(m_impl->strip->captureBtn, &QPushButton::clicked,
            this, &ScanScreen::onCapture);

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
    // Reset status to idle message whenever the camera is (re)started —
    // e.g. after the user cancels the confirmation dialog.
    m_impl->strip->setStatusText(
        PlanScanner::isAvailable()
            ? tr("Pointez l'écran Medtronic et appuyez sur Capturer")
            : tr("OCR non disponible — saisissez les coordonnées manuellement"));
    m_impl->camera->start();
}

void ScanScreen::stopCamera() { if (m_impl->camera) m_impl->camera->stop(); }

void ScanScreen::onCapture()
{
    if (m_impl->lastFrame.empty()) {
        emit planDetected({});
        return;
    }

    m_impl->strip->setStatusText(tr("Analyse en cours…"));
    QCoreApplication::processEvents();   // let the label repaint before OCR blocks

    SurgicalPlan plan = PlanScanner::scan(m_impl->lastFrame);
    emit planDetected(plan);
}
