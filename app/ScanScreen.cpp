#include "ScanScreen.h"
#include "core/ocr/PlanScanner.h"
#include "core/rendering/GLWidget.h"

#include <QCoreApplication>
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
        lay->setContentsMargins(16, 16, 16, 16);
        lay->setSpacing(12);

        m_status = new QLabel(m_inner);
        m_status->setAlignment(Qt::AlignCenter);
        m_status->setWordWrap(true);
        m_status->setStyleSheet(
            "color: #75D0C5; background: rgba(117,208,197,50);"
            "padding: 6px; font-size: 12pt;");
        lay->addWidget(m_status);
        lay->addStretch(1);

        captureBtn = new QPushButton(m_inner);
        captureBtn->setStyleSheet(
            "QPushButton { background:#DE5F5E; color:white; border-radius:8px;"
            "              padding:12px 32px; font-family:'Arial';"
            "              font-size:14pt; font-weight:bold; }"
            "QPushButton:pressed { background:#a33c3f; }");
        lay->addWidget(captureBtn, 0, Qt::AlignHCenter);
        lay->addStretch(1);

        backBtn = new QPushButton("← Retour", m_inner);
        backBtn->setStyleSheet(
            "QPushButton { background:#8A8C8F; color:black; border-radius:8px;"
            "              padding:12px 24px; font-family:'Arial';"
            "              font-size:13pt; font-weight:bold; }"
            "QPushButton:pressed { background:#6d6f72; }");
        lay->addWidget(backBtn, 0, Qt::AlignHCenter);
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
        // translate(0, stripH) + rotate(−90°):
        //   painter's +x → screen upward  (landscape: rightward)
        //   painter's +y → screen rightward (landscape: downward)
        p.translate(0, m_stripH);
        p.rotate(-90);
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
    // Inverse of inner(x,y) → screen(y, stripH−x):  inner = (stripH−sy, sx)
    QPoint toInner(QPoint s) const { return { m_stripH - s.y(), s.x() }; }

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

    // Root layout: camera fills the top; rotated control strip at the bottom.
    auto *root = new QVBoxLayout(this);
    root->setContentsMargins(0, 0, 0, 0);
    root->setSpacing(0);

    // ── Camera preview ────────────────────────────────────────────────────────
    m_impl->preview = new GLWidget(this);
    root->addWidget(m_impl->preview, 1);

    // ── Control strip ─────────────────────────────────────────────────────────
    m_impl->strip = new RotatedStrip(200, this);
    m_impl->strip->captureBtn->setText(
        PlanScanner::isAvailable() ? "Capturer" : "Saisir manuellement");
    m_impl->strip->setStatusText(
        PlanScanner::isAvailable()
            ? "Pointez l'écran Medtronic et appuyez sur Capturer"
            : "OCR non disponible — saisissez les coordonnées manuellement");
    root->addWidget(m_impl->strip);

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

    m_impl->strip->setStatusText("Analyse en cours…");
    QCoreApplication::processEvents();   // let the label repaint before OCR blocks

    SurgicalPlan plan = PlanScanner::scan(m_impl->lastFrame);
    emit planDetected(plan);
}
