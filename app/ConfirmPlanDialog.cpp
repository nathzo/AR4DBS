#include "ConfirmPlanDialog.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QTabWidget>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QDialogButtonBox>
#include <QLabel>
#include <QPushButton>
#include <QKeyEvent>
#include <QFocusEvent>
#include <QPainter>
#include <QPaintEvent>
#include <QGuiApplication>
#include <QScreen>

static constexpr float kConfidenceThreshold = 0.99f;

// ── AutoSelectSpinBox ─────────────────────────────────────────────────────────
// Selects all text on focus so the first keystroke replaces the value.
// Minimum is always -1.0; setValue(-1) shows the special "—" placeholder,
// indicating a field that was not detected by OCR.

class AutoSelectSpinBox : public QDoubleSpinBox {
public:
    using QDoubleSpinBox::QDoubleSpinBox;
protected:
    void focusInEvent(QFocusEvent *e) override {
        QDoubleSpinBox::focusInEvent(e);
        QMetaObject::invokeMethod(this, &QAbstractSpinBox::selectAll,
                                  Qt::QueuedConnection);
    }
};

// Creates a spinbox with -1 as the "not detected" sentinel (shown as " —").
static AutoSelectSpinBox *makeSpinBox(double max, bool detected, double val,
                                      const QString &suffix, QWidget *parent)
{
    auto *sb = new AutoSelectSpinBox(parent);
    sb->setRange(-1.0, max);
    sb->setDecimals(1);
    sb->setSingleStep(0.1);
    sb->setSuffix(suffix);
    sb->setSpecialValueText(" —");          // shown when value == minimum (-1)
    sb->setFixedHeight(52);                 // uniform row height across all fields
    sb->setMinimumWidth(150);
    sb->setValue(detected ? val : sb->minimum());
    return sb;
}

// ── Flag styling ──────────────────────────────────────────────────────────────

static const char *kFlaggedStyle =
    "QDoubleSpinBox {"
    "  background: rgba(196,82,85,0.18);"
    "  border: 1px solid #c45255;"
    "  border-radius: 4px;"
    "  padding: 3px 6px;"
    "  color: #e0e0e0;"
    "}";

// ── buildSide ─────────────────────────────────────────────────────────────────

ConfirmPlanDialog::TargetWidgets ConfirmPlanDialog::buildSide(
    const QString &title, const LeksellTarget &t, QWidget *parent)
{
    TargetWidgets w;

    auto *box = new QGroupBox(title, parent);
    box->setCheckable(false);
    box->setObjectName(title);
    auto *form = new QFormLayout(box);
    form->setLabelAlignment(Qt::AlignRight);
    form->setVerticalSpacing(14);
    form->setHorizontalSpacing(16);
    form->setContentsMargins(16, 16, 16, 16);

    w.enabled = new QCheckBox("Activer", parent);
    w.enabled->setChecked(true);

    // confidence[] < 0 → field not detected; 0–1 → Vision confidence
    bool xDet    = t.confidence[0] >= 0.f;
    bool yDet    = t.confidence[1] >= 0.f;
    bool zDet    = t.confidence[2] >= 0.f;
    bool ringDet = t.confidence[3] >= 0.f;
    bool arcDet  = t.confidence[4] >= 0.f;

    w.x    = makeSpinBox(300, xDet,    t.x_mm,     " mm", parent);
    w.y    = makeSpinBox(300, yDet,    t.y_mm,     " mm", parent);
    w.z    = makeSpinBox(200, zDet,    t.z_mm,     " mm", parent);
    w.ring = makeSpinBox(360, ringDet, t.ring_deg, " °",  parent);
    w.arc  = makeSpinBox(180, arcDet,  t.arc_deg,  " °",  parent);

    form->addRow(w.enabled);
    form->addRow("X (mm) :",        w.x);
    form->addRow("Y (mm) :",        w.y);
    form->addRow("Z (mm) :",        w.z);
    form->addRow("Ring (degrés) :", w.ring);
    form->addRow("Arc  (degrés) :", w.arc);

    // Disable spinboxes when side is unchecked
    auto updateEnabled = [w](bool on) {
        w.x->setEnabled(on);
        w.y->setEnabled(on);
        w.z->setEnabled(on);
        w.ring->setEnabled(on);
        w.arc->setEnabled(on);
    };
    updateEnabled(true);
    QObject::connect(w.enabled, &QCheckBox::toggled, box, updateEnabled);
    // Re-evaluate the Confirmer gate when a side is toggled on/off.
    QObject::connect(w.enabled, &QCheckBox::toggled, this,
                     [this](bool) { updateConfirmButton(); });

    // Flag spinboxes with confidence below threshold (includes undetected: conf = -1).
    auto maybeFlag = [&](QDoubleSpinBox *sb, float conf) {
        if (conf >= kConfidenceThreshold) return; // high confidence: no flag
        sb->setProperty("flagged", true);
        sb->setStyleSheet(kFlaggedStyle);
        ++m_flaggedCount;
        QObject::connect(sb, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                         this, [this, sb](double) { clearFlag(sb); });
    };

    maybeFlag(w.x,    t.confidence[0]);
    maybeFlag(w.y,    t.confidence[1]);
    maybeFlag(w.z,    t.confidence[2]);
    maybeFlag(w.ring, t.confidence[3]);
    maybeFlag(w.arc,  t.confidence[4]);

    return w;
}

// ── Constructor ───────────────────────────────────────────────────────────────

ConfirmPlanDialog::ConfirmPlanDialog(const SurgicalPlan &initial, QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle("Confirmer le plan chirurgical");
    setWindowFlags(Qt::FramelessWindowHint | Qt::Dialog);
    setAttribute(Qt::WA_TranslucentBackground);
    {
        const QRect ag = QGuiApplication::primaryScreen()->availableGeometry();
        setFixedSize(ag.width(), ag.height());
    }
    setStyleSheet(
        "QDialog { background: transparent; }"
        "QGroupBox, QWidget {"
        "  background-color: black;"
        "  color: #e0e0e0;"
        "  font-family: 'Arial'; font-size: 14pt;"
        "}"
        "QGroupBox {"
        "  border: 1px solid #75D0C5;"
        "  border-radius: 8px;"
        "  margin-top: 12px;"
        "  padding-top: 10px;"
        "  font-family: 'Arial'; font-weight: bold;"
        "  color: #75D0C5;"
        "}"
        "QGroupBox::title { subcontrol-origin: margin; left: 12px; }"
        "QTabWidget::pane { border: 1px solid #75D0C5; border-radius: 8px; }"
        "QTabBar::tab {"
        "  background: #2a2b2d; color: #e0e0e0;"
        "  padding: 16px 28px; font-family: 'Arial'; font-size: 14pt;"
        "  border: 1px solid #444; border-bottom: none;"
        "  border-top-left-radius: 8px; border-top-right-radius: 8px;"
        "}"
        "QTabBar::tab:selected { background: #75D0C5; color: black; font-family: 'Arial'; font-weight: bold; }"
        "QDoubleSpinBox {"
        "  background: #2a2b2d; color: #e0e0e0;"
        "  border: 1px solid #444; border-radius: 6px; padding: 10px 12px;"
        "}"
        "QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { width: 0; border: none; }"
        "QCheckBox { color: #75D0C5; font-family: 'Arial'; font-weight: bold; }"
        "QPushButton {"
        "  border-radius: 10px; padding: 16px 40px;"
        "  font-family: 'Arial'; font-size: 15pt; font-weight: bold;"
        "}"
        "QPushButton[text='Annuler'] { background: #75D0C5; color: black; }"
    );

    auto *mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(20, 20, 20, 20);
    mainLayout->setSpacing(14);

    // OCR status banner
    auto *banner = new QLabel(this);
    banner->setWordWrap(true);
    if (initial.hasAny()) {
        banner->setText("✓ Coordonnées détectées. Vérifiez avant de confirmer.");
        banner->setStyleSheet("background: #1e3a3a; color: #75D0C5; padding: 8px; border-radius: 4px;");
    } else {
        banner->setText("Coordonnées non détectées. Saisissez les valeurs manuellement.");
        banner->setStyleSheet("background: #3a1e1f; color: #c45255; padding: 8px; border-radius: 4px;");
    }
    mainLayout->addWidget(banner);

    // Create the confirm button early so buildSide can reference m_confirmBtn
    // through updateConfirmButton() called from the checkbox toggle connection.
    auto *buttons = new QDialogButtonBox(
        QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);

    m_confirmBtn = buttons->button(QDialogButtonBox::Ok);
    m_confirmBtn->setText("Confirmer");
    m_confirmBtn->setAutoDefault(false);
    m_confirmBtn->setDefault(false);
    m_confirmBtn->setStyleSheet(
        "QPushButton { background: #c45255; color: white;"
        "  border-radius: 10px; padding: 16px 40px;"
        "  font-family: 'Arial'; font-size: 15pt; font-weight: bold; }"
        "QPushButton:disabled { background: #5a2e2f; color: #888; }");

    auto *cancelBtn = buttons->button(QDialogButtonBox::Cancel);
    cancelBtn->setText("Annuler");
    cancelBtn->setAutoDefault(false);
    cancelBtn->setDefault(false);

    // Build tabs (buildSide may increment m_flaggedCount and connect valueChanged)
    auto *tabs = new QTabWidget(this);

    auto *leftPage = new QWidget(tabs);
    auto *leftLayout = new QVBoxLayout(leftPage);
    leftLayout->setContentsMargins(16, 16, 16, 16);
    leftLayout->setSpacing(12);
    m_left = buildSide("Gauche (G)", initial.left, leftPage);
    leftLayout->addWidget(leftPage->findChild<QGroupBox *>("Gauche (G)"));
    leftLayout->addStretch();
    tabs->addTab(leftPage, "Gauche (G)");

    auto *rightPage = new QWidget(tabs);
    auto *rightLayout = new QVBoxLayout(rightPage);
    rightLayout->setContentsMargins(16, 16, 16, 16);
    rightLayout->setSpacing(12);
    m_right = buildSide("Droite (D)", initial.right, rightPage);
    rightLayout->addWidget(rightPage->findChild<QGroupBox *>("Droite (D)"));
    rightLayout->addStretch();
    tabs->addTab(rightPage, "Droite (D)");

    mainLayout->addWidget(tabs);
    mainLayout->addWidget(buttons);

    connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);

    // Set initial Confirmer state based on flagged count.
    updateConfirmButton();
}

// ── Flag management ───────────────────────────────────────────────────────────

void ConfirmPlanDialog::clearFlag(QDoubleSpinBox *sb)
{
    if (!sb->property("flagged").toBool()) return;
    if (sb->value() < 0.0) return; // still "—": user hasn't entered a real value

    sb->setProperty("flagged", false);
    sb->setStyleSheet(""); // revert to dialog-level stylesheet
    --m_flaggedCount;
    updateConfirmButton();
}

void ConfirmPlanDialog::updateConfirmButton()
{
    // Only count flagged fields on sides that are currently active.
    auto countFlagged = [](const TargetWidgets &w) -> int {
        if (!w.enabled || !w.enabled->isChecked()) return 0;
        int n = 0;
        for (auto *sb : { w.x, w.y, w.z, w.ring, w.arc })
            if (sb && sb->property("flagged").toBool()) ++n;
        return n;
    };
    const int effective = countFlagged(m_left) + countFlagged(m_right);
    m_confirmBtn->setEnabled(effective == 0);
}

// ── Read-back ─────────────────────────────────────────────────────────────────

LeksellTarget ConfirmPlanDialog::readWidgets(const TargetWidgets &w)
{
    LeksellTarget t;
    t.valid = w.enabled->isChecked();
    // Fields left at the sentinel (-1) were never filled; leave them at 0 default.
    if (w.x->value()    >= 0) t.x_mm     = w.x->value();
    if (w.y->value()    >= 0) t.y_mm     = w.y->value();
    if (w.z->value()    >= 0) t.z_mm     = w.z->value();
    if (w.ring->value() >= 0) t.ring_deg = w.ring->value();
    if (w.arc->value()  >= 0) t.arc_deg  = w.arc->value();
    return t;
}

SurgicalPlan ConfirmPlanDialog::plan() const
{
    SurgicalPlan p;
    p.left  = readWidgets(m_left);
    p.right = readWidgets(m_right);
    return p;
}

// ── Paint / input overrides ───────────────────────────────────────────────────

void ConfirmPlanDialog::paintEvent(QPaintEvent *)
{
    QPainter p(this);
    p.setPen(Qt::NoPen);
    p.setBrush(Qt::black);
    p.drawRect(rect());
}

void ConfirmPlanDialog::showEvent(QShowEvent *e)
{
    QDialog::showEvent(e);
    const QRect ag = QGuiApplication::primaryScreen()->availableGeometry();
    move(ag.topLeft());
}

void ConfirmPlanDialog::keyPressEvent(QKeyEvent *event)
{
    if (event->key() == Qt::Key_Return || event->key() == Qt::Key_Enter)
        return; // "Terminé" on iOS keyboard only hides the keyboard
    QDialog::keyPressEvent(event);
}
