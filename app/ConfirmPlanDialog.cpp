#include "ConfirmPlanDialog.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QTabWidget>
#include <QDoubleSpinBox>
#include <QLineEdit>
#include <QCheckBox>
#include <QDialogButtonBox>
#include <QLabel>
#include <QPushButton>
#include <QKeyEvent>
#include <QFocusEvent>
#include <QInputMethodEvent>
#include <QPainter>
#include <QPaintEvent>
#include <QGuiApplication>
#include <QScreen>
#include <QInputMethod>
#include <functional>

static constexpr float kConfidenceThreshold = 0.99f;

// ── AutoSelectSpinBox ─────────────────────────────────────────────────────────
// Selects all text on focus so the first keystroke replaces the value.
// Minimum is always -1.0; setValue(-1) shows the special "—" placeholder,
// indicating a field that was not detected by OCR.
//
// In Scan mode each spinbox is wired into a focus chain via chainTo():
//   Return/Enter → calls onConfirm callback, then focuses the next field.
//   On the last field (next == nullptr) → hides the virtual keyboard instead.

class AutoSelectSpinBox : public QDoubleSpinBox {
public:
    explicit AutoSelectSpinBox(QWidget *parent = nullptr)
        : QDoubleSpinBox(parent)
    {
        // On iOS the soft keyboard's QInputMethodEvent is delivered directly
        // to the internal QLineEdit (the actual UIKit first responder), never
        // reaching QDoubleSpinBox::inputMethodEvent.  Installing an event
        // filter here lets us intercept and normalise the decimal separator
        // before the QLineEdit sees the event.
        lineEdit()->installEventFilter(this);
    }

    // Wire this spinbox into the scan-mode focus chain.
    // next == nullptr marks this as the last field (Return hides the keyboard).
    // onConfirm is called on every Return press, even when the value hasn't
    // changed, so pre-filled OCR values get confirmed without retyping.
    void chainTo(QWidget *next, std::function<void()> onConfirm = {}) {
        m_nextInChain = next;
        m_inChain     = true;
        m_onConfirm   = std::move(onConfirm);
    }

protected:
    void focusInEvent(QFocusEvent *e) override {
        QDoubleSpinBox::focusInEvent(e);
        // Call selectAll() on the inner QLineEdit directly (not on the
        // QDoubleSpinBox wrapper) so iOS derives the selection-handle
        // positions from the QLineEdit's own coordinate system, which is
        // what it uses when positioning the blue pins.
        QMetaObject::invokeMethod(this, [this]() { lineEdit()->selectAll(); },
                                  Qt::QueuedConnection);
    }

    // eventFilter intercepts events sent to the internal QLineEdit.
    bool eventFilter(QObject *obj, QEvent *ev) override {
        if (obj == lineEdit() && ev->type() == QEvent::InputMethod) {
            auto *ime = static_cast<QInputMethodEvent *>(ev);
            const QString commit = ime->commitString();
            if (!commit.isEmpty()) {
                const QChar dp  = locale().decimalPoint().at(0);
                const QChar alt = (dp == QLatin1Char('.')) ? QLatin1Char(',')
                                                           : QLatin1Char('.');
                if (commit.contains(alt)) {
                    // Replace the wrong decimal separator, then re-deliver
                    // the corrected event to the QLineEdit.  Returning true
                    // discards the original so QAbstractSpinBox never sees it.
                    QString fixed = commit;
                    fixed.replace(alt, dp);
                    QInputMethodEvent mapped(ime->preeditString(), ime->attributes());
                    mapped.setCommitString(fixed,
                                          ime->replacementStart(),
                                          ime->replacementLength());
                    QCoreApplication::sendEvent(lineEdit(), &mapped);
                    return true;
                }
            }
        }
        return QDoubleSpinBox::eventFilter(obj, ev);
    }

    // Normalise the decimal separator at validation time — catches any path
    // (paste, hardware keyboard, programmatic setText) that slips past the
    // event filter or keyPressEvent.
    QValidator::State validate(QString &input, int &pos) const override {
        normalise(input);
        return QDoubleSpinBox::validate(input, pos);
    }
    double valueFromText(const QString &text) const override {
        QString t = text;
        normalise(t);
        return QDoubleSpinBox::valueFromText(t);
    }

    void keyPressEvent(QKeyEvent *e) override {
        // Hardware keyboard: normalise , and . to the locale decimal point.
        if (e->key() == Qt::Key_Comma || e->key() == Qt::Key_Period) {
            const QChar dp = locale().decimalPoint().at(0);
            const int key  = (dp == QLatin1Char('.')) ? Qt::Key_Period : Qt::Key_Comma;
            QKeyEvent mapped(e->type(), key, e->modifiers(), QString(dp));
            QDoubleSpinBox::keyPressEvent(&mapped);
            return;
        }
        // Scan-mode chain: Return/Enter confirms the field and advances focus.
        if (m_inChain && (e->key() == Qt::Key_Return || e->key() == Qt::Key_Enter)) {
            if (m_onConfirm) m_onConfirm();
            if (m_nextInChain) {
                m_nextInChain->setFocus(Qt::TabFocusReason);
            } else {
                clearFocus();
                if (auto *im = QGuiApplication::inputMethod()) im->hide();
            }
            return;
        }
        QDoubleSpinBox::keyPressEvent(e);
    }

private:
    // Replace the "wrong" decimal separator with the locale's one.
    // Called from validate() and valueFromText() so every input path
    // (paste, drag-drop, programmatic setText) is covered.
    void normalise(QString &s) const {
        const QChar dp  = locale().decimalPoint().at(0);
        const QChar alt = (dp == QLatin1Char('.')) ? QLatin1Char(',') : QLatin1Char('.');
        s.replace(alt, dp);
    }

    bool                  m_inChain     = false;
    QWidget              *m_nextInChain = nullptr;
    std::function<void()> m_onConfirm;
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
    "  selection-color: #e0e0e0;"
    "  selection-background-color: #7a2e30;"
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

    // Helper: flag a spinbox red and subscribe to value changes to auto-clear.
    auto doFlag = [&](QDoubleSpinBox *sb) {
        sb->setProperty("flagged", true);
        sb->setStyleSheet(kFlaggedStyle);
        ++m_flaggedCount;
        QObject::connect(sb, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                         this, [this, sb](double) { clearFlag(sb); });
    };

    if (m_scanMode) {
        // Scan mode: every field must be explicitly confirmed regardless of
        // OCR confidence — flag them all.
        for (auto *sb : { w.x, w.y, w.z, w.ring, w.arc })
            doFlag(sb);
    } else {
        // Edit mode: only flag fields whose OCR confidence is below threshold.
        auto maybeFlag = [&](QDoubleSpinBox *sb, float conf) {
            if (conf >= kConfidenceThreshold) return;
            doFlag(sb);
        };
        maybeFlag(w.x,    t.confidence[0]);
        maybeFlag(w.y,    t.confidence[1]);
        maybeFlag(w.z,    t.confidence[2]);
        maybeFlag(w.ring, t.confidence[3]);
        maybeFlag(w.arc,  t.confidence[4]);
    }

    return w;
}

// ── Constructor ───────────────────────────────────────────────────────────────

ConfirmPlanDialog::ConfirmPlanDialog(const SurgicalPlan &initial, Mode mode,
                                     QWidget *parent)
    : QDialog(parent), m_scanMode(mode == Mode::Scan)
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
        "  selection-color: #e0e0e0; selection-background-color: #2d5f7a;"
        "}"
        "QDoubleSpinBox::up-button {"
        "  subcontrol-origin: border; subcontrol-position: top right;"
        "  width: 0; height: 0; border: none; margin: 0;"
        "}"
        "QDoubleSpinBox::down-button {"
        "  subcontrol-origin: border; subcontrol-position: bottom right;"
        "  width: 0; height: 0; border: none; margin: 0;"
        "}"
        "QCheckBox { color: #75D0C5; font-family: 'Arial'; font-weight: bold; }"
        "QPushButton {"
        "  border-radius: 10px; padding: 16px 40px;"
        "  font-family: 'Arial'; font-size: 15pt; font-weight: bold;"
        "}"
        "QPushButton[text='Annuler'] { background: #8A8C8F; color: black; }"
    );

    auto *mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(20, 20, 20, 20);
    mainLayout->setSpacing(14);

    // ── OCR status banner ──────────────────────────────────────────────────────
    auto *banner = new QLabel(this);
    banner->setWordWrap(true);
    if (initial.hasAny()) {
        banner->setText("✓ Coordonnées détectées. Vérifiez avant de confirmer.");
        banner->setStyleSheet("background: #1e3a3a; color: #75D0C5; padding: 8px; border-radius: 4px;");
    } else {
        banner->setText("Coordonnées non détectées. Saisissez les valeurs manuellement.");
        banner->setStyleSheet("background: #3a1e1f; color: #c45255; padding: 8px; border-radius: 4px;");
    }

    // ── Confirm button — created before buildSide so updateConfirmButton works ─
    m_confirmBtn = new QPushButton("Confirmer", this);
    m_confirmBtn->setAutoDefault(false);
    m_confirmBtn->setDefault(false);
    m_confirmBtn->setStyleSheet(
        "QPushButton { background: #c45255; color: white;"
        "  border-radius: 10px; padding: 16px 40px;"
        "  font-family: 'Arial'; font-size: 15pt; font-weight: bold; }"
        "QPushButton:disabled { background: #5a2e2f; color: #888; }");
    connect(m_confirmBtn, &QPushButton::clicked, this, &QDialog::accept);

    // ── Scan mode: Annuler top-left so it stays reachable above the keyboard ──
    if (m_scanMode) {
        auto *cancelBtn = new QPushButton("Annuler", this);
        cancelBtn->setAutoDefault(false);
        cancelBtn->setDefault(false);
        connect(cancelBtn, &QPushButton::clicked, this, &QDialog::reject);

        auto *topRow = new QHBoxLayout();
        topRow->setSpacing(12);
        topRow->addWidget(cancelBtn);
        topRow->addWidget(banner, 1);
        mainLayout->addLayout(topRow);
    } else {
        mainLayout->addWidget(banner);
    }

    // ── Tabs ───────────────────────────────────────────────────────────────────
    // (buildSide may increment m_flaggedCount and connect valueChanged)
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

    // ── Bottom buttons ─────────────────────────────────────────────────────────
    if (m_scanMode) {
        // Annuler is already at the top; only Confirmer goes here.
        auto *bottomRow = new QHBoxLayout();
        bottomRow->addStretch();
        bottomRow->addWidget(m_confirmBtn);
        mainLayout->addLayout(bottomRow);

        // ── Scan-mode focus chain: x → y → z → ring → arc → hide keyboard ──
        // Pressing Return on each field clears its flag (even without retyping)
        // and moves focus to the next one; the last field hides the keyboard.
        auto chainSide = [this](TargetWidgets &w) {
            auto *x    = static_cast<AutoSelectSpinBox *>(w.x);
            auto *y    = static_cast<AutoSelectSpinBox *>(w.y);
            auto *z    = static_cast<AutoSelectSpinBox *>(w.z);
            auto *ring = static_cast<AutoSelectSpinBox *>(w.ring);
            auto *arc  = static_cast<AutoSelectSpinBox *>(w.arc);
            x->chainTo(y,       [this, sb = w.x]()    { clearFlag(sb); });
            y->chainTo(z,       [this, sb = w.y]()    { clearFlag(sb); });
            z->chainTo(ring,    [this, sb = w.z]()    { clearFlag(sb); });
            ring->chainTo(arc,  [this, sb = w.ring]() { clearFlag(sb); });
            arc->chainTo(nullptr, [this, sb = w.arc](){ clearFlag(sb); }); // last
        };
        chainSide(m_left);
        chainSide(m_right);

        // Auto-focus the x field whenever the user switches tabs.
        connect(tabs, &QTabWidget::currentChanged, this, [this](int idx) {
            QMetaObject::invokeMethod(this, [this, idx]() {
                ((idx == 0) ? m_left.x : m_right.x)->setFocus();
            }, Qt::QueuedConnection);
        });

        // Auto-focus left.x when the dialog first appears.
        QMetaObject::invokeMethod(this, [this]() { m_left.x->setFocus(); },
                                  Qt::QueuedConnection);
    } else {
        // Edit mode: Annuler and Confirmer side-by-side at the bottom.
        auto *cancelBtn = new QPushButton("Annuler", this);
        cancelBtn->setAutoDefault(false);
        cancelBtn->setDefault(false);
        connect(cancelBtn, &QPushButton::clicked, this, &QDialog::reject);

        auto *bottomRow = new QHBoxLayout();
        bottomRow->addWidget(cancelBtn);
        bottomRow->addStretch();
        bottomRow->addWidget(m_confirmBtn);
        mainLayout->addLayout(bottomRow);
    }

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
    // Mark every confirmed field with confidence 1.0 so that re-opening the dialog
    // (e.g. "Modifier le plan") pre-fills the previously confirmed values.
    if (w.x->value()    >= 0) { t.x_mm     = w.x->value();    t.confidence[0] = 1.f; }
    if (w.y->value()    >= 0) { t.y_mm     = w.y->value();    t.confidence[1] = 1.f; }
    if (w.z->value()    >= 0) { t.z_mm     = w.z->value();    t.confidence[2] = 1.f; }
    if (w.ring->value() >= 0) { t.ring_deg = w.ring->value(); t.confidence[3] = 1.f; }
    if (w.arc->value()  >= 0) { t.arc_deg  = w.arc->value();  t.confidence[4] = 1.f; }
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
