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

static QDoubleSpinBox *makeSpinBox(double min, double max, double val,
                                   const QString &suffix, QWidget *parent)
{
    auto *sb = new QDoubleSpinBox(parent);
    sb->setRange(min, max);
    sb->setDecimals(1);
    sb->setSingleStep(0.1);
    sb->setValue(val);
    sb->setSuffix(suffix);
    sb->setMinimumWidth(110);
    return sb;
}

ConfirmPlanDialog::TargetWidgets ConfirmPlanDialog::buildSide(
    const QString &title, const LeksellTarget &t, QWidget *parent)
{
    TargetWidgets w;

    auto *box = new QGroupBox(title, parent);
    box->setCheckable(false);
    auto *form = new QFormLayout(box);
    form->setLabelAlignment(Qt::AlignRight);

    w.enabled = new QCheckBox("Activer", parent);
    w.enabled->setChecked(t.valid);

    w.x    = makeSpinBox(0, 300, t.x_mm,     " mm",  parent);
    w.y    = makeSpinBox(0, 300, t.y_mm,     " mm",  parent);
    w.z    = makeSpinBox(0, 200, t.z_mm,     " mm",  parent);
    w.ring = makeSpinBox(0, 360, t.ring_deg, " °",   parent);
    w.arc  = makeSpinBox(0, 180, t.arc_deg,  " °",   parent);

    form->addRow(w.enabled);
    form->addRow("X (mm) :",       w.x);
    form->addRow("Y (mm) :",       w.y);
    form->addRow("Z (mm) :",       w.z);
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
    updateEnabled(t.valid);
    QObject::connect(w.enabled, &QCheckBox::toggled, box, updateEnabled);

    // Store the group box as the managed widget — caller places it
    // We repurpose parent->layout() indirectly; the caller is responsible
    // for adding box to a layout.
    // To allow the caller to retrieve the box, we set it as a child named
    // after the title.
    box->setObjectName(title);

    return w;
}

ConfirmPlanDialog::ConfirmPlanDialog(const SurgicalPlan &initial, QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle("Confirmer le plan chirurgical");
    // No fixed minimum width — must fit a vertical iPhone screen
    setStyleSheet(
        "QDialog, QGroupBox, QWidget {"
        "  background-color: #1a1b1d;"
        "  color: #e0e0e0;"
        "  font-size: 11pt;"
        "}"
        "QGroupBox {"
        "  border: 1px solid #75D0C5;"
        "  border-radius: 6px;"
        "  margin-top: 8px;"
        "  padding-top: 6px;"
        "  font-weight: bold;"
        "  color: #75D0C5;"
        "}"
        "QGroupBox::title { subcontrol-origin: margin; left: 10px; }"
        "QTabWidget::pane {"
        "  border: 1px solid #75D0C5;"
        "  border-radius: 6px;"
        "}"
        "QTabBar::tab {"
        "  background: #2a2b2d;"
        "  color: #e0e0e0;"
        "  padding: 12px 28px;"
        "  font-size: 12pt;"
        "  border: 1px solid #444;"
        "  border-bottom: none;"
        "  border-top-left-radius: 6px;"
        "  border-top-right-radius: 6px;"
        "}"
        "QTabBar::tab:selected {"
        "  background: #75D0C5;"
        "  color: #1a1b1d;"
        "  font-weight: bold;"
        "}"
        "QDoubleSpinBox {"
        "  background: #2a2b2d;"
        "  color: #e0e0e0;"
        "  border: 1px solid #444;"
        "  border-radius: 4px;"
        "  padding: 3px 6px;"
        "}"
        "QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { width: 0; border: none; }"
        "QCheckBox { color: #75D0C5; font-weight: bold; }"
        "QPushButton {"
        "  border-radius: 8px;"
        "  padding: 10px 28px;"
        "  font-size: 12pt;"
        "  font-weight: bold;"
        "}"
        "QPushButton[text='Confirmer'] { background: #c45255; color: white; }"
        "QPushButton[text='Annuler']   { background: #75D0C5; color: #1a1b1d; }"
    );

    auto *mainLayout = new QVBoxLayout(this);

    // OCR status banner
    auto *banner = new QLabel(this);
    if (initial.hasAny()) {
        banner->setText("✓ Coordonnées détectées automatiquement. Vérifiez avant de confirmer.");
        banner->setStyleSheet("background: #1e3a3a; color: #75D0C5; padding: 8px; border-radius: 4px;");
    } else {
        banner->setText("Coordonnées non détectées. Saisissez les valeurs manuellement.");
        banner->setStyleSheet("background: #3a1e1f; color: #c45255; padding: 8px; border-radius: 4px;");
    }
    mainLayout->addWidget(banner);

    // Two tabs — one per side — so the form fits a vertical iPhone screen
    auto *tabs = new QTabWidget(this);
    mainLayout->addWidget(tabs);

    // Left side tab
    auto *leftPage = new QWidget(tabs);
    auto *leftLayout = new QVBoxLayout(leftPage);
    leftLayout->setContentsMargins(8, 8, 8, 8);
    m_left = buildSide("Gauche (G)", initial.left, leftPage);
    leftLayout->addWidget(findChild<QGroupBox *>("Gauche (G)"));
    leftLayout->addStretch();
    tabs->addTab(leftPage, "Gauche (G)");

    // Right side tab
    auto *rightPage = new QWidget(tabs);
    auto *rightLayout = new QVBoxLayout(rightPage);
    rightLayout->setContentsMargins(8, 8, 8, 8);
    m_right = buildSide("Droite (D)", initial.right, rightPage);
    rightLayout->addWidget(findChild<QGroupBox *>("Droite (D)"));
    rightLayout->addStretch();
    tabs->addTab(rightPage, "Droite (D)");

    // Buttons
    auto *buttons = new QDialogButtonBox(
        QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    buttons->button(QDialogButtonBox::Ok)->setText("Confirmer");
    buttons->button(QDialogButtonBox::Cancel)->setText("Annuler");
    mainLayout->addWidget(buttons);

    connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
}

LeksellTarget ConfirmPlanDialog::readWidgets(const TargetWidgets &w)
{
    LeksellTarget t;
    t.valid    = w.enabled->isChecked();
    t.x_mm     = w.x->value();
    t.y_mm     = w.y->value();
    t.z_mm     = w.z->value();
    t.ring_deg = w.ring->value();
    t.arc_deg  = w.arc->value();
    return t;
}

SurgicalPlan ConfirmPlanDialog::plan() const
{
    SurgicalPlan p;
    p.left  = readWidgets(m_left);
    p.right = readWidgets(m_right);
    return p;
}
