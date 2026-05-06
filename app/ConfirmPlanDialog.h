#pragma once
#include <QDialog>
#include "core/math/SurgicalPlan.h"

class QDoubleSpinBox;
class QCheckBox;
class QPushButton;
class QKeyEvent;
class QShowEvent;
class QPaintEvent;

// Shows the surgical plan detected by OCR (or empty fields for manual entry).
// Fields with OCR confidence < 70 % are highlighted in red and block confirmation
// until the user explicitly replaces the value.
class ConfirmPlanDialog : public QDialog
{
    Q_OBJECT
public:
    explicit ConfirmPlanDialog(const SurgicalPlan &initial,
                               QWidget            *parent = nullptr);

    SurgicalPlan plan() const;

private:
    struct TargetWidgets {
        QCheckBox      *enabled = nullptr;
        QDoubleSpinBox *x       = nullptr;
        QDoubleSpinBox *y       = nullptr;
        QDoubleSpinBox *z       = nullptr;
        QDoubleSpinBox *ring    = nullptr;
        QDoubleSpinBox *arc     = nullptr;
    };

    TargetWidgets  m_left, m_right;
    QPushButton   *m_confirmBtn   = nullptr;
    int            m_flaggedCount = 0;

    TargetWidgets buildSide(const QString       &title,
                            const LeksellTarget &initial,
                            class QWidget       *parent);
    static LeksellTarget readWidgets(const TargetWidgets &w);

    // Called from valueChanged on any flagged spinbox.
    void clearFlag(QDoubleSpinBox *sb);
    // Re-evaluates whether Confirmer should be enabled.
    void updateConfirmButton();

protected:
    void paintEvent(QPaintEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;
    void showEvent(QShowEvent *event) override;
};
