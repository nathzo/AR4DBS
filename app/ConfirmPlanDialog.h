#pragma once
#include <QDialog>
#include "core/math/SurgicalPlan.h"

class QDoubleSpinBox;
class QCheckBox;
class QKeyEvent;

// Shows the surgical plan detected by OCR (or empty fields for manual entry).
// The user can edit any value before confirming.
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

    TargetWidgets m_left, m_right;

    TargetWidgets buildSide(const QString       &title,
                            const LeksellTarget &initial,
                            class QWidget       *parent);
    static LeksellTarget readWidgets(const TargetWidgets &w);

protected:
    void keyPressEvent(QKeyEvent *event) override;
};
