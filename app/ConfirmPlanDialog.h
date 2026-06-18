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
//
// Mode::Scan  – used right after OCR: ALL fields are flagged red and the user
//               must confirm each one in sequence (Return advances focus; last
//               field hides the keyboard).  "Annuler" sits at the top-left so
//               it stays reachable while the virtual keyboard is visible.
// Mode::Edit  – used from "Modifier le plan": pre-filled with previously
//               confirmed values; only low-confidence fields are flagged.
class ConfirmPlanDialog : public QDialog
{
    Q_OBJECT
public:
    enum class Mode { Scan, Edit };

    explicit ConfirmPlanDialog(const SurgicalPlan &initial,
                               Mode                mode,
                               QWidget            *parent = nullptr);
    ~ConfirmPlanDialog();

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
    bool           m_scanMode     = false;

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
    bool event(QEvent *event) override;
};
