#pragma once
#include <QWidget>
#include "core/math/SurgicalPlan.h"

class GLWidget;

// Shows a live camera preview and a "Capturer" button.
// When the user taps Capturer, the current frame is processed by PlanScanner
// and planDetected is emitted (targets may be invalid — caller shows dialog).
class ScanScreen : public QWidget
{
    Q_OBJECT
public:
    explicit ScanScreen(QWidget *parent = nullptr);
    ~ScanScreen();

    void startCamera();
    void stopCamera();

signals:
    void planDetected(const SurgicalPlan &plan);
    void cancelled();

private slots:
    void onCapture();

private:
    struct Impl;
    Impl *m_impl;
};
