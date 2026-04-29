#pragma once
#include <QMainWindow>
#include "core/math/SurgicalPlan.h"

class QStackedWidget;
class QPushButton;
class GLWidget;
class AppController;

#ifdef Q_OS_IOS
class IOSCamera;
#else
class DesktopCamera;
#endif

#ifdef FEATURE_PLAN_SCANNER
class StartScreen;
class ScanScreen;
class ConfirmPlanDialog;
#endif

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

protected:
    void closeEvent(QCloseEvent *event) override;

private:
#ifdef FEATURE_PLAN_SCANNER
    void startAR();
    void onPlanDetected(const SurgicalPlan &plan);
    void editPlan();

    StartScreen    *m_startScreen  = nullptr;
    ScanScreen     *m_scanScreen   = nullptr;
    QStackedWidget *m_stack        = nullptr;
#endif

    GLWidget       *m_glWidget        = nullptr;
    AppController  *m_controller      = nullptr;
    QPushButton    *m_btnNextTarget   = nullptr;
    QPushButton    *m_btnEditPlan     = nullptr;
    QPushButton    *m_btnBackToMenu   = nullptr;
    SurgicalPlan    m_currentPlan;

#ifdef Q_OS_IOS
    IOSCamera      *m_arCamera   = nullptr;
#else
    DesktopCamera  *m_arCamera   = nullptr;
#endif
};
