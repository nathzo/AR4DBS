#pragma once
#include <QMainWindow>
#include <QThread>
#include "core/math/SurgicalPlan.h"
#include "core/rendering/OverlayRenderer.h"
#include "app/SettingsDialog.h"

class QLabel;
class QStackedWidget;
class QPushButton;
class GLWidget;
class AppController;
class SettingsDialog;

#ifdef Q_OS_IOS
class ARKitSession;
#else
class DesktopCamera;
#endif

#ifdef FEATURE_PLAN_SCANNER
class StartScreen;
class ScanScreen;
class ConfirmPlanDialog;
#endif

class ARRotatedStrip;

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

    GLWidget        *m_glWidget         = nullptr;
    AppController   *m_controller       = nullptr;
    QThread         *m_controllerThread = nullptr;
    QPushButton     *m_btnEditPlan      = nullptr;
    QPushButton     *m_btnBackToMenu    = nullptr;
    QPushButton     *m_btnSettings      = nullptr;
    ARRotatedStrip  *m_arStrip          = nullptr;
    SurgicalPlan     m_currentPlan;

    // Local copies of current settings — passed to SettingsDialog on open
    OverlayRenderer::Style m_renderStyle;
    double                 m_reprojThreshold = 1.0;
    TagPositions           m_tagPositions;

    void openSettings();

#ifdef Q_OS_IOS
    QLabel *m_calibLabel    = nullptr;
    bool    m_arLocked      = false;
    bool    m_arCalibrating = false;
    void    setArLocked(bool locked);
    void    setArCalibrating(bool calibrating);
#endif

#ifdef Q_OS_IOS
    ARKitSession   *m_arCamera   = nullptr;
#else
    DesktopCamera  *m_arCamera   = nullptr;
#endif
};
