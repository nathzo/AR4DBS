#pragma once
#include <QWidget>

// First screen shown on launch.
// "Nouvelle chirurgie" → scan + confirm flow
// "Mode test AR"       → skip scan, go straight to AR with the last saved plan
class StartScreen : public QWidget
{
    Q_OBJECT
public:
    explicit StartScreen(QWidget *parent = nullptr);

signals:
    void newSurgeryRequested();
    void directARRequested();
};
