#pragma once
#include <QDialog>

class QTextEdit;
class QPushButton;

class DebugLogDialog : public QDialog {
    Q_OBJECT
public:
    explicit DebugLogDialog(QWidget *parent = nullptr);
    ~DebugLogDialog();

private:
    void onCopyToClipboard();
    void onOpenMailApp();
    void onClear();

    QTextEdit *m_logText = nullptr;
    QPushButton *m_btnCopy = nullptr;
    QPushButton *m_btnMail = nullptr;
    QPushButton *m_btnClear = nullptr;
};
