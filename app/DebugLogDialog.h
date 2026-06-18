#pragma once
#include <QDialog>

class QTextEdit;
class QPushButton;

class DebugLogDialog : public QDialog {
    Q_OBJECT
public:
    explicit DebugLogDialog(QWidget *parent = nullptr);
    ~DebugLogDialog();

protected:
    bool event(QEvent *event) override;

private:
    void onCopyToClipboard();
    void onClear();

    QTextEdit *m_logText = nullptr;
    QPushButton *m_btnCopy = nullptr;
    QPushButton *m_btnClear = nullptr;
};
