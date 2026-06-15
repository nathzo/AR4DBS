#include "DebugLogDialog.h"
#include "EmailLogger.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QTextEdit>
#include <QPushButton>
#include <QClipboard>
#include <QGuiApplication>
#include <QDesktopServices>
#include <QUrl>
#include <QDateTime>

DebugLogDialog::DebugLogDialog(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(tr("Debug Logs"));
    setGeometry(100, 100, 600, 400);

    auto *mainLayout = new QVBoxLayout;

    // Text edit to display logs
    m_logText = new QTextEdit;
    m_logText->setReadOnly(true);
    m_logText->setPlainText(EmailLogger::getAllLogs());
    m_logText->setStyleSheet("font-family: monospace; font-size: 9pt;");
    mainLayout->addWidget(m_logText);

    // Buttons layout
    auto *btnLayout = new QHBoxLayout;

    m_btnCopy = new QPushButton(tr("Copy to Clipboard"));
    connect(m_btnCopy, &QPushButton::clicked, this, &DebugLogDialog::onCopyToClipboard);
    btnLayout->addWidget(m_btnCopy);

    m_btnMail = new QPushButton(tr("Open Mail App"));
    connect(m_btnMail, &QPushButton::clicked, this, &DebugLogDialog::onOpenMailApp);
    btnLayout->addWidget(m_btnMail);

    m_btnClear = new QPushButton(tr("Clear Logs"));
    connect(m_btnClear, &QPushButton::clicked, this, &DebugLogDialog::onClear);
    btnLayout->addWidget(m_btnClear);

    auto *btnClose = new QPushButton(tr("Close"));
    connect(btnClose, &QPushButton::clicked, this, &QDialog::accept);
    btnLayout->addWidget(btnClose);

    mainLayout->addLayout(btnLayout);
    setLayout(mainLayout);

    EmailLogger::logEvent("DebugLogDialog", "opened");
}

void DebugLogDialog::onCopyToClipboard()
{
    QString logs = EmailLogger::getAllLogs();
    QGuiApplication::clipboard()->setText(logs);
    EmailLogger::logEvent("DebugLogDialog", "logs copied to clipboard");

    // Update the text to show it was copied
    m_logText->setPlainText("✓ Logs copied to clipboard!\n\n" + logs);
}

void DebugLogDialog::onOpenMailApp()
{
    QString logs = EmailLogger::getLogsForEmail();
    QGuiApplication::clipboard()->setText(logs);
    EmailLogger::logEvent("DebugLogDialog", "opening mail app");

    // Open default mail client with subject line
    // Note: mailto URLs don't reliably support body parameter, so we copy to clipboard
    QString subject = "DBSAR Debug Logs - " + QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss");
    QString mailtoUrl = "mailto:?subject=" + subject.replace(" ", "%20");
    QDesktopServices::openUrl(QUrl(mailtoUrl));

    // Show message that logs are in clipboard
    m_logText->setPlainText("✓ Mail app opened!\n\nThe logs have been copied to your clipboard.\n"
                           "Paste them into the email body.\n\n" + logs);
}

void DebugLogDialog::onClear()
{
    EmailLogger::clearLogs();
    EmailLogger::logEvent("DebugLogDialog", "logs cleared");
    m_logText->setPlainText("✓ All logs cleared!");
}
