#include "DebugLogDialog.h"
#include "EmailLogger.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QTextEdit>
#include <QPushButton>
#include <QLabel>
#include <QClipboard>
#include <QGuiApplication>
#include <QScreen>
#include <QDateTime>

DebugLogDialog::DebugLogDialog(QWidget *parent)
    : QDialog(parent)
{
    EmailLogger::logEvent("DebugLogDialog", "constructor: started");
    setWindowTitle(tr("Debug Logs"));

    // Fill the entire screen
    const QRect ag = QGuiApplication::primaryScreen()->availableGeometry();
    setGeometry(ag);

    // Set window flags for proper modal behavior
    setWindowFlags(Qt::Dialog);
    setAttribute(Qt::WA_TranslucentBackground);

    auto *mainLayout = new QVBoxLayout;
    mainLayout->setContentsMargins(12, 12, 12, 12);
    mainLayout->setSpacing(10);

    // ── Header with title and close button ──────────────────────────────────
    auto *headerLayout = new QHBoxLayout;
    auto *titleLabel = new QLabel(tr("Debug Logs"));
    titleLabel->setStyleSheet("color: white; font-weight: bold; font-size: 14pt;");
    headerLayout->addWidget(titleLabel);
    headerLayout->addStretch();

    auto *btnCloseTop = new QPushButton("✕");
    btnCloseTop->setMaximumWidth(40);
    btnCloseTop->setMaximumHeight(40);
    btnCloseTop->setStyleSheet(
        "QPushButton {"
        "  background: #555; color: white;"
        "  border-radius: 4px; padding: 4px;"
        "  font-weight: bold; font-size: 16pt;"
        "}"
        "QPushButton:hover { background: #777; }"
        "QPushButton:pressed { background: #333; }");
    connect(btnCloseTop, &QPushButton::clicked, this, &QDialog::accept);
    headerLayout->addWidget(btnCloseTop);

    mainLayout->addLayout(headerLayout);

    // ── Text edit to display logs ──────────────────────────────────────────
    EmailLogger::logEvent("DebugLogDialog", "constructor: creating text edit widget");
    m_logText = new QTextEdit;
    m_logText->setReadOnly(true);
    m_logText->setPlainText(EmailLogger::getAllLogs());
    m_logText->setStyleSheet(
        "QTextEdit {"
        "  background: #1a1a1a; color: #e0e0e0;"
        "  border: 1px solid #444; border-radius: 4px;"
        "  font-family: monospace; font-size: 9pt;"
        "  padding: 6px;"
        "}");
    mainLayout->addWidget(m_logText);

    // ── Buttons layout (vertical stack so all fit on narrow screens) ────────
    auto *btnLayout = new QVBoxLayout;
    btnLayout->setSpacing(8);

    EmailLogger::logEvent("DebugLogDialog", "constructor: creating buttons");
    m_btnCopy = new QPushButton(tr("Copy to Clipboard"));
    m_btnCopy->setMinimumHeight(36);
    m_btnCopy->setStyleSheet(
        "QPushButton {"
        "  background: #DE5F5E; color: white;"  // IMPULSE_RED
        "  border-radius: 6px; padding: 8px;"
        "  font-size: 11pt; font-weight: bold;"
        "}"
        "QPushButton:pressed { background: #b84a48; }");
    connect(m_btnCopy, &QPushButton::clicked, this, &DebugLogDialog::onCopyToClipboard);
    btnLayout->addWidget(m_btnCopy);

    m_btnClear = new QPushButton(tr("Clear Logs"));
    m_btnClear->setMinimumHeight(36);
    m_btnClear->setStyleSheet(
        "QPushButton {"
        "  background: #75D0C5; color: black;"  // ARC_BLUE
        "  border-radius: 6px; padding: 8px;"
        "  font-size: 11pt; font-weight: bold;"
        "}"
        "QPushButton:pressed { background: #5ab8ae; }");
    connect(m_btnClear, &QPushButton::clicked, this, &DebugLogDialog::onClear);
    btnLayout->addWidget(m_btnClear);

    mainLayout->addLayout(btnLayout);
    setLayout(mainLayout);

    EmailLogger::logEvent("DebugLogDialog", "constructor: complete, showing dialog");
}

DebugLogDialog::~DebugLogDialog()
{
    EmailLogger::logEvent("DebugLogDialog", "destructor: entered");
    EmailLogger::logEvent("DebugLogDialog", "destructor: about to return");
}

void DebugLogDialog::onCopyToClipboard()
{
    EmailLogger::logEvent("DebugLogDialog", "copy button: clicked");
    QString logs = EmailLogger::getAllLogs();
    QGuiApplication::clipboard()->setText(logs);
    EmailLogger::logEvent("DebugLogDialog", "copy button: logs copied to clipboard (" + QString::number(logs.length()) + " chars)");

    // Update the text to show it was copied
    m_logText->setPlainText("✓ Logs copied to clipboard!\n\n" + logs);
    EmailLogger::logEvent("DebugLogDialog", "copy button: display updated");
}

void DebugLogDialog::onClear()
{
    EmailLogger::logEvent("DebugLogDialog", "clear button: clicked");
    EmailLogger::clearLogs();
    EmailLogger::logEvent("DebugLogDialog", "clear button: logs cleared from memory and QSettings");
    m_logText->setPlainText("✓ All logs cleared!");
    EmailLogger::logEvent("DebugLogDialog", "clear button: display updated");
}
