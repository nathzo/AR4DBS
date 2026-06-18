#include "DebugLogDialog.h"
#include "DebugLogger.h"
#include <QVBoxLayout>
#include <sys/resource.h>
#include <QHBoxLayout>
#include <QTextEdit>
#include <QPushButton>
#include <QLabel>
#include <QClipboard>
#include <QGuiApplication>
#include <QScreen>
#include <QDateTime>

// ── Memory logging helper ────────────────────────────────────────────────────

static void logMemoryState(const QString &context)
{
#ifdef Q_OS_IOS
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        long maxRss = usage.ru_maxrss;
        DebugLogger::logEvent("Memory", context + ": peak_rss=" + QString::number(maxRss / 1024 / 1024) + "MB");
    }
#endif
}

DebugLogDialog::DebugLogDialog(QWidget *parent)
    : QDialog(parent)
{
    DebugLogger::logEvent("DebugLogDialog", "constructor: started");
    setWindowTitle(tr("Debug Logs"));

    // Fill the entire screen (100%)
    const QRect screenGeom = QGuiApplication::primaryScreen()->geometry();
    setGeometry(screenGeom);

    // Set window flags for proper fullscreen behavior
    setWindowFlags(Qt::Dialog | Qt::FramelessWindowHint);
    setAttribute(Qt::WA_TranslucentBackground);

    auto *mainLayout = new QVBoxLayout;
    mainLayout->setContentsMargins(12, 12, 12, 12);
    mainLayout->setSpacing(10);

    // ── Header with title and close button ──────────────────────────────────
    auto *headerLayout = new QHBoxLayout;
    int lineCount = DebugLogger::getLogLineCount();
    auto *titleLabel = new QLabel(tr("Debug Logs (%1 lines)").arg(lineCount));
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
    DebugLogger::logEvent("DebugLogDialog", "constructor: creating text edit widget");
    m_logText = new QTextEdit;
    m_logText->setReadOnly(true);
    m_logText->setPlainText(DebugLogger::getAllLogs());
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

    DebugLogger::logEvent("DebugLogDialog", "constructor: creating buttons");
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

    DebugLogger::logEvent("DebugLogDialog", "constructor: complete, showing dialog");
}

DebugLogDialog::~DebugLogDialog()
{
    logMemoryState("DebugLogDialog::destructor");
    DebugLogger::logEvent("DebugLogDialog", "destructor: entered, thread=" + QString::number((quint64)QThread::currentThreadId()));

    int childCount = findChildren<QObject*>().count();
    DebugLogger::logEvent("DebugLogDialog", "destructor: found " + QString::number(childCount) + " child objects");

    // Disable accessibility on all children to prevent iOS crash when destroying buttons
    int widgetCount = 0;
    for (QObject *child : findChildren<QObject*>()) {
        if (QWidget *w = qobject_cast<QWidget*>(child)) {
            w->setAccessibleDescription(QString());
            widgetCount++;
        }
    }
    DebugLogger::logEvent("DebugLogDialog", "destructor: cleared accessibility on " + QString::number(widgetCount) + " widgets");
    DebugLogger::logEvent("DebugLogDialog", "destructor: about to return");
}

bool DebugLogDialog::event(QEvent *event)
{
    // Block all accessibility events to prevent iOS crash during widget destruction
    // Accessibility events are in the range 1000-1030
    if (event->type() >= 1000 && event->type() <= 1030) {
        return true;  // Consume the event, don't process it
    }
    return QDialog::event(event);
}

void DebugLogDialog::onCopyToClipboard()
{
    DebugLogger::logEvent("DebugLogDialog", "copy button: clicked");
    QString logs = DebugLogger::getAllLogs();
    QGuiApplication::clipboard()->setText(logs);
    DebugLogger::logEvent("DebugLogDialog", "copy button: logs copied to clipboard (" + QString::number(logs.length()) + " chars)");

    // Update the text to show it was copied
    m_logText->setPlainText("✓ Logs copied to clipboard!\n\n" + logs);
    DebugLogger::logEvent("DebugLogDialog", "copy button: display updated");
}

void DebugLogDialog::onClear()
{
    DebugLogger::logEvent("DebugLogDialog", "clear button: clicked");
    DebugLogger::clearLogs();
    DebugLogger::logEvent("DebugLogDialog", "clear button: logs cleared from memory and QSettings");
    m_logText->setPlainText("✓ All logs cleared!");
    DebugLogger::logEvent("DebugLogDialog", "clear button: display updated");
}
