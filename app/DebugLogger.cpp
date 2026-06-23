#include "DebugLogger.h"
#include <QDateTime>
#include <QDebug>
#include <QSettings>

// Static member initialization
QStringList DebugLogger::s_logQueue;
QMutex DebugLogger::s_mutex;
const char *DebugLogger::SETTINGS_KEY = "debugLogs/queue";
const int DebugLogger::MAX_LOG_LINES = 10000;

void DebugLogger::initialize()
{
    QMutexLocker lock(&s_mutex);
    loadLogs();
    addLog("[" + QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss.zzz") + "] APP STARTED");
}

void DebugLogger::logEvent(const QString &dialog, const QString &event)
{
    QMutexLocker lock(&s_mutex);
    QString timestamp = QDateTime::currentDateTime().toString("hh:mm:ss.zzz");
    QString logLine = "[" + timestamp + "] " + dialog + ": " + event;
    addLog(logLine);
}

static int s_logsSinceLastPersist = 0;
static const int PERSIST_BATCH_SIZE = 50;  // Only persist to storage every 50 logs

void DebugLogger::addLog(const QString &logLine)
{
    // When logs reach the limit, trim the oldest 20% to make room for new logs
    // This is more efficient than removing one line at a time
    if (s_logQueue.size() >= MAX_LOG_LINES) {
        const int linesToRemove = MAX_LOG_LINES / 5;  // Remove oldest 20% (2000 lines)
        const QString clearMarker = QString(
            "[%1] ════════════════════════════════════════════════════════════════"
            "[%2] ╔══ LOGS ROTATED: Removed %3 oldest lines (buffer at capacity) ══╗"
            "[%4] ╚════════════════════════════════════════════════════════════════╝")
            .arg(QDateTime::currentDateTime().toString("hh:mm:ss.zzz"))
            .arg(QDateTime::currentDateTime().toString("hh:mm:ss.zzz"))
            .arg(linesToRemove)
            .arg(QDateTime::currentDateTime().toString("hh:mm:ss.zzz"));

        // Remove oldest lines in batch
        for (int i = 0; i < linesToRemove && !s_logQueue.isEmpty(); ++i) {
            s_logQueue.removeFirst();
        }

        // Add a marker so we can see in the logs when rotation happened
        s_logQueue.append(clearMarker);

        qDebug() << "DebugLogger: LOG ROTATION - removed" << linesToRemove << "oldest lines,"
                 << "current size:" << s_logQueue.size() << "/" << MAX_LOG_LINES;

        // Force persist on rotation to ensure it's saved
        persistLogs();
        s_logsSinceLastPersist = 0;
        return;
    }

    s_logQueue.append(logLine);
    qDebug() << logLine;  // Also print to debug console

    // Only persist to storage in batches to reduce I/O pressure
    // This prevents storage bottleneck on devices with limited I/O bandwidth
    ++s_logsSinceLastPersist;
    if (s_logsSinceLastPersist >= PERSIST_BATCH_SIZE) {
        persistLogs();
        s_logsSinceLastPersist = 0;
    }
}

void DebugLogger::persistLogs()
{
    // Save logs to QSettings (called after each log addition)
    QSettings settings;
    settings.setValue(SETTINGS_KEY, s_logQueue);
}

void DebugLogger::loadLogs()
{
    // Load logs from QSettings on app start
    QSettings settings;
    s_logQueue = settings.value(SETTINGS_KEY, QStringList()).toStringList();
}

QString DebugLogger::getAllLogs()
{
    QMutexLocker lock(&s_mutex);
    return s_logQueue.join("\n");
}

int DebugLogger::getLogLineCount()
{
    QMutexLocker lock(&s_mutex);
    return s_logQueue.size();
}

void DebugLogger::clearLogs()
{
    QMutexLocker lock(&s_mutex);
    s_logQueue.clear();
    s_logsSinceLastPersist = 0;
    QSettings settings;
    settings.remove(SETTINGS_KEY);
    qDebug() << "DebugLogger: logs cleared";
}

void DebugLogger::flushLogs()
{
    QMutexLocker lock(&s_mutex);
    if (s_logsSinceLastPersist > 0) {
        persistLogs();
        s_logsSinceLastPersist = 0;
    }
}
