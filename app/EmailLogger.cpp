#include "EmailLogger.h"
#include <QDateTime>
#include <QDebug>
#include <QSettings>

// Static member initialization
QStringList EmailLogger::s_logQueue;
QMutex EmailLogger::s_mutex;
const char *EmailLogger::SETTINGS_KEY = "debugLogs/queue";
const int EmailLogger::MAX_LOG_LINES = 10000;

void EmailLogger::initialize()
{
    QMutexLocker lock(&s_mutex);
    loadLogs();
    addLog("[" + QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss.zzz") + "] APP STARTED");
}

void EmailLogger::logEvent(const QString &dialog, const QString &event)
{
    QMutexLocker lock(&s_mutex);
    QString timestamp = QDateTime::currentDateTime().toString("hh:mm:ss.zzz");
    QString logLine = "[" + timestamp + "] " + dialog + ": " + event;
    addLog(logLine);
}

void EmailLogger::addLog(const QString &logLine)
{
    // Keep only the last MAX_LOG_LINES to avoid unbounded growth
    if (s_logQueue.size() >= MAX_LOG_LINES) {
        s_logQueue.removeFirst();
    }
    s_logQueue.append(logLine);
    qDebug() << logLine;  // Also print to debug console
    persistLogs();
}

void EmailLogger::persistLogs()
{
    // Save logs to QSettings (called after each log addition)
    QSettings settings;
    settings.setValue(SETTINGS_KEY, s_logQueue);
}

void EmailLogger::loadLogs()
{
    // Load logs from QSettings on app start
    QSettings settings;
    s_logQueue = settings.value(SETTINGS_KEY, QStringList()).toStringList();
}

QString EmailLogger::getAllLogs()
{
    QMutexLocker lock(&s_mutex);
    return s_logQueue.join("\n");
}

void EmailLogger::clearLogs()
{
    QMutexLocker lock(&s_mutex);
    s_logQueue.clear();
    QSettings settings;
    settings.remove(SETTINGS_KEY);
    qDebug() << "EmailLogger: logs cleared";
}
