#include "EmailLogger.h"
#include <QDateTime>
#include <QDebug>

// Static member initialization
QStringList EmailLogger::s_logQueue;
QMutex EmailLogger::s_mutex;

void EmailLogger::logEvent(const QString &dialog, const QString &event)
{
    QMutexLocker lock(&s_mutex);
    QString timestamp = QDateTime::currentDateTime().toString("hh:mm:ss.zzz");
    QString logLine = "[" + timestamp + "] " + dialog + ": " + event;
    addLog(logLine);
}

void EmailLogger::addLog(const QString &logLine)
{
    s_logQueue.append(logLine);
    qDebug() << logLine;  // Also print to debug console
}

QString EmailLogger::getAllLogs()
{
    QMutexLocker lock(&s_mutex);
    return s_logQueue.join("\n");
}

QString EmailLogger::getLogsForEmail()
{
    QMutexLocker lock(&s_mutex);

    QString subject = "[DBSAR Debug Log] " + QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss");
    QString body = "Application Session Log\n";
    body += "=======================\n";
    body += "Generated: " + QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss") + "\n";
    body += "Total events: " + QString::number(s_logQueue.size()) + "\n";
    body += "\n";
    body += s_logQueue.join("\n");

    return body;
}

void EmailLogger::clearLogs()
{
    QMutexLocker lock(&s_mutex);
    s_logQueue.clear();
    qDebug() << "EmailLogger: logs cleared after sending";
}
