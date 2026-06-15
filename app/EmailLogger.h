#pragma once
#include <QString>
#include <QDateTime>
#include <QMutex>
#include <QStringList>

class EmailLogger {
public:
    // Log an event (thread-safe)
    static void logEvent(const QString &dialog, const QString &event);

    // Get all accumulated logs as a single string
    static QString getAllLogs();

    // Get logs formatted as email body (with summary)
    static QString getLogsForEmail();

    // Clear the log queue (call after user sends email)
    static void clearLogs();

private:
    static QStringList s_logQueue;
    static QMutex s_mutex;

    static void addLog(const QString &logLine);
};
