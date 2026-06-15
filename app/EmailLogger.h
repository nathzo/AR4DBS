#pragma once
#include <QString>
#include <QDateTime>
#include <QMutex>
#include <QStringList>

class EmailLogger {
public:
    // Initialize — loads persisted logs from QSettings
    static void initialize();

    // Log an event (thread-safe, persisted to QSettings)
    static void logEvent(const QString &dialog, const QString &event);

    // Get all accumulated logs as a single string
    static QString getAllLogs();

    // Get logs formatted as email body (with summary)
    static QString getLogsForEmail();

    // Clear the log queue and QSettings (call after user sends email)
    static void clearLogs();

private:
    static QStringList s_logQueue;
    static QMutex s_mutex;
    static const char *SETTINGS_KEY;
    static const int MAX_LOG_LINES;

    static void addLog(const QString &logLine);
    static void persistLogs();
    static void loadLogs();
};
