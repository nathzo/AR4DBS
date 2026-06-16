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

    // Clear the log queue and QSettings
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
