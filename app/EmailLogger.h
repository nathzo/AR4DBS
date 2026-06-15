#pragma once
#include <QString>
#include <QDateTime>
#include <QMutex>
#include <QStringList>

class EmailLogger {
public:
    // Initialize with Gmail credentials (app password, not regular password)
    static void initialize(const QString &emailAddress, const QString &appPassword);

    // Log an event (thread-safe)
    static void logEvent(const QString &dialog, const QString &event);

    // Send all collected logs via email and clear the queue
    // Called on app close (before destructor of MainWindow)
    static void sendOnExit();

    // Get current logs as string (for debugging)
    static QString getCurrentLogs();

private:
    static QString s_emailAddress;
    static QString s_appPassword;
    static QStringList s_logQueue;
    static QMutex s_mutex;
    static bool s_initialized;

    static void addLog(const QString &logLine);
    static bool sendViaSmtp(const QString &recipient, const QString &subject, const QString &body);
};
