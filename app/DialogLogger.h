#pragma once
#include <QString>
#include <QDateTime>
#include <QFile>
#include <QDir>
#include <QStandardPaths>
#include <QTextStream>
#include <QMutex>
#include <QDebug>

class DialogLogger {
private:
    static QString getLogFilePath() {
        QString logDir = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation);

        // Fallback if Documents location is empty
        if (logDir.isEmpty()) {
            logDir = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
        }

        // Ensure directory exists
        QDir dir(logDir);
        if (!dir.exists()) {
            dir.mkpath(".");
        }

        return QDir(logDir).filePath("dbsar_debug.log");
    }

public:
    static void clearLog() {
        QString logFile = getLogFilePath();
        QFile file(logFile);
        if (!file.remove() && file.exists()) {
            qDebug() << "Failed to remove log file:" << logFile;
        }
    }

    static void logEvent(const QString &dialog, const QString &event) {
        static QMutex mutex;
        QMutexLocker lock(&mutex);

        QString logFile = getLogFilePath();
        QFile file(logFile);

        if (!file.open(QIODevice::Append | QIODevice::Text)) {
            qDebug() << "Failed to open log file for writing:" << logFile << "Error:" << file.errorString();
            return;
        }

        QTextStream stream(&file);
        QString timestamp = QDateTime::currentDateTime().toString("hh:mm:ss.zzz");
        stream << "[" << timestamp << "] " << dialog << ": " << event << "\n";
        stream.flush();
        file.close();
    }
};
