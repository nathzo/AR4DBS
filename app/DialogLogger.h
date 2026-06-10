#pragma once
#include <QString>
#include <QDateTime>
#include <QFile>
#include <QStandardPaths>
#include <QTextStream>
#include <QMutex>

class DialogLogger {
public:
    static void clearLog() {
        QString logDir = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation);
        QString logFile = logDir + "/dbsar_debug.log";
        QFile file(logFile);
        file.remove();
    }

    static void logEvent(const QString &dialog, const QString &event) {
        static QMutex mutex;
        QMutexLocker lock(&mutex);

        QString logDir = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation);
        QString logFile = logDir + "/dbsar_debug.log";

        QFile file(logFile);
        if (file.open(QIODevice::Append | QIODevice::Text)) {
            QTextStream stream(&file);
            QString timestamp = QDateTime::currentDateTime().toString("hh:mm:ss.zzz");
            stream << "[" << timestamp << "] " << dialog << ": " << event << "\n";
            stream.flush();
            file.close();
        }
    }
};
