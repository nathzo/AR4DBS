#pragma once
#include <QString>
#include <QDateTime>
#include <QFile>
#include <QDir>
#include <QStandardPaths>
#include <QTextStream>
#include <QMutex>

class DialogLogger {
public:
    static QString getSuccessfulLogPath() {
        QStringList paths = getPossibleLogPaths();
        for (const QString &path : paths) {
            QFile file(path);
            if (file.open(QIODevice::Append | QIODevice::Text)) {
                file.close();
                return path;
            }
        }
        return "";
    }

    static QString getDiagnosticsString() {
        QStringList paths = getPossibleLogPaths();
        QString result = "Logs: ";

        for (int i = 0; i < paths.size(); ++i) {
            const QString &path = paths.at(i);
            QFile file(path);
            QDir dir(QFileInfo(path).dir());

            // Ensure directory exists
            if (!dir.exists()) {
                dir.mkpath(".");
            }

            // Try to actually write a test line and verify it exists
            if (file.open(QIODevice::Append | QIODevice::Text)) {
                QTextStream stream(&file);
                stream << "";  // Write empty content to test
                stream.flush();
                file.close();

                // Verify file exists after closing
                if (file.exists()) {
                    result += "✓ WRITING to ";
                    // Extract just the last part of the path for readability
                    QString displayPath = QFileInfo(path).dir().dirName() + "/" + QFileInfo(path).fileName();
                    result += displayPath;
                    return result;
                }
            }

            // Diagnose why it failed
            if (!dir.exists()) {
                result += "✗ dir-missing ";
            } else {
                result += "✗ file-persist-failed ";
            }
        }

        result += "| Tried: docs, appdata, cache, temp";
        return result;
    }

private:
    static void writeToFile(const QString &filePath, const QString &content) {
        QFile file(filePath);
        if (file.open(QIODevice::Append | QIODevice::Text)) {
            QTextStream stream(&file);
            stream << content;
            stream.flush();
            file.close();
        }
    }

    static QStringList getPossibleLogPaths() {
        QStringList paths;

        // Try Documents first (for file sharing)
        QString docsDir = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation);
        if (!docsDir.isEmpty()) {
            paths << QDir(docsDir).filePath("dbsar_debug.log");
        }

        // Try AppDataLocation (app's private directory)
        QString appDataDir = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
        if (!appDataDir.isEmpty()) {
            paths << QDir(appDataDir).filePath("dbsar_debug.log");
        }

        // Try CacheLocation (usually writable)
        QString cacheDir = QStandardPaths::writableLocation(QStandardPaths::CacheLocation);
        if (!cacheDir.isEmpty()) {
            paths << QDir(cacheDir).filePath("dbsar_debug.log");
        }

        // Try TempLocation (last resort)
        QString tempDir = QStandardPaths::writableLocation(QStandardPaths::TempLocation);
        if (!tempDir.isEmpty()) {
            paths << QDir(tempDir).filePath("dbsar_debug.log");
        }

        return paths;
    }

public:
    static void clearLog() {
        QStringList paths = getPossibleLogPaths();
        for (const QString &path : paths) {
            QFile file(path);
            file.remove();
        }

        // Write startup indicator to all locations
        QString timestamp = QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss.zzz");
        QString startupMsg = "[" + timestamp + "] APP STARTED - Clearing logs\n";

        for (const QString &path : paths) {
            QDir dir(QFileInfo(path).dir());
            dir.mkpath(".");
            writeToFile(path, startupMsg);
        }
    }

    static void logEvent(const QString &dialog, const QString &event) {
        static QMutex mutex;
        QMutexLocker lock(&mutex);

        QString timestamp = QDateTime::currentDateTime().toString("hh:mm:ss.zzz");
        QString logLine = "[" + timestamp + "] " + dialog + ": " + event + "\n";

        QStringList paths = getPossibleLogPaths();
        for (const QString &path : paths) {
            QDir dir(QFileInfo(path).dir());
            dir.mkpath(".");
            writeToFile(path, logLine);
        }
    }
};
