#include "EmailLogger.h"
#include <QDateTime>
#include <QSslSocket>
#include <QSslConfiguration>
#include <QEventLoop>
#include <QTimer>
#include <QDebug>
#include <QCoreApplication>

// Static member initialization
QString EmailLogger::s_emailAddress;
QString EmailLogger::s_appPassword;
QStringList EmailLogger::s_logQueue;
QMutex EmailLogger::s_mutex;
bool EmailLogger::s_initialized = false;

void EmailLogger::initialize(const QString &emailAddress, const QString &appPassword)
{
    QMutexLocker lock(&s_mutex);
    s_emailAddress = emailAddress;
    s_appPassword = appPassword;
    s_initialized = true;

    // Log initialization
    QString timestamp = QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss.zzz");
    addLog("[" + timestamp + "] EMAILLOGGER INITIALIZED for " + emailAddress);
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
    s_logQueue.append(logLine);
    qDebug() << logLine;  // Also print to debug console
}

QString EmailLogger::getCurrentLogs()
{
    QMutexLocker lock(&s_mutex);
    return s_logQueue.join("\n");
}

bool EmailLogger::sendViaSmtp(const QString &recipient, const QString &subject, const QString &body)
{
    QSslSocket socket;
    socket.connectToHostEncrypted("smtp.gmail.com", 587);

    // Wait for encrypted connection (timeout 5 seconds)
    if (!socket.waitForEncrypted(5000)) {
        qDebug() << "SSL connection failed:" << socket.errorString();
        return false;
    }

    qDebug() << "SSL connected, reading greeting...";

    // Read 220 greeting
    if (!socket.waitForReadyRead(2000)) {
        qDebug() << "No greeting received";
        return false;
    }
    QByteArray response = socket.readAll();
    qDebug() << "Server:" << response;

    // Send EHLO
    socket.write("EHLO localhost\r\n");
    socket.waitForBytesWritten(1000);
    socket.waitForReadyRead(2000);
    response = socket.readAll();
    qDebug() << "EHLO response:" << response;

    // Auth login
    socket.write("AUTH LOGIN\r\n");
    socket.waitForBytesWritten(1000);
    socket.waitForReadyRead(2000);
    response = socket.readAll();
    qDebug() << "AUTH LOGIN response:" << response;

    // Send email (base64 encoded)
    QByteArray email = s_emailAddress.toUtf8().toBase64();
    socket.write(email + "\r\n");
    socket.waitForBytesWritten(1000);
    socket.waitForReadyRead(2000);
    response = socket.readAll();
    qDebug() << "Email response:" << response;

    // Send password (base64 encoded)
    QByteArray password = s_appPassword.toUtf8().toBase64();
    socket.write(password + "\r\n");
    socket.waitForBytesWritten(1000);
    socket.waitForReadyRead(2000);
    response = socket.readAll();
    qDebug() << "Password response:" << response;

    // Check if auth succeeded (235 = Auth successful)
    if (!response.contains("235")) {
        qDebug() << "Authentication failed";
        socket.write("QUIT\r\n");
        socket.waitForBytesWritten(1000);
        socket.disconnectFromHost();
        return false;
    }

    // MAIL FROM
    socket.write("MAIL FROM:<" + s_emailAddress.toUtf8() + ">\r\n");
    socket.waitForBytesWritten(1000);
    socket.waitForReadyRead(2000);
    response = socket.readAll();
    qDebug() << "MAIL FROM response:" << response;

    // RCPT TO
    socket.write("RCPT TO:<" + recipient.toUtf8() + ">\r\n");
    socket.waitForBytesWritten(1000);
    socket.waitForReadyRead(2000);
    response = socket.readAll();
    qDebug() << "RCPT TO response:" << response;

    // DATA
    socket.write("DATA\r\n");
    socket.waitForBytesWritten(1000);
    socket.waitForReadyRead(2000);
    response = socket.readAll();
    qDebug() << "DATA response:" << response;

    // Compose email headers and body
    QString emailContent =
        "From: " + s_emailAddress + "\r\n" +
        "To: " + recipient + "\r\n" +
        "Subject: " + subject + "\r\n" +
        "Date: " + QDateTime::currentDateTime().toString(Qt::RFC2822Date) + "\r\n" +
        "Content-Type: text/plain; charset=utf-8\r\n" +
        "\r\n" +
        body + "\r\n";

    socket.write(emailContent.toUtf8());
    socket.write("\r\n.\r\n");
    socket.waitForBytesWritten(2000);
    socket.waitForReadyRead(2000);
    response = socket.readAll();
    qDebug() << "Final response:" << response;

    bool success = response.contains("250");

    // QUIT
    socket.write("QUIT\r\n");
    socket.waitForBytesWritten(1000);
    socket.disconnectFromHost();

    return success;
}

void EmailLogger::sendOnExit()
{
    QMutexLocker lock(&s_mutex);

    if (!s_initialized) {
        qDebug() << "EmailLogger not initialized, skipping send";
        return;
    }

    if (s_logQueue.isEmpty()) {
        qDebug() << "No logs to send";
        return;
    }

    QString allLogs = s_logQueue.join("\n");
    QString subject = "[AR4DBS CRASH LOG] " + QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss");

    qDebug() << "Attempting to send crash logs via email...";
    qDebug() << "Email address: " << s_emailAddress;

    bool success = sendViaSmtp(s_emailAddress, subject, allLogs);

    if (success) {
        qDebug() << "Email sent successfully";
        s_logQueue.clear();
    } else {
        qDebug() << "Email send failed";
    }
}
