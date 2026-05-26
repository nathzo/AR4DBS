#include "StartScreen.h"
#include <QVBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QFont>
#include <QFontDatabase>
#include <QPixmap>
#include <QGuiApplication>
#include <QScreen>

static constexpr auto DARK_BG      = "#1a1b1d";
static constexpr auto IMPULSE_RED  = "#DE5F5E";
static constexpr auto ARC_BLUE     = "#75D0C5";

StartScreen::StartScreen(QWidget *parent) : QWidget(parent)
{
    setStyleSheet("background-color: black; color: white;");
    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(12, 12, 12, 20);
    layout->setSpacing(24);

    // Gear button — top-right corner
    auto *topRow = new QHBoxLayout;
    topRow->setContentsMargins(0, 0, 0, 0);
    topRow->addStretch(1);
    auto *gearBtn = new QPushButton("⚙", this);
    gearBtn->setFixedSize(44, 44);
    gearBtn->setStyleSheet(
        "QPushButton { background:#2a2b2d; color:#e0e0e0; border-radius:8px;"
        "              font-size:18pt; border:1px solid #444; }"
        "QPushButton:pressed { background:#3a3b3d; }");
    topRow->addWidget(gearBtn);
    layout->addLayout(topRow);
    connect(gearBtn, &QPushButton::clicked, this, &StartScreen::settingsRequested);

    auto *logo = new QLabel(this);
    QPixmap logoPixmap(":/resources/logo.png");
    const qreal dpr = QGuiApplication::primaryScreen()->devicePixelRatio();
    const int px = qRound(150 * dpr);
    QPixmap scaledPix = logoPixmap.scaled(px, px, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    scaledPix.setDevicePixelRatio(dpr);
    logo->setPixmap(scaledPix);
    logo->setAlignment(Qt::AlignCenter);

    auto *title = new QLabel("AR4DBS", this);
    int fontId = QFontDatabase::addApplicationFont(":/resources/Diagramm-Bold.ttf");
    QString family = fontId != -1 ? QFontDatabase::applicationFontFamilies(fontId).first() : "Arial";
    QFont tf(family, 36, QFont::Bold);
    title->setFont(tf);
    title->setAlignment(Qt::AlignCenter);
    title->setStyleSheet(QString("color: %1;").arg(IMPULSE_RED));

    auto btnStyle = [](const char *bg, const char *fg = "white") {
        return QString(
                   "QPushButton {"
                   "  background-color: %1;"
                   "  color: %2;"
                   "  border: none;"
                   "  border-radius: 12px;"
                   "  padding: 18px 48px;"
                   "  font-family: 'Arial';"
                   "  font-size: 16pt;"
                   "  font-weight: bold;"
                   "}"
                   "QPushButton:pressed { padding: 20px 46px; }"
                   ).arg(bg, fg);
    };

    auto *btnNew = new QPushButton("Nouvelle chirurgie", this);
    btnNew->setStyleSheet(btnStyle(IMPULSE_RED));
    auto *btnTest = new QPushButton("Mode test AR", this);
    btnTest->setStyleSheet(btnStyle(ARC_BLUE, "#000000"));

    layout->addStretch(1);
    layout->addWidget(logo,    0, Qt::AlignCenter);
    layout->addWidget(title,   0, Qt::AlignCenter);
    layout->addSpacing(32);
    layout->addWidget(btnNew,  0, Qt::AlignCenter);
    layout->addWidget(btnTest, 0, Qt::AlignCenter);
    layout->addStretch(1);

    connect(btnNew,  &QPushButton::clicked, this, &StartScreen::newSurgeryRequested);
    connect(btnTest, &QPushButton::clicked, this, &StartScreen::directARRequested);
}