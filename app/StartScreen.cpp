#include "StartScreen.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QFont>
#include <QFontDatabase>
#include <QPixmap>
#include <QPainter>
#include <QPainterPath>
#include <QGuiApplication>
#include <QScreen>
#include <cmath>

static constexpr auto DARK_BG      = "#1a1b1d";
static constexpr auto IMPULSE_RED  = "#DE5F5E";
static constexpr auto ARC_BLUE     = "#75D0C5";

static QPixmap makeGearPixmap(int size, const QColor &color)
{
    QPixmap pm(size, size);
    pm.fill(Qt::transparent);
    QPainter p(&pm);
    p.setRenderHint(QPainter::Antialiasing);

    const double cx      = size * 0.5;
    const double cy      = size * 0.5;
    const int    nTeeth  = 8;
    const double outerR  = size * 0.46;
    const double innerR  = size * 0.33;
    const double toothA  = M_PI / nTeeth * 0.55;
    const double holeR   = size * 0.13;

    QPainterPath gear;
    for (int i = 0; i < nTeeth; ++i) {
        const double mid = 2.0 * M_PI * i / nTeeth;
        const double a0  = mid - M_PI / nTeeth;
        const double a1  = mid - toothA;
        const double a2  = mid + toothA;
        const double a3  = mid + M_PI / nTeeth;

        auto pt = [&](double r, double a) {
            return QPointF(cx + r * std::cos(a), cy + r * std::sin(a));
        };

        if (i == 0) gear.moveTo(pt(innerR, a0));
        else        gear.lineTo(pt(innerR, a0));
        gear.lineTo(pt(outerR, a1));
        gear.lineTo(pt(outerR, a2));
        gear.lineTo(pt(innerR, a3));
    }
    gear.closeSubpath();

    QPainterPath hole;
    hole.addEllipse(QPointF(cx, cy), holeR, holeR);

    p.setPen(Qt::NoPen);
    p.setBrush(color);
    p.drawPath(gear.subtracted(hole));

    return pm;
}

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
    auto *gearBtn = new QPushButton(this);
    gearBtn->setFixedSize(44, 44);
    gearBtn->setIcon(QIcon(makeGearPixmap(256, QColor(0x8A, 0x8C, 0x8F))));
    gearBtn->setIconSize(QSize(30, 30));
    gearBtn->setStyleSheet(
        "QPushButton { background: transparent; border: none; }"
        "QPushButton:pressed { background: rgba(255,255,255,20); border-radius:8px; }");
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