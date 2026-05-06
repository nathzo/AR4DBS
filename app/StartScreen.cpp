#include "StartScreen.h"
#include <QVBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QFont>
#include <QPixmap>

static constexpr auto DARK_BG      = "#1a1b1d";
static constexpr auto IMPULSE_RED  = "#c45255";
static constexpr auto ARC_BLUE     = "#75D0C5";

StartScreen::StartScreen(QWidget *parent) : QWidget(parent)
{
    setStyleSheet("background-color: black; color: white;");
    auto *layout = new QVBoxLayout(this);
    layout->setAlignment(Qt::AlignCenter);
    layout->setSpacing(24);

    auto *logo = new QLabel(this);
    QPixmap logoPixmap(":/resources/logo.png");
    logo->setPixmap(logoPixmap.scaled(200, 200,
                                      Qt::KeepAspectRatio,
                                      Qt::SmoothTransformation));
    logo->setAlignment(Qt::AlignCenter);

    auto *title = new QLabel("AR4DBS", this);
    QFont tf;
    tf.setPointSize(36);
    tf.setBold(true);
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

    layout->addStretch();
    layout->addWidget(logo,    0, Qt::AlignCenter);
    layout->addWidget(title);
    layout->addSpacing(32);
    layout->addWidget(btnNew,  0, Qt::AlignCenter);
    layout->addWidget(btnTest, 0, Qt::AlignCenter);
    layout->addStretch();

    connect(btnNew,  &QPushButton::clicked, this, &StartScreen::newSurgeryRequested);
    connect(btnTest, &QPushButton::clicked, this, &StartScreen::directARRequested);
}