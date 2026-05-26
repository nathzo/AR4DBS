#include "SettingsDialog.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QLabel>
#include <QPushButton>
#include <QButtonGroup>
#include <QDoubleSpinBox>
#include <QFrame>
#include <QScrollArea>
#include <QPainter>
#include <QPaintEvent>
#include <QGuiApplication>
#include <QScreen>

// ── Shared palette ────────────────────────────────────────────────────────────

static constexpr QColor kImpulseRed  {222,  95,  94};
static constexpr QColor kArcBlue     {117, 208, 197};
static constexpr QColor kVoltYellow  {233, 223,  77};

static const QColor kPalette[3] = { kImpulseRed, kArcBlue, kVoltYellow };

static int colorIndex(const QColor &c)
{
    int best = 0, bestDist = INT_MAX;
    for (int i = 0; i < 3; ++i) {
        int d = std::abs(c.red()   - kPalette[i].red())
              + std::abs(c.green() - kPalette[i].green())
              + std::abs(c.blue()  - kPalette[i].blue());
        if (d < bestDist) { bestDist = d; best = i; }
    }
    return best;
}

// ── Common stylesheet fragments ───────────────────────────────────────────────

static const char *kBaseSS =
    "QDialog  { background: transparent; }"
    "QWidget#content, QWidget#innerContent { background: #0d0d0d; }"
    "QLabel   { background: transparent; color: #e0e0e0;"
    "           font-family: 'Arial'; font-size: 13pt; }"
    "QGroupBox {"
    "  border: 1px solid #333; border-radius: 8px;"
    "  margin-top: 14px; padding-top: 10px;"
    "  color: #aaa; font-family: 'Arial'; font-size: 11pt; font-weight: bold;"
    "}"
    "QGroupBox::title { subcontrol-origin: margin; left: 12px; }"
    "QDoubleSpinBox {"
    "  background: #222; border: 1px solid #555; border-radius: 6px;"
    "  color: #e0e0e0; font-family: 'Arial'; font-size: 13pt;"
    "  padding: 4px 8px; min-height: 44px;"
    "}"
    "QDoubleSpinBox::up-button   { width: 28px; }"
    "QDoubleSpinBox::down-button { width: 28px; }";

static QPushButton *makePrimaryBtn(const QString &text, QWidget *parent)
{
    auto *b = new QPushButton(text, parent);
    b->setStyleSheet(
        "QPushButton { background:#75D0C5; color:black; border-radius:8px;"
        "              padding:12px 30px; font-family:'Arial';"
        "              font-size:13pt; font-weight:bold; }"
        "QPushButton:pressed { background:#5ab8ae; }");
    return b;
}

static QPushButton *makeSecondaryBtn(const QString &text, QWidget *parent)
{
    auto *b = new QPushButton(text, parent);
    b->setStyleSheet(
        "QPushButton { background:#2a2b2d; color:#e0e0e0; border-radius:8px;"
        "              padding:12px 30px; font-family:'Arial';"
        "              font-size:13pt; border: 1px solid #444; }"
        "QPushButton:pressed { background:#3a3b3d; }");
    return b;
}

static QFrame *makeSeparator(QWidget *parent)
{
    auto *f = new QFrame(parent);
    f->setFrameShape(QFrame::HLine);
    f->setFixedHeight(1);
    f->setStyleSheet("background: #333;");
    return f;
}

// ── paintEvent shared helper ──────────────────────────────────────────────────

static void paintBlack(QWidget *w, QPaintEvent *)
{
    QPainter p(w);
    p.setPen(Qt::NoPen);
    p.setBrush(Qt::black);
    p.drawRect(w->rect());
}

// ─────────────────────────────────────────────────────────────────────────────
// GraphicsSettingsDialog
// ─────────────────────────────────────────────────────────────────────────────

class GraphicsSettingsDialog : public QDialog
{
    Q_OBJECT
public:
    explicit GraphicsSettingsDialog(const OverlayRenderer::Style &style,
                                    QWidget *parent = nullptr)
        : QDialog(parent)
    {
        setWindowFlags(Qt::FramelessWindowHint | Qt::Dialog);
        setAttribute(Qt::WA_TranslucentBackground);
        const QRect ag = QGuiApplication::primaryScreen()->availableGeometry();
        setFixedSize(ag.width(), ag.height());

        auto *root = new QWidget(this);
        root->setObjectName("content");
        root->setFixedSize(ag.width(), ag.height());
        root->setStyleSheet(kBaseSS);

        auto *vbox = new QVBoxLayout(root);
        vbox->setContentsMargins(20, 60, 20, 30);
        vbox->setSpacing(16);

        auto *title = new QLabel("Paramètres graphiques", root);
        title->setAlignment(Qt::AlignCenter);
        title->setStyleSheet(
            "color:white; font-family:'Arial'; font-size:18pt; font-weight:bold;");
        vbox->addWidget(title);
        vbox->addWidget(makeSeparator(root));
        vbox->addSpacing(8);

        // ── Color picker grid ─────────────────────────────────────────────────
        auto *grid = new QGridLayout;
        grid->setHorizontalSpacing(12);
        grid->setVerticalSpacing(16);
        grid->setColumnStretch(0, 2);
        grid->setColumnStretch(1, 1);
        grid->setColumnStretch(2, 1);
        grid->setColumnStretch(3, 1);

        // Column headers
        const char *colorLabels[] = { "Impulse\nRed", "Arc\nBlue", "Volt\nYellow" };
        for (int c = 0; c < 3; ++c) {
            auto *lbl = new QLabel(colorLabels[c], root);
            lbl->setAlignment(Qt::AlignCenter);
            lbl->setStyleSheet(
                "color:#888; font-family:'Arial'; font-size:10pt;");
            grid->addWidget(lbl, 0, c + 1);
        }

        // Row data: label and initial selection index
        struct RowDef { const char *label; int *idx; };
        m_lineIdx     = colorIndex(style.lineColor);
        m_incisionIdx = colorIndex(style.incisionColor);
        m_targetIdx   = colorIndex(style.targetColor);

        RowDef rows[] = {
            { "Trajectoire",           &m_lineIdx     },
            { "Marqueur d'incision",   &m_incisionIdx },
            { "Cible",                 &m_targetIdx   },
        };

        for (int r = 0; r < 3; ++r) {
            auto *lbl = new QLabel(rows[r].label, root);
            lbl->setStyleSheet(
                "color:#e0e0e0; font-family:'Arial'; font-size:13pt;");
            grid->addWidget(lbl, r + 1, 0, Qt::AlignVCenter | Qt::AlignLeft);

            auto *grp = new QButtonGroup(this);
            grp->setExclusive(true);
            m_groups[r] = grp;

            for (int c = 0; c < 3; ++c) {
                auto *btn = new QPushButton(root);
                btn->setFixedSize(58, 58);
                btn->setCheckable(true);
                btn->setChecked(c == *rows[r].idx);
                grp->addButton(btn, c);
                grid->addWidget(btn, r + 1, c + 1, Qt::AlignCenter);

                const QColor col = kPalette[c];
                auto applyStyle = [btn, col](bool checked) {
                    btn->setStyleSheet(QString(
                        "QPushButton {"
                        "  background:%1; border-radius:8px;"
                        "  border:%2;"
                        "}"
                        "QPushButton:pressed { background:%1; }")
                        .arg(col.name())
                        .arg(checked ? "3px solid white" : "2px solid #555"));
                };
                applyStyle(c == *rows[r].idx);
                connect(btn, &QPushButton::toggled, btn, applyStyle);
            }
        }

        vbox->addLayout(grid);
        vbox->addStretch(1);
        vbox->addWidget(makeSeparator(root));
        vbox->addSpacing(8);

        // ── Buttons ───────────────────────────────────────────────────────────
        auto *btnRow = new QHBoxLayout;
        btnRow->setSpacing(16);
        auto *btnCancel = makeSecondaryBtn("Annuler", root);
        auto *btnApply  = makePrimaryBtn("Appliquer", root);
        btnRow->addWidget(btnCancel);
        btnRow->addWidget(btnApply);
        vbox->addLayout(btnRow);

        connect(btnCancel, &QPushButton::clicked, this, &QDialog::reject);
        connect(btnApply,  &QPushButton::clicked, this, [this]() {
            OverlayRenderer::Style s;
            s.lineColor     = kPalette[m_groups[0]->checkedId()];
            s.incisionColor = kPalette[m_groups[1]->checkedId()];
            s.targetColor   = kPalette[m_groups[2]->checkedId()];
            emit styleApplied(s);
            accept();
        });
    }

signals:
    void styleApplied(OverlayRenderer::Style style);

protected:
    void paintEvent(QPaintEvent *e) override { paintBlack(this, e); }

private:
    int          m_lineIdx     = 0;
    int          m_incisionIdx = 2;
    int          m_targetIdx   = 1;
    QButtonGroup *m_groups[3]  = {};
};

// ─────────────────────────────────────────────────────────────────────────────
// CalibrationSettingsDialog
// ─────────────────────────────────────────────────────────────────────────────

class CalibrationSettingsDialog : public QDialog
{
    Q_OBJECT
public:
    explicit CalibrationSettingsDialog(double             reprojThresh,
                                       const TagPositions &tagPos,
                                       QWidget            *parent = nullptr)
        : QDialog(parent)
    {
        setWindowFlags(Qt::FramelessWindowHint | Qt::Dialog);
        setAttribute(Qt::WA_TranslucentBackground);
        const QRect ag = QGuiApplication::primaryScreen()->availableGeometry();
        setFixedSize(ag.width(), ag.height());

        auto *root = new QWidget(this);
        root->setObjectName("content");
        root->setFixedSize(ag.width(), ag.height());
        root->setStyleSheet(kBaseSS);

        auto *vbox = new QVBoxLayout(root);
        vbox->setContentsMargins(20, 60, 20, 30);
        vbox->setSpacing(16);

        auto *title = new QLabel("Paramètres de calibration", root);
        title->setAlignment(Qt::AlignCenter);
        title->setStyleSheet(
            "color:white; font-family:'Arial'; font-size:18pt; font-weight:bold;");
        vbox->addWidget(title);
        vbox->addWidget(makeSeparator(root));
        vbox->addSpacing(8);

        // ── Reprojection threshold ─────────────────────────────────────────────
        auto *reprojBox = new QGroupBox("Seuil d'erreur de reprojection", root);
        auto *reprojForm = new QFormLayout(reprojBox);
        reprojForm->setContentsMargins(16, 20, 16, 16);
        reprojForm->setVerticalSpacing(12);
        reprojForm->setHorizontalSpacing(16);

        m_reprojSB = new QDoubleSpinBox(reprojBox);
        m_reprojSB->setRange(0.1, 10.0);
        m_reprojSB->setDecimals(2);
        m_reprojSB->setSingleStep(0.1);
        m_reprojSB->setSuffix(" px");
        m_reprojSB->setValue(reprojThresh);
        reprojForm->addRow("Seuil (px) :", m_reprojSB);
        vbox->addWidget(reprojBox);

        // ── Tag positions ──────────────────────────────────────────────────────
        const char *tagLabels[] = { "Tag 0 — gauche", "Tag 1 — droit" };
        for (int t = 0; t < 2; ++t) {
            auto *box  = new QGroupBox(tagLabels[t], root);
            auto *form = new QFormLayout(box);
            form->setContentsMargins(16, 20, 16, 16);
            form->setVerticalSpacing(12);
            form->setHorizontalSpacing(16);

            for (int ax = 0; ax < 3; ++ax) {
                m_tagSB[t][ax] = new QDoubleSpinBox(box);
                m_tagSB[t][ax]->setRange(-500.0, 500.0);
                m_tagSB[t][ax]->setDecimals(1);
                m_tagSB[t][ax]->setSingleStep(1.0);
                m_tagSB[t][ax]->setSuffix(" mm");
            }
            // Values are stored in meters; display in mm
            m_tagSB[t][0]->setValue(tagPos.tx_m[t] * 1000.0);
            m_tagSB[t][1]->setValue(tagPos.ty_m[t] * 1000.0);
            m_tagSB[t][2]->setValue(tagPos.tz_m[t] * 1000.0);

            form->addRow("tx (mm) :", m_tagSB[t][0]);
            form->addRow("ty (mm) :", m_tagSB[t][1]);
            form->addRow("tz (mm) :", m_tagSB[t][2]);
            vbox->addWidget(box);
        }

        vbox->addStretch(1);
        vbox->addWidget(makeSeparator(root));
        vbox->addSpacing(8);

        // ── Buttons ───────────────────────────────────────────────────────────
        auto *btnRow = new QHBoxLayout;
        btnRow->setSpacing(16);
        auto *btnCancel = makeSecondaryBtn("Annuler", root);
        auto *btnApply  = makePrimaryBtn("Appliquer", root);
        btnRow->addWidget(btnCancel);
        btnRow->addWidget(btnApply);
        vbox->addLayout(btnRow);

        connect(btnCancel, &QPushButton::clicked, this, &QDialog::reject);
        connect(btnApply,  &QPushButton::clicked, this, [this]() {
            emit reprojThresholdApplied(m_reprojSB->value());
            for (int t = 0; t < 2; ++t) {
                emit tagPositionApplied(t,
                    m_tagSB[t][0]->value() / 1000.0,
                    m_tagSB[t][1]->value() / 1000.0,
                    m_tagSB[t][2]->value() / 1000.0);
            }
            accept();
        });
    }

signals:
    void reprojThresholdApplied(double px);
    void tagPositionApplied(int tagId, double tx_m, double ty_m, double tz_m);

protected:
    void paintEvent(QPaintEvent *e) override { paintBlack(this, e); }

private:
    QDoubleSpinBox *m_reprojSB       = nullptr;
    QDoubleSpinBox *m_tagSB[2][3]   = {};
};

// ─────────────────────────────────────────────────────────────────────────────
// SettingsDialog
// ─────────────────────────────────────────────────────────────────────────────

SettingsDialog::SettingsDialog(const OverlayRenderer::Style &currentStyle,
                               double                        currentReprojThreshold,
                               const TagPositions           &currentTagPositions,
                               QWidget                      *parent)
    : QDialog(parent)
{
    setWindowFlags(Qt::FramelessWindowHint | Qt::Dialog);
    setAttribute(Qt::WA_TranslucentBackground);
    const QRect ag = QGuiApplication::primaryScreen()->availableGeometry();
    setFixedSize(ag.width(), ag.height());

    auto *root = new QWidget(this);
    root->setObjectName("content");
    root->setFixedSize(ag.width(), ag.height());
    root->setStyleSheet(kBaseSS);

    auto *vbox = new QVBoxLayout(root);
    vbox->setContentsMargins(20, 60, 20, 30);
    vbox->setSpacing(0);

    // ── Title ─────────────────────────────────────────────────────────────────
    auto *title = new QLabel("Paramètres", root);
    title->setAlignment(Qt::AlignCenter);
    title->setStyleSheet(
        "color:white; font-family:'Arial'; font-size:20pt; font-weight:bold;");
    vbox->addWidget(title);
    vbox->addSpacing(20);
    vbox->addWidget(makeSeparator(root));
    vbox->addSpacing(20);

    // ── Language selector (disabled) ──────────────────────────────────────────
    auto *langRow = new QHBoxLayout;
    langRow->setSpacing(12);
    auto *langLbl = new QLabel("Langue :", root);
    langLbl->setStyleSheet(
        "color:#666; font-family:'Arial'; font-size:13pt;");
    auto *btnFr = new QPushButton("Français", root);
    auto *btnEn = new QPushButton("English",  root);
    for (auto *b : {btnFr, btnEn}) {
        b->setFixedHeight(44);
        b->setEnabled(false);
        b->setStyleSheet(
            "QPushButton { background:#1a1a1a; color:#444; border-radius:8px;"
            "              padding:8px 20px; font-family:'Arial'; font-size:12pt;"
            "              border:1px solid #333; }");
    }
    btnFr->setStyleSheet(  // mark Français as the active choice (greyed-out)
        "QPushButton { background:#1a3030; color:#3a6060; border-radius:8px;"
        "              padding:8px 20px; font-family:'Arial'; font-size:12pt;"
        "              border:1px solid #2a5050; }");
    langRow->addWidget(langLbl);
    langRow->addStretch(1);
    langRow->addWidget(btnFr);
    langRow->addWidget(btnEn);
    vbox->addLayout(langRow);
    vbox->addSpacing(16);

    // ── Coordinates reference selector (disabled) ─────────────────────────────
    auto *refRow = new QHBoxLayout;
    refRow->setSpacing(12);
    auto *refLbl   = new QLabel("Référentiel :", root);
    refLbl->setStyleSheet(
        "color:#666; font-family:'Arial'; font-size:13pt;");
    auto *btnMdt = new QPushButton("Medtronic", root);
    auto *btnBL  = new QPushButton("BrainLab",  root);
    for (auto *b : {btnMdt, btnBL}) {
        b->setFixedHeight(44);
        b->setEnabled(false);
        b->setStyleSheet(
            "QPushButton { background:#1a1a1a; color:#444; border-radius:8px;"
            "              padding:8px 20px; font-family:'Arial'; font-size:12pt;"
            "              border:1px solid #333; }");
    }
    refRow->addWidget(refLbl);
    refRow->addStretch(1);
    refRow->addWidget(btnMdt);
    refRow->addWidget(btnBL);
    vbox->addLayout(refRow);
    vbox->addSpacing(24);
    vbox->addWidget(makeSeparator(root));
    vbox->addSpacing(24);

    // ── Navigation buttons ────────────────────────────────────────────────────
    auto makeNavBtn = [&](const QString &label) {
        auto *b = new QPushButton(label, root);
        b->setFixedHeight(60);
        b->setStyleSheet(
            "QPushButton {"
            "  background:#1e1e1e; color:#e0e0e0; border-radius:10px;"
            "  padding:0px 20px; font-family:'Arial'; font-size:14pt;"
            "  text-align:left; border:1px solid #333;"
            "}"
            "QPushButton:pressed { background:#2e2e2e; }");
        return b;
    };

    auto *btnGraphics = makeNavBtn("  Paramètres graphiques   ›");
    auto *btnCalib    = makeNavBtn("  Paramètres de calibration   ›");
    vbox->addWidget(btnGraphics);
    vbox->addSpacing(12);
    vbox->addWidget(btnCalib);
    vbox->addStretch(1);
    vbox->addWidget(makeSeparator(root));
    vbox->addSpacing(16);

    // ── Close button ──────────────────────────────────────────────────────────
    auto *btnClose = makePrimaryBtn("Fermer", root);
    vbox->addWidget(btnClose, 0, Qt::AlignHCenter);

    connect(btnClose, &QPushButton::clicked, this, &QDialog::accept);

    // ── Open sub-dialogs ──────────────────────────────────────────────────────
    connect(btnGraphics, &QPushButton::clicked, this, [this, currentStyle]() {
        auto *dlg = new GraphicsSettingsDialog(currentStyle, this);
        connect(dlg, &GraphicsSettingsDialog::styleApplied,
                this, &SettingsDialog::styleChanged);
        dlg->exec();
        dlg->deleteLater();
    });

    connect(btnCalib, &QPushButton::clicked, this,
            [this, currentReprojThreshold, currentTagPositions]() {
        auto *dlg = new CalibrationSettingsDialog(
            currentReprojThreshold, currentTagPositions, this);
        connect(dlg, &CalibrationSettingsDialog::reprojThresholdApplied,
                this, &SettingsDialog::reprojThresholdChanged);
        connect(dlg, &CalibrationSettingsDialog::tagPositionApplied,
                this, &SettingsDialog::tagPositionChanged);
        dlg->exec();
        dlg->deleteLater();
    });
}

void SettingsDialog::paintEvent(QPaintEvent *e) { paintBlack(this, e); }

#include "SettingsDialog.moc"
