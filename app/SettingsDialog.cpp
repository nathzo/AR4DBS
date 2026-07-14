#include "SettingsDialog.h"
#include "DebugLogger.h"
#include "DebugLogDialog.h"

#include <QVBoxLayout>
#include <QApplication>
#include <QSysInfo>
#include <QThread>
#include <sys/resource.h>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QLabel>
#include <QPushButton>
#include <QButtonGroup>
#include <QDoubleSpinBox>
#include <QLineEdit>
#include <QFrame>
#include <QScrollArea>
#include <QPainter>
#include <QPaintEvent>
#include <QGuiApplication>
#include <QInputMethod>
#include <QSettings>
#include <QMessageBox>
#include <QInputMethodEvent>
#include <QKeyEvent>
#include <QCoreApplication>
#include <QScreen>
#include <QComboBox>

// ── Comprehensive memory and system logging helper ────────────────────────────

static void logMemoryState(const QString &context)
{
#ifdef Q_OS_IOS
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        // Memory information
        long maxRss = usage.ru_maxrss;  // Peak memory in bytes
        long ixRss = usage.ru_ixrss;    // Shared memory
        long idrss = usage.ru_idrss;    // Unshared data
        long isrss = usage.ru_isrss;    // Unshared stack

        // Page fault information
        long majFlt = usage.ru_majflt;  // Major page faults
        long minFlt = usage.ru_minflt;  // Minor page faults

        // CPU time
        long userSec = usage.ru_utime.tv_sec;
        long userUsec = usage.ru_utime.tv_usec;
        long sysSec = usage.ru_stime.tv_sec;
        long sysUsec = usage.ru_stime.tv_usec;

        // Context switches
        long volCtx = usage.ru_nvcsw;   // Voluntary context switches
        long invCtx = usage.ru_nivcsw;  // Involuntary context switches

        QString memLog = context + ": "
            "peak_rss=" + QString::number(maxRss / 1024 / 1024) + "MB, "
            "shared=" + QString::number(ixRss / 1024 / 1024) + "MB, "
            "unshared_data=" + QString::number(idrss / 1024 / 1024) + "MB, "
            "unshared_stack=" + QString::number(isrss / 1024 / 1024) + "MB, "
            "majflt=" + QString::number(majFlt) + ", "
            "minflt=" + QString::number(minFlt) + ", "
            "user_cpu=" + QString::number(userSec) + "." + QString::number(userUsec / 100000) + "s, "
            "sys_cpu=" + QString::number(sysSec) + "." + QString::number(sysUsec / 100000) + "s, "
            "vol_ctx=" + QString::number(volCtx) + ", "
            "inv_ctx=" + QString::number(invCtx);

        DebugLogger::logEvent("Memory", memLog);

        // System information
        QString sysInfo = context + ": "
            "ios=" + QSysInfo::productVersion() + ", "
            "device=" + QSysInfo::machineHostName() + ", "
            "thread_id=0x" + QString::number((quint64)QThread::currentThreadId(), 16) + ", "
            "timestamp=" + QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss.zzz");

        DebugLogger::logEvent("System", sysInfo);
    }
#endif
}

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
    "           font-size: 13pt; }"
    "QGroupBox {"
    "  border: 1px solid #333; border-radius: 8px;"
    "  margin-top: 14px; padding-top: 10px;"
    "  color: #aaa; font-size: 11pt; font-weight: bold;"
    "}"
    "QGroupBox::title { subcontrol-origin: margin; left: 12px; }"
    "QDoubleSpinBox {"
    "  background: #2a2b2d; border: 1px solid #444; border-radius: 6px;"
    "  color: #e0e0e0; font-size: 13pt;"
    "  padding: 6px 10px; min-height: 36px;"
    "  selection-color: #e0e0e0; selection-background-color: #2d5f7a;"
    "}"
    "QDoubleSpinBox::up-button {"
    "  subcontrol-origin: border; subcontrol-position: top right;"
    "  width: 0; height: 0; border: none; margin: 0;"
    "}"
    "QDoubleSpinBox::down-button {"
    "  subcontrol-origin: border; subcontrol-position: bottom right;"
    "  width: 0; height: 0; border: none; margin: 0;"
    "}";

static QPushButton *makePrimaryBtn(const QString &text, QWidget *parent)
{
    auto *b = new QPushButton(text, parent);
    b->setStyleSheet(
        "QPushButton { background:#75D0C5; color:black; border-radius:8px;"
        "              padding:12px 30px;"
        "              font-size:13pt; font-weight:bold; }"
        "QPushButton:pressed { background:#5ab8ae; }");
    return b;
}

static QPushButton *makeSecondaryBtn(const QString &text, QWidget *parent)
{
    auto *b = new QPushButton(text, parent);
    b->setStyleSheet(
        "QPushButton { background:#2a2b2d; color:#e0e0e0; border-radius:8px;"
        "              padding:12px 30px;"
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
// ToggleSwitch — iOS-style pill toggle
// ─────────────────────────────────────────────────────────────────────────────

class ToggleSwitch : public QWidget
{
    Q_OBJECT
    Q_PROPERTY(bool checked READ isChecked WRITE setChecked NOTIFY toggled)
public:
    explicit ToggleSwitch(bool initialState = true, QWidget *parent = nullptr)
        : QWidget(parent), m_checked(initialState)
    {
        setFixedSize(58, 30);
        setCursor(Qt::PointingHandCursor);
    }

    bool isChecked() const { return m_checked; }

    void setChecked(bool on)
    {
        if (m_checked == on) return;
        m_checked = on;
        update();
        emit toggled(on);
    }

    void setEnabled(bool enabled)
    {
        QWidget::setEnabled(enabled);
        update();
    }

signals:
    void toggled(bool checked);

protected:
    void mousePressEvent(QMouseEvent *) override
    {
        if (isEnabled()) setChecked(!m_checked);
    }

    void paintEvent(QPaintEvent *) override
    {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);

        const QRectF track(0, 3, width(), height() - 6);
        const qreal  radius = track.height() / 2.0;

        if (!isEnabled()) {
            p.setBrush(QColor(0x2a, 0x2a, 0x2a));
        } else if (m_checked) {
            p.setBrush(QColor(0x75, 0xD0, 0xC5));
        } else {
            p.setBrush(QColor(0x3a, 0x3a, 0x3c));
        }
        p.setPen(Qt::NoPen);
        p.drawRoundedRect(track, radius, radius);

        const qreal  knobD   = track.height() - 4;
        const qreal  knobY   = track.top() + 2;
        const qreal  knobX   = m_checked
                                ? track.right() - knobD - 2
                                : track.left()  + 2;
        const QColor knobCol = isEnabled() ? QColor(0xff, 0xff, 0xff)
                                           : QColor(0x55, 0x55, 0x55);
        p.setBrush(knobCol);
        p.drawEllipse(QRectF(knobX, knobY, knobD, knobD));
    }

private:
    bool m_checked;
};

// ─────────────────────────────────────────────────────────────────────────────
// GraphicsSettingsDialog
// ─────────────────────────────────────────────────────────────────────────────

class GraphicsSettingsDialog : public QDialog
{
    Q_OBJECT
public:
    explicit GraphicsSettingsDialog(const OverlayRenderer::Style &style,
                                    bool     hasLidar,
                                    bool     arTestDepthOverlay,
                                    QWidget *parent = nullptr)
        : QDialog(parent)
        , m_arTestDepthOverlay(arTestDepthOverlay)
    {
        DebugLogger::logEvent("GraphicsSettingsDialog", "constructor started");
        setWindowFlags(Qt::Dialog);
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

        auto *title = new QLabel(tr("Paramètres graphiques"), root);
        title->setAlignment(Qt::AlignCenter);
        title->setStyleSheet(
            "color:white; font-size:18pt; font-weight:bold;");
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
                "color:#888; font-size:10pt;");
            grid->addWidget(lbl, 0, c + 1);
        }

        // Row data: label and initial selection index
        struct RowDef { QString label; int *idx; };
        m_lineIdx     = colorIndex(style.lineColor);
        m_incisionIdx = colorIndex(style.incisionColor);
        m_targetIdx   = colorIndex(style.targetColor);

        RowDef rows[] = {
            { tr("Trajectoire"),           &m_lineIdx     },
            { tr("Marqueur d'incision"),   &m_incisionIdx },
            { tr("Cible"),                 &m_targetIdx   },
        };

        for (int r = 0; r < 3; ++r) {
            auto *lbl = new QLabel(rows[r].label, root);  // label is already tr()'d above
            lbl->setStyleSheet(
                "color:#e0e0e0; font-size:13pt;");
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
        vbox->addSpacing(20);
        vbox->addWidget(makeSeparator(root));
        vbox->addSpacing(16);

        // ── AR test depth overlay toggle ──────────────────────────────────────
        auto *overlayRow = new QHBoxLayout;
        overlayRow->setSpacing(12);

        auto *overlayLbl = new QLabel(tr("Visualisation profondeur test AR"), root);
        overlayLbl->setStyleSheet(hasLidar
            ? "color:#e0e0e0; font-size:13pt;"
            : "color:#555;    font-size:13pt;");
        overlayRow->addWidget(overlayLbl, 1);

        if (!hasLidar) {
            m_arTestDepthOverlay = false;
            auto *noLidarLbl = new QLabel(tr("(LiDAR requis)"), root);
            noLidarLbl->setStyleSheet(
                "color:#555; font-size:10pt; font-style:italic;");
            overlayRow->addWidget(noLidarLbl);
        }

        auto *toggle = new ToggleSwitch(m_arTestDepthOverlay, root);
        toggle->setEnabled(hasLidar);
        connect(toggle, &ToggleSwitch::toggled, root, [this](bool on) {
            m_arTestDepthOverlay = on;
        });
        overlayRow->addWidget(toggle);
        vbox->addLayout(overlayRow);

        vbox->addStretch(1);
        vbox->addWidget(makeSeparator(root));
        vbox->addSpacing(8);

        // ── Buttons ───────────────────────────────────────────────────────────
        auto *btnRow = new QHBoxLayout;
        btnRow->setSpacing(16);
        auto *btnCancel = makeSecondaryBtn(tr("Annuler"), root);
        auto *btnApply  = makePrimaryBtn(tr("Appliquer"), root);
        btnRow->addWidget(btnCancel);
        btnRow->addWidget(btnApply);
        vbox->addLayout(btnRow);

        connect(btnCancel, &QPushButton::clicked, this, &QDialog::reject);
        connect(btnApply,  &QPushButton::clicked, this, [this]() {
            DebugLogger::logEvent("GraphicsSettingsDialog", "apply clicked");
            OverlayRenderer::Style s;
            s.lineColor     = kPalette[m_groups[0]->checkedId()];
            s.incisionColor = kPalette[m_groups[1]->checkedId()];
            s.targetColor   = kPalette[m_groups[2]->checkedId()];
            emit styleApplied(s);
            emit arTestDepthOverlayChanged(m_arTestDepthOverlay);
            accept();
        });
    }

    ~GraphicsSettingsDialog() {
        DebugLogger::logEvent("GraphicsSettingsDialog", "destructor starting");
        // Disable accessibility on all children to prevent iOS crash when destroying buttons
        for (QObject *child : findChildren<QObject*>()) {
            if (QWidget *w = qobject_cast<QWidget*>(child))
                w->setAccessibleDescription(QString());
        }
        DebugLogger::logEvent("GraphicsSettingsDialog", "destructor completed");
    }

signals:
    void styleApplied(OverlayRenderer::Style style);
    void arTestDepthOverlayChanged(bool enabled);

protected:
    void paintEvent(QPaintEvent *e) override { paintBlack(this, e); }

    bool event(QEvent *ev) override {
        // Block all accessibility events to prevent iOS crash during widget destruction
        // Accessibility events are in the range 1000-1030
        if (ev->type() >= 1000 && ev->type() <= 1030) {
            return true;
        }
        return QDialog::event(ev);
    }

private:
    int          m_lineIdx            = 0;
    int          m_incisionIdx        = 2;
    int          m_targetIdx          = 1;
    QButtonGroup *m_groups[3]         = {};
    bool         m_arTestDepthOverlay = true;
};

// ─────────────────────────────────────────────────────────────────────────────
// CalibSpinBox — QDoubleSpinBox with:
//   • inputMethodQuery fix so iOS selection handles appear at the correct position
//   • Enter/Return closes the keyboard (never advances focus)
//   • decimal-separator normalisation (like AutoSelectSpinBox in ConfirmPlanDialog)
//   • NO auto-select-all on focus
// ─────────────────────────────────────────────────────────────────────────────

class CalibSpinBox : public QDoubleSpinBox
{
public:
    explicit CalibSpinBox(QWidget *parent = nullptr) : QDoubleSpinBox(parent)
    {
        lineEdit()->installEventFilter(this);
    }

    // Translate cursor/anchor rectangles so iOS selection handles appear inside
    // the visible text area rather than at the QDoubleSpinBox origin.
    QVariant inputMethodQuery(Qt::InputMethodQuery query) const override
    {
        QVariant v = QDoubleSpinBox::inputMethodQuery(query);
        if (query == Qt::ImCursorRectangle    ||
            query == Qt::ImAnchorRectangle    ||
            query == Qt::ImInputItemClipRectangle) {
            if (v.canConvert<QRect>())
                return v.toRect().translated(lineEdit()->pos());
        }
        return v;
    }

protected:
    void focusInEvent(QFocusEvent *e) override
    {
        QDoubleSpinBox::focusInEvent(e);
        QMetaObject::invokeMethod(this, [this]() { lineEdit()->selectAll(); },
                                  Qt::QueuedConnection);
    }

    void keyPressEvent(QKeyEvent *e) override
    {
        if (e->key() == Qt::Key_Return || e->key() == Qt::Key_Enter) {
            clearFocus();
            if (auto *im = QGuiApplication::inputMethod()) im->hide();
            return;
        }
        QDoubleSpinBox::keyPressEvent(e);
    }

    bool eventFilter(QObject *obj, QEvent *ev) override
    {
        if (obj == lineEdit() && ev->type() == QEvent::InputMethod) {
            auto *ime = static_cast<QInputMethodEvent *>(ev);
            const QString commit = ime->commitString();
            if (!commit.isEmpty()) {
                const QChar dp  = locale().decimalPoint().at(0);
                const QChar alt = (dp == QLatin1Char('.')) ? QLatin1Char(',')
                                                           : QLatin1Char('.');
                if (commit.contains(alt)) {
                    QString fixed = commit;
                    fixed.replace(alt, dp);
                    QInputMethodEvent mapped(ime->preeditString(), ime->attributes());
                    mapped.setCommitString(fixed,
                                          ime->replacementStart(),
                                          ime->replacementLength());
                    QCoreApplication::sendEvent(lineEdit(), &mapped);
                    return true;
                }
            }
        }
        return QDoubleSpinBox::eventFilter(obj, ev);
    }

    QValidator::State validate(QString &input, int &pos) const override
    {
        normalise(input);
        return QDoubleSpinBox::validate(input, pos);
    }

    double valueFromText(const QString &text) const override
    {
        QString t = text;
        normalise(t);
        return QDoubleSpinBox::valueFromText(t);
    }

private:
    void normalise(QString &s) const
    {
        const QChar dp  = locale().decimalPoint().at(0);
        const QChar alt = (dp == QLatin1Char('.')) ? QLatin1Char(',') : QLatin1Char('.');
        s.replace(alt, dp);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// CalibrationSettingsDialog
// ─────────────────────────────────────────────────────────────────────────────

class CalibrationSettingsDialog : public QDialog
{
    Q_OBJECT
public:
    // Predefined configurations: flat and tilted
    struct Configuration {
        QString name;
        double  tx_m[2];
        double  ty_m[2];
        double  tz_m[2];
    };

    explicit CalibrationSettingsDialog(double             moveTransMm,
                                       double             moveRotDeg,
                                       const TagPositions &tagPos,
                                       QWidget            *parent = nullptr)
        : QDialog(parent)
    {
        DebugLogger::logEvent("CalibrationSettingsDialog", "constructor started");
        setWindowFlags(Qt::Dialog);
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

        auto *title = new QLabel(tr("Paramètres de calibration"), root);
        title->setAlignment(Qt::AlignCenter);
        title->setStyleSheet(
            "color:white; font-size:18pt; font-weight:bold;");
        vbox->addWidget(title);
        vbox->addWidget(makeSeparator(root));
        vbox->addSpacing(16);

        // ── Configuration selector ─────────────────────────────────────────────
        auto *configRow = new QHBoxLayout;
        configRow->setSpacing(12);
        auto *configLbl = new QLabel(tr("Configuration :"), root);
        configLbl->setStyleSheet(
            "color:#e0e0e0; font-size:13pt;");

        m_configCombo = new QComboBox(root);
        m_configCombo->setStyleSheet(
            "QComboBox { background:#2a2b2d; border:1px solid #444; border-radius:6px;"
            "            color:#e0e0e0; font-size:13pt; padding:8px 10px; min-height:36px; }"
            "QComboBox::drop-down { border:none; width:0; }"
            "QComboBox QAbstractItemView { background:#2a2b2d; color:#e0e0e0;"
            "                             border:1px solid #444; border-radius:6px; }");
        m_configCombo->addItem("Flat (v9)");
        m_configCombo->addItem("Tilted 45° (v10)");
        m_configCombo->setCurrentIndex(0);

        configRow->addWidget(configLbl);
        configRow->addStretch(1);
        configRow->addWidget(m_configCombo, 2);
        vbox->addLayout(configRow);
        vbox->addSpacing(16);

        // ── Scrollable area for settings ────────────────────────────────────────
        auto *scrollArea = new QScrollArea(root);
        scrollArea->setStyleSheet(
            "QScrollArea { border:none; background:transparent; }"
            "QScrollBar:vertical { background:#0d0d0d; width:10px; }"
            "QScrollBar::handle:vertical { background:#3a3a3c; border-radius:5px; min-height:30px; }"
            "QScrollBar::handle:vertical:hover { background:#4a4a4c; }"
            "QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical { border:none; }");
        scrollArea->setWidgetResizable(true);

        auto *scrollContent = new QWidget(scrollArea);
        scrollContent->setObjectName("innerContent");
        scrollContent->setStyleSheet("QWidget#innerContent { background:#0d0d0d; }");
        auto *scrollVBox = new QVBoxLayout(scrollContent);
        scrollVBox->setContentsMargins(0, 0, 0, 0);
        scrollVBox->setSpacing(16);

        // ── Movement thresholds (non-LiDAR recalibration trigger) ─────────────
        auto *moveBox  = new QGroupBox(tr("Seuils de mouvement (recalibration)"), scrollContent);
        auto *moveForm = new QFormLayout(moveBox);
        moveForm->setContentsMargins(16, 20, 16, 16);
        moveForm->setVerticalSpacing(8);
        moveForm->setHorizontalSpacing(16);
        moveForm->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);

        m_moveTransSB = new CalibSpinBox(moveBox);
        m_moveTransSB->setRange(0.1, 100.0);
        m_moveTransSB->setDecimals(1);
        m_moveTransSB->setSingleStep(0.5);
        m_moveTransSB->setSuffix(" mm");
        m_moveTransSB->setValue(moveTransMm);
        moveForm->addRow(tr("Translation (mm) :"), m_moveTransSB);

        m_moveRotSB = new CalibSpinBox(moveBox);
        m_moveRotSB->setRange(0.1, 30.0);
        m_moveRotSB->setDecimals(1);
        m_moveRotSB->setSingleStep(0.1);
        m_moveRotSB->setSuffix(" °");
        m_moveRotSB->setValue(moveRotDeg);
        moveForm->addRow(tr("Rotation (°) :"), m_moveRotSB);

        scrollVBox->addWidget(moveBox);

        // ── Tag positions and rotations ────────────────────────────────────────
        const QString tagLabels[] = { tr("Tag 0 — gauche"), tr("Tag 1 — droit") };
        for (int t = 0; t < 2; ++t) {
            auto *box  = new QGroupBox(tagLabels[t], scrollContent);
            auto *form = new QFormLayout(box);
            form->setContentsMargins(16, 20, 16, 16);
            form->setVerticalSpacing(8);
            form->setHorizontalSpacing(16);
            form->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);

            // Position spinboxes (tx, ty, tz in mm)
            for (int ax = 0; ax < 3; ++ax) {
                m_tagPosSB[t][ax] = new CalibSpinBox(box);
                m_tagPosSB[t][ax]->setRange(-500.0, 500.0);
                m_tagPosSB[t][ax]->setDecimals(1);
                m_tagPosSB[t][ax]->setSingleStep(1.0);
                m_tagPosSB[t][ax]->setSuffix(" mm");
            }
            // Values are stored in meters; display in mm
            m_tagPosSB[t][0]->setValue(tagPos.tx_m[t] * 1000.0);
            m_tagPosSB[t][1]->setValue(tagPos.ty_m[t] * 1000.0);
            m_tagPosSB[t][2]->setValue(tagPos.tz_m[t] * 1000.0);

            form->addRow(tr("tx (mm) :"), m_tagPosSB[t][0]);
            form->addRow(tr("ty (mm) :"), m_tagPosSB[t][1]);
            form->addRow(tr("tz (mm) :"), m_tagPosSB[t][2]);

            // Rotation spinboxes (rx, ry, rz in radians)
            for (int ax = 0; ax < 3; ++ax) {
                m_tagRotSB[t][ax] = new CalibSpinBox(box);
                m_tagRotSB[t][ax]->setRange(-3.14159, 3.14159);
                m_tagRotSB[t][ax]->setDecimals(6);
                m_tagRotSB[t][ax]->setSingleStep(0.017453); // ~1°
                m_tagRotSB[t][ax]->setSuffix(" rad");
            }
            // Initialize with flat configuration defaults (will be updated by preset)
            m_tagRotSB[t][0]->setValue(0.0);           // rx
            m_tagRotSB[t][1]->setValue(3.141592653589793); // ry = π
            m_tagRotSB[t][2]->setValue(0.0);           // rz

            form->addRow(tr("rx (rad) :"), m_tagRotSB[t][0]);
            form->addRow(tr("ry (rad) :"), m_tagRotSB[t][1]);
            form->addRow(tr("rz (rad) :"), m_tagRotSB[t][2]);
            scrollVBox->addWidget(box);
        }

        scrollVBox->addStretch(1);
        scrollArea->setWidget(scrollContent);
        vbox->addWidget(scrollArea, 1);

        vbox->addSpacing(8);
        vbox->addWidget(makeSeparator(root));
        vbox->addSpacing(8);

        // ── Buttons ───────────────────────────────────────────────────────────
        auto *btnRow = new QHBoxLayout;
        btnRow->setSpacing(16);
        auto *btnCancel = makeSecondaryBtn(tr("Annuler"), root);
        auto *btnApply  = makePrimaryBtn(tr("Appliquer"), root);
        btnCancel->setAutoDefault(false);
        btnCancel->setDefault(false);
        btnApply->setAutoDefault(false);
        btnApply->setDefault(false);
        btnRow->addWidget(btnCancel);
        btnRow->addWidget(btnApply);
        vbox->addLayout(btnRow);

        // Load saved configuration preference and apply it
        const QString savedConfig = QSettings().value("tagFrameConfiguration", "Flat (v9)").toString();
        int savedIndex = m_configCombo->findText(savedConfig);
        if (savedIndex >= 0) {
            m_configCombo->setCurrentIndex(savedIndex);
        } else {
            applyConfigurationPreset(0); // Default to flat if saved config not found
        }

        // ── Connect configuration selector ──────────────────────────────────────
        connect(m_configCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
                this, [this](int index) {
            applyConfigurationPreset(index);
        });

        connect(btnCancel, &QPushButton::clicked, this, &QDialog::reject);
        connect(btnApply,  &QPushButton::clicked, this, [this]() {
            DebugLogger::logEvent("CalibrationSettingsDialog", "apply clicked");
            emit movementThresholdsApplied(m_moveTransSB->value(), m_moveRotSB->value());
            for (int t = 0; t < 2; ++t) {
                emit tagPositionApplied(t,
                    m_tagPosSB[t][0]->value() / 1000.0,
                    m_tagPosSB[t][1]->value() / 1000.0,
                    m_tagPosSB[t][2]->value() / 1000.0);
                emit tagRotationApplied(t,
                    m_tagRotSB[t][0]->value(),
                    m_tagRotSB[t][1]->value(),
                    m_tagRotSB[t][2]->value());
            }
            // Save selected configuration
            QSettings().setValue("tagFrameConfiguration", m_configCombo->currentText());
            accept();
        });

        // Prevent auto-focus of first spinbox on open
        btnCancel->setFocus();
    }

private:
    void applyConfigurationPreset(int index)
    {
        // Flat configuration (v9)
        if (index == 0) {
            const double flat_tx[] = {0.2325, -0.0325};
            const double flat_ty[] = {0.100,  0.100};
            const double flat_tz[] = {0.171,  0.171};
            const double flat_rx[] = {0.0, 0.0};
            const double flat_ry[] = {3.141592653589793, 3.141592653589793}; // π
            const double flat_rz[] = {0.0, 0.0};
            for (int t = 0; t < 2; ++t) {
                m_tagPosSB[t][0]->setValue(flat_tx[t] * 1000.0);
                m_tagPosSB[t][1]->setValue(flat_ty[t] * 1000.0);
                m_tagPosSB[t][2]->setValue(flat_tz[t] * 1000.0);
                m_tagRotSB[t][0]->setValue(flat_rx[t]);
                m_tagRotSB[t][1]->setValue(flat_ry[t]);
                m_tagRotSB[t][2]->setValue(flat_rz[t]);
            }
        }
        // Tilted configuration (v10) — 45° around X axis
        else if (index == 1) {
            const double tilted_tx[] = {0.2475, -0.0475};
            const double tilted_ty[] = {0.1077,  0.1077};
            const double tilted_tz[] = {0.1671,  0.1671};
            const double tilted_rx[] = {0.785398163397448, 0.785398163397448}; // π/4 (45°)
            const double tilted_ry[] = {3.141592653589793, 3.141592653589793}; // π
            const double tilted_rz[] = {0.0, 0.0};
            for (int t = 0; t < 2; ++t) {
                m_tagPosSB[t][0]->setValue(tilted_tx[t] * 1000.0);
                m_tagPosSB[t][1]->setValue(tilted_ty[t] * 1000.0);
                m_tagPosSB[t][2]->setValue(tilted_tz[t] * 1000.0);
                m_tagRotSB[t][0]->setValue(tilted_rx[t]);
                m_tagRotSB[t][1]->setValue(tilted_ry[t]);
                m_tagRotSB[t][2]->setValue(tilted_rz[t]);
            }
        }
    }

    ~CalibrationSettingsDialog()
    {
        DebugLogger::logEvent("CalibrationSettingsDialog", "destructor starting");
        // Disable accessibility on all children to prevent iOS crash when destroying buttons
        for (QObject *child : findChildren<QObject*>()) {
            if (QWidget *w = qobject_cast<QWidget*>(child))
                w->setAccessibleDescription(QString());
        }
        DebugLogger::logEvent("CalibrationSettingsDialog", "destructor completed");
    }

signals:
    void movementThresholdsApplied(double transMm, double rotDeg);
    void tagPositionApplied(int tagId, double tx_m, double ty_m, double tz_m);
    void tagRotationApplied(int tagId, double rx_rad, double ry_rad, double rz_rad);

protected:
    void paintEvent(QPaintEvent *e) override { paintBlack(this, e); }

    void showEvent(QShowEvent *e) override
    {
        QDialog::showEvent(e);
        const QRect ag = QGuiApplication::primaryScreen()->availableGeometry();
        move(ag.topLeft());
        // Clear any auto-focus that Qt may have assigned to the first spinbox.
        if (auto *im = QGuiApplication::inputMethod()) im->hide();
    }

    void keyPressEvent(QKeyEvent *e) override
    {
        if (e->key() == Qt::Key_Return || e->key() == Qt::Key_Enter)
            return; // handled per-spinbox; dialog-level Enter does nothing
        QDialog::keyPressEvent(e);
    }

    bool event(QEvent *ev) override {
        // Block all accessibility events to prevent iOS crash during widget destruction
        // Accessibility events are in the range 1000-1030
        if (ev->type() >= 1000 && ev->type() <= 1030) {
            return true;
        }
        return QDialog::event(ev);
    }

private:
    CalibSpinBox *m_moveTransSB  = nullptr;
    CalibSpinBox *m_moveRotSB    = nullptr;
    CalibSpinBox *m_tagPosSB[2][3] = {};
    CalibSpinBox *m_tagRotSB[2][3] = {};
    QComboBox    *m_configCombo  = nullptr;
};

// ─────────────────────────────────────────────────────────────────────────────
// SettingsDialog
// ─────────────────────────────────────────────────────────────────────────────

SettingsDialog::SettingsDialog(const OverlayRenderer::Style &currentStyle,
                               double                        currentMoveTransMm,
                               double                        currentMoveRotDeg,
                               const TagPositions           &currentTagPositions,
                               bool                          hasLidar,
                               bool                          arTestDepthOverlay,
                               QWidget                      *parent)
    : QDialog(parent)
    , m_style(currentStyle)
    , m_moveTransMm(currentMoveTransMm)
    , m_moveRotDeg(currentMoveRotDeg)
    , m_tagPositions(currentTagPositions)
    , m_hasLidar(hasLidar)
    , m_arTestDepthOverlay(arTestDepthOverlay)
{
    logMemoryState("SettingsDialog::constructor");
    DebugLogger::logEvent("SettingsDialog", "constructor started, parent=" + QString::number((quint64)parent) + ", thread=" + QString::number((quint64)QThread::currentThreadId()));
    setWindowFlags(Qt::Dialog);
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
    auto *title = new QLabel(tr("Paramètres"), root);
    title->setAlignment(Qt::AlignCenter);
    title->setStyleSheet(
        "color:white; font-size:20pt; font-weight:bold;");
    vbox->addWidget(title);
    vbox->addSpacing(20);
    vbox->addWidget(makeSeparator(root));
    vbox->addSpacing(20);

    // ── Language selector ─────────────────────────────────────────────────────
    auto *langRow = new QHBoxLayout;
    langRow->setSpacing(12);
    auto *langLbl = new QLabel(tr("Langue :"), root);
    langLbl->setStyleSheet(
        "color:#e0e0e0; font-size:13pt;");
    auto *btnFr = new QPushButton("Français", root);
    auto *btnEn = new QPushButton("English",  root);

    const QString activeLangStyle =
        "QPushButton { background:#1a3030; color:#75D0C5; border-radius:8px;"
        "              padding:8px 20px; font-size:12pt;"
        "              border:1px solid #2a7a70; }"
        "QPushButton:pressed { background:#153030; }";
    const QString inactiveLangStyle =
        "QPushButton { background:#1a1a1a; color:#888; border-radius:8px;"
        "              padding:8px 20px; font-size:12pt;"
        "              border:1px solid #333; }"
        "QPushButton:pressed { background:#222; }";

    {
        const QString cur = QSettings().value("language", "fr").toString();
        btnFr->setStyleSheet(cur == "fr" ? activeLangStyle : inactiveLangStyle);
        btnEn->setStyleSheet(cur == "en" ? activeLangStyle : inactiveLangStyle);
    }
    for (auto *b : {btnFr, btnEn})
        b->setFixedHeight(44);

    langRow->addWidget(langLbl);
    langRow->addStretch(1);
    langRow->addWidget(btnFr);
    langRow->addWidget(btnEn);
    vbox->addLayout(langRow);

    vbox->addSpacing(16);

    // Show a native alert in the TARGET language, save + quit on confirm.
    // Strings are hardcoded per language so the user reads the message in
    // the language they are about to switch to.
    auto applyLang = [this](const QString &lang) {
        if (QSettings().value("language", "fr").toString() == lang) return;

        QString title, message, yesText, noText;
        if (lang == "en") {
            title   = "Apply language?";
            message = "The app will close to apply English.\nReopen it to continue.";
            yesText = "Close now";
            noText  = "Cancel";
        } else {
            title   = "Appliquer la langue ?";
            message = "L'application va se fermer pour appliquer le français.\n"
                      "Rouvrez-la pour continuer.";
            yesText = "Fermer maintenant";
            noText  = "Annuler";
        }

        QMessageBox box(QMessageBox::NoIcon, title, QString(), QMessageBox::NoButton, this);
        auto *yesBtn = box.addButton(yesText, QMessageBox::YesRole);
        auto *noBtn = box.addButton(noText, QMessageBox::NoRole);
        box.setDefaultButton(noBtn);

        // HTML body: title bold + message below
        box.setText(
            "<span style='color:#e0e0e0; font-size:13pt; font-weight:bold;'>"
            + title.toHtmlEscaped() +
            "</span><br><br>"
            "<span style='color:#a0a0a0; font-size:11pt;'>"
            + QString(message).replace('\n', "<br>") +
            "</span>");

        // Dark-theme stylesheet for the whole dialog
        box.setStyleSheet(
            "QMessageBox {"
            "  background-color: #1a1b1d;"
            "  border: 1px solid #3a3b3d;"
            "  border-radius: 10px;"
            "}"
            "QMessageBox QLabel {"
            "  color: #e0e0e0;"
            "  min-width: 280px;"
            "}"
            "QMessageBox QPushButton {"
            "  font-size: 12pt;"
            "  font-weight: bold;"
            "  border-radius: 8px;"
            "  padding: 10px 14px;"
            "}"
        );

        // Style individual buttons
        yesBtn->setStyleSheet(
            "QPushButton { background:#DE5F5E; color:white; border-radius:8px;"
            "              padding:10px 14px;"
            "              font-size:12pt; font-weight:bold; }"
            "QPushButton:pressed { background:#a33c3f; }");
        noBtn->setStyleSheet(
            "QPushButton { background:#2a2b2d; color:#c0c0c0; border-radius:8px;"
            "              padding:10px 14px;"
            "              font-size:12pt; font-weight:bold; }"
            "QPushButton:pressed { background:#3a3b3d; }");

        if (box.exec() && box.clickedButton() == yesBtn) {
            QSettings().setValue("language", lang);
            QCoreApplication::quit();
        }
    };

    connect(btnFr, &QPushButton::clicked, this, [=]() { applyLang("fr"); });
    connect(btnEn, &QPushButton::clicked, this, [=]() { applyLang("en"); });

    // ── Coordinates reference selector (disabled) ─────────────────────────────
    auto *refRow = new QHBoxLayout;
    refRow->setSpacing(12);
    auto *refLbl   = new QLabel(tr("Référentiel :"), root);
    refLbl->setStyleSheet(
        "color:#666; font-size:13pt;");
    auto *btnMdt = new QPushButton("Medtronic", root);
    auto *btnBL  = new QPushButton("BrainLab",  root);
    for (auto *b : {btnMdt, btnBL}) {
        b->setFixedHeight(44);
        b->setEnabled(false);
        b->setStyleSheet(
            "QPushButton { background:#1a1a1a; color:#444; border-radius:8px;"
            "              padding:8px 20px; font-size:12pt;"
            "              border:1px solid #333; }");
    }
    // Medtronic is the default selection — highlight it like the active choice
    btnMdt->setStyleSheet(
        "QPushButton { background:#1a3030; color:#3a6060; border-radius:8px;"
        "              padding:8px 20px; font-size:12pt;"
        "              border:1px solid #2a5050; }");
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
            "  padding:0px 20px; font-size:14pt;"
            "  text-align:left; border:1px solid #333;"
            "}"
            "QPushButton:pressed { background:#2e2e2e; }");
        return b;
    };

    auto *btnGraphics = makeNavBtn("  " + tr("Paramètres graphiques") + "   ›");
    auto *btnCalib    = makeNavBtn("  " + tr("Paramètres de calibration") + "   ›");
    auto *btnDebugLogs = makeNavBtn("  " + tr("Journaux de débogage") + "   ›");
    vbox->addWidget(btnGraphics);
    vbox->addSpacing(12);
    vbox->addWidget(btnCalib);
    vbox->addSpacing(12);
    vbox->addWidget(btnDebugLogs);
    vbox->addStretch(1);
    vbox->addWidget(makeSeparator(root));
    vbox->addSpacing(16);

    // ── Close button ──────────────────────────────────────────────────────────
    auto *btnClose = makePrimaryBtn(tr("Fermer"), root);
    vbox->addWidget(btnClose, 0, Qt::AlignHCenter);

    connect(btnClose, &QPushButton::clicked, this, &QDialog::accept);

    // ── Open sub-dialogs ──────────────────────────────────────────────────────
    connect(btnGraphics, &QPushButton::clicked, this, [this]() {
        DebugLogger::logEvent("SettingsDialog", "btnGraphics clicked: creating GraphicsSettingsDialog");
        auto *dlg = new GraphicsSettingsDialog(m_style, m_hasLidar, m_arTestDepthOverlay, nullptr);
        DebugLogger::logEvent("SettingsDialog", "btnGraphics: created dialog ptr=" + QString::number((quint64)dlg) + ", connecting signals");
        int signalCount = 0;
        connect(dlg, &GraphicsSettingsDialog::styleApplied, this,
                [this](OverlayRenderer::Style s) {
            DebugLogger::logEvent("SettingsDialog", "btnGraphics: styleApplied signal received");
            m_style = s;
            emit styleChanged(s);
        });
        connect(dlg, &GraphicsSettingsDialog::arTestDepthOverlayChanged, this,
                [this](bool enabled) {
            DebugLogger::logEvent("SettingsDialog", "btnGraphics: arTestDepthOverlayChanged signal received");
            m_arTestDepthOverlay = enabled;
            emit arTestDepthOverlayChanged(enabled);
        });
        DebugLogger::logEvent("SettingsDialog", "btnGraphics: showing dialog via exec()");
        dlg->exec();
        DebugLogger::logEvent("SettingsDialog", "btnGraphics: dialog closed, calling deleteLater()");
        dlg->deleteLater();
        DebugLogger::logEvent("SettingsDialog", "btnGraphics: deleteLater() called, returning from click handler");
    });

    connect(btnCalib, &QPushButton::clicked, this, [this]() {
        DebugLogger::logEvent("SettingsDialog", "btnCalib clicked: creating CalibrationSettingsDialog");
        auto *dlg = new CalibrationSettingsDialog(m_moveTransMm, m_moveRotDeg,
                                                  m_tagPositions, nullptr);
        DebugLogger::logEvent("SettingsDialog", "btnCalib: created dialog ptr=" + QString::number((quint64)dlg) + ", connecting signals");
        connect(dlg, &CalibrationSettingsDialog::movementThresholdsApplied, this,
                [this](double transMm, double rotDeg) {
            DebugLogger::logEvent("SettingsDialog", "btnCalib: movementThresholdsApplied signal received");
            m_moveTransMm = transMm;
            m_moveRotDeg  = rotDeg;
            emit movementThresholdsChanged(transMm, rotDeg);
        });
        connect(dlg, &CalibrationSettingsDialog::tagPositionApplied, this,
                [this](int tagId, double tx, double ty, double tz) {
            DebugLogger::logEvent("SettingsDialog", "btnCalib: tagPositionApplied signal received");
            m_tagPositions.tx_m[tagId] = tx;
            m_tagPositions.ty_m[tagId] = ty;
            m_tagPositions.tz_m[tagId] = tz;
            emit tagPositionChanged(tagId, tx, ty, tz);
        });
        connect(dlg, &CalibrationSettingsDialog::tagRotationApplied, this,
                [this](int tagId, double rx, double ry, double rz) {
            DebugLogger::logEvent("SettingsDialog", "btnCalib: tagRotationApplied signal received");
            emit tagRotationChanged(tagId, rx, ry, rz);
        });
        DebugLogger::logEvent("SettingsDialog", "btnCalib: showing dialog via exec()");
        dlg->exec();
        DebugLogger::logEvent("SettingsDialog", "btnCalib: dialog closed, calling deleteLater()");
        dlg->deleteLater();
        DebugLogger::logEvent("SettingsDialog", "btnCalib: deleteLater() called, returning from click handler");
    });

    // ── Debug Logs dialog ─────────────────────────────────────────────────────
    connect(btnDebugLogs, &QPushButton::clicked, this, [this]() {
        DebugLogger::logEvent("SettingsDialog", "btnDebugLogs clicked: creating DebugLogDialog");
        auto *dlg = new DebugLogDialog(this);
        DebugLogger::logEvent("SettingsDialog", "btnDebugLogs: created dialog ptr=" + QString::number((quint64)dlg) + ", showing via exec()");
        dlg->exec();
        DebugLogger::logEvent("SettingsDialog", "btnDebugLogs: dialog closed, calling deleteLater()");
        dlg->deleteLater();
        DebugLogger::logEvent("SettingsDialog", "btnDebugLogs: deleteLater() called, returning from click handler");
    });
}

SettingsDialog::~SettingsDialog()
{
    logMemoryState("SettingsDialog::destructor");
    DebugLogger::logEvent("SettingsDialog", "destructor: entered, thread=" + QString::number((quint64)QThread::currentThreadId()));

    // Log widget count and memory state
    int childCount = findChildren<QObject*>().count();
    DebugLogger::logEvent("SettingsDialog", "destructor: found " + QString::number(childCount) + " child objects");

    // Disable accessibility on all children to prevent iOS crash when destroying buttons
    int widgetCount = 0;
    for (QObject *child : findChildren<QObject*>()) {
        if (QWidget *w = qobject_cast<QWidget*>(child)) {
            w->setAccessibleDescription(QString());
            widgetCount++;
        }
    }
    DebugLogger::logEvent("SettingsDialog", "destructor: cleared accessibility on " + QString::number(widgetCount) + " widgets");
    DebugLogger::logEvent("SettingsDialog", "destructor: about to return (destruction complete)");
}

void SettingsDialog::paintEvent(QPaintEvent *e) { paintBlack(this, e); }

bool SettingsDialog::event(QEvent *event)
{
    // Block all accessibility events to prevent iOS crash during widget destruction
    // Accessibility events are in the range 1000-1030
    if (event->type() >= 1000 && event->type() <= 1030) {
        static int accessibilityEventCount = 0;
        if (++accessibilityEventCount % 10 == 0) {
            DebugLogger::logEvent("SettingsDialog", "event: blocked accessibility type=" + QString::number(event->type()) + " (count=" + QString::number(accessibilityEventCount) + ")");
        }
        return true;
    }
    return QDialog::event(event);
}

#include "SettingsDialog.moc"
