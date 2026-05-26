#pragma once
#include <QDialog>
#include "core/rendering/OverlayRenderer.h"

struct TagPositions {
    double tx_m[2] = {0.2325, -0.0325};
    double ty_m[2] = {0.100,   0.100};
    double tz_m[2] = {0.170,   0.170};
};

class SettingsDialog : public QDialog
{
    Q_OBJECT
public:
    explicit SettingsDialog(const OverlayRenderer::Style &currentStyle,
                            double                        currentReprojThreshold,
                            const TagPositions           &currentTagPositions,
                            QWidget                      *parent = nullptr);

signals:
    void styleChanged(OverlayRenderer::Style style);
    void reprojThresholdChanged(double px);
    void tagPositionChanged(int tagId, double tx_m, double ty_m, double tz_m);

protected:
    void paintEvent(QPaintEvent *event) override;
};
