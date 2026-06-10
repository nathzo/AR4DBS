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
                            double                        currentMoveTransMm,
                            double                        currentMoveRotDeg,
                            const TagPositions           &currentTagPositions,
                            bool                          hasLidar,
                            bool                          arTestDepthOverlay,
                            QWidget                      *parent = nullptr);
    ~SettingsDialog();

signals:
    void styleChanged(OverlayRenderer::Style style);
    void reprojThresholdChanged(double px);
    void movementThresholdsChanged(double transMm, double rotDeg);
    void tagPositionChanged(int tagId, double tx_m, double ty_m, double tz_m);
    void arTestDepthOverlayChanged(bool enabled);

protected:
    void paintEvent(QPaintEvent *event) override;

private:
    OverlayRenderer::Style m_style;
    double                 m_reprojThreshold = 1.0;
    double                 m_moveTransMm     = 10.0;
    double                 m_moveRotDeg      = 1.0;
    TagPositions           m_tagPositions;
    bool                   m_hasLidar             = false;
    bool                   m_arTestDepthOverlay   = true;
};
