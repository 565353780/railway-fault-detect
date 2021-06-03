#ifndef SHOWIMAGEWIDGET_H
#define SHOWIMAGEWIDGET_H

#include <QWidget>
#include <QPainter>

#include "DarknetDetector.h"

namespace Ui {
class ShowImageWidget;
}

class ShowImageWidget : public QWidget
{
    Q_OBJECT

public:
    explicit ShowImageWidget(QWidget *parent = nullptr);
    ~ShowImageWidget();

    QImage* getQImage();

    QVector<BBox>* getBBoxVec();

private:
    void paintEvent(QPaintEvent *event);

public slots:
    void slot_updateShowImage();

private:
    Ui::ShowImageWidget *ui;

    QImage* show_image_;
    QVector<BBox>* bbox_vec_;
};

#endif // SHOWIMAGEWIDGET_H
