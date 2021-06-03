#include "ShowImageWidget.h"
#include "ui_showimagewidget.h"

ShowImageWidget::ShowImageWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::ShowImageWidget)
{
    ui->setupUi(this);

    show_image_ = new QImage();

    bbox_vec_ = new QVector<BBox>();
}

ShowImageWidget::~ShowImageWidget()
{
    delete ui;
}

QImage* ShowImageWidget::getQImage()
{
    return show_image_;
}

QVector<BBox>* ShowImageWidget::getBBoxVec()
{
    return bbox_vec_;
}

void ShowImageWidget::paintEvent(QPaintEvent *event)
{
    Q_UNUSED(event);

    QPainter painter(this);

    if(show_image_->width() > 0 && show_image_->height() > 0)
    {
        painter.drawImage(QRect(0, 0, this->width(), this->height()), *show_image_);

        painter.setPen(QPen(QColor(0, 160, 230), 3));

        double scale_x = 1.0 * this->width() / show_image_->width();
        double scale_y = 1.0 * this->height() / show_image_->height();

        if(bbox_vec_->size() > 0)
        {
            for(int i = 0; i < bbox_vec_->size(); ++i)
            {
                painter.drawRect(int((*bbox_vec_)[i].x * scale_x), int((*bbox_vec_)[i].y * scale_y), int((*bbox_vec_)[i].w * scale_x), int((*bbox_vec_)[i].h * scale_y));
            }
        }
    }
}

void ShowImageWidget::slot_updateShowImage()
{
    this->repaint();
    this->update();
}
