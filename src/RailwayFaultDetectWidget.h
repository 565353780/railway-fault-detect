#ifndef RAILWAYFAULTDETECTWIDGET_H
#define RAILWAYFAULTDETECTWIDGET_H

#include <QWidget>
#include <QFile>
#include <QString>
#include <QPainter>
#include <QImage>
#include <QVector>
#include <QThread>
#include <QKeyEvent>

#include "DataReader.h"
#include "DarknetDetector.h"
#include "LapnetDetector.h"
#include "PostProcesser.h"

namespace Ui {
class RailwayFaultDetectWidget;
}

class RailwayFaultDetectWidget : public QWidget
{
    Q_OBJECT

public:
    explicit RailwayFaultDetectWidget(QWidget *parent = nullptr);
    ~RailwayFaultDetectWidget();

private:
    bool init_model();

private:
    Ui::RailwayFaultDetectWidget *ui;

    DataReader* data_reader_;
    QThread* data_reader_thread_;

    DarknetDetector* darknet_detector_;
    QThread* darknet_detector_thread_;

    LapnetDetector* lapnet_line_detector_;
    QThread* lapnet_line_detector_thread_;

    LapnetDetector* lapnet_point_detector_;
    QThread* lapnet_point_detector_thread_;

    PostProcesser* post_processer_;
    QThread* post_processer_thread_;

    QImage* data_image_;

    QVector<BBox>* darknet_bbox_vec_;
    QImage* lapnet_output_image_line_;
    QImage* lapnet_output_image_point_;

private slots:
    void on_Btn_SelectFile_clicked();
    void on_Btn_SelectFolder_clicked();

    void on_Btn_InitData_clicked();

    void on_Btn_StartDetect_clicked();

    void slot_getData_finished(bool succeed);

    void slot_darknetDetect_finished(bool succeed);

    void slot_lapnetDetectLine_finished(bool succeed);

    void slot_lapnetDetectPoint_finished(bool succeed);

    void slot_postProcess_finished(bool succeed);

signals:
    void signal_getData();

    void signal_darknetDetect();

    void signal_lapnetLineDetect();

    void signal_lapnetPointDetect();

    void signal_postProcess();
};

#endif // RAILWAYFAULTDETECTWIDGET_H
