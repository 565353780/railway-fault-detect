#ifndef DARKNETDETECTOR_H
#define DARKNETDETECTOR_H

#include <QObject>
#include <QMainWindow>
#include <QWidget>
#include <QImage>
#include <QFileDialog>
#include <QTimer>
#include <QDebug>
#include <QVector>

#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#ifdef WIN32
#include <yolo_v2_class.hpp>
#endif

#ifdef Linux
//extern "C"
//{
#include "../../include/darknet.h"
//}
#include <vector>
#include <string.h>
#endif

struct BBox
{
    double x = 0.0;
    double y = 0.0;
    double w = 0.0;
    double h = 0.0;
    int label = 0;
    double score = 0.0;
};

#ifdef WIN32
class DarknetDetector : public QObject
{
    Q_OBJECT

public:
    explicit DarknetDetector(const std::string &yolov3_cfg, const std::string &yolov3_weights, const std::string &coco_names, QObject *parent = nullptr);
    ~DarknetDetector();

public:
    void slot_LoadImage(QString image_path);
    void slot_LoadVideo(QString video_path);

    static QImage cvMat2QImage(const cv::Mat mat);

    static cv::Mat QImage2cvMat(QImage image);

    bool darknet_process(cv::Mat data);

    bool darknet_process(QImage data);

    void setImage(QImage* image){detect_image_ = image;}
    QVector<BBox>* getBBoxVec(){return &bbox_vec_;}

private:
    void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names,
        int current_det_fps = -1, int current_cap_fps = -1);

    std::vector<std::string> objects_names_from_file(std::string const filename);

    std::vector<bbox_t> getDarknetResult(cv::Mat mat);

private:
    QImage* detect_image_;
    QVector<BBox> bbox_vec_;

    cv::Mat mat_;
    cv::VideoCapture capture_;

    Detector* detector_;
    std::vector<std::string> obj_names_;

public slots:
    void slot_darknetDetect();

signals:
    void signal_darknetDetect_finished(bool);
};
#endif

#ifdef Linux
class DarknetDetector : public QObject
{
    Q_OBJECT

public:
    explicit DarknetDetector(const std::string &yolov3_cfg, const std::string &yolov3_weights, const std::string &coco_data, QObject *parent = nullptr);

    ~DarknetDetector();

    std::vector<std::pair<char *, std::vector<float>>> getDarknetResult(image img, float thresh=0.5, float hier_thresh=0.5, float nms=0.45);

    std::vector<std::pair<char *, std::vector<float>>> getDarknetResult(char *img, float thresh=0.5, float hier_thresh=0.5, float nms=0.45);

    std::vector<std::pair<char *, std::vector<float>>> getDarknetResult(float *img, int w, int h, int c, float thresh=0.5, float hier_thresh=0.5, float nms=0.45);

private:
    std::vector<std::pair<char *, std::vector<float>>> detect(image im, float thresh=0.5, float hier_thresh=0.5, float nms=0.45);

    std::vector<std::pair<char *, std::vector<float>>> detect(char *img, float thresh=0.5, float hier_thresh=0.5, float nms=0.45);

    std::vector<std::pair<char *, std::vector<float>>> detect(float *img, int w, int h, int c, float thresh=0.5, float hier_thresh=0.5, float nms=0.45);

    bool darknet_process(ConnectedData &data);

    network *net;
    metadata meta;
};
#endif

#endif // DARKNETDETECTOR_H
