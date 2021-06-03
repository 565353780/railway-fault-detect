#ifndef OPENCVCAP_H
#define OPENCVCAP_H

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <QThread>
#include <QString>
#include <QDebug>
#include <QQueue>

using namespace cv;
using namespace std;

class OpencvCap :public QThread
{
    Q_OBJECT

public:
    explicit OpencvCap(QObject *parent = 0);
    ~OpencvCap();

    void set_param(QString url, bool dequeue_frame);

    void put_frame(Mat frame);
    bool get_frame(Mat &frame);

    bool is_read_finished();

private:
    void run() override;

public:
    VideoCapture m_cap;

    QString url_;

private:
    QQueue<Mat> m_queue_frame_;
    bool dequeue_frame_;

    bool read_finished_;
};

#endif // OPENCVCAP_H
