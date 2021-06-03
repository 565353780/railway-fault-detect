#include <iostream>
#include <stdlib.h>
#include <QObject>
#include <QString>
#include <QDir>
#include <QDebug>
#include <QImage>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<time.h>

#include "OpenCVCap.h"

class VideoCap : public QObject
{
    Q_OBJECT

public:
    explicit VideoCap(QObject *parent = 0);
    ~VideoCap();

public:
    void set_param(QString video_path, int camera_idx, int cap_time, QString cap_video_path, bool dequeue_frame);
    bool connect_camera();
    bool update_video_writer();
    void set_save_video_mode(bool is_save_video=false);

    QString local_index();

    QImage cvMat2QImage(const cv::Mat& mat);

    cv::Mat QImage2cvMat(QImage image);

    QImage receive_mjpg();

private:
    QString video_path_;
    int camera_idx_;
    int cap_time_;
    QString cap_video_path_;

    QString file_title_;

    cv::VideoWriter videowriter_;

    int img_width_;
    int img_height_;
    int fps_;

    bool start_video_cap_;
    bool is_save_video_;
    time_t time_start_;
    time_t time_now_;

    OpencvCap *m_opencvcap;

public slots:
    bool slot_updateVideoImage(QImage& camera_image);

signals:
    void signal_updateVideoImage_finished(bool);
};
