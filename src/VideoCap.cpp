#include "VideoCap.h"

VideoCap::VideoCap(QObject *parent) : QObject(parent)
{
    m_opencvcap = new OpencvCap(this);
}

VideoCap::~VideoCap()
{
    delete m_opencvcap;
}

void VideoCap::set_param(QString video_path, int camera_idx, int cap_time, QString cap_video_path, bool dequeue_frame)
{
    video_path_ = video_path;
    camera_idx_= camera_idx;
    cap_time_ = cap_time;
    cap_video_path_ = cap_video_path;

    QDir dir(cap_video_path_);

    cap_video_path_ = dir.absoluteFilePath(cap_video_path_);

    if(!dir.exists())
    {
        dir.mkdir(cap_video_path_);
    }

    file_title_ = cap_video_path_ + "/Camera" + QString::number(camera_idx) + "_";

    start_video_cap_ = false;
    is_save_video_ = false;

    m_opencvcap->set_param(video_path_, dequeue_frame);
}

bool VideoCap::connect_camera()
{
    m_opencvcap->start();

    img_width_ = int(m_opencvcap->m_cap.get(3));

    if(img_width_ == 0)
    {
        qDebug() << "Connecting to the web camera failed. VideoCap system is stopped. Please try again." << endl;
        return false;
    }
    qDebug() << "Camera connected.";

    img_height_ = int(m_opencvcap->m_cap.get(4));

    fps_ = int(m_opencvcap->m_cap.get(5));

    time(&time_start_);

    time_now_ = time_start_;

    return true;
}

bool VideoCap::update_video_writer()
{
    videowriter_.open((file_title_ + local_index() + ".avi").toStdString(), cv::CAP_OPENCV_MJPEG, fps_, cv::Size(img_width_, img_height_));

    return true;
}

void VideoCap::set_save_video_mode(bool is_save_video)
{
    is_save_video_ = is_save_video;
}

QString VideoCap::local_index()
{
    QString local_index;

    struct tm *p;

    time_t time_p;

    time(&time_p);

    p = gmtime(&time_p);

    local_index += QString::number(1900+p->tm_year) + "_";
    local_index += QString::number(1+p->tm_mon) + "_";
    local_index += QString::number(p->tm_mday) + "_";
    local_index += QString::number(8+p->tm_hour) + "_";
    local_index += QString::number(p->tm_min) + "_";
    local_index += QString::number(p->tm_sec);

    return local_index;
}

QImage VideoCap::cvMat2QImage(const cv::Mat& mat)
{
    // 8-bits unsigned, NO. OF CHANNELS = 1
    if(mat.type() == CV_8UC1)
    {
        QImage image(mat.cols, mat.rows, QImage::Format_Indexed8);
        // Set the color table (used to translate colour indexes to qRgb values)
        image.setColorCount(256);
        for(int i = 0; i < 256; i++)
        {
            image.setColor(i, qRgb(i, i, i));
        }
        // Copy input Mat
        uchar *pSrc = mat.data;
        for(int row = 0; row < mat.rows; row ++)
        {
            uchar *pDest = image.scanLine(row);
            memcpy(pDest, pSrc, mat.cols);
            pSrc += mat.step;
        }
        return image;
    }
    // 8-bits unsigned, NO. OF CHANNELS = 3
    else if(mat.type() == CV_8UC3)
    {
        // Copy input Mat
        const uchar *pSrc = (const uchar*)mat.data;
        // Create QImage with same dimensions as input Mat
        QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
        return image.rgbSwapped();
    }
    else if(mat.type() == CV_8UC4)
    {
        qDebug() << "CV_8UC4";
        // Copy input Mat
        const uchar *pSrc = (const uchar*)mat.data;
        // Create QImage with same dimensions as input Mat
        QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32);
        return image.copy();
    }
    else
    {
        qDebug() << "ERROR: Mat could not be converted to QImage.";
        return QImage();
    }
}

cv::Mat VideoCap::QImage2cvMat(QImage image)
{
    cv::Mat mat;

//    qDebug() << "VideoCap::QImage2cvMat : image.format() = " << image.format();

    switch(image.format())
    {
    case QImage::Format_ARGB32:
    case QImage::Format_RGB32:
    case QImage::Format_ARGB32_Premultiplied:
        mat = cv::Mat(image.height(), image.width(), CV_8UC4, (void*)image.constBits(), image.bytesPerLine());
        break;
    case QImage::Format_RGB888:
        mat = cv::Mat(image.height(), image.width(), CV_8UC3, (void*)image.constBits(), image.bytesPerLine());
        cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);
        break;
    case QImage::Format_Indexed8:
        mat = cv::Mat(image.height(), image.width(), CV_8UC1, (void*)image.constBits(), image.bytesPerLine());
        break;
    }
    return mat;
}

QImage VideoCap::receive_mjpg()
{
    cv::Mat frame;

    if(m_opencvcap->m_cap.isOpened())
    {
        int retry_times = 0;

        while(!m_opencvcap->get_frame(frame))
        {
            ++retry_times;

            if(m_opencvcap->is_read_finished())
            {
                return QImage();
            }
        }

//        qDebug() << "trying get_frame " << retry_times << " times";

        if(is_save_video_)
        {
            if(time_now_ == time_start_)
            {
                update_video_writer();

                time(&time_start_);

                time_now_ = time_start_ + 1;
            }
            else
            {
                time(&time_now_);
            }

            if(time_now_ - time_start_ > cap_time_)
            {
                update_video_writer();

                time_start_ = time_now_;
                time_now_ = time_start_ + 1;
            }

            videowriter_ << frame;
        }
    }

    return cvMat2QImage(frame);

    QImage img;

    if(frame.channels()==3)
    {
        cv::Mat rgb;

        cv::cvtColor(frame,rgb,cv::COLOR_BGR2RGB);

        img = QImage((const unsigned char*)(rgb.data), rgb.cols ,rgb.rows, rgb.cols*rgb.channels(), QImage::Format_RGB888);
    }
    else
    {
        img = QImage((const unsigned char*)(frame.data), frame.cols, frame.rows, frame.cols*frame.channels(), QImage::Format_RGB888);
    }

    return img;
}

bool VideoCap::slot_updateVideoImage(QImage& camera_image)
{
    if(!start_video_cap_)
    {
        start_video_cap_ = connect_camera();
    }

    if(start_video_cap_)
    {
        camera_image = receive_mjpg();

        if(camera_image.isNull())
        {
            qDebug() << "VideoCap::slot_updateVideoImage : connect_camera_failed.";

            emit signal_updateVideoImage_finished(false);

            return false;
        }
        else
        {
            emit signal_updateVideoImage_finished(true);

            return true;
        }
    }
    else
    {
        qDebug() << "VideoCap::slot_updateVideoImage : connect_camera_failed.";

        emit signal_updateVideoImage_finished(false);

        return false;
    }
}
