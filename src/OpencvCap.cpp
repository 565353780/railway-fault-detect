#include "OpencvCap.h"

OpencvCap::OpencvCap(QObject* parent)// : QThread(parent)
{
    dequeue_frame_ = false;
}

OpencvCap::~OpencvCap()
{
    if(m_cap.isOpened())
    {
        m_cap.release();
    }
}

void OpencvCap::set_param(QString url, bool dequeue_frame)
{
    url_ = url;
    dequeue_frame_ = dequeue_frame;

    m_queue_frame_.clear();

    if(m_cap.isOpened())
    {
        m_cap.release();
    }

    m_cap.open(url_.toStdString());
}


void OpencvCap::put_frame(Mat frame)
{
    if(m_queue_frame_.size() > 5)
    {
        if(dequeue_frame_)
        {
            m_queue_frame_.dequeue();
        }
    }
    //存入容器
    m_queue_frame_.enqueue(frame);
    return;
}

bool OpencvCap::get_frame(Mat &frame)
{
    if(m_queue_frame_.size() < 1)
    {
        QThread::msleep(1);
        return false;
    }

    //从容器中取图像
    frame = m_queue_frame_.dequeue();
    return true;
}

bool OpencvCap::is_read_finished()
{
    if(m_queue_frame_.size() == 0 && read_finished_)
    {
        return true;
    }

    return false;
}

void OpencvCap::run()
{
    read_finished_ = false;

    if(!m_cap.isOpened())
    {
        m_cap.open(url_.toStdString());
        if(!m_cap.isOpened())
        {
            qDebug("cannot open the videocapture\n");
            return ;
        }
    }
    Mat current_frame;
    while(true)
    {
        m_cap >> current_frame;
        if(current_frame.empty())
        {
            qDebug("frame empty\n");
            read_finished_ = true;
            return ;
        }
        put_frame(current_frame);
    }
}
