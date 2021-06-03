#include "DataReader.h"

DataReader::DataReader(QObject *parent) : QObject (parent)
{
    video_cap_ = new VideoCap(this);

    data_mode_ = Data_Free;
}

DataReader::~DataReader()
{

}

bool DataReader::readCamera(QString camera_path)
{
    data_mode_ = Data_Camera;

    video_cap_->set_param(camera_path, 0, 300, "../Cap_Video", true);

    qDebug() << "init camera finished!";

    return true;
}

bool DataReader::readImage(QString image_path)
{
    data_mode_ = Data_Image;

    image_path_queue_.clear();

    QDir dir_;

    if(!dir_.exists(image_path))
    {
        return false;
    }

    image_path_queue_.enqueue(image_path);

    return true;
}

bool DataReader::readVideo(QString video_path)
{
    data_mode_ = Data_Video;

    video_cap_->set_param(video_path, 0, 300, "../Cap_Video", false);

    return true;
}

bool DataReader::readFolder(QString folder_path)
{
    data_mode_ = Data_Folder;

    image_path_queue_.clear();

    QDir dir_;

    if(!dir_.exists(folder_path))
    {
        return false;
    }

    QStringList image_filter;
    image_filter << "*.jpg";

    dir_.setNameFilters(image_filter);
    dir_.setFilter(QDir::NoDotAndDotDot | QDir::Files);

    dir_.setPath(folder_path);

    QStringList image_path_list = dir_.entryList();

    if(image_path_list.size() == 0)
    {
        qDebug() << "DataReader::readVideo : no image found.";
        return false;
    }

    for(QString image_path : image_path_list)
    {
        image_path_queue_.enqueue(folder_path + image_path);
    }

    return true;
}

void DataReader::setDataMode(DataMode data_mode)
{
    data_mode_ = data_mode;
}

DataMode DataReader::getDataMode()
{
    return data_mode_;
}

bool DataReader::readData(QString input_path)
{
    bool read_data_succeed = false;

    switch(data_mode_)
    {
    case Data_Free:
        break;
    case Data_Camera:

        read_data_succeed = readCamera(input_path);

        break;
    case Data_Image:

        read_data_succeed = readImage(input_path);

        break;
    case Data_Video:

        read_data_succeed = readVideo(input_path);

        break;
    case Data_Folder:

        read_data_succeed = readFolder(input_path);

        break;
    }

    return read_data_succeed;
}

bool DataReader::slot_getData()
{
    bool get_data_succeed = false;

    switch(data_mode_)
    {
    case Data_Free:
        break;
    case Data_Camera:

        get_data_succeed = video_cap_->slot_updateVideoImage(image_);

        if(image_.format() != QImage::Format_RGB888)
        {
            image_ = image_.convertToFormat(QImage::Format_RGB888);
        }

        break;
    case Data_Image:

        if(image_path_queue_.size() > 0)
        {
            current_iamge_path_ = image_path_queue_.dequeue();

            image_ = QImage(current_iamge_path_).copy();

            if(image_.format() != QImage::Format_RGB888)
            {
                image_ = image_.convertToFormat(QImage::Format_RGB888);
            }

            get_data_succeed = true;
        }

        break;
    case Data_Video:

        get_data_succeed = video_cap_->slot_updateVideoImage(image_);

        break;
    case Data_Folder:

        if(image_path_queue_.size() > 0)
        {
            current_iamge_path_ = image_path_queue_.dequeue();

            image_ = QImage(current_iamge_path_).copy();

            if(image_.format() != QImage::Format_RGB888)
            {
                image_ = image_.convertToFormat(QImage::Format_RGB888);
            }

            get_data_succeed = true;
        }

        break;
    }

    emit signal_getData_finished(get_data_succeed);

    return get_data_succeed;
}

