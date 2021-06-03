#ifndef DATAREADER_H
#define DATAREADER_H

#include <QObject>
#include <QString>
#include <QQueue>
#include <QDir>
#include <QDebug>
#include <QImage>

#include "VideoCap.h"

enum DataMode
{
    Data_Free = 0,
    Data_Camera = 1,
    Data_Image = 2,
    Data_Video = 3,
    Data_Folder = 4
};

class DataReader : public QObject
{
    Q_OBJECT

public:
    explicit DataReader(QObject *parent = nullptr);
    ~DataReader();

    bool readCamera(QString camera_path);

    bool readImage(QString image_path);

    bool readVideo(QString video_path);

    bool readFolder(QString folder_path);

    void setDataMode(DataMode data_mode);
    DataMode getDataMode();

    bool readData(QString input_path);

    QImage* getImage(){return &image_;}

    QString getCurrentImagePath(){return current_iamge_path_;}

public slots:
    bool slot_getData();

signals:
    void signal_getData_finished(bool);

private:
    DataMode data_mode_;

    VideoCap* video_cap_;
    bool need_update_camera_image_;
    QImage image_;

    QQueue<QString> image_path_queue_;
    QString current_iamge_path_;
};

#endif // DATAREADER_H
