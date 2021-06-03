#ifndef LAPNETDETECTOR_H
#define LAPNETDETECTOR_H

#include <QObject>
#include <QImage>
#include <QString>
#include <QJsonObject>
#include <QJsonDocument>
#include <QByteArray>
#include <QDebug>
#include <QDir>
#include <QBuffer>

#include "HttpApi.h"

class LapnetDetector : public QObject
{
    Q_OBJECT

public:
    explicit LapnetDetector(QObject* parent = nullptr);
    ~LapnetDetector();

    void setPort(QString port);

    bool lapnet_process();

    void setImage(QImage* input_image){input_image_ = input_image;}

    QImage* getOutputImage(){return &output_image_;}

private:
    QImage* input_image_;

    QString port_;
    QString url_;

    GCL::HttpApi* http_api_;

    QImage output_image_;

    QDir dir_;

public slots:
    void slot_lapnetDetect();

signals:
    void signal_lapnetDetect_finished(bool);
};

#endif // LAPNETDETECTOR_H
