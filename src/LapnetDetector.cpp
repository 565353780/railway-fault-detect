#include "LapnetDetector.h"

LapnetDetector::LapnetDetector(QObject* parent) : QObject(parent)
{
    http_api_ = new GCL::HttpApi(this);
}

LapnetDetector::~LapnetDetector()
{

}

void LapnetDetector::setPort(QString port)
{
    port_ = port;
    url_ = "http://127.0.0.1:" + port_ + "/predict";
}

bool LapnetDetector::lapnet_process()
{
    QByteArray image_bytes;

    QBuffer buffer(&image_bytes);

    buffer.open(QFile::WriteOnly);

    input_image_->save(&buffer, "JPG", 100);

    image_bytes = image_bytes.toBase64();

    QJsonObject input_Obj;
    input_Obj.insert("Image", image_bytes.data());

    QJsonDocument doc = QJsonDocument(input_Obj);

    QByteArray bytes = doc.toJson();

    QJsonObject result_json = http_api_->post(url_, bytes);

    if(result_json.empty())
    {
        return false;
    }

    QByteArray output_bytes = QByteArray::fromBase64(result_json.value("OutputImage").toString().toUtf8());

    output_image_.loadFromData(output_bytes);

    return true;
}

void LapnetDetector::slot_lapnetDetect()
{
    bool succeed = lapnet_process();

    emit signal_lapnetDetect_finished(succeed);
}
