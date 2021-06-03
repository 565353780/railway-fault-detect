#include "HttpApi.h"
#include <QDebug>
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QEventLoop>
#include <QJsonDocument>
#include <QFile>
#include <QFileInfo>
#include <QDir>
#include <QDataStream>
#include <QTimer>
namespace GCL {
#define DOWNLOAD_TEMP_SUFFIX ".tmp"
#define DOWNLOAD_PROGRESS_SUFFIX ".tpp"
HttpApi::HttpApi(QObject *parent, int timeout):
    QObject(parent),isRunning_(false),download_reply_(nullptr),bytesLastReceived_(0),bytesReceived_(0),
    timeout_(timeout)
{
    download_manager_ = new QNetworkAccessManager(this);
}

HttpApi::~HttpApi()
{

}

void HttpApi::error(QNetworkReply::NetworkError e)
{
    qDebug()<<e;
}

void HttpApi::onDownloadFinished()
{
    if(!isRunning_) return;
    isRunning_ = false;

    if(QFile::exists(save_filename_+DOWNLOAD_TEMP_SUFFIX))
    {
        if(QFile::exists(filename_))
        {
            QFile::remove(filename_);
        }

        if(!download_reply_->error())
        {
            QFile::rename(save_filename_+DOWNLOAD_TEMP_SUFFIX,filename_);
        }
        else
        {
            QFile::remove(save_filename_+DOWNLOAD_TEMP_SUFFIX);
        }
    }
    if(QFile::exists(save_filename_+DOWNLOAD_PROGRESS_SUFFIX))
    {
        QFile::remove(save_filename_+DOWNLOAD_PROGRESS_SUFFIX);
    }
    //qDebug()<<"Download Done";
    emit downloadFinished();
}

void HttpApi::onDownloadReadyRead()
{
    if(!isRunning_) return;
    if(download_reply_->error())
    {
        return;
    }

    QFile file(save_filename_+DOWNLOAD_TEMP_SUFFIX);
    if (file.open(QIODevice::WriteOnly | QIODevice::Append))
    {
        QByteArray arr = download_reply_->readAll();

        bytesWrited_ += arr.length();
        file.write(arr);
    }
    file.close();

    QFile ppfile(save_filename_+DOWNLOAD_PROGRESS_SUFFIX);
    if(ppfile.open(QIODevice::WriteOnly))
    {
        ppfile.write(QByteArray::number(bytesWrited_));
    }
    ppfile.close();
}

void HttpApi::onDownloadProgress(qint64 bytesReceived, qint64 bytesTotal)
{
    bytesReceived_ = bytesReceived;
    bytesTotal_ = bytesTotal;
//    qDebug()<<bytesReceived<<" "<<bytesTotal;
    emit downloadProgress(bytesReceived+bytesLastReceived_, bytesTotal+bytesLastReceived_);
}

void HttpApi::stopDownload()
{
    if(!isRunning_) return;

    bytesLastReceived_ += bytesReceived_;
    isRunning_ = false;
    if(download_reply_)
    {
        disconnect(download_reply_, SIGNAL(downloadProgress(qint64, qint64)), this, SLOT(onDownloadProgress(qint64, qint64)));
        disconnect(download_reply_, SIGNAL(readyRead()), this, SLOT(onDownloadReadyRead()));
        disconnect(download_reply_, SIGNAL(finished()), this, SLOT(onDownloadFinished()));
        disconnect(download_reply_, SIGNAL(error(QNetworkReply::NetworkError)), this, SLOT(error(QNetworkReply::NetworkError)));

        download_reply_->abort();
        download_reply_->deleteLater();
        download_reply_ = nullptr;
    }
}

void HttpApi::post(const QString &url, const QByteArray &arr, QByteArray &res)
{
    QNetworkAccessManager manager;
    QNetworkRequest request;
    request.setUrl(QUrl(url));
    request.setHeader(QNetworkRequest::ContentTypeHeader,"application/x-www-form-urlencoded");
    request.setHeader(QNetworkRequest::ContentLengthHeader,arr.length());

    QNetworkReply *reply = manager.post(request,arr);
    QEventLoop eventLoop;
    connect(this,SIGNAL(stop()), &eventLoop,SLOT(quit()));
    connect(&manager, SIGNAL(finished(QNetworkReply*)), &eventLoop, SLOT(quit()));
    QTimer *event_timeout_timer=nullptr;
    if(timeout_>0){
        event_timeout_timer=new QTimer(this);
        connect(event_timeout_timer,&QTimer::timeout,&eventLoop, &QEventLoop::quit);
        event_timeout_timer->setSingleShot(true);
        event_timeout_timer->start(timeout_);
    }
    eventLoop.exec();       //block until finish

//    qDebug()<<reply->readAll();

    res = reply->readAll();

    if(event_timeout_timer){
        event_timeout_timer->stop();
        delete event_timeout_timer;
    }
//    reply->deleteLater();
}

QJsonObject HttpApi::post(const QString &url, const QByteArray &arr)
{
    QByteArray res;
    this->post(url,arr,res);
//    qDebug()<<"finish post";
    QJsonParseError parse_error;
    QJsonDocument doc = QJsonDocument::fromJson(res, &parse_error);
    if(parse_error.error!=QJsonParseError::NoError){
        qDebug()<<"error:"<<parse_error.errorString();
    }
    return doc.object();
}

QString HttpApi::postToGetString(const QString &url, const QByteArray &arr)
{
    QByteArray res;
    this->post(url,arr,res);
    return QString(res);
}

void HttpApi::get(const QString &url, QByteArray &res){
    QNetworkAccessManager manager;
    QNetworkRequest request;
    request.setUrl(QUrl(url));

    QNetworkReply *reply = manager.get(request);
    QEventLoop eventLoop;
    connect(this,SIGNAL(stop()), &eventLoop,SLOT(quit()));
    connect(&manager, SIGNAL(finished(QNetworkReply*)), &eventLoop, SLOT(quit()));
    eventLoop.exec();       //block until finish

    res = reply->readAll();
}

QJsonObject HttpApi::getToGetObject(const QString &url){
    QByteArray res;
    this->get(url,res);
    QJsonDocument doc = QJsonDocument::fromJson(res);
    return doc.object();
}

QString HttpApi::getToGetString(const QString &url){
    QByteArray res;
    this->get(url,res);
    return QString(res);
}

void HttpApi::upload(const QString &url, const QString &filename, QJsonObject &res)
{

    QFile file(filename);
    if(!file.exists())
    {
        qDebug()<<"file does not exsit!";
    }
    if(!file.open (QIODevice::ReadOnly))
    {
        qDebug()<<"file open failed.";
        return;
    }
    int file_len = file.size();
    QDataStream in(&file);
    char* m_buf = new char[file_len];
    in.readRawData( m_buf, file_len);
    file.close();
    QByteArray arr = QByteArray(m_buf, file_len);
    delete [] m_buf;



    QNetworkRequest req;
    req.setUrl(QUrl(url));
    req.setHeader(QNetworkRequest::ContentTypeHeader, "multipart/form-data");
    req.setHeader(QNetworkRequest::ContentTypeHeader, "application/octet-stream");
    req.setRawHeader("Content-Disposition","form-data;name='file';");




    req.setHeader(QNetworkRequest::ContentLengthHeader,arr.length());
    QNetworkAccessManager manager;

    QNetworkReply *reply =  manager.post(req , arr);

    connect(reply, SIGNAL(error(QNetworkReply::NetworkError)), this, SLOT(error(QNetworkReply::NetworkError)));
    connect(reply, SIGNAL(uploadProgress( qint64 ,qint64 )), this,SIGNAL( uploadProgress(qint64 ,qint64 )));

    QEventLoop eventLoop;
    connect(this,SIGNAL(stop()), &eventLoop,SLOT(quit()));
    connect(&manager, SIGNAL(finished(QNetworkReply*)), &eventLoop, SLOT(quit()));
    eventLoop.exec();       //block until finish

    QJsonDocument doc = QJsonDocument::fromJson(reply->readAll());
    res = doc.object();

//    reply->deleteLater();
}

QJsonObject HttpApi::upload(const QString &url, const QString &filename)
{
    QJsonObject obj;
    this->upload(url,filename,obj);
    return obj;
}

QNetworkReply::NetworkError HttpApi::downloadError() const
{
    return download_reply_->error();
}

void HttpApi::startDownload(const QString &url, const QString &filename)
{
    if(isRunning_) return;

    QNetworkRequest request;
    request.setUrl(url);

    filename_ = filename;


    QFileInfo fileInfo(filename_);
    save_filename_ = fileInfo.absoluteDir().absoluteFilePath( QString(url.toUtf8().toBase64()).right(50));

    checkDownloadFile();

    if(bytesLastReceived_ > 0)
    {
        QString strRange = QString("bytes=%1-").arg(bytesLastReceived_);
        request.setRawHeader("Range", strRange.toLatin1());
    }
    bytesWrited_ = bytesLastReceived_;
    isRunning_ = true;

    QNetworkReply *reply =  download_manager_->get(request);
    connect(reply, SIGNAL(downloadProgress(qint64, qint64)), this, SLOT(onDownloadProgress(qint64, qint64)));
    connect(reply, SIGNAL(readyRead()), this, SLOT(onDownloadReadyRead()));
    connect(reply, SIGNAL(finished()), this, SLOT(onDownloadFinished()));
    connect(reply, SIGNAL(error(QNetworkReply::NetworkError)), this, SLOT(error(QNetworkReply::NetworkError)));
    download_reply_ = reply;

}

void HttpApi::checkDownloadFile()
{
    bytesLastReceived_ = 0;
    if(QFile::exists(save_filename_+DOWNLOAD_TEMP_SUFFIX))
    {
        if(QFile::exists(save_filename_+DOWNLOAD_PROGRESS_SUFFIX))
        {
            QFile ppfile(save_filename_+DOWNLOAD_PROGRESS_SUFFIX);
            if(ppfile.open(QFile::ReadOnly))
            {
                QByteArray b = ppfile.readAll();
                bytesLastReceived_ = b.toLongLong();
                qDebug()<<"Read Progress FILE: "<<bytesLastReceived_;
            }
            ppfile.close();
        }
        else
        {
            QFile::remove(save_filename_+DOWNLOAD_TEMP_SUFFIX);
        }
    }
}


}
