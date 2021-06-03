#ifndef HTTPAPI_H
#define HTTPAPI_H
#include <QObject>
#include <QJsonObject>
#include <QNetworkReply>
class QNetworkAccessManager;
namespace GCL {
class HttpApi : public QObject
{
    Q_OBJECT
public:
    explicit HttpApi(QObject *parent = 0, int timeout=-1);
    virtual ~HttpApi();
signals:
    void stop();
    void uploadProgress( qint64 ,qint64);
    void downloadProgress( qint64, qint64);
    void downloadFinished();

public slots:
    void error(QNetworkReply::NetworkError e);

    void onDownloadFinished();
    void onDownloadReadyRead();
    void onDownloadProgress(qint64 bytesReceived, qint64 bytesTotal);

    void stopDownload();

public:
    void post(const QString &url, const QByteArray &arr, QByteArray &res);
    QJsonObject post(const QString& url, const QByteArray &arr);
    QString postToGetString(const QString& url, const QByteArray &arr);

    void get(const QString &url, QByteArray &res);
    QJsonObject getToGetObject(const QString& url);
    QString getToGetString(const QString& url);

    void upload(const QString &url, const QString &filename, QJsonObject &res);
    QJsonObject upload(const QString &url, const QString &filename);





public:
    qint64 getTotalBytes() const {return bytesWrited_;}
    QNetworkReply::NetworkError downloadError() const;
    void startDownload(const QString &url, const QString &filename);

private:
    void checkDownloadFile();

private:
    int isRunning_;
    QString save_filename_;
    QString filename_;
    qint64 bytesLastReceived_;
    qint64 bytesReceived_;
    qint64 bytesTotal_;
    qint64 bytesWrited_;
    QNetworkAccessManager *download_manager_;
    QNetworkReply *download_reply_;
    int     timeout_;//ms

};
}
#endif // HTTPAPI_H
