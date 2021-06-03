#ifndef POSTPROCESSER_H
#define POSTPROCESSER_H

#include <QObject>
#include <QString>
#include <QDebug>
#include <QImage>
#include <QVector>
#include <QQueue>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "DarknetDetector.h"

enum ImageType
{
    Empty = 0,
    Normal = 1,
    OnlyDarknetError = 2,
    OnlyLapnetError = 3,
    DarknetAndLapnetError = 4
};

class PostProcesser : public QObject
{
    Q_OBJECT

public:
    explicit PostProcesser(QObject *parent = nullptr);
    ~PostProcesser();

    void setSourceImageRootPath(QString root_path);

    float getDistToLine(cv::Point point, cv::Vec4f line);

    std::vector<std::vector<cv::Point>> pointCluster(QQueue<cv::Point> point_queue, int max_dist_to_cluster);

    std::vector<cv::Vec4f> fitLines(std::vector<std::vector<cv::Point>> point_cluster_set, int min_point_num_per_cluster);

    void showFitLines(cv::Mat background, std::vector<cv::Vec4f> lines);

    std::vector<cv::Point> getAveragePoints(std::vector<std::vector<cv::Point>> point_cluster_set, int min_point_num_per_cluster);

    void showAveragePoints(cv::Mat background, std::vector<cv::Point> points);

    std::vector<int> matchLineAndPoint(std::vector<cv::Vec4f> lines, std::vector<cv::Point> points, int max_dist_between_point_and_line);

    void showMatchResult(cv::Mat background, std::vector<cv::Vec4f> lines, std::vector<cv::Point> points, std::vector<int> line_id_of_point);

    std::vector<std::vector<int>> getValidMatch(int lines_num, std::vector<int> line_id_of_point);

    void showValidMatch(cv::Mat background, std::vector<cv::Vec4f> lines, std::vector<cv::Point> points, std::vector<std::vector<int>> valid_match_set);

    std::vector<bool> checkValidMatchParallel(std::vector<cv::Vec4f> lines, std::vector<cv::Point> points, std::vector<std::vector<int>> valid_match_set, float max_k_error);

    void showParallelValidMatch(cv::Mat background, std::vector<cv::Vec4f> lines, std::vector<cv::Point> points, std::vector<std::vector<int>> valid_match_set, std::vector<bool> valid_match_parallel);

    cv::Mat createSaveImage(std::vector<cv::Point> points, std::vector<std::vector<int>> valid_match_set, std::vector<bool> valid_match_parallel);

    ImageType getImageType(std::vector<bool> valid_match_parallel);

    void saveImage(cv::Mat save_image, ImageType image_type);

    bool postProcess();

    void setSourceImage(QImage* source_image){source_image_ = source_image;}

    void setSourceImagePath(QString source_image_path){source_image_path_ = source_image_path;}

    void setBBoxVec(QVector<BBox>* darknet_bbox_vec){darknet_bbox_vec_ = darknet_bbox_vec;}

    void setLapnetLineImage(QImage* lapnet_line_image){lapnet_line_image_ = lapnet_line_image;}

    void setLapnetPointImage(QImage* lapnet_point_image){lapnet_point_image_ = lapnet_point_image;}

private:
    QString root_path_;
    QString image_save_root_path_;
    QString normal_image_save_root_path_;
    QString error_image_save_root_path_;

    QString source_image_path_;
    QImage* source_image_;
    QVector<BBox>* darknet_bbox_vec_;
    QImage* lapnet_line_image_;
    QImage* lapnet_point_image_;

public slots:
    void slot_postProcess();

signals:
    void signal_postProcess_finished(bool);
};

#endif // POSTPROCESSER_H
