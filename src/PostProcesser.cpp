#include "PostProcesser.h"

PostProcesser::PostProcesser(QObject *parent) : QObject (parent)
{

}

PostProcesser::~PostProcesser()
{

}

void PostProcesser::setSourceImageRootPath(QString root_path)
{
    root_path_ = root_path;

    image_save_root_path_ = root_path_ + "DetectStation/";

    normal_image_save_root_path_ = image_save_root_path_ + "Normal/";

    error_image_save_root_path_ = image_save_root_path_ + "Error/";

    QDir dir_;

    if(!dir_.exists(image_save_root_path_))
    {
        dir_.mkdir(image_save_root_path_);
    }

    if(!dir_.exists(normal_image_save_root_path_))
    {
        dir_.mkdir(normal_image_save_root_path_);
    }

    if(!dir_.exists(error_image_save_root_path_))
    {
        dir_.mkdir(error_image_save_root_path_);
    }
}

float PostProcesser::getDistToLine(cv::Point point, cv::Vec4f line)
{
    float d = abs(line[0] * point.x - line[1] * point.y + line[1] * line[2] - line[0] * line[3]);

    return d / sqrt(line[0] * line[0] + line[1] * line[1]);
}

std::vector<std::vector<cv::Point>> PostProcesser::pointCluster(QQueue<cv::Point> point_queue, int max_dist_to_cluster)
{
    std::vector<std::vector<cv::Point>> point_cluster_set;

    while(point_queue.size() > 0)
    {
        cv::Point current_point = point_queue.dequeue();

        bool have_inserted = false;

        if(point_cluster_set.size() > 0)
        {
            for(int i = 0; i < point_cluster_set.size(); ++i)
            {
                for(int j = 0; j < point_cluster_set[i].size(); ++j)
                {
                    cv::Point dif_point = current_point - point_cluster_set[i][j];
                    int current_dist = abs(dif_point.x) + abs(dif_point.y);
                    if(current_dist <= max_dist_to_cluster)
                    {
                        point_cluster_set[i].emplace_back(current_point);

                        have_inserted = true;

                        break;
                    }
                }
                if(have_inserted)
                {
                    break;
                }
            }
        }

        if(have_inserted)
        {
            continue;
        }

        std::vector<cv::Point> new_point_vec;
        new_point_vec.emplace_back(current_point);
        point_cluster_set.emplace_back(new_point_vec);
    }

    return point_cluster_set;
}

std::vector<cv::Vec4f> PostProcesser::fitLines(std::vector<std::vector<cv::Point>> point_cluster_set, int min_point_num_per_cluster)
{
    std::vector<cv::Vec4f> lines;

    for(std::vector<cv::Point> point_vec : point_cluster_set)
    {
        if(point_vec.size() < min_point_num_per_cluster)
        {
            continue;
        }

        cv::Vec4f line;
        cv::fitLine(point_vec, line, cv::DIST_L2, 0, 1e-2, 1e-2);
        lines.emplace_back(line);
    }

    return lines;
}

void PostProcesser::showFitLines(cv::Mat background, std::vector<cv::Vec4f> lines)
{
    cv::Mat mat = background.clone();

    for(cv::Vec4f line : lines)
    {
        //获取点斜式的点和斜率
        cv::Point point0;
        point0.x = int(line[3]);
        point0.y = int(line[2]);

        float k = line[1] / line[0];

        //计算直线的端点(x = k(y - y0) + x0)
        cv::Point point1, point2;
        point1.y = 0;
        point1.x = int(k * (0 - point0.y) + point0.x);
        point2.y = mat.cols;
        point2.x = int(k * (mat.cols - point0.y) + point0.x);

        cv::line(mat, point1, point2, cv::Scalar(255), 2, 8, 0);
    }

    cv::imshow("fit lines", mat);

    cv::waitKey(0);
}

std::vector<cv::Point> PostProcesser::getAveragePoints(std::vector<std::vector<cv::Point>> point_cluster_set, int min_point_num_per_cluster)
{
    std::vector<cv::Point> points;

    for(std::vector<cv::Point> point_vec : point_cluster_set)
    {
        if(point_vec.size() < min_point_num_per_cluster)
        {
            continue;
        }

        float x_sum = 0;
        float y_sum = 0;

        for(cv::Point point : point_vec)
        {
            x_sum += point.x;
            y_sum += point.y;
        }

        cv::Point avg_point;

        avg_point.y = int(x_sum / point_vec.size());
        avg_point.x = int(y_sum / point_vec.size());

        points.emplace_back(avg_point);
    }

    return points;
}

void PostProcesser::showAveragePoints(cv::Mat background, std::vector<cv::Point> points)
{
    cv::Mat mat = background.clone();

    for(cv::Point point : points)
    {
        cv::circle(mat, point, 4, cv::Scalar(255), 2, 8, 0);
    }

    cv::imshow("fit points", mat);

    cv::waitKey(0);
}

std::vector<int> PostProcesser::matchLineAndPoint(std::vector<cv::Vec4f> lines, std::vector<cv::Point> points, int max_dist_between_point_and_line)
{
    std::vector<int> line_id_of_point;

    for(cv::Point point : points)
    {
        int current_line_id = -1;
        float min_dist_to_line = -1;

        for(int i = 0; i < lines.size(); ++i)
        {
            float current_dist_to_line = getDistToLine(point, lines[i]);

            if(min_dist_to_line == -1)
            {
                current_line_id = i;
                min_dist_to_line = current_dist_to_line;
            }
            else if(current_dist_to_line < min_dist_to_line)
            {
                current_line_id = i;
                min_dist_to_line = current_dist_to_line;
            }
        }

        if(min_dist_to_line < max_dist_between_point_and_line)
        {
            line_id_of_point.emplace_back(current_line_id);
        }
        else
        {
            line_id_of_point.emplace_back(-1);
        }
    }

    return line_id_of_point;
}

void PostProcesser::showMatchResult(cv::Mat background, std::vector<cv::Vec4f> lines, std::vector<cv::Point> points, std::vector<int> line_id_of_point)
{
    cv::Mat show_image;

    cv::cvtColor(background, show_image, cv::COLOR_GRAY2BGR);

    std::vector<cv::Scalar> color_set;

    for(int i = 0; i < lines.size(); ++i)
    {
        color_set.emplace_back(cv::Scalar(rand()%255, rand()%255, rand()%255));
    }

    for(int i = 0; i < lines.size(); ++i)
    {
        //获取点斜式的点和斜率
        cv::Point point0;
        point0.x = int(lines[i][3]);
        point0.y = int(lines[i][2]);

        float k = lines[i][1] / lines[i][0];

        //计算直线的端点(x = k(y - y0) + x0)
        cv::Point point1, point2;
        point1.y = 0;
        point1.x = int(k * (0 - point0.y) + point0.x);
        point2.y = show_image.cols;
        point2.x = int(k * (show_image.cols - point0.y) + point0.x);

        cv::line(show_image, point1, point2, color_set[i], 2, 8, 0);
    }

    cv::Scalar white(255, 255, 255);

    for(int i = 0; i < points.size(); ++i)
    {
        if(line_id_of_point[i] != -1)
        {
            cv::circle(show_image, points[i], 4, color_set[line_id_of_point[i]], 2, 8, 0);
        }
        else
        {
            cv::circle(show_image, points[i], 4, white, 2, 8, 0);
        }
    }

    cv::imshow("match result", show_image);

    cv::waitKey(0);
}

std::vector<std::vector<int>> PostProcesser::getValidMatch(int lines_num, std::vector<int> line_id_of_point)
{
    std::vector<std::vector<int>> valid_match_set;

    for(int i = 0; i < lines_num; ++i)
    {
        std::vector<int> valid_match;
        valid_match.emplace_back(i);

        int point_num_on_current_line = 0;

        for(int j = 0; j < line_id_of_point.size(); ++j)
        {
            if(line_id_of_point[j] == i)
            {
                valid_match.emplace_back(j);
                ++point_num_on_current_line;
            }
        }

        if(point_num_on_current_line == 2)
        {
            valid_match_set.emplace_back(valid_match);
        }
    }

    return valid_match_set;
}

void PostProcesser::showValidMatch(cv::Mat background, std::vector<cv::Vec4f> lines, std::vector<cv::Point> points, std::vector<std::vector<int>> valid_match_set)
{
    cv::Mat show_image;

    cv::cvtColor(background, show_image, cv::COLOR_GRAY2BGR);

    std::vector<cv::Scalar> color_set;

    for(int i = 0; i < valid_match_set.size(); ++i)
    {
        color_set.emplace_back(cv::Scalar(rand()%255, rand()%255, rand()%255));
    }

    for(int i = 0; i < valid_match_set.size(); ++i)
    {
        int line_id = valid_match_set[i][0];

        //获取点斜式的点和斜率
        cv::Point point0;
        point0.x = int(lines[line_id][3]);
        point0.y = int(lines[line_id][2]);

        float k = lines[line_id][1] / lines[line_id][0];

        //计算直线的端点(x = k(y - y0) + x0)
        cv::Point point1, point2;
        point1.y = 0;
        point1.x = int(k * (0 - point0.y) + point0.x);
        point2.y = show_image.cols;
        point2.x = int(k * (show_image.cols - point0.y) + point0.x);

        cv::line(show_image, point1, point2, color_set[i], 2, 8, 0);

        cv::circle(show_image, points[valid_match_set[i][1]], 4, color_set[i], 2, 8, 0);
        cv::circle(show_image, points[valid_match_set[i][2]], 4, color_set[i], 2, 8, 0);
    }

    cv::imshow("valid match", show_image);

    cv::waitKey(0);
}

std::vector<bool> PostProcesser::checkValidMatchParallel(std::vector<cv::Vec4f> lines, std::vector<cv::Point> points, std::vector<std::vector<int>> valid_match_set, float max_k_error)
{
    std::vector<bool> valid_match_parallel;

    for(int i = 0; i < valid_match_set.size(); ++i)
    {
        int line_id = valid_match_set[i][0];
        int point1_id = valid_match_set[i][1];
        int point2_id = valid_match_set[i][2];

        float k_line = lines[line_id][1] / lines[line_id][0];
        float k_point = 1.0 * (points[point1_id].x - points[point2_id].x) / (points[point1_id].y - points[point2_id].y);

        float abs_k_dist = abs(k_line - k_point);

        if(abs_k_dist <= max_k_error)
        {
            valid_match_parallel.emplace_back(true);
        }
        else
        {
            valid_match_parallel.emplace_back(false);
        }
    }

    return valid_match_parallel;
}

void PostProcesser::showParallelValidMatch(cv::Mat background, std::vector<cv::Vec4f> lines, std::vector<cv::Point> points, std::vector<std::vector<int>> valid_match_set, std::vector<bool> valid_match_parallel)
{
    cv::Mat show_image;

    cv::cvtColor(background, show_image, cv::COLOR_GRAY2BGR);

    std::vector<cv::Scalar> color_set;

    for(int i = 0; i < valid_match_set.size(); ++i)
    {
        color_set.emplace_back(cv::Scalar(rand()%255, rand()%255, rand()%255));
    }

    for(int i = 0; i < valid_match_set.size(); ++i)
    {
        if(!valid_match_parallel[i])
        {
            continue;
        }

        int line_id = valid_match_set[i][0];

        //获取点斜式的点和斜率
        cv::Point point0;
        point0.x = int(lines[line_id][3]);
        point0.y = int(lines[line_id][2]);

        float k = lines[line_id][1] / lines[line_id][0];

        //计算直线的端点(x = k(y - y0) + x0)
        cv::Point point1, point2;
        point1.y = 0;
        point1.x = int(k * (0 - point0.y) + point0.x);
        point2.y = show_image.cols;
        point2.x = int(k * (show_image.cols - point0.y) + point0.x);

        cv::line(show_image, point1, point2, color_set[i], 2, 8, 0);

        cv::circle(show_image, points[valid_match_set[i][1]], 4, color_set[i], 2, 8, 0);
        cv::circle(show_image, points[valid_match_set[i][2]], 4, color_set[i], 2, 8, 0);
    }

    cv::imshow("parallel valid match", show_image);

    cv::waitKey(0);
}

cv::Mat PostProcesser::createSaveImage(std::vector<cv::Point> points, std::vector<std::vector<int>> valid_match_set, std::vector<bool> valid_match_parallel)
{
    cv::Mat save_image = DarknetDetector::QImage2cvMat(*source_image_);

    if(darknet_bbox_vec_->size() > 0)
    {
        for(int i = 0; i < darknet_bbox_vec_->size(); ++i)
        {
            BBox bbox = darknet_bbox_vec_->at(i);

            if(bbox.score > 0.5)
            {
                cv::rectangle(save_image,
                              cv::Point(int(bbox.x) - 20, int(bbox.y) - 20),
                              cv::Point(int(bbox.x + bbox.w) + 20, int(bbox.y + bbox.h) + 20),
                              cv::Scalar(0, 0, 255),
                              2,
                              8,
                              0);
            }
        }
    }

    for(int i = 0; i < valid_match_parallel.size(); ++i)
    {
        if(!valid_match_parallel[i])
        {
            int point1_id = valid_match_set[i][1];
            int point2_id = valid_match_set[i][2];

            cv::rectangle(save_image,
                          cv::Point(int(points[point1_id].x * save_image.cols / 1024.0) - 40, int(points[point1_id].y * save_image.rows / 512.0) - 40),
                          cv::Point(int(points[point2_id].x * save_image.cols / 1024.0) + 40, int(points[point2_id].y * save_image.rows / 512.0) + 40),
                          cv::Scalar(0, 255, 0),
                          2,
                          8,
                          0);
        }
    }

    return save_image;
}

ImageType PostProcesser::getImageType(std::vector<bool> valid_match_parallel)
{
    ImageType image_type = Normal;

    bool found_darknet_error = false;
    bool found_lapnet_error = false;

    if(darknet_bbox_vec_->size() > 0)
    {
        found_darknet_error = true;
    }

    for(int i = 0; i < valid_match_parallel.size(); ++i)
    {
        if(!valid_match_parallel[i])
        {
            found_lapnet_error = true;
            break;
        }
    }

    if(found_darknet_error)
    {
        if(found_lapnet_error)
        {
            image_type = DarknetAndLapnetError;
        }
        else
        {
            image_type = OnlyDarknetError;
        }
    }
    else if(found_lapnet_error)
    {
        image_type = OnlyLapnetError;
    }

    return image_type;
}

void PostProcesser::saveImage(cv::Mat save_image, ImageType image_type)
{
    switch(image_type)
    {
    case Empty:
        break;
    case Normal:

        cv::imwrite((normal_image_save_root_path_ + source_image_path_.replace(":", "_").replace("/", "_")).toStdString(), save_image);

        break;
    case OnlyDarknetError:

        cv::imwrite((error_image_save_root_path_ + source_image_path_.replace(":", "_").replace("/", "_")).toStdString(), save_image);

        break;
    case OnlyLapnetError:

        cv::imwrite((error_image_save_root_path_ + source_image_path_.replace(":", "_").replace("/", "_")).toStdString(), save_image);

        break;
    case DarknetAndLapnetError:

        cv::imwrite((error_image_save_root_path_ + source_image_path_.replace(":", "_").replace("/", "_")).toStdString(), save_image);

        break;
    }
}

bool PostProcesser::postProcess()
{
    int min_point_num_per_line_cluster = 20;
    int max_dist_to_line_cluster = 20;

    int min_point_num_per_point_cluster = 1;
    int max_dist_to_point_cluster = 10;

    int max_dist_between_point_and_line = 5;

    float max_k_error = 0.1;

    cv::Mat mat_lapnet_line = DarknetDetector::QImage2cvMat(*lapnet_line_image_);
    cv::Mat mat_lapnet_point = DarknetDetector::QImage2cvMat(*lapnet_point_image_);

//    cv::imshow("line", mat_lapnet_line);
//    cv::imshow("point", mat_lapnet_point);

    cv::Mat gray_line;

    cv::Mat gray_point;

    cv::cvtColor(mat_lapnet_line, gray_line, cv::COLOR_BGR2GRAY);

    cv::cvtColor(mat_lapnet_point, gray_point, cv::COLOR_BGR2GRAY);

    cv::Mat binary_line;

    cv::Mat binary_point;

    cv::threshold(gray_line, binary_line, 10, 255, cv::THRESH_BINARY);

    cv::threshold(gray_point, binary_point, 10, 255, cv::THRESH_BINARY);

    cv::Mat edges_line;

    cv::Canny(binary_line, edges_line, 50, 150, 3);

    QQueue<cv::Point> point_queue_of_lines;

    for(int i = 0; i < edges_line.rows; ++i)
    {
        for(int j = 0; j < edges_line.cols; ++j)
        {
            if(edges_line.at<uchar>(i, j) > 200)
            {
                point_queue_of_lines.enqueue(cv::Point(i, j));
            }
        }
    }

    std::vector<std::vector<cv::Point>> point_cluster_set_of_lines = pointCluster(point_queue_of_lines, max_dist_to_line_cluster);

    std::vector<cv::Vec4f> lines = fitLines(point_cluster_set_of_lines, min_point_num_per_line_cluster);

//    showFitLines(edges_line, lines);

    QQueue<cv::Point> point_queue_of_points;

    for(int i = 0; i < binary_point.rows; ++i)
    {
        for(int j = 0; j < binary_point.cols; ++j)
        {
            if(binary_point.at<uchar>(i, j) > 200)
            {
                point_queue_of_points.enqueue(cv::Point(i, j));
            }
        }
    }

    std::vector<std::vector<cv::Point>> point_cluster_set_of_points = pointCluster(point_queue_of_points, max_dist_to_point_cluster);

    std::vector<cv::Point> points = getAveragePoints(point_cluster_set_of_points, min_point_num_per_point_cluster);

//    showAveragePoints(binary_point, points);

    std::vector<int> line_id_of_point = matchLineAndPoint(lines, points, max_dist_between_point_and_line);

//    showMatchResult(binary_point, lines, points, line_id_of_point);

    std::vector<std::vector<int>> valid_match_set = getValidMatch(lines.size(), line_id_of_point);

//    showValidMatch(binary_point, lines, points, valid_match_set);

    std::vector<bool> valid_match_parallel = checkValidMatchParallel(lines, points, valid_match_set, max_k_error);

//    showParallelValidMatch(binary_point, lines, points, valid_match_set, valid_match_parallel);

    cv::Mat save_image = createSaveImage(points, valid_match_set, valid_match_parallel);

    ImageType image_type = getImageType(valid_match_parallel);

    saveImage(save_image, image_type);

    return true;
}

void PostProcesser::slot_postProcess()
{
    bool succeed = postProcess();

    emit signal_postProcess_finished(succeed);
}
