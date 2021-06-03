#include "DarknetDetector.h"

#define OPENCV
#define GPU

using namespace cv;

#ifdef WIN32
#pragma comment(lib, "D:/chLi/Project/ABACI/FilterSocket/bin_win/darknet-win-gpu/yolo_cpp_dll.lib")//引入链接库

DarknetDetector::DarknetDetector(const std::string &yolov3_cfg, const std::string &yolov3_weights, const std::string &coco_names, QObject *parent) : QObject (parent)
{
    std::string names_file = coco_names;
    std::string cfg_file = yolov3_cfg;
    std::string weights_file = yolov3_weights;
    detector_ = new Detector(cfg_file,weights_file,0);
    obj_names_.clear();
    //obj_names_ = objects_names_from_file(names_file); //调用获得分类对象名称
    //或者使用以下四行代码也可实现读入分类对象文件
    std::ifstream ifs(names_file.c_str());
    std::string line;
    while (getline(ifs, line))
    {
        obj_names_.push_back(line);
    }
}

DarknetDetector::~DarknetDetector()
{

}

void DarknetDetector::slot_LoadImage(QString image_path)
{
    mat_ = imread(image_path.toStdString());
    QImage disImage;
    disImage= QImage((const unsigned char*)(mat_.data), mat_.cols, mat_.rows, QImage::Format_RGB888);

    std::vector<bbox_t> result_vec = detector_->detect(mat_);
    draw_boxes(mat_, result_vec, obj_names_);

    cv::namedWindow("test", CV_WINDOW_NORMAL);
    cv::imshow("test", mat_);
    cv::waitKey(3);
}

void DarknetDetector::slot_LoadVideo(QString video_path)
{
    capture_.open(video_path.toStdString());
    if (!capture_.isOpened())
    {
        printf("文件打开失败");
    }
    cv::Mat frame;

    while (true)
    {
        capture_ >> frame;
        std::vector<bbox_t> result_vec = detector_->detect(frame);
        draw_boxes(frame, result_vec, obj_names_);

        cv::namedWindow("test", CV_WINDOW_NORMAL);
        cv::imshow("test", frame);
        cv::waitKey(3);
    }
}

QImage DarknetDetector::cvMat2QImage(const cv::Mat mat)
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

cv::Mat DarknetDetector::QImage2cvMat(QImage image)
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
        cv::cvtColor(mat, mat, CV_BGR2RGB);
        break;
    case QImage::Format_Indexed8:
        mat = cv::Mat(image.height(), image.width(), CV_8UC1, (void*)image.constBits(), image.bytesPerLine());
        break;
    case QImage::Format_Grayscale8:
        mat = cv::Mat(image.height(), image.width(), CV_8UC1, (void*)image.constBits(), image.bytesPerLine());
        break;
    }
    return mat;
}

bool DarknetDetector::darknet_process(cv::Mat data)
{
    clock_t start = clock();
    std::vector<bbox_t> result = this->getDarknetResult(data);
//    std::cout << int((clock() - start)) << " ms" << std::endl;

    bbox_vec_.clear();

    for(int i=0; i < result.size(); i++)
    {
        BBox current_bbox;

        current_bbox.x = result[i].x;
        current_bbox.y = result[i].y;
        current_bbox.w = result[i].w;
        current_bbox.h = result[i].h;
        current_bbox.label = result[i].obj_id;
        current_bbox.score = result[i].prob;

        bbox_vec_.append(current_bbox);
    }


    return true;
}

bool DarknetDetector::darknet_process(QImage data)
{
    cv::Mat mat = this->QImage2cvMat(data);

    // process

    clock_t start = clock();
    std::vector<bbox_t> result = this->getDarknetResult(mat);
//    std::cout << int((clock() - start)) << " ms" << std::endl;

    bbox_vec_.clear();

    for(int i=0; i < result.size(); i++)
    {
        BBox current_bbox;

        current_bbox.x = result[i].x;
        current_bbox.y = result[i].y;
        current_bbox.w = result[i].w;
        current_bbox.h = result[i].h;
        current_bbox.label = result[i].obj_id;
        current_bbox.score = result[i].prob;

        bbox_vec_.append(current_bbox);
    }

    return true;
}

void DarknetDetector::draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names,
    int current_det_fps, int current_cap_fps)
{
    int const colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };

    for (auto &i : result_vec) {
        cv::Scalar color = obj_id_to_color(i.obj_id);
        cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 2);
        if (obj_names.size() > i.obj_id) {
            std::string obj_name = obj_names[i.obj_id];
            if (i.track_id > 0) obj_name += " - " + std::to_string(i.track_id);
            cv::Size const text_size = getTextSize(obj_name, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
            int const max_width = (text_size.width > i.w + 2) ? text_size.width : (i.w + 2);
            cv::rectangle(mat_img, cv::Point2f(std::max((int)i.x - 1, 0), std::max((int)i.y - 30, 0)),
                cv::Point2f(std::min((int)i.x + max_width, mat_img.cols-1), std::min((int)i.y, mat_img.rows-1)),
                color, CV_FILLED, 8, 0);
            putText(mat_img, obj_name, cv::Point2f(i.x, i.y - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 0), 2);
        }
    }
    if (current_det_fps >= 0 && current_cap_fps >= 0) {
        std::string fps_str = "FPS detection: " + std::to_string(current_det_fps) + "   FPS capture: " + std::to_string(current_cap_fps);
        putText(mat_img, fps_str, cv::Point2f(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(50, 255, 0), 2);
    }
}

std::vector<std::string> DarknetDetector::objects_names_from_file(std::string const filename)
{
    std::ifstream file(filename);
    std::vector<std::string> file_lines;
    if (!file.is_open()) return file_lines;
    for(std::string line; getline(file, line);) file_lines.push_back(line);
    std::cout << "object names loaded \n";
    return file_lines;
}

std::vector<bbox_t> DarknetDetector::getDarknetResult(cv::Mat mat)
{
    std::vector<bbox_t> result_vec = detector_->detect(mat);

    return result_vec;
}

void DarknetDetector::slot_darknetDetect()
{
    bool succeed = darknet_process(*detect_image_);

    emit signal_darknetDetect_finished(succeed);
}
#endif

#ifdef Linux
DarknetDetector::DarknetDetector(const std::string &yolov3_cfg, const std::string &yolov3_weights, const std::string &coco_data, QObject *parent) : QObject (parent)
{
    const char *c_str_yolov3_cfg = yolov3_cfg.c_str();
    const char *c_str_yolov3_weights = yolov3_weights.c_str();
    const char *c_str_coco_cfg = coco_data.c_str();

    char *str_yolov3_cfg = new char[strlen(c_str_yolov3_cfg) + 1];
    char *str_yolov3_weights = new char[strlen(c_str_yolov3_weights) + 1];
    char *str_coco_cfg = new char[strlen(c_str_coco_cfg) + 1];

    strcpy(str_yolov3_cfg, c_str_yolov3_cfg);
    strcpy(str_yolov3_weights, c_str_yolov3_weights);
    strcpy(str_coco_cfg, c_str_coco_cfg);

    cuda_set_device(gpu_index);

    net = load_network(str_yolov3_cfg, str_yolov3_weights, 0);

    meta = get_metadata(str_coco_cfg);
}

DarknetDetector::~DarknetDetector()
{
    delete net;
}

std::vector<std::pair<char *, std::vector<float>>> DarknetDetector::getDarknetResult(image img, float thresh, float hier_thresh, float nms)
{
    return detect(img, thresh, hier_thresh, nms);
}

std::vector<std::pair<char *, std::vector<float>>> DarknetDetector::getDarknetResult(char *img, float thresh, float hier_thresh, float nms)
{
    return detect(img, thresh, hier_thresh, nms);
}

std::vector<std::pair<char *, std::vector<float>>> DarknetDetector::getDarknetResult(float *img, int w, int h, int c, float thresh, float hier_thresh, float nms)
{
    return detect(img, w, h, c, thresh, hier_thresh, nms);
}

std::vector<std::pair<char *, std::vector<float>>> DarknetDetector::detect(image im, float thresh, float hier_thresh, float nms)
{
    int num = 0;

    int *pnum = &num;

    network_predict_image(net, im);

    detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, nullptr, 0, pnum);

    num = pnum[0];

    if (nms)
    {
        do_nms_obj(dets, num, meta.classes, nms);
    }

    std::vector<std::pair<char *, std::vector<float>>> res;

    for(int j = 0; j < num; ++j)
    {
        for(int i = 0; i < meta.classes; ++i)
        {
            if(dets[j].prob[i] > 0)
            {
                box b = dets[j].bbox;

                std::pair<char *, std::vector<float>> temp_data;

                temp_data.first = meta.names[i];

                temp_data.second.emplace_back(dets[j].prob[i]);
                temp_data.second.emplace_back(b.x - b.w / 2.0);
                temp_data.second.emplace_back(b.y - b.h / 2.0);
                temp_data.second.emplace_back(b.w);
                temp_data.second.emplace_back(b.h);
                temp_data.second.emplace_back(i);

                res.emplace_back(temp_data);
            }
        }
    }

    if(res.size() > 1)
    {
        for(int i = 0; i < res.size() - 1; ++i)
        {
            for(int j = i + 1; j < res.size(); ++j)
            {
                if(res[i].second[0] < res[j].second[0])
                {
                    std::pair<char *, std::vector<float>> exchange_data = res[i];

                    res[i] = res[j];
                    res[j] = exchange_data;
                }
            }
        }
    }

    free_image(im);

    free_detections(dets, num);

    return res;
}

std::vector<std::pair<char *, std::vector<float>>> DarknetDetector::detect(char *img, float thresh, float hier_thresh, float nms)
{
    image im = load_image_color(img, 0, 0);

    return detect(im, thresh, hier_thresh, nms);
}

std::vector<std::pair<char *, std::vector<float>>> DarknetDetector::detect(float *img, int w, int h, int c, float thresh, float hier_thresh, float nms)
{
    image im;
    im.w = w;
    im.h = h;
    im.c = c;

    int im_size = w * h * c;

    im.data = new float[im_size];

    memcpy(im.data, img, im_size * sizeof(float));

    return detect(im, thresh, hier_thresh, nms);
}

bool DarknetDetector::darknet_process(ConnectedData &data)
{
    float *image = new float[data.info.img_height_ * data.info.img_width_ * data.info.img_format_];

    for(int j=0; j < data.info.img_height_; j++)
    {
        for(int i=0; i < data.info.img_width_; i++)
        {
            // pixel(i,j)
            int color = data.getRed(i, j);
            if(color < 0)
            {
                color += 255;
            }
            image[j * data.info.img_width_ + i] = float(color) / 255.0;
        }
    }
    for(int j=0; j < data.info.img_height_; j++)
    {
        for(int i=0; i < data.info.img_width_; i++)
        {
            // pixel(i,j)
            int color = data.getGreen(i, j);
            if(color < 0)
            {
                color += 255;
            }
            image[data.info.img_height_ * data.info.img_width_ + j * data.info.img_width_ + i] = float(color) / 255.0;
        }
    }
    for(int j=0; j < data.info.img_height_; j++)
    {
        for(int i=0; i < data.info.img_width_; i++)
        {
            // pixel(i,j)
            int color = data.getBlue(i, j);
            if(color < 0)
            {
                color += 255;
            }
            image[2 * data.info.img_height_ * data.info.img_width_ + j * data.info.img_width_ + i] = float(color) / 255.0;
        }
    }

    // process

    clock_t start = clock();
    std::vector<std::pair<char *, std::vector<float>>> result = this->getDarknetResult(image, data.info.img_width_, data.info.img_height_, data.info.img_format_);
    delete(image);
    std::cout << int((clock() - start)/1000) << " ms" << std::endl;

    data.resetBBoxNum(result.size());

    for(int i=0; i < data.info.bbox_num_; i++)
    {
        data.setBBox(i, result[i].second[1], result[i].second[2], result[i].second[3], result[i].second[4], result[i].second[5]);
    }

    return true;
}
#endif
