#include "RailwayFaultDetectWidget.h"
#include "ui_railwayfaultdetectwidget.h"

RailwayFaultDetectWidget::RailwayFaultDetectWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::RailwayFaultDetectWidget)
{
    ui->setupUi(this);

    data_reader_ = new DataReader();
    data_reader_thread_ = new QThread();
    data_reader_->moveToThread(data_reader_thread_);
    data_image_ = data_reader_->getImage();

    connect(this, SIGNAL(signal_getData()), data_reader_, SLOT(slot_getData()), Qt::DirectConnection);
    connect(data_reader_, SIGNAL(signal_getData_finished(bool)), this, SLOT(slot_getData_finished(bool)), Qt::DirectConnection);

    init_model();

    post_processer_ = new PostProcesser();
    post_processer_thread_ = new QThread();
    post_processer_->moveToThread(post_processer_thread_);
    post_processer_->setSourceImage(data_image_);
    post_processer_->setBBoxVec(darknet_detector_->getBBoxVec());
    post_processer_->setLapnetLineImage(lapnet_line_detector_->getOutputImage());
    post_processer_->setLapnetPointImage(lapnet_point_detector_->getOutputImage());

    connect(this, SIGNAL(signal_postProcess()), post_processer_, SLOT(slot_postProcess()), Qt::DirectConnection);
    connect(post_processer_, SIGNAL(signal_postProcess_finished(bool)), this, SLOT(slot_postProcess_finished(bool)), Qt::DirectConnection);

    this->ui->lineEdit_DataPath->setText("Z:/chLi/tiedaobu/TestStation/123/");
    data_reader_->setDataMode(Data_Folder);
}

RailwayFaultDetectWidget::~RailwayFaultDetectWidget()
{
    delete ui;
}

bool RailwayFaultDetectWidget::init_model()
{
    std::string coco_names = "../thirdparty/darknet-gpu/yolov3_train_2c_detect_2class/coco.names";
    std::string yolov3_cfg = "../thirdparty/darknet-gpu/yolov3_train_2c_detect_2class/yolov3.cfg";
    std::string yolov3_weights = "../thirdparty/darknet-gpu/yolov3_train_2c_detect_2class/yolov3_train_2c_detect_2class.backup";
    std::string coco_data = "../thirdparty/darknet-gpu/yolov3_train_2c_detect_2class/coco.data";

    QFile file_yolo(QString::fromStdString(yolov3_cfg));
    if(!file_yolo.exists())
    {
        qDebug() << "RailwayFaultDetectWidget::init : failed.";
        return false;
    }

#ifdef WIN32
    darknet_detector_ = new DarknetDetector(yolov3_cfg, yolov3_weights, coco_names);
#endif

#ifdef Linux
    darknet_detector_ = new DarknetDetector(yolov3_cfg, yolov3_weights, coco_data);
#endif

    darknet_detector_thread_ = new QThread();
    darknet_detector_->moveToThread(darknet_detector_thread_);
    darknet_detector_->setImage(data_image_);
    darknet_bbox_vec_ = darknet_detector_->getBBoxVec();

    connect(this, SIGNAL(signal_darknetDetect()), darknet_detector_, SLOT(slot_darknetDetect()), Qt::DirectConnection);
    connect(darknet_detector_, SIGNAL(signal_darknetDetect_finished(bool)), this, SLOT(slot_darknetDetect_finished(bool)), Qt::DirectConnection);

    lapnet_line_detector_ = new LapnetDetector();
    lapnet_line_detector_thread_ = new QThread();
    lapnet_line_detector_->moveToThread(lapnet_line_detector_thread_);
    lapnet_line_detector_->setImage(data_image_);
    lapnet_line_detector_->setPort("9360");
    lapnet_output_image_line_ = lapnet_line_detector_->getOutputImage();

    connect(this, SIGNAL(signal_lapnetLineDetect()), lapnet_line_detector_, SLOT(slot_lapnetDetect()), Qt::DirectConnection);
    connect(lapnet_line_detector_, SIGNAL(signal_lapnetDetect_finished(bool)), this, SLOT(slot_lapnetDetectLine_finished(bool)), Qt::DirectConnection);

    lapnet_point_detector_ = new LapnetDetector();
    lapnet_point_detector_thread_ = new QThread();
    lapnet_point_detector_->moveToThread(lapnet_point_detector_thread_);
    lapnet_point_detector_->setImage(data_image_);
    lapnet_point_detector_->setPort("9361");
    lapnet_output_image_point_ = lapnet_point_detector_->getOutputImage();

    connect(this, SIGNAL(signal_lapnetPointDetect()), lapnet_point_detector_, SLOT(slot_lapnetDetect()), Qt::DirectConnection);
    connect(lapnet_point_detector_, SIGNAL(signal_lapnetDetect_finished(bool)), this, SLOT(slot_lapnetDetectPoint_finished(bool)), Qt::DirectConnection);

    qDebug() << "init model finished!";

    return true;
}

void RailwayFaultDetectWidget::on_Btn_SelectFile_clicked()
{
    QString file_path = QFileDialog::getOpenFileName(0, "", "", "*.jpg || *.mp4 || *.avi");

    if(file_path == "")
    {
        return;
    }

    this->ui->lineEdit_DataPath->setText(file_path);

    if(file_path.contains(".jpg"))
    {
        data_reader_->setDataMode(Data_Image);
    }
    else if(file_path.contains(".mp4") || file_path.contains(".avi"))
    {
        data_reader_->setDataMode(Data_Video);
    }
}

void RailwayFaultDetectWidget::on_Btn_SelectFolder_clicked()
{
    QString folder_path = QFileDialog::getExistingDirectory(this, "choose src Directory", "/");

    if(folder_path == "")
    {
        return;
    }

    if(folder_path.lastIndexOf("/") != folder_path.size() - 1)
    {
        folder_path += "/";
    }

    this->ui->lineEdit_DataPath->setText(folder_path);

    data_reader_->setDataMode(Data_Folder);

    post_processer_->setSourceImageRootPath(folder_path);
}

void RailwayFaultDetectWidget::on_Btn_InitData_clicked()
{
    QString input_path = this->ui->lineEdit_DataPath->text();

    if(input_path != "")
    {
        if(data_reader_->getDataMode() == Data_Free)
        {
            data_reader_->setDataMode(Data_Camera);
        }

        data_reader_->readData(input_path);

        if(data_reader_->getDataMode() == Data_Folder)
        {
            post_processer_->setSourceImageRootPath(input_path);
        }
    }

    qDebug() << "RailwayFaultDetectWidget::on_Btn_InitData_clicked : init Data finished.";
}

void RailwayFaultDetectWidget::on_Btn_StartDetect_clicked()
{
    emit signal_getData();
}

void RailwayFaultDetectWidget::slot_getData_finished(bool succeed)
{
    if(succeed)
    {
        emit signal_darknetDetect();
    }
    else
    {
        qDebug() << "RailwayFaultDetectWidget::slot_getData_finished : get Data failed.";
    }
}

void RailwayFaultDetectWidget::slot_darknetDetect_finished(bool succeed)
{
    if(succeed)
    {
        emit signal_lapnetLineDetect();
    }
    else
    {
        qDebug() << "RailwayFaultDetectWidget::slot_darknetDetect_finished : darknet detect failed.";
    }
}

void RailwayFaultDetectWidget::slot_lapnetDetectLine_finished(bool succeed)
{
    if(succeed)
    {
        emit signal_lapnetPointDetect();
    }
    else
    {
        qDebug() << "RailwayFaultDetectWidget::slot_lapnetDetectLine_finished : lapnet line detect failed.";
    }
}

void RailwayFaultDetectWidget::slot_lapnetDetectPoint_finished(bool succeed)
{
    if(succeed)
    {
        post_processer_->setSourceImagePath(data_reader_->getCurrentImagePath());

        emit signal_postProcess();
    }
    else
    {
        qDebug() << "RailwayFaultDetectWidget::slot_lapnetDetectPoint_finished : lapnet point detect failed.";
    }
}

void RailwayFaultDetectWidget::slot_postProcess_finished(bool succeed)
{
    if(succeed)
    {
        emit signal_getData();
    }
    else
    {
        qDebug() << "RailwayFaultDetectWidget::slot_postProcess_finished : post process failed.";
    }
}

//signal_getData
//slot_getData_finished
//signal_darknetDetect
//slot_darknetDetect_finished
//signal_lapnetDetectLine
//slot_lapnetDetectLine_finished
//signal_lapnetDetectPoint
//slot_lapnetDetectPoint_finished
