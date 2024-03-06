#include "mainwindow.h"
#include "ui_mainwindow.h"


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    setWindowTitle(QStringLiteral("手势检测软件"));

    ui->startdetect->setEnabled(false);
    ui->stopdetect->setEnabled(false);
    this->model = new DetectModel();

    connect(this->model->thread, &VideoDetectThread::reportProgress, this, &MainWindow::detectReport);
    connect(this->model->thread, &VideoDetectThread::done, this, &MainWindow::playDone);


    // 使用宏定义的版本号
    int major = PROJECT_VERSION_MAJOR;
    int minor = PROJECT_VERSION_MINOR;
    int patch = PROJECT_VERSION_PATCH;
    ui->textEditlog->append(QString("Project Version: %1.%2.%3").arg(major).arg(minor).arg(patch));
}

MainWindow::~MainWindow()
{
    delete model;
    delete ui;
}



void MainWindow::on_openfile_clicked()
{
    this->filename = QFileDialog::getOpenFileName(this,QStringLiteral("打开文件"),".","*.png *.jpg *.jpeg *.bmp;;*.mp4 *.avi");
    if(!QFile::exists(this->filename)){
        return;
    }
    ui->statusbar->showMessage(this->filename);

    QMimeDatabase db;
    QMimeType mime = db.mimeTypeForFile(this->filename);
    if (mime.name().startsWith("image/")) {

        ui->textEditlog->append(QString("Picture opened successfully!"));
        QPixmap pixmap(this->filename.toLatin1().data());
        ui->label->setPixmap(pixmap);
        ui->label->setScaledContents(true);

        // 设置图片选择事件
        event_group.setBits(SELECT_PICTURE_EVT);
        event_group.clearBits(SELECT_VIDEO_EVT);

    } else if (mime.name().startsWith("video/")) {
        if (!model->openVideo(filename.toLatin1().data())){
            ui->textEditlog->append("fail to open video!");
            return;
        }

        ui->textEditlog->append(QString("Video opened succesfully!"));

        //获取整个帧数QStringLiteral
        long totalFrame = model->capture->get(cv::CAP_PROP_FRAME_COUNT);
        int width = model->capture->get(cv::CAP_PROP_FRAME_WIDTH);
        int height = model->capture->get(cv::CAP_PROP_FRAME_HEIGHT);
        ui->textEditlog->append(QStringLiteral("整个视频共 %1 帧, 宽=%2 高=%3 ").arg(totalFrame).arg(width).arg(height));
        // ui->label->resize(QSize(width, height));

        //设置开始帧()
        unsigned long frameToStart = 0;
        model->setVideoStartFrame(frameToStart);
        ui->textEditlog->append(QStringLiteral("从第 %1 帧开始读").arg(frameToStart));

        //获取帧率
        double rate = model->capture->get(cv::CAP_PROP_FPS);
        ui->textEditlog->append(QStringLiteral("帧率为: %1 ").arg(rate));

        // 设置视频选择事件
        event_group.setBits(SELECT_VIDEO_EVT);
        event_group.clearBits(SELECT_PICTURE_EVT);
    }

    // 判断事件条件，启动预测按钮
    event_group.setBits(OPENED_FILE_EVT);
    if (event_group.waitBits(OPENED_FILE_EVT) && event_group.waitBits(OPENED_ONNX_EVT)) {
        ui->startdetect->setEnabled(true);
    }

}



void MainWindow::on_loadfile_clicked()
{
    QString onnxFile = QFileDialog::getOpenFileName(this,QStringLiteral("选择模型"),".","*.onnx");
    if(!QFile::exists(onnxFile)){
        return;
    }
    ui->statusbar->showMessage(onnxFile);
    int status = model->loadOnnxModel(onnxFile.toLatin1().data());

    switch (status) {
    case DetectModel::STATUS_MODEL_INVALID:
        ui->textEditlog->append(QStringLiteral("模型类型不符！"));
        break;
    case DetectModel::STATUS_LABEL_INVALID:
        ui->textEditlog->append(QStringLiteral("模型标签获取失败！"));
        break;
    case DetectModel::STATUS_FILE_INVALID:
        ui->textEditlog->append(QStringLiteral("模型打开失败！"));
        break;
    case DetectModel::STATUS_PROCESS_OK: {
        ui->textEditlog->append(QString("OnnxFile opened succesfully!"));
        QStringList stringList;
        for (int64_t value : model->output_shape) {
            stringList.append(QString::number(value)); // 将int64_t转换为QString
        }
        ui->textEditlog->append("Out shape:["+stringList.join(",")+"]");

        QStringList labels;
        for (const std::string &name : model->output_labels) {
            labels.append(name.c_str());
        }
        ui->textEditlog->append("class name:["+labels.join(",")+"]");

        if (this->model->cudaEnableStatus) {
            ui->textEditlog->append(QString("Use the CUDA inference!"));
        } else {
            ui->textEditlog->append(QString("Use the CPU inference!"));
        }
        // 判断事件条件，启动预测按钮
        event_group.setBits(OPENED_ONNX_EVT);
        if (event_group.waitBits(OPENED_FILE_EVT) && event_group.waitBits(OPENED_ONNX_EVT)) {
            ui->startdetect->setEnabled(true);
        }
        break;
    }
    default : break;
    }

}

void MainWindow::on_startdetect_clicked()
{
    if (event_group.waitBits(SELECT_PICTURE_EVT)) {
        timer.start();
        QPixmap output = model->pictureDetect(this->filename.toLatin1().data());
        int elapsedTime = timer.elapsed();
        ui->textEditlog->append(QString("cost_time: %1 ms").arg(elapsedTime));
        ui->label->setPixmap(output);
        ui->label->setScaledContents(true);
    }

    if (event_group.waitBits(SELECT_VIDEO_EVT)) {
        ui->startdetect->setEnabled(false);
        ui->stopdetect->setEnabled(true);
        ui->openfile->setEnabled(false);
        ui->loadfile->setEnabled(false);
        ui->textEditlog->append(QStringLiteral("=========开始检测=========\n"));

        // 开始计时
        timer.start();
        this->model->startVideoDetect();
    }
}

void MainWindow::on_stopdetect_clicked()
{

    if (event_group.waitBits(SELECT_PICTURE_EVT)) {
        ui->startdetect->setEnabled(true);
        ui->stopdetect->setEnabled(false);
        ui->openfile->setEnabled(true);
        ui->loadfile->setEnabled(true);
        ui->textEditlog->append(QStringLiteral("=========结束检测=========\n"));

    } else {
        this->model->pauseVideoDetect();
    }

}


void MainWindow::detectReport(const QPixmap &output)
{
    // 停止计时，并获取经过的时间（以毫秒为单位）
    int elapsedTime = timer.elapsed();
    current_frames++;
    ui->textEditlog->append(QString("[%1] cost_time: %2 ms").arg(current_frames)
                                .arg(elapsedTime));

    ui->label->setPixmap(output);
    ui->label->setScaledContents(true);

    timer.start();
}

void MainWindow::playDone()
{
    ui->startdetect->setEnabled(true);
    ui->stopdetect->setEnabled(false);
    ui->openfile->setEnabled(true);
    ui->loadfile->setEnabled(true);
    ui->textEditlog->append(QStringLiteral("=========结束检测=========\n"));
    timer.elapsed();
    current_frames = 0;
}
