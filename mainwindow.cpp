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
    this->inference = new InferenceEngine();

    connect(this->inference->thread, &InferenceThread::reportProgress, this, &MainWindow::detectReport);
    connect(this->inference->thread, &InferenceThread::done, this, &MainWindow::playDone);


    // 使用宏定义的版本号
    int major = PROJECT_VERSION_MAJOR;
    int minor = PROJECT_VERSION_MINOR;
    int patch = PROJECT_VERSION_PATCH;
    ui->textEditlog->append(QString("Project Version: %1.%2.%3").arg(major).arg(minor).arg(patch));

#ifdef USE_MINGW_COMPILER
    // 配置模型输出信息
    this->inference->classes = {"fist", "one", "two", "three", "five", "four"};
#endif
}

MainWindow::~MainWindow()
{
    delete inference;
    delete ui;
}


void MainWindow::on_openfile_clicked()
{
    QString open_file = QFileDialog::getOpenFileName(this,QStringLiteral("打开文件"),".","*.png *.jpg *.jpeg *.bmp;;*.mp4 *.avi");
    if(!QFile::exists(open_file)){
        return;
    }
    this->filename = open_file;
    ui->statusbar->showMessage(this->filename);

    QMimeDatabase db;
    QMimeType mime = db.mimeTypeForFile(this->filename);
    if (mime.name().startsWith("image/")) {

        ui->textEditlog->append(QString("Picture opened successfully!"));
        QPixmap pixmap(this->filename.toLatin1().data());
        QPixmap picture = pixmap.scaled(ui->label->size(), Qt::KeepAspectRatio);
        ui->label->setPixmap(picture);

        // 设置图片选择事件
        event_group.setBits(SELECT_PICTURE_EVT);
        event_group.clearBits(SELECT_VIDEO_EVT);

    } else if (mime.name().startsWith("video/")) {
        if (!inference->openVideo(filename.toStdString())){
            ui->textEditlog->append("fail to open video!");
            return;
        }

        ui->textEditlog->append(QString("Video opened succesfully!"));

        QPixmap pixmap = inference->readVideoFirstFrame();
        QPixmap picture = pixmap.scaled(ui->label->size(), Qt::KeepAspectRatio);
        ui->label->setPixmap(picture);

        //设置开始帧()
        unsigned long frameToStart = 0;
        inference->setVideoStartFrame(frameToStart);

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
    int status = inference->loadOnnxModel(onnxFile.toStdString());

    switch (status) {
    case InferenceEngine::STATUS_MODEL_INVALID:
        ui->textEditlog->append(QStringLiteral("模型类型不符！"));
        break;
    case InferenceEngine::STATUS_LABEL_INVALID:
        ui->textEditlog->append(QStringLiteral("模型标签获取失败！"));
        break;
    case InferenceEngine::STATUS_FILE_INVALID:
        ui->textEditlog->append(QStringLiteral("模型打开失败！"));
        break;
    case InferenceEngine::STATUS_PROCESS_OK: {
        ui->textEditlog->append(QString("OnnxFile opened succesfully!"));
        QStringList stringList;
        for (int64_t value : inference->output_shape) {
            stringList.append(QString::number(value)); // 将int64_t转换为QString
        }
        ui->textEditlog->append("Out shape:["+stringList.join(",")+"]");

        QStringList labels;
        for (const std::string &name : inference->classes) {
            labels.append(name.c_str());
        }
        ui->textEditlog->append("class name:["+labels.join(",")+"]");

        if (this->inference->cudaEnableStatus) {
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
        ui->startdetect->setEnabled(false);
        ui->stopdetect->setEnabled(false);
        ui->openfile->setEnabled(false);
        ui->loadfile->setEnabled(false);
        ui->textEditlog->append(QStringLiteral("======start detect======"));

        // 开始计时
        timer.start();
        this->inference->imageOutSizeConfig(ui->label->size());
        this->inference->pictureThreadDetect(this->filename.toStdString());
        this->inference->startThreadDetect();
    }

    if (event_group.waitBits(SELECT_VIDEO_EVT)) {
        ui->startdetect->setEnabled(false);
        ui->stopdetect->setEnabled(true);
        ui->openfile->setEnabled(false);
        ui->loadfile->setEnabled(false);
        ui->textEditlog->append(QStringLiteral("======start detect======"));

        // 开始计时
        timer.start();
        this->inference->imageOutSizeConfig(ui->label->size());
        this->inference->startThreadDetect();
    }
}

void MainWindow::on_stopdetect_clicked()
{

    if (event_group.waitBits(SELECT_PICTURE_EVT)) {
        ui->startdetect->setEnabled(true);
        ui->stopdetect->setEnabled(false);
        ui->openfile->setEnabled(true);
        ui->loadfile->setEnabled(true);
        ui->textEditlog->append(QStringLiteral("=======end detect=======\n"));

    } else {
        this->inference->pauseThreadDetect();
    }

}


void MainWindow::detectReport(const QPixmap &output)
{
    // 停止计时，并获取经过的时间（以毫秒为单位）
    int elapsedTime = timer.elapsed();
    ui->textEditlog->append(QString("cost_time: %1 ms").arg(elapsedTime));

    QPixmap picture = output.scaled(ui->label->size(), Qt::KeepAspectRatio);
    this->inference->imageOutSizeConfig(ui->label->size());
    ui->label->setPixmap(picture);

    timer.start();
}

void MainWindow::playDone()
{
    ui->startdetect->setEnabled(true);
    ui->stopdetect->setEnabled(false);
    ui->openfile->setEnabled(true);
    ui->loadfile->setEnabled(true);
    ui->textEditlog->append(QStringLiteral("=======end detect=======\n"));
    timer.elapsed();
}
