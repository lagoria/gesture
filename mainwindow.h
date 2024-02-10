#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include <QFileDialog>
#include <QFile>
#include <QMainWindow>
#include <QTimer>
#include <QImage>
#include <QPixmap>
#include <QDateTime>
#include <QMutex>
#include <QMutexLocker>
#include <QMimeDatabase>
#include <iostream>
#include <QElapsedTimer>
#include <QThread>

#include <yolov5.h>
#include "eventgroup.h"


QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    enum {
        OPENED_FILE_EVT = 0,
        OPENED_ONNX_EVT,
        SELECT_PICTURE_EVT,
        SELECT_VIDEO_EVT
    };

    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:

    void on_openfile_clicked();

    void on_loadfile_clicked();

    void on_startdetect_clicked();

    void on_stopdetect_clicked();

    void detectReport(const QPixmap &output);
    void playDone();


private:
    std::vector<std::string> class_name = {"0", "1", "2", "3", "5"};
    Ui::MainWindow *ui;
    QElapsedTimer timer;
    DetectModel *model;
    QString filename;
    EventGroup event_group;
    unsigned long current_frames = 0;

};
#endif // MAINWINDOW_H
