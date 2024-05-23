#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include <QFileDialog>
#include <QFile>
#include <QMainWindow>
#include <QTimer>
#include <QImage>
#include <QPixmap>
#include <QMimeDatabase>
#include <iostream>
#include <QElapsedTimer>
#include <QThread>

#include <inferenceEngine.h>
#include "eventgroup.h"
#include "VersionConfig.h"

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

    void on_openCamera_clicked();

    void on_startdetect_clicked();

    void on_stopdetect_clicked();

    void on_clearButton_clicked();

    void detectReport(const QPixmap &output);
    void playDone();


private:
    Ui::MainWindow *ui;
    QElapsedTimer timer;
    InferenceEngine *inference;
    QString filename;
    EventGroup event_group;
    bool setup_time_flag = false;
    bool camera_setup_flag = false;

};
#endif // MAINWINDOW_H
