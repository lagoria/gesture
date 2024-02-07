#ifndef YOLOV5_H
#define YOLOV5_H
#include <opencv2/opencv.hpp>
#include <QImage>
#include <QPixmap>
#include <QThread>

class VideoDetectThread;


class DetectModel
{
public:

    typedef struct
    {
        int input_width;
        int input_height;
        float score_threshold;
        float nms_threshold;
        float confidence_threshold;
    } modelConfig_t;

    DetectModel();
    ~DetectModel();
    bool loadOnnx(const char* onnxfile);
    void thresholdConfig(modelConfig_t &conf);
    void classNameConfig(std::vector<std::string> &classes);
    QPixmap pictureDetect(const char *file);
    bool openVideo(const char *file);
    void setVideoStartFrame(unsigned long startPoint);
    void startVideoDetect();
    void pauseVideoDetect();

    QPixmap frameDetect(cv::Mat frame);

    VideoDetectThread *thread;
    cv::VideoCapture *capture;

private:

    void pre_process(cv::Mat& image, cv::Mat& blob);
    cv::Mat post_process(cv::Mat& image, std::vector<cv::Mat>& outputs,
                         std::vector<std::string> &class_name);
    void scale_boxes(cv::Rect& box, cv::Size size);
    void draw_result(cv::Mat& image, std::string label, cv::Rect box);
    QPixmap cvMatToQPixmap(const cv::Mat &cvMat);

    modelConfig_t config;
    cv::dnn::Net net;
    std::vector<std::string> class_name;
    unsigned long start_frame;

};


class VideoDetectThread : public QThread
{
    Q_OBJECT
public:
    explicit VideoDetectThread();
    ~VideoDetectThread();
    void configure(DetectModel *model_);
    void pauseThread();
protected:
    void run() override; // 重写QThread类的虚函数，也是线程子类的入口函数
signals:
    void done(); // 完成信号
    void reportProgress(QPixmap output); // 报告完成进度

private:
    DetectModel *model;
    bool pauseFlag = false;
};


#endif // YOLOV5_H
