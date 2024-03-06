#ifndef YOLOV5_H
#define YOLOV5_H
#include <QImage>
#include <QPixmap>
#include <QThread>
// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include "VersionConfig.h"

#ifndef USE_MINGW_COMPILER
// ONNX Runtime
#include <onnxruntime_cxx_api.h>
#endif

#include "QDebug"

class VideoDetectThread;


class DetectModel
{
public:

    enum {
        STATUS_MODEL_INVALID = -3,
        STATUS_LABEL_INVALID,
        STATUS_FILE_INVALID,
        STATUS_PROCESS_OK = 0,
    };

    // 模型配置结构
    typedef struct
    {
        int input_width;
        int input_height;
        float score_threshold;
        float nms_threshold;
        float confidence_threshold;
    } modelConfig_t;

    DetectModel();      // 构造函数
    ~DetectModel();     // 析构函数

    // 加载ONNX模型
    int loadOnnxModel(const char* onnxfile);
    // 配置模型阈值
    void thresholdConfig(modelConfig_t &conf);
    // 检测一张图片
    QPixmap pictureDetect(const char *file);
    // 打开视频文件
    bool openVideo(const char *file);
    // 设置视频开始帧起点
    void setVideoStartFrame(unsigned long startPoint);
    // 开始视频检测
    void startVideoDetect();
    // 中止视频检测
    void pauseVideoDetect();

    // 图像帧检测
    QPixmap frameDetect(cv::Mat frame);

    VideoDetectThread *thread;              // 视频检测线程
    cv::VideoCapture *capture;              // 视频抓取器
    std::vector<int64_t> output_shape;      // 模型输出形状
    std::vector<std::string> output_labels;    // 模型输出类名
    bool cudaEnableStatus;                  // CUDA使能状态

private:

#ifndef USE_MINGW_COMPILER
    // 解析onnx模型
    int parseOnnxModel(const char *onnxfile);
#endif

    // 检测预处理
    void pre_process(cv::Mat& image, cv::Mat& blob);
    // 检测后处理
    cv::Mat post_process(cv::Mat& image, std::vector<cv::Mat>& outputs,
                         std::vector<std::string> &output_labels);
    // 缩放检测框
    void scale_boxes(cv::Rect& box, cv::Size size);
    // 绘制检测结果
    void draw_result(cv::Mat& image, std::string label, cv::Rect box);
    // 转化cv::Mat 到 QPixmap
    QPixmap cvMatToQPixmap(const cv::Mat &cvMat);

    modelConfig_t config;                   // 模型配置
    cv::dnn::Net net;                       // 模型
    unsigned long start_frame;              // 视频帧起点

};


class VideoDetectThread : public QThread
{
    Q_OBJECT
public:
    explicit VideoDetectThread() {}
    ~VideoDetectThread() {}
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
