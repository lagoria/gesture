#ifndef YOLOV5_H
#define YOLOV5_H
#include <QImage>
#include <QPixmap>
#include <QThread>
// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#ifndef USE_MINGW_COMPILER
// ONNX Runtime
#include <onnxruntime_cxx_api.h>
#endif


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

    /**
     * @brief loadOnnxModel 加载ONNX模型
     * @param onnxfile 文件路径
     * @return 0：成功，其他：失败
     */
    int loadOnnxModel(const char* onnxfile);

    /**
     * @brief thresholdConfig 配置模型阈值
     * @param conf 配置结构体
     */
    void thresholdConfig(modelConfig_t &conf);

    /**
     * @brief imageOutSizeConfig 配置图像输出大小
     * @param size 大小
     */
    void imageOutSizeConfig(QSize size);

    /**
     * @brief pictureDetect 检测一张图片
     * @param file 图片路径
     * @return
     */
    QPixmap pictureDetect(const char *file);

    /**
     * @brief pictureThreadDetect 在线程中检测一张图片
     * @param file 图片路径
     */
    void pictureThreadDetect(const char *file);

    /**
     * @brief openVideo 打开视频文件
     * @param file 文件路径
     * @return
     */
    bool openVideo(const char *file);

    /**
     * @brief readVideoFirstFrame 读取视频的第一帧
     * @return 输出图片
     */
    QPixmap readVideoFirstFrame();

    /**
     * @brief setVideoStartFrame 设置视频开始帧起点
     * @param startPoint 起始点
     */
    void setVideoStartFrame(unsigned long startPoint);

    /**
     * @brief startThreadDetect 开始线程检测
     */
    void startThreadDetect();

    /**
     * @brief pauseThreadDetect 中止线程检测
     */
    void pauseThreadDetect();


    friend class VideoDetectThread;         // 申明友元类
    VideoDetectThread *thread;              // 视频检测线程
    std::vector<int64_t> output_shape;      // 模型输出形状
    std::vector<std::string> output_labels; // 模型输出类名
    bool cudaEnableStatus;                  // CUDA使能状态

private:

#ifndef USE_MINGW_COMPILER
    /**
     * @brief parseOnnxModel 解析ONNX模型文件，获取信息
     * @param onnxfile  文件路径
     * @return 解析结果
     */
    int parseOnnxModel(const char *onnxfile);
#endif

    //
    /**
     * @brief pre_process 检测预处理
     * @param image 图像输入
     * @param blob 处理输出
     */
    void pre_process(cv::Mat& image, cv::Mat& blob);
    //
    /**
     * @brief post_process 检测后处理
     * @param image 源图像
     * @param outputs 检测输出图像
     * @param output_labels 输出标签
     * @return 输出图像
     */
    cv::Mat post_process(cv::Mat& image, std::vector<cv::Mat>& outputs,
                         std::vector<std::string> &output_labels);

    /**
     * @brief scale_boxes 缩放检测框
     * @param box 框
     * @param size 大小
     */
    void scale_boxes(cv::Rect& box, cv::Size size);

    /**
     * @brief draw_result 绘制检测结果
     * @param image 图像
     * @param label 标签
     * @param box 检测框
     */
    void draw_result(cv::Mat& image, std::string label, cv::Rect box);

    /**
     * @brief cvMatToQPixmap 转化cv::Mat 到 QPixmap
     * @param cvMat 源图像
     * @return 转化后图像
     */
    QPixmap cvMatToQPixmap(const cv::Mat &cvMat);

    /**
     * @brief resizeImage 图像缩放
     * @param image 图像
     * @param maxWidth 缩放最大宽度
     * @param maxHeight 缩放最大高度
     * @return 缩放后图像
     */
    cv::Mat resizeImage(const cv::Mat &image, int maxWidth, int maxHeight);

    /**
     * @brief frameDetect 图像帧检测
     * @param frame 图像帧
     * @return
     */
    QPixmap frameDetect(cv::Mat frame);

    modelConfig_t config;                   // 模型配置
    cv::dnn::Net net;                       // 模型
    cv::VideoCapture *capture;              // 视频抓取器
    QSize display_size;                     // 图像输出大小

};


class VideoDetectThread : public QThread
{
    Q_OBJECT
public:
    explicit VideoDetectThread() {}
    ~VideoDetectThread() {}

    friend class DetectModel;         // 申明友元类
protected:
    void run() override; // 重写QThread类的虚函数，也是线程子类的入口函数
signals:
    void done(); // 完成信号
    void reportProgress(QPixmap output); // 报告完成进度

private:
    DetectModel *model;
    bool pause_flag = false;
    std::string picture_path;
};


#endif // YOLOV5_H
