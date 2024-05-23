#ifndef InferenceEngine_H
#define InferenceEngine_H
// Cpp native
#include <string>
// Qt
#include <QImage>
#include <QPixmap>
#include <QThread>
#include <QDebug>
// OpenCV
#include <opencv2/opencv.hpp>

#ifdef USE_ONNXRUNTIME_LIB
// ONNX Runtime
#include <onnxruntime_cxx_api.h>
#endif


class InferenceThread;


class InferenceEngine
{
public:

    enum {
        // model parse status
        STATUS_MODEL_INVALID = -3,
        STATUS_LABEL_INVALID,
        STATUS_FILE_INVALID,
        STATUS_PROCESS_OK = 0,

        // model type
        MODEL_TYPE_UNKNOWN,
        MODEL_TYPE_YOLOV5,
        MODEL_TYPE_YOLOV8,
    };

    // 模型配置结构
    typedef struct
    {
        int input_width;
        int input_height;
        float score_threshold;
        float nms_threshold;            // 非极大值抑制（Non-Maximum Suppression，NMS）
        float confidence_threshold;
        bool enable_cuda;
    } modelConfig_t;

    // 推理检测输出结构
    struct DetectResult
    {
        int class_id{0};
        std::string className{};
        float confidence{0.0};
        cv::Rect box{};
    };


    InferenceEngine();      // 构造函数
    ~InferenceEngine();     // 析构函数

    /**
     * @brief loadOnnxModel 加载ONNX模型
     * @param onnxfile 文件路径
     * @return 0：成功，其他：失败
     */
    int loadOnnxModel(const std::string &onnxfile);

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
    QPixmap pictureDetect(const std::string &file);

    /**
     * @brief pictureThreadDetect 在线程中检测一张图片
     * @param file 图片路径
     */
    void pictureThreadDetect(const std::string &file);

    /**
     * @brief openVideo 打开视频文件
     * @param file 文件路径
     * @return
     */
    bool openVideo(const std::string &file);

    /**
     * @brief openDefaultCamera 打开默认摄像头
     * @return
     */
    bool openDefaultCamera();

    /**
     * @brief readCameraVideoStream 读取摄像视频流
     * @return
     */
    int readCameraVideoStream();

    /**
     * @brief stopCameraVideoStream 停止摄像视频流
     * @return
     */
    int stopCameraVideoStream();

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



    friend class InferenceThread;           // 申明友元类
    InferenceThread *thread;                // 视频检测线程
    std::vector<int64_t> output_shape;      // 模型输出形状
    std::vector<std::string> classes;       // 模型输出类名
    bool cudaEnableStatus;                  // CUDA使能状态

private:

#ifndef USE_MINGW_COMPILER
    /**
     * @brief parseOnnxModel 解析ONNX模型文件，获取信息
     * @param onnxfile  文件路径
     * @return 解析结果
     */
    int parseOnnxModel(const std::string &onnxfile);
#endif

    /**
     * @brief runInference 启动推理
     * @param input 图像帧
     * @return 推理检测结果向量
     */
    std::vector<DetectResult> runInference(const cv::Mat &input);

    /**
     * @brief drawDetectResult 绘制检测结果
     * @param image 源图像
     * @param detection 推理检测输出
     * @return 绘制后图像
     */
    cv::Mat drawDetectResult(cv::Mat& image, std::vector<DetectResult> detect_result);

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
     * @brief scaleToLetterBox 缩放成像信封的图像
     * @param image 原图像
     * @param newShape 新图像形状
     * @param color 边框填充颜色
     * @return 输出图像
     */
    cv::Mat scaleToLetterBox(const cv::Mat& image, const cv::Size& newShape, const cv::Scalar& color);

    /**
     * @brief scaleBoxToSource 缩放监测框到原图像
     * @param box 检测框
     * @param source_size 原图像尺寸
     * @return 输出框
     */
    cv::Rect scaleBoxToSource(const cv::Rect &box, cv::Size source_size);

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
    int model_type;                         // 模型类型

    // 边界框颜色分类定义
    std::vector<cv::Scalar> label_color_map = {
        cv::Scalar(255, 0, 0),      // 蓝色
        cv::Scalar(0, 255, 0),      // 绿色
        cv::Scalar(0, 0, 255),      // 红色
        cv::Scalar(0, 255, 255),    // 黄色
        cv::Scalar(255, 255, 0),    // 青色
        cv::Scalar(255, 0, 255),    // 紫色
    };
};



class InferenceThread : public QThread
{
    Q_OBJECT
public:
    explicit InferenceThread() {}
    ~InferenceThread() {}

    friend class InferenceEngine;           // 申明友元类
protected:
    void run() override;                    // 重写QThread类的虚函数，也是线程子类的入口函数
signals:
    void done();                            // 完成信号
    void reportProgress(QPixmap output);    // 报告进度

private:
    InferenceEngine *model;
    bool pause_thread_flag = false;         // 中断线程标志
    std::string picture_path;
    bool setup_detect_flag = false;         // 启动检测标志
    bool video_stream_flag = false;         // 摄像视频流标志
};


#endif // InferenceEngine_H
