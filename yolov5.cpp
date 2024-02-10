#include "yolov5.h"



DetectModel::DetectModel()
{
    this->config.input_width = 640;
    this->config.input_height = 640;
    this->config.score_threshold = 0.5;
    this->config.nms_threshold = 0.45;
    this->config.confidence_threshold = 0.45;

    cudaEnableStatus = false;

    this->start_frame = 0;

    this->thread = new VideoDetectThread();
    this->capture = new cv::VideoCapture();
}

DetectModel::~DetectModel()
{
    capture->release();
    delete capture;
    delete thread;
}

void DetectModel::thresholdConfig(modelConfig_t &conf)
{
    this->config = conf;
}


void DetectModel::classNameConfig(std::vector<std::string> &classes)
{
    this->class_name = classes;
}

bool DetectModel::loadOnnx(const char *onnxfile)
{
    this->net = cv::dnn::readNet(onnxfile);
    if (net.empty()) {
        return false;
    }
    int device_count = cv::cuda::getCudaEnabledDeviceCount();
    if (device_count >= 1){
        cudaEnableStatus = true;
        this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }else{
        cudaEnableStatus = false;
    }
    return true;
}

// LetterBox处理
void LetterBox(const cv::Mat& image, cv::Mat& outImage,
               cv::Vec4d& params, // [ratio_x,ratio_y,dw,dh]
               const cv::Size& newShape = cv::Size(640, 640),
               bool autoShape = false,
               bool scaleFill = false,
               bool scaleUp = true,
               int stride = 32,
               const cv::Scalar& color = cv::Scalar(114, 114, 114))
{
    cv::Size shape = image.size();
    float r = std::min((float)newShape.height / (float)shape.height, (float)newShape.width / (float)shape.width);
    if (!scaleUp)
    {
        r = std::min(r, 1.0f);
    }

    float ratio[2]{ r, r };
    int new_un_pad[2] = { (int)std::round((float)shape.width * r),(int)std::round((float)shape.height * r) };

    auto dw = (float)(newShape.width - new_un_pad[0]);
    auto dh = (float)(newShape.height - new_un_pad[1]);

    if (autoShape)
    {
        dw = (float)((int)dw % stride);
        dh = (float)((int)dh % stride);
    }
    else if (scaleFill)
    {
        dw = 0.0f;
        dh = 0.0f;
        new_un_pad[0] = newShape.width;
        new_un_pad[1] = newShape.height;
        ratio[0] = (float)newShape.width / (float)shape.width;
        ratio[1] = (float)newShape.height / (float)shape.height;
    }

    dw /= 2.0f;
    dh /= 2.0f;

    if (shape.width != new_un_pad[0] && shape.height != new_un_pad[1])
        cv::resize(image, outImage, cv::Size(new_un_pad[0], new_un_pad[1]));
    else
        outImage = image.clone();

    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));
    params[0] = ratio[0];
    params[1] = ratio[1];
    params[2] = left;
    params[3] = top;
    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}

void DetectModel::pre_process(cv::Mat &image, cv::Mat &blob)
{
    cv::Vec4d params;
    cv::Mat letterbox;
    LetterBox(image, letterbox, params, cv::Size(this->config.input_width, this->config.input_height));
    cv::dnn::blobFromImage(letterbox, blob, 1. / 255., cv::Size(this->config.input_width, this->config.input_height), cv::Scalar(), true, false);
}


//box缩放到原图尺寸
void DetectModel::scale_boxes(cv::Rect& box, cv::Size size)
{
    float gain = std::min(this->config.input_width * 1.0 / size.width, this->config.input_height * 1.0 / size.height);
    int pad_w = (this->config.input_width - size.width * gain) / 2;
    int pad_h = (this->config.input_height - size.height * gain) / 2;
    box.x -= pad_w;
    box.y -= pad_h;
    box.x /= gain;
    box.y /= gain;
    box.width /= gain;
    box.height /= gain;
}

//可视化函数
void draw_result(cv::Mat& image, std::string label, cv::Rect box)
{
    cv::rectangle(image, box, cv::Scalar(255, 0, 0), 2);
    int baseLine;
    cv::Size label_size = cv::getTextSize(label, 0.8, 0.8, 1, &baseLine);
    cv::Point tlc = cv::Point(box.x, box.y);
    cv::Point brc = cv::Point(box.x, box.y + label_size.height + baseLine);
    cv::putText(image, label, cv::Point(box.x, box.y), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 1);
}

void DetectModel::draw_result(cv::Mat &image, std::string label, cv::Rect box)
{
    cv::rectangle(image, box, cv::Scalar(255, 0, 0), 2);
    int baseLine;
    cv::Size label_size = cv::getTextSize(label, 0.8, 0.8, 1, &baseLine);
    cv::Point tlc = cv::Point(box.x, box.y);
    cv::Point brc = cv::Point(box.x, box.y + label_size.height + baseLine);
    cv::putText(image, label, cv::Point(box.x, box.y), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 1);
}


cv::Mat DetectModel::post_process(cv::Mat &image, std::vector<cv::Mat> &outputs, std::vector<std::string> &class_name)
{
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    float* data = (float*)outputs[0].data;

    const int dimensions = 10;  //5+5
    const int rows = 25200;		//(640/8)*(640/8)*3+(640/16)*(640/16)*3+(640/32)*(640/32)*3
    for (int i = 0; i < rows; ++i)
    {
        float confidence = data[4];
        if (confidence >= this->config.confidence_threshold)
        {
            float* classes_scores = data + 5;
            cv::Mat scores(1, class_name.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > this->config.score_threshold)
            {
                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                int left = int(x - 0.5 * w);
                int top = int(y - 0.5 * h);
                int width = int(w);
                int height = int(h);
                cv::Rect box = cv::Rect(left, top, width, height);
                scale_boxes(box, image.size());
                boxes.push_back(box);
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);
            }
        }
        data += dimensions;
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, this->config.score_threshold, this->config.nms_threshold, indices);
    for (int i = 0; i < indices.size(); i++)
    {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        std::string label = class_name[class_ids[idx]] + ":" + cv::format("%.2f", confidences[idx]);
        draw_result(image, label, box);
    }
    return image;
}


QPixmap DetectModel::cvMatToQPixmap(const cv::Mat &cvMat)
{
    // 将 BGR 转换为 RGB
    cv::Mat rgbMat;
    cv::cvtColor(cvMat, rgbMat, cv::COLOR_BGR2RGB);

    // 将 OpenCV Mat 转换为 QImage
    QImage img((const uchar*) rgbMat.data, rgbMat.cols, rgbMat.rows, rgbMat.step, QImage::Format_RGB888);

    // 将 QImage 转换为 QPixmap
    return QPixmap::fromImage(img);
}


QPixmap DetectModel::frameDetect(cv::Mat frame)
{
    cv::Mat blob;
    this->pre_process(frame, blob);

    std::vector<cv::Mat> outputs;
    this->net.setInput(blob);
    this->net.forward(outputs, net.getUnconnectedOutLayersNames());

    cv::Mat result = this->post_process(frame, outputs, this->class_name);
    return cvMatToQPixmap(result);
}


QPixmap DetectModel::pictureDetect(const char *file)
{
    cv::Mat image = cv::imread(cv::String(file));
    QPixmap result = this->frameDetect(image);

    return result;
}

bool DetectModel::openVideo(const char *file)
{
    if (this->capture->isOpened()) {
        capture->release();
    }
    this->capture->open(file);
    if (!this->capture->isOpened()){
        return false;
    }
    this->start_frame = 0;
    return true;
}


void DetectModel::setVideoStartFrame(unsigned long startPoint)
{
    long totalFrame = this->capture->get(cv::CAP_PROP_FRAME_COUNT);
    if (startPoint < totalFrame) {
        this->start_frame = startPoint;
    }
}

void DetectModel::startVideoDetect()
{
    this->capture->set(cv::CAP_PROP_POS_FRAMES, this->start_frame);
    this->thread->configure(this);
    this->thread->start();
}

void DetectModel::pauseVideoDetect()
{
    this->thread->pauseThread();
}


/* class DetectThread */

VideoDetectThread::VideoDetectThread()
{

}

VideoDetectThread::~VideoDetectThread()
{

}


void VideoDetectThread::pauseThread()
{
    pauseFlag = true;
}

void VideoDetectThread::configure(DetectModel *model_)
{
    this->model = model_;
}

void VideoDetectThread::run()
{
    cv::Mat frame;
    pauseFlag = false;
    while (this->model->capture->read(frame)) {
        QPixmap output = this->model->frameDetect(frame);
        emit reportProgress(output);

        if (pauseFlag == true) {
            break;
        }
    }

    emit done();
}

