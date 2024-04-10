#include "yolov5.h"


DetectModel::DetectModel()
{
    // default param settings
    this->config.input_width = 640;
    this->config.input_height = 640;
    this->config.score_threshold = 0.5;
    this->config.nms_threshold = 0.45;
    this->config.confidence_threshold = 0.45;

    cudaEnableStatus = false;

    this->display_size = QSize(640, 480);

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


#ifndef USE_MINGW_COMPILER
int DetectModel::parseOnnxModel(const char *onnxfile)
{
    QString fileNamePath = onnxfile;
    const wchar_t* model_path = reinterpret_cast<const wchar_t *>(fileNamePath.utf16());

    Ort::Env env; // 创建env
    Ort::Session session(nullptr); // 创建一个空会话
    Ort::SessionOptions sessionOptions{ nullptr }; // 创建会话配置
    Ort::AllocatorWithDefaultOptions allocator;
    session = Ort::Session(env, model_path, sessionOptions);

    // 检测模型输出
    if (session.GetOutputCount() == 1) {
        // 保存模型输出维度
        this->output_shape.clear();
        this->output_shape = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    } else {
        return STATUS_MODEL_INVALID;      // 模型类型不符
    }

    // 获取模型元数据
    Ort::ModelMetadata model_metadata = session.GetModelMetadata();

    // 获取标签名
    int64_t num_elements;
    char** data_keys = model_metadata.GetCustomMetadataMapKeys(allocator, num_elements);

    for (int64_t i = 0; i < num_elements; ++i) {
        // 在元数据中找到 names 的键值
        if (QString(data_keys[i]) == QString("names")) {
            this->output_labels.clear();
            std::string label_names = model_metadata.LookupCustomMetadataMap(data_keys[i], allocator);
            // 移除开头的'{'和结尾的'}'
            label_names.erase(0, 1); // 移除'{'
            label_names.pop_back();  // 移除'}'

            // 使用stringstream和getline来分割字符串
            std::stringstream ss(label_names);
            std::string item;
            while (std::getline(ss, item, ',')) {
                // 移除每个键值对中的冒号及其前面的部分
                size_t colonPos = item.find(':');
                if (colonPos != std::string::npos) {
                    item.erase(0, colonPos + 1); // 移除冒号及其前面的部分
                }

                // 移除可能存在的空格
                item.erase(std::remove(item.begin(), item.end(), ' '), item.end());

                // 检查是否有单引号
                if (item.front() == '\'' && item.back() == '\'') {
                    item.erase(0, 1); // 去除开头的单引号
                    item.erase(item.length() - 1, 1);   // 去除结尾的单引号
                }

                // 将处理后的值添加到vector中
                this->output_labels.push_back(item);
            }
            allocator.Free(data_keys[i]);
            break;
        }
    }
    allocator.Free(data_keys);

    if ((this->output_labels.size() + 5) == this->output_shape[2]) {
        return STATUS_PROCESS_OK;         // 模型信息获取成功
    }


    return STATUS_LABEL_INVALID;      // 模型标签读取失败
}
#endif


int DetectModel::loadOnnxModel(const char *onnxfile)
{
    int result = STATUS_PROCESS_OK;
#ifndef USE_MINGW_COMPILER
    result = parseOnnxModel(onnxfile);
    if (result != STATUS_PROCESS_OK) {
        return result;
    }
#endif

    this->net = cv::dnn::readNet(onnxfile);
    if (net.empty()) {
        return STATUS_FILE_INVALID;
    }
    int device_count = cv::cuda::getCudaEnabledDeviceCount();
    if (device_count >= 1){
        cudaEnableStatus = true;
        this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }else{
        cudaEnableStatus = false;
    }
    return result;
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
void DetectModel::draw_result(cv::Mat &image, std::string label, cv::Rect box)
{
    cv::rectangle(image, box, cv::Scalar(0, 0, 255), 2);
    int baseLine;
    cv::Size label_size = cv::getTextSize(label, 0.8, 0.8, 1, &baseLine);
    cv::Point tlc = cv::Point(box.x, box.y);
    cv::Point brc = cv::Point(box.x, box.y + label_size.height + baseLine);
    cv::putText(image, label, cv::Point(box.x, box.y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
}


cv::Mat DetectModel::post_process(cv::Mat &image, std::vector<cv::Mat> &outputs, std::vector<std::string> &output_labels)
{
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    float* data = (float*)outputs[0].data;

    const int dimensions = 5 + this->output_labels.size();  // 5+class
    const int rows = output_shape[1];		//(640/8)*(640/8)*3+(640/16)*(640/16)*3+(640/32)*(640/32)*3
    for (int i = 0; i < rows; ++i)
    {
        float confidence = data[4];
        if (confidence >= this->config.confidence_threshold)
        {
            float* classes_scores = data + 5;
            cv::Mat scores(1, output_labels.size(), CV_32FC1, classes_scores);
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
        std::string label = output_labels[class_ids[idx]] + ":" + cv::format("%.2f", confidences[idx]);
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


cv::Mat DetectModel::resizeImage(const cv::Mat &image, int maxWidth, int maxHeight)
{
    int height = image.rows;
    int width = image.cols;

    // 计算原始长宽比
    double ratio = static_cast<double>(width) / height;

    // 根据最大宽度和高度确定缩放后的尺寸
    if (width > maxWidth || height > maxHeight) {
        if (ratio > 1.0) {
            width = maxWidth;
            height = static_cast<int>(width / ratio);
        } else {
            height = maxHeight;
            width = static_cast<int>(height * ratio);
        }

        // 如果计算后的高度或宽度仍然超出限制，则取最大限制值
        if (height > maxHeight) {
            height = maxHeight;
            width = static_cast<int>(height * ratio);
        }
        if (width > maxWidth) {
            width = maxWidth;
            height = static_cast<int>(width / ratio);
        }
    }

    // 创建缩放后的图像矩阵
    cv::Mat resizedImage(height, width, image.type());

    // 使用cv::resize进行缩放
    cv::resize(image, resizedImage, resizedImage.size());

    return resizedImage;
}

void DetectModel::imageOutSizeConfig(QSize size)
{
    this->display_size = size;
}


QPixmap DetectModel::frameDetect(cv::Mat frame)
{
    cv::Mat blob;
    std::vector<cv::Mat> outputs;

    cv::Mat resizedImage = this->resizeImage(frame, this->display_size.width(), this->display_size.height());
    this->pre_process(resizedImage, blob);


    this->net.setInput(blob);
    this->net.forward(outputs, net.getUnconnectedOutLayersNames());

    cv::Mat result = this->post_process(resizedImage, outputs, this->output_labels);
    return cvMatToQPixmap(result);
}


QPixmap DetectModel::pictureDetect(const char *file)
{
    cv::Mat image = cv::imread(cv::String(file));
    QPixmap result = this->frameDetect(image);

    return result;
}


void DetectModel::pictureThreadDetect(const char *file)
{
    this->thread->picture_path = file;
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
    return true;
}


QPixmap DetectModel::readVideoFirstFrame()
{
    cv::Mat frame;
    this->capture->read(frame);
    cv::Mat resizedImage = this->resizeImage(frame, this->display_size.width(), this->display_size.height());
    this->setVideoStartFrame(0);

    return cvMatToQPixmap(resizedImage);
}


void DetectModel::setVideoStartFrame(unsigned long startPoint)
{
    long totalFrame = this->capture->get(cv::CAP_PROP_FRAME_COUNT);
    if (startPoint < totalFrame) {
        this->capture->set(cv::CAP_PROP_POS_FRAMES, startPoint);
    }
}

void DetectModel::startThreadDetect()
{
    this->thread->model = this;
    this->thread->start();
}

void DetectModel::pauseThreadDetect()
{
    this->thread->pause_flag = true;
}


/* class DetectThread */


void VideoDetectThread::run()
{
    cv::Mat frame;
    pause_flag = false;
    if (this->picture_path.empty() == false) {
        frame = cv::imread(cv::String(this->picture_path));
        QPixmap output = this->model->frameDetect(frame);
        emit reportProgress(output);
        this->picture_path.clear();
    } else {
        while (this->model->capture->read(frame)) {
            QPixmap output = this->model->frameDetect(frame);
            emit reportProgress(output);

            if (pause_flag == true) {
                break;
            }
        }
    }


    emit done();
}

