#include "inferenceEngine.h"


InferenceEngine::InferenceEngine()
{
    // default param settings
    this->config.input_width = 640;
    this->config.input_height = 640;
    this->config.score_threshold = 0.5;
    this->config.nms_threshold = 0.45;
    this->config.confidence_threshold = 0.45;
    this->config.enable_cuda = true;

    cudaEnableStatus = false;

    this->display_size = QSize(640, 480);

    this->thread = new InferenceThread();
    this->capture = new cv::VideoCapture();
}

InferenceEngine::~InferenceEngine()
{
    capture->release();
    delete capture;
    delete thread;
}

void InferenceEngine::thresholdConfig(modelConfig_t &conf)
{
    this->config = conf;
}

#ifdef USE_ONNXRUNTIME_LIB
int InferenceEngine::parseOnnxModel(const std::string &onnxfile)
{
    QString fileNamePath = QString::fromStdString(onnxfile);
    const wchar_t* model_path = reinterpret_cast<const wchar_t *>(fileNamePath.utf16());

    Ort::Env env;                                   // 创建env
    Ort::Session session(nullptr);                  // 创建一个空会话
    Ort::SessionOptions sessionOptions(nullptr);    // 创建会话配置
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
    Ort::AllocatedStringPtr names_ptr = model_metadata.LookupCustomMetadataMapAllocated("names", allocator);
    if (names_ptr != nullptr) {
        this->classes.clear();
        std::string label_names = names_ptr.get();
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
            this->classes.push_back(item);
        }
    }


    int result;
    if ((this->classes.size() + 5) == this->output_shape[2]) {
        result = STATUS_PROCESS_OK;         // YOLOv5模型
        this->model_type = MODEL_TYPE_YOLOV5;
    } else if ((this->classes.size() + 4) == this->output_shape[1]) {
        result = STATUS_PROCESS_OK;         // YOLOv8模型
        this->model_type = MODEL_TYPE_YOLOV8;
    } else {
        result = STATUS_LABEL_INVALID;      // 模型标签读取失败
        this->model_type = MODEL_TYPE_UNKNOWN;
    }

    return result;
}
#endif


int InferenceEngine::loadOnnxModel(const std::string &onnxfile)
{
    int result = STATUS_PROCESS_OK;
#ifdef USE_ONNXRUNTIME_LIB
    result = parseOnnxModel(onnxfile);
    if (result != STATUS_PROCESS_OK) {
        return result;
    }
#endif

    this->net = cv::dnn::readNet(onnxfile);
    if (net.empty()) {
        return STATUS_FILE_INVALID;
    }

    /**
     * OpenCV >= 4.8.0 时，使用Cuda推理YOLOv8模型会出现检测框为0的问题，这里设置为CPU推理。
     **/
    if (this->config.enable_cuda && this->model_type != MODEL_TYPE_YOLOV8 && cv::cuda::getCudaEnabledDeviceCount() >= 1) {
        cudaEnableStatus = true;
        this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    } else {
        cudaEnableStatus = false;
        this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    return result;
}



std::vector<InferenceEngine::DetectResult> InferenceEngine::runInference(const cv::Mat &input)
{
    cv::Mat modelInput = input;

    // 信封缩放
    modelInput = this->scaleToLetterBox(modelInput, cv::Size(this->config.input_width, this->config.input_height), cv::Scalar(114, 114, 114));

    cv::Mat blob;
    // 缩放裁切
    cv::dnn::blobFromImage(modelInput, blob, 1.0/255.0, cv::Size(this->config.input_width, this->config.input_height), cv::Scalar(), true, false);
    this->net.setInput(blob);

    std::vector<cv::Mat> outputs;
    this->net.forward(outputs, net.getUnconnectedOutLayersNames());

    int rows = outputs[0].size[1];
    int dimensions = outputs[0].size[2];

    bool yolov8 = false;
    // yolov5 has an output of shape (batchSize, 25200, 85) (Num classes + box[x,y,w,h] + confidence[c])
    // yolov8 has an output of shape (batchSize, 84,  8400) (Num classes + box[x,y,w,h])
    if (dimensions > rows) // Check if the shape[2] is more than shape[1] (yolov8)
    {
        yolov8 = true;
        rows = outputs[0].size[2];
        dimensions = outputs[0].size[1];

        outputs[0] = outputs[0].reshape(1, dimensions);
        cv::transpose(outputs[0], outputs[0]);
    }
    float *data = (float *)outputs[0].data;

    float x_factor = modelInput.cols / this->config.input_width;
    float y_factor = modelInput.rows / this->config.input_height;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i)
    {
        if (yolov8)
        {
            float *classes_scores = data + 4;

            cv::Mat scores(1, classes.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double maxClassScore;

            cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

            if (maxClassScore > this->config.score_threshold)
            {
                confidences.push_back(maxClassScore);
                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);

                int width = int(w * x_factor);
                int height = int(h * y_factor);

                cv::Rect box = cv::Rect(left, top, width, height);
                box = this->scaleBoxToSource(box, input.size());
                boxes.push_back(box);
            }
        }
        else // yolov5
        {
            float confidence = data[4];

            if (confidence >= this->config.confidence_threshold)
            {
                float *classes_scores = data + 5;

                cv::Mat scores(1, classes.size(), CV_32FC1, classes_scores);
                cv::Point class_id;
                double max_class_score;

                cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

                if (max_class_score > this->config.score_threshold)
                {
                    confidences.push_back(confidence);
                    class_ids.push_back(class_id.x);

                    float x = data[0];
                    float y = data[1];
                    float w = data[2];
                    float h = data[3];

                    int left = int((x - 0.5 * w) * x_factor);
                    int top = int((y - 0.5 * h) * y_factor);

                    int width = int(w * x_factor);
                    int height = int(h * y_factor);

                    cv::Rect box = cv::Rect(left, top, width, height);
                    box = this->scaleBoxToSource(box, input.size());
                    boxes.push_back(box);
                }
            }
        }

        data += dimensions;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, this->config.score_threshold, this->config.nms_threshold, nms_result);

    std::vector<InferenceEngine::DetectResult> detections{};
    for (unsigned long i = 0; i < nms_result.size(); ++i)
    {
        int idx = nms_result[i];

        DetectResult result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.className = classes[result.class_id];
        result.box = boxes[idx];

        detections.push_back(result);
    }

    return detections;
}


cv::Mat InferenceEngine::scaleToLetterBox(const cv::Mat& image, const cv::Size& newShape, const cv::Scalar& color)
{
    cv::Mat outImage;
    cv::Size shape = image.size();
    float r = std::min((float)newShape.height / (float)shape.height, (float)newShape.width / (float)shape.width);

    int new_un_pad[2] = { (int)std::round((float)shape.width * r),(int)std::round((float)shape.height * r) };

    auto dw = (float)(newShape.width - new_un_pad[0]);
    auto dh = (float)(newShape.height - new_un_pad[1]);

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

    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);

    return outImage;
}


cv::Rect InferenceEngine::scaleBoxToSource(const cv::Rect &box, cv::Size source_size)
{
    cv::Rect output_box;
    float gain = std::min(this->config.input_width * 1.0 / source_size.width, this->config.input_height * 1.0 / source_size.height);
    int pad_w = (this->config.input_width - source_size.width * gain) / 2;
    int pad_h = (this->config.input_height - source_size.height * gain) / 2;
    output_box.x = (box.x - pad_w) / gain;
    output_box.y = (box.y - pad_h) / gain;
    output_box.width = box.width / gain;
    output_box.height = box.height / gain;

    return output_box;
}


cv::Mat InferenceEngine::drawDetectResult(cv::Mat &image, std::vector<DetectResult> detect_result)
{
    cv::Mat output = image;
    int detections = detect_result.size();

    for (int i = 0; i < detections; ++i)
    {
        DetectResult detection = detect_result[i];

        cv::Rect box = detection.box;

        cv::Scalar color;

        if (this->label_color_map.size() - 1 < detection.class_id) {
            // box color (b, g, r)
            color = cv::Scalar(128, 128, 128);
        } else {
            color = label_color_map[detection.class_id];
        }


        // Detection box
        cv::rectangle(output, box, color, 2);

        // Detection box text
        std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
        cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
        cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

        cv::rectangle(output, textBox, color, cv::FILLED);
        cv::putText(output, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
    }
    return output;
}


QPixmap InferenceEngine::cvMatToQPixmap(const cv::Mat &cvMat)
{
    // 将 BGR 转换为 RGB
    cv::Mat rgbMat;
    cv::cvtColor(cvMat, rgbMat, cv::COLOR_BGR2RGB);

    // 将 OpenCV Mat 转换为 QImage
    QImage img((const uchar*) rgbMat.data, rgbMat.cols, rgbMat.rows, rgbMat.step, QImage::Format_RGB888);

    // 将 QImage 转换为 QPixmap
    return QPixmap::fromImage(img);
}


cv::Mat InferenceEngine::resizeImage(const cv::Mat &image, int maxWidth, int maxHeight)
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

void InferenceEngine::imageOutSizeConfig(QSize size)
{
    this->display_size = size;
}


QPixmap InferenceEngine::frameDetect(cv::Mat frame)
{
    cv::Mat resizedImage = this->resizeImage(frame, this->display_size.width(), this->display_size.height());
    std::vector<DetectResult> detections = this->runInference(resizedImage);

    cv::Mat output = this->drawDetectResult(resizedImage, detections);
    return cvMatToQPixmap(output);
}


QPixmap InferenceEngine::pictureDetect(const std::string &file)
{
    cv::Mat image = cv::imread(file);
    QPixmap result = this->frameDetect(image);

    return result;
}


void InferenceEngine::pictureThreadDetect(const std::string &file)
{
    this->thread->picture_path = file;
}

bool InferenceEngine::openVideo(const std::string &file)
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

bool InferenceEngine::openDefaultCamera()
{
    if (this->capture->isOpened()) {
        this->capture->release();
    }
    // open default camera
    this->capture->open(0);

    return this->capture->isOpened();
}

int InferenceEngine::readCameraVideoStream()
{
    if (this->capture->isOpened()) {
        this->thread->video_stream_flag = true;
        this->thread->model = this;
        this->thread->start();
        return 0;
    } else {
        return -1;
    }
}

int InferenceEngine::stopCameraVideoStream()
{
    if (this->thread->setup_detect_flag == true) {
        return -1;
    }

    this->thread->pause_thread_flag = true;
    return 0;
}

QPixmap InferenceEngine::readVideoFirstFrame()
{
    cv::Mat frame;
    this->capture->read(frame);
    cv::Mat resizedImage = this->resizeImage(frame, this->display_size.width(), this->display_size.height());
    this->setVideoStartFrame(0);

    return cvMatToQPixmap(resizedImage);
}


void InferenceEngine::setVideoStartFrame(unsigned long startPoint)
{
    unsigned long totalFrame = this->capture->get(cv::CAP_PROP_FRAME_COUNT);
    if (startPoint < totalFrame) {
        this->capture->set(cv::CAP_PROP_POS_FRAMES, startPoint);
    }
}

void InferenceEngine::startThreadDetect()
{
    this->thread->setup_detect_flag = true;
    if (this->thread->isRunning() == false) {
        this->thread->model = this;
        this->thread->start();
    }
}

void InferenceEngine::pauseThreadDetect()
{
    if (this->thread->video_stream_flag == true) {
        this->thread->setup_detect_flag = false;
    } else {
        this->thread->pause_thread_flag = true;
    }
}


/* class DetectThread */


void InferenceThread::run()
{
    cv::Mat frame;
    QPixmap output;
    pause_thread_flag = false;
    if (this->picture_path.empty() == false) {
        frame = cv::imread(this->picture_path);
        output = this->model->frameDetect(frame);
        emit reportProgress(output);
        this->picture_path.clear();
    } else {
        while (this->model->capture->read(frame)) {
            if (setup_detect_flag == true) {
                output = this->model->frameDetect(frame);
            } else {
                output = this->model->cvMatToQPixmap(frame);
            }
            emit reportProgress(output);

            if (pause_thread_flag == true) {
                if (video_stream_flag == true) {
                    this->model->capture->release();
                }
                break;
            }
        }
    }

    this->video_stream_flag = false;
    this->setup_detect_flag = false;

    emit done();

}

