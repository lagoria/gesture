# Gesture

A image hand gesture number detection of Qt application

# Environment

- Windows 10 x64
- Qt5.15.2
- OpenCV >= 4.7.0
- ONNX Runtime 1.17.3
- VC16 or MinGW 8.10

# Feature

- supported YOLOv5  and YOLOv8 export onnx detection model

- surported picture , video (.mp4, .avi) and camera

- gesture model characteristic:

| class | name  |
|:-----:|:-----:|
| 0     | fist  |
| 1     | one   |
| 2     | two   |
| 3     | three |
| 4     | five  |
| 5     | four  |

## Attention

when OpenCV >= 4.8.0 the YOLOv8 onnx model inference  boxs always 0 use the CUDA. And OpenCV == 4.7.0, the YOLOv5 onnx model inference ineffectiveness use the CPU