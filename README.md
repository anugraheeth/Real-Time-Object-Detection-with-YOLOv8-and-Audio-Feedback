# Real-time Object Detection with Audio Feedback (GPU ACCELERATED USING CUDA)

## Overview
This Python program performs real-time object detection using a webcam feed, leveraging the YOLOv8 model. It provides audio feedback for detected objects and offers directional guidance to avoid obstacles.

## Features
- Real-time object detection using YOLOv8
- Audio feedback for detected objects
- Directional guidance to avoid obstacles
- CUDA support for GPU acceleration

## Requirements
- Python 3.6+
- CUDA-capable GPU (optional, for improved performance)

## Dependencies
- OpenCV (cv2)
- Ultralytics YOLO
- PyTorch
- pyttsx3

## Installation

1. Clone this repository or download the script.

2. Install the required dependencies:
   ```
   pip install opencv-python ultralytics torch pyttsx3
   ```

3. Download the YOLOv8 model weights:
   - The script uses `yolov8n.pt` by default. You can download it from the [Ultralytics YOLO repository](https://github.com/ultralytics/yolov8).
   - Place the model file in the same directory as the script.

## CUDA Setup (for GPU acceleration)

To enable CUDA for GPU acceleration:

1. Ensure you have a CUDA-capable GPU.
2. Install the CUDA Toolkit from the [NVIDIA website](https://developer.nvidia.com/cuda-downloads).
3. Install the cuDNN library from the [NVIDIA Developer website](https://developer.nvidia.com/cudnn).
4. Install the CUDA-enabled version of PyTorch:
   ```
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
   (Replace `cu118` with your CUDA version if different)

The script will automatically use CUDA if available.

Want to know more of CUDA installation visit ---[My CUDA installation guide](https://github.com/anugraheeth/OpenCV-with-CUDA-Accelerating-Deep-Learning-on-GPU-)

## Usage

Run the script using Python:

```
python object-detection.py
```

- The program will access your default webcam and start detecting objects in real-time.
- Detected objects will be announced via audio feedback.
- Directional guidance will be provided to avoid obstacles.
- Press 'q' to quit the program.

## Customization

- Adjust the `confidence_threshold` variable to change the detection sensitivity.
- Modify the `speech_interval` to change how often audio feedback is provided.
- Change the YOLOv8 model by replacing `yolov8n.pt` with other variants like `yolov8s.pt` or `yolov8m.pt` for different performance/accuracy trade-offs.

## How It Works

1. The script initializes the YOLO model and the text-to-speech engine.
2. It captures frames from the webcam in real-time.
3. Each frame is processed by the YOLO model for object detection.
4. Detected objects are announced via audio, with a cooldown period between announcements.
5. The program analyzes the position of detected objects and provides directional guidance to avoid obstacles.

## Limitations

- Audio feedback may overlap if many objects are detected in quick succession.
- The accuracy of object detection depends on the chosen YOLO model and the `confidence_threshold`.
- Directional guidance is basic and may not account for complex environments.

## Contributing

Feel free to fork this project and submit pull requests with improvements or bug fixes.

