# Advanced Object Detection using YOLO11

A Streamlit-based application for real-time, video, and image object detection using **YOLOv11** (Ultralytics) combined with **DeepSORT** tracking.
This project enables accurate detection, tracking, and visualization of objects from images, uploaded videos, and webcam streams.

---

## Features

* **YOLOv11 Inference**

  * High-speed, high-accuracy object detection.
* **DeepSORT Tracking**

  * Real-time multi-object tracking with unique IDs.
* **Streamlit GUI**

  * Simple and intuitive interface to run detections.
* **Supports Multiple Inputs**

  * Image upload
  * Video upload
  * Live webcam stream (local execution)
* **Configurable Confidence Threshold**

  * Adjustable detection sensitivity.

---

## Project Structure

```
Advanced-Object-Detection-using-YOLO11/
│
├── weights/
│   └── yolo11n.pt                 # YOLO model file
│
├── app.py                         # Main Streamlit application
├── requirements.txt               # Python dependencies
├── packages.txt                   # System packages (for Streamlit Cloud)
└── README.md                      # Project documentation
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/FouaadAI/Advanced-Object-Detection-using-YOLO11.git
cd Advanced-Object-Detection-using-YOLO11
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install System Packages (If Required)

For Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install libgl1 libglib2.0-0
```

---

## Usage

### Start the Application

```bash
streamlit run app.py
```

### Input Options

* **Image Mode:** Upload an image and view detected objects.
* **Video Mode:** Upload a video (mp4, avi, mov, mkv) and process it frame-by-frame.
* **Live Stream:** Use your webcam for real-time detection (local run only).

---

## Tech Stack

| Component                 | Description                             |
| ------------------------- | --------------------------------------- |
| **YOLOv11 (Ultralytics)** | State-of-the-art object detection model |
| **DeepSORT**              | Object tracking with unique IDs         |
| **OpenCV**                | Video processing and frame reading      |
| **Streamlit**             | Web interface for the application       |
| **NumPy / PIL**           | Array and image manipulation            |

---

## Requirements

### Python Dependencies (requirements.txt)

```
streamlit>=1.23
opencv-python-headless>=4.8.1
ultralytics>=8.3.0
numpy<2
Pillow>=9.0
json5>=0.9.6
deep_sort_realtime>=0.2.0
torch>=2.0
torchvision>=0.15
```

### System Packages (packages.txt)

```
libgl1
libglib2.0-0
```

---

## Model

Download the YOLOv11 weights and place them in the `/weights` directory:

```
weights/yolo11n.pt
```

You may use different YOLOv11 variants depending on accuracy/speed requirements.

App_Link : [lets try it](https://fouad-ai-advanced-object-detection-using-yolo11.streamlit.app/)

---


## License

This project is released under the MIT License.
You are free to modify and distribute it.
