# Import Required Libraries
import cv2
import streamlit as st
from pathlib import Path
import sys
from ultralytics import YOLO
from PIL import Image
import json
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

# Get the absolute path of the current file
FILE = Path(__file__).resolve()
ROOT = FILE.parent

# Add the root path to sys.path
if ROOT not in sys.path:
    sys.path.append(str(ROOT))

ROOT = ROOT.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'
VIDEO = 'Video'
LIVE = 'Live Stream'

SOURCES_LIST = [IMAGE, VIDEO, LIVE]

# Model Configurations
MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'yolo11n.pt'
SEGMENTATION_MODEL = MODEL_DIR / 'yolo11n-seg.pt'
POSE_ESTIMATION_MODEL = MODEL_DIR / 'yolo11n-pose.pt'

# Page Layout
st.set_page_config(
    page_title="YOLO11 Object Detection",
    page_icon="🤖"
)

# Header
st.header("Advanced Object Detection using YOLO11")

# Sidebar Configurations
st.sidebar.header("Model Configurations")

# Choose Model: Detection, Segmentation, or Pose Estimation
model_type = st.sidebar.radio("Task", ["Detection", "Segmentation", "Pose Estimation"])

# Select Confidence Value
confidence_value = float(st.sidebar.slider("Select Model Confidence Value", 25, 100, 40)) / 100

# Selecting Detection, Segmentation, Pose Estimation Model
if model_type == 'Detection':
    model_path = Path(DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(SEGMENTATION_MODEL)
elif model_type == 'Pose Estimation':
    model_path = Path(POSE_ESTIMATION_MODEL)

# Load YOLO Model
try:
    model = YOLO(model_path)
except Exception as e:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(e)

# Image / Video Configuration
st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio("Select Source", SOURCES_LIST)

# Initialize Object Tracker
tracker = DeepSort(max_age=50)

if source_radio == IMAGE:
    source_image = st.sidebar.file_uploader("Choose an Image...", type=("jpg", "png", "jpeg", "bmp", "webp"))

    if source_image is not None:
        uploaded_image = Image.open(source_image)
        st.image(source_image, caption="Uploaded Image", use_container_width=True)

        if st.sidebar.button("Detect Objects"):
            result = model.predict(uploaded_image, conf=confidence_value)
            result_plotted = result[0].plot()[:, :, ::-1]
            st.image(result_plotted, caption="Detected Image", use_container_width=True)

            detection_data = {"objects": []}
            for box in result[0].boxes:
                detection_data["objects"].append({
                    "class": int(box.cls),
                    "confidence": float(box.conf),
                    "bbox": box.xyxy.tolist()
                })

            with open("detection_results.json", "w") as f:
                json.dump(detection_data, f)

            with st.expander("Detection Results"):
                for box in result[0].boxes:
                    st.write(box.data)

elif source_radio == VIDEO:
    source_video = st.sidebar.file_uploader("Upload a Video...", type=("mp4", "avi", "mov", "mkv"))

    if source_video is not None:
        st.video(source_video)

        if st.sidebar.button("Detect Video Objects"):
            try:
                video_cap = cv2.VideoCapture(source_video.name)
                st_frame = st.empty()
                while video_cap.isOpened():
                    success, frame = video_cap.read()
                    if success:
                        frame = cv2.resize(frame, (720, int(720 * (9 / 16))))
                        result = model.predict(frame, conf=confidence_value)
                        result_plotted = result[0].plot()

                        bboxes = result[0].boxes.xyxy.cpu().numpy()
                        confidences = result[0].boxes.conf.cpu().numpy()

                        detections = [(bbox, conf) for bbox, conf in zip(bboxes, confidences)]
                        tracks = tracker.update_tracks(detections, frame=frame)

                        for track in tracks:
                            bbox = track.to_tlbr()
                            cv2.rectangle(result_plotted, (int(bbox[0]), int(bbox[1])),
                                          (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)

                        st_frame.image(result_plotted, caption="Detected Video", channels="BGR", use_container_width=True)
                    else:
                        video_cap.release()
                        break
            except Exception as e:
                st.sidebar.error("Error Processing Video: " + str(e))

elif source_radio == LIVE:
    st.sidebar.write("Using webcam for live detection.")
    if st.sidebar.button("Start Live Detection"):
        video_cap = cv2.VideoCapture(0)
        st_frame = st.empty()
        while video_cap.isOpened():
            success, frame = video_cap.read()
            if success:
                result = model.predict(frame, conf=confidence_value)
                result_plotted = result[0].plot()

                st_frame.image(result_plotted, caption="Live Detection", channels="BGR", use_container_width=True)
            else:
                video_cap.release()
                break