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

# Page Layout
st.set_page_config(
    page_title="YOLO11 Object Detection",
    page_icon="🤖"
)

# Header
st.header("Advanced Object Detection using YOLO11")

# Sidebar Configurations
st.sidebar.header("Model Configurations")

# Select Confidence Value
confidence_value = float(st.sidebar.slider("Select Model Confidence Value", 25, 100, 40)) / 100

# Load YOLO Model
try:
    model = YOLO(DETECTION_MODEL)
except Exception as e:
    st.error(f"Unable to load model. Check the specified path: {DETECTION_MODEL}")
    st.error(e)

# Image / Video Configuration
st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio("Select Source", SOURCES_LIST)

# Initialize Object Tracker
tracker = DeepSort(max_age=50)

if source_radio == LIVE:
    st.sidebar.write("Using webcam for live detection.")

    if st.sidebar.button("Start Live Detection"):
        # Try different camera indexes (0 or 1)
        cam_index = 0  
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)  # Improved compatibility for Windows

        if not cap.isOpened():
            st.error("⚠️ Unable to access webcam. Retrying with index 1...")
            cap = cv2.VideoCapture(1)

        if not cap.isOpened():
            st.error("❌ Failed to open webcam. Try checking device settings.")
        else:
            st.success("✅ Webcam accessed successfully!")

            # Stream frames
            st_frame = st.empty()
            while True:
                success, frame = cap.read()

                if not success:
                    st.error("⚠️ Failed to retrieve frame. Restart webcam.")
                    break

                frame = cv2.resize(frame, (720, int(720 * (9 / 16))))  # Resize for consistency

                try:
                    result = model.predict(frame, conf=confidence_value)
                    result_plotted = result[0].plot()
                    st_frame.image(result_plotted, caption="Live Detection", channels="BGR", use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Error during detection: {e}")
                    break

            cap.release()
