import cv2
import streamlit as st
from pathlib import Path
import tempfile
from ultralytics import YOLO
from PIL import Image
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

# Paths
MODEL_PATH = Path('weights/yolo11n.pt')

# Sidebar + Header
st.set_page_config(page_title="YOLO11 Object Detection", page_icon="🤖")
st.header("🧠 Advanced Object Detection using YOLOv11")
st.sidebar.header("🛠️ Model Configurations")

# Confidence
confidence_value = float(st.sidebar.slider("Confidence Threshold (%)", 25, 100, 40)) / 100

# Source Select
source_radio = st.sidebar.radio("Choose Input Source", ["Image", "Video", "Live Stream"])

# Load Model
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    st.error("🚨 Failed to load YOLO model.")
    st.error(e)
    st.stop()

# Tracker
tracker = DeepSort(max_age=50)

# === Image Upload ===
if source_radio == "Image":
    image_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if image_file:
        image = Image.open(image_file).convert("RGB")
        st.image(image, caption="📷 Uploaded Image", use_column_width=True)
        frame = np.array(image)

        try:
            result = model.predict(frame, conf=confidence_value)[0]
            plotted = result.plot()
            st.image(plotted, caption="🔍 Detection Result", channels="BGR", use_column_width=True)
        except Exception as e:
            st.error(f"Detection failed: {e}")

# === Video Upload ===
elif source_radio == "Video":
    video_file = st.sidebar.file_uploader("Upload a Video", type=["mp4", "avi", "mov", "mkv"])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            try:
                results = model.predict(frame, conf=confidence_value)
                plotted = results[0].plot()
                st_frame.image(plotted, channels="BGR", use_column_width=True)
            except Exception as e:
                st.error(f"Detection failed: {e}")
                break

        cap.release()

# === Live Stream (Webcam) ===
elif source_radio == "Live Stream":
    st.sidebar.write("⚠️ Live stream requires local run. Not supported on Streamlit Cloud.")
    if st.sidebar.button("Start Webcam"):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Cannot access webcam.")
        else:
            st.success("✅ Webcam started.")
            st_frame = st.empty()

            while True:
                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠️ Frame capture failed.")
                    break

                frame = cv2.resize(frame, (720, int(720 * 9 / 16)))
                try:
                    results = model.predict(frame, conf=confidence_value)
                    plotted = results[0].plot()
                    st_frame.image(plotted, channels="BGR", use_column_width=True)
                except Exception as e:
                    st.error(f"Live detection error: {e}")
                    break

            cap.release()
