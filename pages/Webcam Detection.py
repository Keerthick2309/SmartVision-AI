import os
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

st.set_page_config(page_title="YOLO Detection", layout="wide")

st.title("🔍 YOLOv8 Object Detection")

@st.cache_resource
def load_model():
    return YOLO("models/best.pt")

model = load_model()
class VideoProcessor(VideoProcessorBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img)
        annotated_frame = results[0].plot()
        return annotated_frame

option = st.radio(
    "Choose Input Method:",
    ["📷 Webcam (may not work on HF)", "🖼️ Upload Image"]
)

if option == "📷 Webcam (may not work on HF)":
    st.warning("⚠️ Webcam may not work properly on Hugging Face Spaces")

    try:
        webrtc_streamer(
            key="yolo-webcam",
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )
    except Exception as e:
        st.error(f"Webcam failed: {e}")
else:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        st.image(img, caption="Uploaded Image", channels="BGR")

        with st.spinner("Detecting..."):
            results = model(img)
            output = results[0].plot()

        st.image(output, caption="Detection Result", channels="BGR")

st.markdown("---")
st.markdown("Built with YOLOv8 + Streamlit")