import streamlit as st
import cv2
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

st.title("Live Webcam Object Detection (YOLOv8)")

@st.cache_resource
def load_model():
    return YOLO("models/best.pt")

model = load_model()

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
        ]
    }
)

class VideoProcessor(VideoProcessorBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img)
        annotated_frame = results[0].plot()
        return annotated_frame

st.write("Click start to begin webcam detection")

webrtc_streamer(
    key="yolo-webcam",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration=RTC_CONFIGURATION,
    async_processing=True
)