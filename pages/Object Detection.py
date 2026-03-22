import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

st.set_page_config(layout="wide")
st.title("Object Detection (YOLOv8)")
st.write("Upload an image to detect multiple objects with bounding boxes")

@st.cache_resource
def load_yolo():
    return YOLO("models/best.pt")

model = load_yolo()

confidence_threshold = st.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05
)

uploaded_file = st.file_uploader("Choose an image")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image=image, caption="Uploaded Image", use_container_width=True)

    img_array = np.array(image)
    result = model(img_array)

    boxes = result[0].boxes
    names = model.names

    detected_object = []

    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = names[cls]

        if  conf >= confidence_threshold:
            detected_object.append((label, conf))

    annotated_img = result[0].plot()

    st.subheader("Detection Result")
    st.image(annotated_img, use_container_width=True)

    st.subheader("Detected Objects")
    if detected_object:
        for label, conf in detected_object:
            st.write(f"{label} ({conf*100:.2f}%)")
            st.progress(conf)
    else:
        st.warning("No objects detected above threshold")
