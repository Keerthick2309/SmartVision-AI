import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json

st.title("Image Classification")
st.write("Upload an image to classify using CNN models")
classes = [
    "airplane", "bed", "bench", "bicycle", "bird", "bottle",
    "bowl", "bus", "cake", "car", "cat", "chair", "couch",
    "cow", "cup", "dog", "elephant", "horse", "motorcycle",
    "person", "pizza", "potted plant", "stop sign",
    "traffic light", "truck"
]

st.write(", ".join(classes))

@st.cache_resource
def load_models():
    models = {
        "VGG16": tf.keras.models.load_model("models/VGG16_best.h5"),
        "ResNet50": tf.keras.models.load_model("models/resnet50_best.h5"),
        "MobileNetV2": tf.keras.models.load_model("models/mobilenet_model.h5"),
        "EfficientNetB0": tf.keras.models.load_model("models/efficientnet_best.h5")
    }
    return models

models = load_models()
with open("models/class_indices.json") as f:
    class_indices = json.load(f)

class_names = {v: k for k, v in class_indices.items()}

def predict(image, model, model_name):
    img = image.resize((224, 224))
    img_array = np.expand_dims(img, axis=0)
    if model_name == "ResNet50":
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    elif model_name == "EfficientNetB0":
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    elif model_name == "VGG16":
        img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
    else:
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    preds = model.predict(img_array)
    class_idx = np.argmax(preds)
    confidence = np.max(preds)

    return class_names[class_idx], confidence

uploaded_file = st.file_uploader("Choose an image")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.subheader("🔍 Model Predictions")
    for model_name, model in models.items():
        label, conf = predict(image, model, model_name)

        st.markdown(f"### {model_name}")
        st.write(f"Prediction: **{label}**")
        st.progress(float(conf))
        st.write(f"Confidence: {conf*100:.2f}%")

        st.divider()