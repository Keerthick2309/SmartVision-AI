import streamlit as st

st.title("About SmartVision AI")

st.markdown("## SmartVision AI - Intelligent Multi-Class Object Recognition System")

st.write("""
SmartVision AI is a computer vision-based application that performs both 
image classification and object detection using deep learning models.

The system is designed to identify and classify multiple objects in images 
across 25 different categories using state-of-the-art CNN architectures 
and YOLOv8.
""")

st.markdown("## Key Features")
st.write("""
- Image Classification using VGG16, ResNet50, MobileNetV2, EfficientNetB0
- Object Detection using YOLOv8
- Multi-model comparison
- Real-time webcam detection
- Interactive Streamlit UI
""")

st.markdown("## Models Used")
st.write("""
- VGG16 (Transfer Learning)
- ResNet50
- MobileNetV2
- EfficientNetB0
- YOLOv8 (Object Detection)
""")

st.markdown("## Dataset")
st.write("""
- COCO Dataset (25 selected classes)
- 2,500 images (100 per class)
- Includes vehicles, animals, people, food, and indoor objects
""")

st.markdown("## Technologies Used")
st.write("""
- Python
- TensorFlow / Keras
- Ultralytics YOLOv8
- OpenCV
- Streamlit
""")

st.markdown("## Use Cases")
st.write("""
- Smart surveillance systems
- Retail product recognition
- Traffic monitoring
- Wildlife tracking
- Smart home automation
""")

st.markdown("## Developer")
st.write("""
- **Name:** Keerthick
- **Project:** SmartVision AI  
""")

st.markdown("---")
st.caption("© 2026 SmartVision AI Project")