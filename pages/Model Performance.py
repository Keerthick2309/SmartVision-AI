import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Model Performance Comparison")

st.write("Comparison of different CNN models used for image classification")

data = {
    "Model": ["VGG16", "ResNet50", "MobileNetV2", "EfficientNetB0"],
    "Accuracy": [0.80, 0.81, 0.82, 0.83]
}

df = pd.DataFrame(data)

st.subheader("Performance Table")
st.dataframe(df, use_container_width=True)

st.subheader("Accuracy Comparison")

fig, ax = plt.subplots()
ax.bar(df["Model"], df["Accuracy"])
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1)

st.pyplot(fig)