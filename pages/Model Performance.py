import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📊 Model Performance Comparison")

st.write("Comparison of different CNN models used for image classification")

data = {
    "Model": ["VGG16", "ResNet50", "MobileNetV2", "EfficientNetB0"],
    "Accuracy": [0.74, 0.85, 0.82, 0.88],
    "Inference Time (ms)": [150, 100, 50, 80]
}

df = pd.DataFrame(data)

st.subheader("📋 Performance Table")
st.dataframe(df, use_container_width=True)

st.subheader("📈 Accuracy Comparison")

fig, ax = plt.subplots()
ax.bar(df["Model"], df["Accuracy"])
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1)

st.pyplot(fig)

st.subheader("⚡ Inference Speed Comparison")

fig2, ax2 = plt.subplots()
ax2.bar(df["Model"], df["Inference Time (ms)"])
ax2.set_ylabel("Time (ms)")

st.pyplot(fig2)