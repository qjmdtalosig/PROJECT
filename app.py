import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("🐴🦌 Horse vs Deer Classifier")

model = tf.keras.models.load_model("horse_deer_model.h5")

CLASS_NAMES = ["Horse", "Deer"]

file = st.file_uploader("Upload a Horse or Deer image", type=["jpg", "png", "jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, width=300)

    img_resized = img.resize((96, 96))
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

    pred = model.predict(img_array)[0][0]

    if pred < 0.5:
        label = "Horse"
        confidence = (1 - pred)
    else:
        label = "Deer"
        confidence = pred

    st.write(f"### Prediction: **{label}**")
    st.write(f"Confidence: **{confidence*100:.2f}%**")
