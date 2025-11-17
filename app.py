import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

st.title("🐴🦌 Horse vs Deer Classifier (TFLite Version)")

interpreter = tflite.Interpreter(model_path="horse_deer_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

file = st.file_uploader("Upload a Horse or Deer image", type=["jpg", "png", "jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, width=300)

    img_resized = img.resize((96, 96))
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    pred = interpreter.get_tensor(output_details[0]['index'])[0][0]

    if pred < 0.5:
        label = "Horse"
        confidence = (1 - pred)
    else:
        label = "Deer"
        confidence = pred

    st.write(f"### Prediction: **{label}**")
    st.write(f"Confidence: **{confidence*100:.2f}%**")
