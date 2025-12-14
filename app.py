import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import requests


# Download model from Hugging Face if not exists

MODEL_FILENAME = "pneumonia_model.h5"
HF_URL = "https://huggingface.co/jahangi/Pneumonia_Detection_System/resolve/main/pneumonia_model.h5"

if not os.path.exists(MODEL_FILENAME):
    st.info("Downloading model from Hugging Face...")
    r = requests.get(HF_URL)
    with open(MODEL_FILENAME, "wb") as f:
        f.write(r.content)
    st.success("Model downloaded successfully!")


# Load Model

model = tf.keras.models.load_model(MODEL_FILENAME)


# Streamlit UI

st.title("Pneumonia Detection System (AI Powered)")

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    # Prepare image for model
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Detect button
    if st.button("Detect Pneumonia"):
        prediction = model.predict(img_array)[0][0]

        if prediction > 0.5:
            st.error(f"Pneumonia Detected | Confidence: {prediction*100:.2f}%")
        else:
            st.success(f"Normal Lungs | Confidence: {(1-prediction)*100:.2f}%")

