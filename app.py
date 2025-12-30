import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# -----------------------------
# Load the model and class names
# -----------------------------
model = tf.keras.models.load_model("models/alzheimer_cnn.keras", compile=False)

with open("models/class_names.json", "r") as f:
    class_names = json.load(f)

# -----------------------------
# Sidebar info
# -----------------------------
st.sidebar.title("Alzheimer Stages Info")
st.sidebar.markdown("""
1. **Mild Demented:** Early signs of memory loss, slight confusion.
2. **Moderate Demented:** Increased memory loss, difficulty performing complex tasks.
3. **Non Demented:** Normal cognitive function, no noticeable memory issues.
4. **Very Mild Demented:** Subtle memory lapses, mostly normal daily life.
""")

# -----------------------------
# App title
# -----------------------------
st.title("🧠 Alzheimer Stage Prediction")
st.write("Upload a brain MRI image and the model will predict the Alzheimer stage.")

# -----------------------------
# Image upload
# -----------------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")  # ensure 3 channels
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # -----------------------------
    # Preprocess image
    # -----------------------------
    img_size = (224, 224)  # Must match your trained model input
    img = image.resize(img_size)  # Resize image
    img_array = np.array(img)  # Shape: (224, 224, 3)
    img_array = img_array / 255.0  # Normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)
    
    # -----------------------------
    # Predict
    # -----------------------------
    predictions = model.predict(img_array)
    pred_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    
    # -----------------------------
    # Display results
    # -----------------------------
    st.subheader("Prediction Result")
    st.write(f"**Stage:** {pred_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")
    
    st.subheader("Prediction Probabilities")
    for i, cname in enumerate(class_names):
        st.write(f"{cname}: {predictions[0][i]*100:.2f}%")
