import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# ------------------------------
# Load model (cached for speed)
# ------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/waste_model.h5")
    return model


model = load_model()


# ------------------------------
# Preprocessing function
# ------------------------------
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))  # adjust this if your model uses another size
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ------------------------------
# Prediction function
# ------------------------------
def predict(image: Image.Image):
    processed = preprocess_image(image)
    preds = model.predict(processed)[0]

    # Binary classification → sigmoid
    prob = float(preds[0])

    if prob >= 0.5:
        return "Recyclable", prob
    else:
        return "Organic", 1 - prob


# ------------------------------
# Streamlit UI
# ------------------------------
st.title("♻️ Waste Classification AI")
st.write("Upload an image of waste to classify it as **Organic** or **Recyclable**.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    if st.button("Predict"):
        label, confidence = predict(image)

        st.subheader("Prediction")
        st.write(f"**Class:** {label}")
        st.write(f"**Confidence:** {confidence:.2f}")
