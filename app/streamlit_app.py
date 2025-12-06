import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(layout="centered")


# ------------------------------
# Load model (cached for speed)
# ------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/waste_model.keras")
    return model


model = load_model()


# ------------------------------
# Preprocessing function
# ------------------------------
def preprocess_image(image: Image.Image):
    image = image.resize((150, 150))  # adjust this if your model uses another size
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
with st.container(border=True):
    st.markdown("### ♻️ Waste Classification AI")
    st.write("Upload an image to classify it as Organic or Recyclable.")


# Track uploaded file for "Clear" button
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    key=f"file_uploader_{st.session_state.uploader_key}",
)

if uploaded_file:
    st.session_state.uploaded_file = uploaded_file

# Clear button
if st.session_state.get("uploaded_file"):
    if st.button("Clear Image", use_container_width=True):
        st.session_state.uploaded_file = None
        st.session_state.uploader_key += 1  # 🔥 reset widget
        st.rerun()

# If image exists, display + predict
if st.session_state.uploaded_file:
    image = Image.open(st.session_state.uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.divider()

    # Predict automatically (I can't validate the difference, only in production I guess)
    with st.spinner("Analyzing image..."):
        label, confidence = predict(image)

    st.markdown("### Prediction")

    if label == "Recyclable":
        st.markdown(
            f"""
        <div style="padding:10px;border-radius:10px;background:#1E90FF">
            <b>♻️ Recyclable</b><br>Confidence: {confidence:.2f}
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
        <div style="padding:10px;border-radius:10px;background:#6B8E23">
            <b>🌱 Organic</b><br>Confidence: {confidence:.2f}
        </div>
        """,
            unsafe_allow_html=True,
        )

    # 🔥 Confidence bar
    st.write("Confidence level:")
    st.progress(confidence)

    # 🔥 Full probability distribution
    st.markdown("### Probability Distribution")

    if label == "Recyclable":
        st.write(f"**Recyclable:** {confidence:.2f}")
        st.write(f"**Organic:** {1 - confidence:.2f}")
    else:
        st.write(f"**Organic:** {confidence:.2f}")
        st.write(f"**Recyclable:** {1 - confidence:.2f}")

# Example images
col1, col2 = st.columns(2)

with col1:
    if st.button("Try Recyclable Example", key="ex_r", use_container_width=True):
        st.session_state.uploaded_file = "notebooks/o-vs-r-split/test/R/R_71.jpg"
        st.session_state.uploader_key += 1
        st.rerun()

with col2:
    if st.button("Try Organic Example", key="ex_o", use_container_width=True):
        st.session_state.uploaded_file = "notebooks/o-vs-r-split/test/O/O_1.jpg"
        st.session_state.uploader_key += 1
        st.rerun()


st.markdown(
    "<p style='text-align:center; color: grey;'>Made with ❤️ by Lampros</p>",
    unsafe_allow_html=True,
)
