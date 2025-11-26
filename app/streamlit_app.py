import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.express as px

# -----------------------------------------------------
# Page config
# -----------------------------------------------------
st.set_page_config(page_title="Waste Classifier", page_icon="♻️", layout="centered")

# -----------------------------------------------------
# Header
# -----------------------------------------------------
st.markdown(
    "<h1 style='text-align: center;'>♻️ Waste Classifier</h1>", unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Classify waste as Organic or Recyclable</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# -----------------------------------------------------
# Sidebar
# -----------------------------------------------------
st.sidebar.title("About this app")
st.sidebar.info(
    """
    Upload an image of waste, and the model will classify it as **Organic** or **Recyclable**.

    **Model:** VGG16 Transfer Learning  
    **Creator:** Lampros Velentzas
    """
)

st.sidebar.markdown("---")
st.sidebar.caption("Micro-SaaS Experiment • Built with Streamlit + TensorFlow")


# -----------------------------------------------------
# Load Model
# -----------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/waste_model.keras")


model = load_model()

IMG_SIZE = (150, 150)


# -----------------------------------------------------
# Prediction helper
# -----------------------------------------------------
def preprocess_image(image: Image.Image):
    image = image.resize(IMG_SIZE)
    img_array = np.array(image).astype("float32") / 255.0
    return np.expand_dims(img_array, axis=0)


def predict(image):
    processed = preprocess_image(image)
    probs = model.predict(processed)[0]

    # Assuming output = p(recyclable)
    p_recyclable = float(probs)
    p_organic = 1 - p_recyclable
    label = "Recyclable" if p_recyclable >= 0.5 else "Organic"

    return label, p_organic, p_recyclable


# -----------------------------------------------------
# UI — Upload Image
# -----------------------------------------------------
uploaded_file = st.file_uploader("Upload a waste image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)

    # Display uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict
    label, p_organic, p_recyclable = predict(image)

    st.markdown("### Result")

    # -----------------------------------------------------
    #  Color-coded output
    # -----------------------------------------------------
    if label == "Recyclable":
        st.success(f"♻️ **Recyclable**")
    else:
        st.warning(f"🌱 **Organic**")

    # -----------------------------------------------------
    # Confidence chart using Plotly
    # -----------------------------------------------------
    st.markdown("### Confidence Levels")

    fig = px.bar(
        x=["Organic", "Recyclable"],
        y=[p_organic, p_recyclable],
        labels={"x": "Class", "y": "Probability"},
        range_y=[0, 1],
        text=[f"{p_organic:.2f}", f"{p_recyclable:.2f}"],
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------
    # Reset Button
    # -----------------------------------------------------
    if st.button("Reset"):
        st.experimental_rerun()

else:
    st.info("Please upload an image to begin.")


# -----------------------------------------------------
# Footer
# -----------------------------------------------------
st.markdown("---")
st.caption("Built by Lampros Velentzas • Waste Classifier Micro-SaaS Prototype")
