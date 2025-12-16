import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(layout="centered")

# Custom CSS for thick bars
st.markdown(
    """
<style>
.prob-row {
    display: flex;
    align-items: center;
    margin-bottom: 10px;
}

.prob-label {
    width: 120px;
    font-weight: 600;
}

.prob-bar {
    flex: 1;
    height: 20px;
    background-color: #eee;
    border-radius: 10px;
    margin: 0 10px;
    position: relative;
    overflow: hidden;
}

.prob-fill {
    height: 100%;
    background-color: #1E90FF;  /* default (Recyclable) */
    border-radius: 10px;
}

.prob-fill-organic {
    background-color: #6B8E23;
}

.prob-value {
    width: 60px;
    text-align: right;
    font-weight: 600;
}
</style>
""",
    unsafe_allow_html=True,
)


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

    # 🔥 Probability Distribution (clean version)
    st.markdown("### Probability Distribution")

    # Compute probabilities
    if label == "Recyclable":
        p_recyclable = confidence
        p_organic = 1 - confidence
    else:
        p_organic = confidence
        p_recyclable = 1 - confidence

    # Convert to %
    r_pct = f"{p_recyclable * 100:.1f}%"
    o_pct = f"{p_organic * 100:.1f}%"

    # Build UI
    st.markdown(
        f"""
    <div class="prob-row">
        <div class="prob-label">♻️ Recyclable</div>
        <div class="prob-bar">
            <div class="prob-fill" style="width:{p_recyclable * 100}%;"></div>
        </div>
        <div class="prob-value">{r_pct}</div>
    </div>

    <div class="prob-row">
        <div class="prob-label">🌱 Organic</div>
        <div class="prob-bar">
            <div class="prob-fill prob-fill-organic" style="width:{p_organic * 100}%;"></div>
        </div>
        <div class="prob-value">{o_pct}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# Example images
col1, col2 = st.columns(2)

with col1:
    if st.button("Try Recyclable Example", key="ex_r", use_container_width=True):
        st.session_state.uploaded_file = "data/R_71.jpg"
        st.session_state.uploader_key += 1
        st.rerun()

with col2:
    if st.button("Try Organic Example", key="ex_o", use_container_width=True):
        st.session_state.uploaded_file = "data/O_1.jpg"
        st.session_state.uploader_key += 1
        st.rerun()


# ------------------------------
# "Upgrade to Pro" Button
# ------------------------------
import requests
import webbrowser
import os
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:4242")

st.subheader("🚀 Upgrade to Waste Classifier Pro")

if st.button("Upgrade for $5"):
    try:
        r = requests.post(f"{BACKEND_URL}/create-checkout-session", timeout=10)
        r.raise_for_status()
        js = r.json()
        checkout_url = js.get("url")
        if checkout_url:
            # Open in a new tab — Streamlit will allow the link.
            st.markdown(
                f"[Open Stripe Checkout]({checkout_url})", unsafe_allow_html=True
            )
            # Optionally open automatically in user's browser (works locally)
            try:
                webbrowser.open_new_tab(checkout_url)
            except:
                pass
        else:
            st.error("Could not create checkout session.")
    except Exception as e:
        st.error(f"Error creating checkout session: {e}")


# ------------------------------
# Footer
# ------------------------------
st.markdown(
    "<p style='text-align:center; color: grey;'>Made with ❤️ by Lampros</p>",
    unsafe_allow_html=True,
)
