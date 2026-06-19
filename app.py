"""
Brain Tumor MRI Classifier — Streamlit app
==========================================
Upload a brain MRI scan and the fine-tuned ResNet50 model predicts one of:
    glioma, meningioma, notumor, pituitary
plus a Grad-CAM heatmap showing where the model looked.

The model already contains its own preprocessing (ResNet `preprocess_input`),
so we feed raw 128x128 RGB images straight in — no manual /255 needed.

Run:  streamlit run app.py
"""

import io
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.cm as cm
import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess

# ---------------------------------------------------------------- config -----
CLASSES   = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_DIMS  = (128, 128)                 # must match training
MODEL_DEFAULT = "ResNet50_finetuned.keras"

CLASS_INFO = {
    "glioma":     "Tumor arising from glial cells.",
    "meningioma": "Tumor of the meninges (brain/spinal-cord membranes).",
    "notumor":    "No tumor detected in this scan.",
    "pituitary":  "Tumor in the pituitary gland region.",
}

st.set_page_config(page_title="Brain Tumor MRI Classifier",
                   page_icon="🧠", layout="wide")

# Clean, medical look ---------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #f7fafc; }
    h1, h2, h3 { color: #14315e; }
    .block-container { padding-top: 2rem; max-width: 1100px; }
    .result-card {
        background:#ffffff; border:1px solid #e2e8f0; border-radius:14px;
        padding:1.2rem 1.4rem; box-shadow:0 1px 3px rgba(0,0,0,.06);
    }
    .pred-label { font-size:1.9rem; font-weight:700; color:#14315e; }
    .muted { color:#64748b; font-size:.9rem; }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------- model loading ---
@st.cache_resource(show_spinner="Loading model…")
def load_model(path):
    return tf.keras.models.load_model(path, compile=False)


def find_backbone(model):
    """The transfer model nests the ResNet50 backbone as a sub-Model."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    raise ValueError("No nested backbone Model found.")


def find_last_conv(base):
    for layer in reversed(base.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found.")


@st.cache_resource(show_spinner=False)
def build_cam_model(_model_path):
    """Rebuild a Grad-CAM model: input -> [last conv feature map, predictions].
    Re-applies the head onto the backbone output to avoid 'graph disconnected'."""
    model = load_model(_model_path)
    base  = find_backbone(model)
    last_conv = find_last_conv(base)
    feat = base.get_layer(last_conv).output
    x = base.output
    started = False
    for layer in model.layers:
        if layer.name == base.name:
            started = True
            continue
        if started:
            x = layer(x)
    cam = tf.keras.Model(base.input, [feat, x])
    return cam


# --------------------------------------------------------------- inference ---
def preprocess_image(pil_img):
    """PIL image -> (1,128,128,3) float32 in [0,255] (model preprocesses inside)."""
    img = pil_img.convert("RGB").resize(IMG_DIMS)
    arr = np.asarray(img, dtype="float32")
    return arr, np.expand_dims(arr, 0)


def predict(model, batch):
    probs = model.predict(batch, verbose=0)[0]
    return probs


def gradcam_heatmap(cam_model, raw_batch, pred_index):
    pre = resnet_preprocess(tf.identity(raw_batch))
    with tf.GradientTape() as tape:
        conv_out, preds = cam_model(pre)
        class_channel = preds[:, pred_index]
    grads  = tape.gradient(class_channel, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.squeeze(conv_out[0] @ pooled[..., tf.newaxis])
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = tf.image.resize(heatmap[..., tf.newaxis], IMG_DIMS).numpy().squeeze()
    return heatmap


def overlay_heatmap(raw_arr, heatmap, alpha=0.4):
    colored = cm.jet(heatmap)[..., :3]          # (H,W,3) in [0,1]
    base = raw_arr / 255.0
    blended = (1 - alpha) * base + alpha * colored
    return np.clip(blended, 0, 1)


# ------------------------------------------------------------------- UI -------
st.title("🧠 Brain Tumor MRI Classifier")
st.markdown(
    "<p class='muted'>Fine-tuned ResNet50 · 4-class MRI classification · "
    "with Grad-CAM explainability</p>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    model_path = st.text_input("Model file (.keras)", value=MODEL_DEFAULT)
    show_cam   = st.toggle("Show Grad-CAM heatmap", value=True)
    st.markdown("---")
    st.caption("Research/educational use only — not a medical device. "
               "Predictions must not be used for real diagnosis.")

# Load model (with friendly error if missing)
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Could not load model from **{model_path}**.\n\n"
             "Download `ResNet50_finetuned.keras` from the training notebook "
             "and place it next to this app (or set the correct path in the "
             f"sidebar).\n\nDetails: {e}")
    st.stop()

uploaded = st.file_uploader("Upload a brain MRI image",
                            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])

if uploaded is None:
    st.info("⬆️ Upload an MRI scan to get a prediction.")
    st.stop()

pil_img = Image.open(io.BytesIO(uploaded.read()))
raw_arr, batch = preprocess_image(pil_img)
probs = predict(model, batch)
pred_idx = int(np.argmax(probs))
pred_cls = CLASSES[pred_idx]
confidence = float(probs[pred_idx])

col_img, col_res = st.columns([1, 1.1], gap="large")

with col_img:
    st.subheader("Input scan")
    st.image(pil_img, use_container_width=True)

with col_res:
    st.subheader("Prediction")
    st.markdown(
        f"<div class='result-card'>"
        f"<div class='pred-label'>{pred_cls.upper()}</div>"
        f"<div class='muted'>{CLASS_INFO[pred_cls]}</div>"
        f"<div style='margin-top:.6rem;font-size:1.1rem;'>"
        f"Confidence: <b>{confidence*100:.2f}%</b></div>"
        f"</div>", unsafe_allow_html=True)

    st.markdown("##### Class probabilities")
    order = np.argsort(probs)[::-1]
    for i in order:
        st.write(f"**{CLASSES[i]}** — {probs[i]*100:.2f}%")
        st.progress(float(probs[i]))

    if pred_cls == "notumor":
        st.success("No tumor detected in this scan.")
    else:
        st.warning("Possible tumor detected — clinical confirmation required.")

# Grad-CAM ---------------------------------------------------------------------
if show_cam:
    st.markdown("---")
    st.subheader("Grad-CAM — where the model looked")
    try:
        cam_model = build_cam_model(model_path)
        heatmap = gradcam_heatmap(cam_model, batch, pred_idx)
        overlay = overlay_heatmap(raw_arr, heatmap)
        g1, g2 = st.columns(2)
        with g1:
            st.image(raw_arr.astype("uint8"), caption="Resized input (128×128)",
                     use_container_width=True)
        with g2:
            st.image(overlay, caption=f"Grad-CAM — predicted: {pred_cls}",
                     use_container_width=True, clamp=True)
        st.caption("Warm regions = areas that most influenced the prediction. "
                   "This is a qualitative check, not a performance metric.")
    except Exception as e:
        st.info(f"Grad-CAM unavailable for this model: {e}")
