"""
NeuroScan — Brain Tumor MRI Classifier
======================================
Premium SaaS-style multi-page Streamlit app around a fine-tuned ResNet50.

Pages
-----
- Analysis     : upload a scan -> predicted class + Grad-CAM heatmap
- Dashboard    : real test-set metrics (accuracy, macro-F1, confusion matrix)
- About        : method, data, disclaimer

The saved model does NOT preprocess internally — the app applies ResNet
`preprocess_input` before predict & Grad-CAM. Train via train_model.py.
Launch via ./run_app.sh (sets the required protobuf env var).
"""
import io
import os
import json

import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.cm as cm
import matplotlib.pyplot as plt

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess

# ================================================================== config ====
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_DIMS = (128, 128)
MODEL_DEFAULT = "ResNet50_finetuned.keras"
METRICS_FILE = "model_metrics.json"

CLASS_INFO = {
    "glioma":     "Tumor arising from glial cells of the brain or spine.",
    "meningioma": "Tumor of the meninges — the membranes around brain & cord.",
    "notumor":    "No tumor detected in this scan.",
    "pituitary":  "Tumor in the pituitary gland region at the skull base.",
}
CLASS_COLOR = {
    "glioma":     "#ef4444",
    "meningioma": "#f59e0b",
    "notumor":    "#10b981",
    "pituitary":  "#8b5cf6",
}

st.set_page_config(page_title="NeuroScan · Brain Tumor MRI",
                   page_icon="🧠", layout="wide",
                   initial_sidebar_state="expanded")

# =============================================================== styling =====
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{
    --bg:#f5f7fb;
    --surface:#ffffff;
    --surface-2:#f8fafc;
    --border:#e6eaf2;
    --border-strong:#d4dbe7;
    --text:#0b1424;
    --text-muted:#5b6577;
    --text-dim:#8892a6;
    --indigo:#4f46e5;
    --indigo-2:#6366f1;
    --indigo-deep:#1e1b4b;
    --indigo-soft:#eef2ff;
    --good:#10b981;
    --warn:#f59e0b;
    --bad:#ef4444;
    --shadow-sm: 0 1px 2px rgba(15,23,42,.04), 0 1px 3px rgba(15,23,42,.06);
    --shadow-md: 0 4px 10px -2px rgba(15,23,42,.08), 0 2px 6px -1px rgba(15,23,42,.06);
    --shadow-lg: 0 20px 40px -12px rgba(30,27,75,.18), 0 8px 16px -8px rgba(30,27,75,.12);
}

html, body, [class*="css"], .stApp, .stMarkdown, .stText {
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif !important;
    color: var(--text);
}
.stApp { background:
    radial-gradient(1200px 600px at 20% -10%, #e0e7ff 0%, transparent 60%),
    radial-gradient(800px 500px at 100% 0%, #ede9fe 0%, transparent 55%),
    var(--bg);
}
.block-container { padding-top: 1.4rem !important; max-width: 1240px; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { right: 1rem; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1024 0%, #161b3a 100%);
    border-right: 1px solid rgba(255,255,255,.06);
}
section[data-testid="stSidebar"] * { color: #cbd2e3; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #ffffff; }
section[data-testid="stSidebar"] [data-baseweb="radio"] label {
    padding: .55rem .75rem; border-radius: 10px; margin: 2px 0;
    transition: background .15s ease, transform .15s ease;
}
section[data-testid="stSidebar"] [data-baseweb="radio"] label:hover {
    background: rgba(99,102,241,.12); transform: translateX(2px);
}
section[data-testid="stSidebar"] input[type="text"]{
    background: rgba(255,255,255,.06); color:#fff; border: 1px solid rgba(255,255,255,.12);
    border-radius: 10px;
}

/* ---------- shared typography ---------- */
h1, h2, h3, h4 { color: var(--text); letter-spacing: -0.01em; }
h1 { font-weight: 800; }
h2, h3 { font-weight: 700; }
p, span, div { color: var(--text); }
.muted { color: var(--text-muted); font-size: .9rem; }
.dim   { color: var(--text-dim);   font-size: .82rem; }

/* ---------- HERO ---------- */
.hero {
    position: relative;
    background:
       radial-gradient(600px 300px at 90% -20%, rgba(139,92,246,.35) 0%, transparent 60%),
       radial-gradient(500px 250px at -10% 110%, rgba(59,130,246,.35) 0%, transparent 60%),
       linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
    color:#fff;
    padding: 2.2rem 2.4rem;
    border-radius: 24px;
    margin-bottom: 1.6rem;
    overflow: hidden;
    box-shadow: var(--shadow-lg);
    border: 1px solid rgba(255,255,255,.08);
    animation: fadeUp .55s ease both;
}
.hero::after{
    content:""; position:absolute; inset:0;
    background-image:
      radial-gradient(circle at 1px 1px, rgba(255,255,255,.08) 1px, transparent 0);
    background-size: 22px 22px;
    opacity: .35; pointer-events:none;
}
.hero h1 { color:#fff; font-size: 2.1rem; margin:0; font-weight: 800; letter-spacing:-.02em; }
.hero .eyebrow {
    display:inline-block; padding:.25rem .7rem; border-radius:999px;
    background: rgba(255,255,255,.12); color:#e0e7ff; font-size:.75rem;
    font-weight:600; letter-spacing:.08em; text-transform:uppercase;
    margin-bottom:.7rem; border:1px solid rgba(255,255,255,.18);
}
.hero p  { color:#c7d2fe; margin:.4rem 0 0; font-size:1.02rem; max-width: 700px; }

/* ---------- generic cards ---------- */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 1.6rem 1.8rem;
    box-shadow: var(--shadow-md);
    transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
    animation: fadeUp .5s ease both;
}
.card:hover { transform: translateY(-2px); box-shadow: var(--shadow-lg); border-color: var(--border-strong); }
.card-title{ font-size:.8rem; font-weight:600; color:var(--text-muted); letter-spacing:.08em; text-transform:uppercase; margin-bottom:.6rem;}

/* ---------- prediction hero card ---------- */
.pred-card {
    position: relative;
    background: linear-gradient(135deg, #ffffff 0%, #fafbff 100%);
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 1.6rem 1.8rem 1.6rem 2.2rem;
    box-shadow: var(--shadow-lg);
    overflow: hidden;
    animation: fadeUp .5s ease both;
}
.pred-card::before{
    content:""; position:absolute; left:0; top:0; bottom:0; width:6px;
    background: var(--accent, var(--indigo));
    border-radius: 22px 0 0 22px;
}
.pred-pill {
    display:inline-block; padding:.25rem .7rem; border-radius:999px;
    font-size:.72rem; font-weight:700; color:#fff; letter-spacing:.06em;
    background: var(--accent, var(--indigo)); text-transform:uppercase;
}
.pred-label{
    font-size: 2.6rem; font-weight: 800; letter-spacing:-.02em; margin-top:.55rem;
    color: var(--accent, var(--indigo)); line-height:1.1;
}
.pred-desc{ color: var(--text-muted); margin-top:.25rem; font-size:.95rem; }
.confidence-row{
    display:flex; align-items:baseline; gap:.5rem; margin-top:1rem;
    padding-top:1rem; border-top:1px dashed var(--border);
}
.confidence-row .v{ font-size:1.7rem; font-weight:800; color:var(--text); letter-spacing:-.02em; }
.confidence-row .l{ color:var(--text-muted); font-size:.85rem; }

/* ---------- probability bars ---------- */
.prob-row{ margin: .5rem 0 1rem; }
.prob-head{
    display:flex; justify-content:space-between; align-items:baseline;
    font-size:.88rem; margin-bottom:.3rem;
}
.prob-head .n{ color: var(--text); font-weight:600; text-transform:capitalize;}
.prob-head .v{ color: var(--text-muted); font-variant-numeric: tabular-nums;}
.prob-track{
    height: 8px; background: #eef2f7; border-radius:999px; overflow:hidden;
}
.prob-fill{
    height:100%; border-radius:999px;
    background: linear-gradient(90deg, var(--c) 0%, color-mix(in srgb, var(--c) 70%, white) 100%);
    width: var(--w);
    animation: grow .8s cubic-bezier(.2,.7,.2,1) both;
    box-shadow: 0 0 0 2px rgba(255,255,255,.5) inset;
}

/* ---------- KPI cards (dashboard) ---------- */
.kpi {
    background: linear-gradient(180deg, #ffffff 0%, #fbfbff 100%);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.2rem 1.3rem;
    box-shadow: var(--shadow-sm);
    transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
    position: relative;
    overflow: hidden;
    animation: fadeUp .5s ease both;
}
.kpi:hover { transform: translateY(-3px); box-shadow: var(--shadow-md); border-color: #c7d2fe; }
.kpi::after{
    content:""; position:absolute; top:0; left:0; right:0; height:3px;
    background: linear-gradient(90deg, var(--indigo) 0%, #8b5cf6 100%);
    opacity:.85;
}
.kpi .l{
    font-size:.74rem; font-weight:600; color:var(--text-muted);
    text-transform:uppercase; letter-spacing:.08em;
}
.kpi .v{
    font-size:2.1rem; font-weight:800; color:var(--indigo-deep);
    letter-spacing:-.02em; line-height:1.1; margin-top:.25rem;
    background: linear-gradient(90deg, #1e1b4b 0%, var(--indigo) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.kpi .s{ color: var(--text-dim); font-size:.78rem; margin-top:.3rem;}

/* ---------- file uploader ---------- */
[data-testid="stFileUploader"] section {
    border: 1.5px dashed var(--border-strong) !important;
    background: var(--surface) !important;
    border-radius: 16px !important;
    padding: 1.3rem !important;
    transition: border-color .2s ease, background .2s ease;
}
[data-testid="stFileUploader"] section:hover {
    border-color: var(--indigo) !important;
    background: var(--indigo-soft) !important;
}
[data-testid="stFileUploader"] button{
    background: var(--indigo) !important; color:#fff !important;
    border: none !important; border-radius: 10px !important;
    font-weight:600 !important; box-shadow: 0 1px 2px rgba(79,70,229,.3);
}

/* ---------- progress + tweaks ---------- */
.stProgress > div > div > div > div { background: var(--indigo) !important; }
.stImage img { border-radius: 14px; box-shadow: var(--shadow-md); }
.stAlert { border-radius: 14px; border: 1px solid var(--border); }

/* table */
.stDataFrame { border-radius: 14px; overflow: hidden; }

/* ---------- animations ---------- */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes grow {
    from { width: 0; } to { width: var(--w); }
}
</style>
""", unsafe_allow_html=True)


# ======================================================== model utilities =====
@st.cache_resource(show_spinner="Loading model…")
def load_model(path):
    return tf.keras.models.load_model(path, compile=False)


def find_backbone(model):
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
    model = load_model(_model_path)
    base = find_backbone(model)
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
    return tf.keras.Model(base.input, [feat, x])


def preprocess_image(pil_img):
    img = pil_img.convert("RGB").resize(IMG_DIMS)
    raw = np.asarray(img, dtype="float32")
    batch = resnet_preprocess(np.expand_dims(raw.copy(), 0))
    return raw, batch


def gradcam_heatmap(cam_model, raw_arr, pred_index):
    pre = resnet_preprocess(np.expand_dims(raw_arr.copy(), 0))
    pre = tf.convert_to_tensor(pre)
    with tf.GradientTape() as tape:
        conv_out, preds = cam_model(pre)
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.squeeze(conv_out[0] @ pooled[..., tf.newaxis])
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = tf.image.resize(heatmap[..., tf.newaxis], IMG_DIMS).numpy().squeeze()
    return heatmap


def overlay_heatmap(raw_arr, heatmap, alpha=0.4):
    colored = cm.jet(heatmap)[..., :3]
    blended = (1 - alpha) * (raw_arr / 255.0) + alpha * colored
    return np.clip(blended, 0, 1)


# ----------------------------------------------------- severity colour scale --
# Green -> yellow -> orange -> red, indexed by a 0..1 "danger" score so the
# probability bars get warmer the more likely a tumor is.
_SEVERITY_STOPS = [
    (0.00, (34, 197, 94)),    # green  #22c55e  (safe)
    (0.40, (234, 179, 8)),    # yellow #eab308
    (0.70, (249, 115, 22)),   # orange #f97316
    (1.00, (239, 68, 68)),    # red    #ef4444  (dangerous)
]


def severity_color(score):
    """Map a 0..1 danger score to a hex colour along green→yellow→orange→red."""
    score = float(np.clip(score, 0.0, 1.0))
    for (s0, c0), (s1, c1) in zip(_SEVERITY_STOPS, _SEVERITY_STOPS[1:]):
        if score <= s1:
            t = 0.0 if s1 == s0 else (score - s0) / (s1 - s0)
            rgb = [round(a + (b - a) * t) for a, b in zip(c0, c1)]
            return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    return "#ef4444"


def danger_score(cls, prob):
    """Tumor classes are dangerous at high prob; 'notumor' is safe at high prob."""
    return (1.0 - prob) if cls == "notumor" else prob


@st.cache_data(show_spinner=False)
def load_metrics(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ================================================================ sidebar =====
with st.sidebar:
    st.markdown(
        "<div style='display:flex;align-items:center;gap:.6rem;margin:.4rem 0 .2rem'>"
        "<div style='width:38px;height:38px;border-radius:11px;"
        "background:linear-gradient(135deg,#6366f1,#8b5cf6);display:flex;"
        "align-items:center;justify-content:center;font-size:20px;"
        "box-shadow:0 6px 14px -4px rgba(99,102,241,.6)'>🧠</div>"
        "<div><div style='font-weight:800;font-size:1.1rem;color:#fff'>NeuroScan</div>"
        "<div style='font-size:.72rem;color:#9aa3bd;letter-spacing:.06em;"
        "text-transform:uppercase'>MRI Classifier</div></div></div>",
        unsafe_allow_html=True)

    st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)
    page = st.radio("Menu",
                    ["🔬  Analysis", "📊  Model Dashboard", "ℹ️  About"],
                    label_visibility="collapsed")

    st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,.08);"
                "margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:.72rem;letter-spacing:.08em;color:#8892a6;"
                "text-transform:uppercase;margin-bottom:.4rem'>Settings</div>",
                unsafe_allow_html=True)
    model_path = st.text_input("Model file", value=MODEL_DEFAULT,
                               label_visibility="collapsed")
    show_cam = st.toggle("Show Grad-CAM heatmap", value=True)

    st.markdown("<div style='position:absolute;bottom:1.4rem;left:1.2rem;right:1.2rem;"
                "font-size:.72rem;color:#6b7390;line-height:1.5'>"
                "Research / educational use only.<br>Not a medical device.</div>",
                unsafe_allow_html=True)

metrics = load_metrics(METRICS_FILE)


# =============================================================== ANALYSIS =====
def page_analysis():
    st.markdown("""
    <div class='hero'>
        <span class='eyebrow'>AI · ResNet50 · Grad-CAM</span>
        <h1>Brain Tumor MRI Classifier</h1>
        <p>Upload a brain MRI scan and get an instant 4-class prediction with
        a visual explanation of the regions that influenced the model.</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Could not load model from **{model_path}**. "
                 f"Train it first with `python train_model.py`.\n\nDetails: {e}")
        st.stop()

    uploaded = st.file_uploader("Drop a brain MRI image here, or click to browse",
                                type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])

    if uploaded is None:
        st.markdown("""
        <div class='card' style='margin-top:1rem;text-align:center'>
            <div style='font-size:2.4rem'>⬆️</div>
            <div style='font-weight:700;font-size:1.1rem;margin-top:.4rem'>
                Upload an MRI scan to begin</div>
            <div class='muted' style='margin-top:.3rem'>
                Supported formats: PNG, JPG, BMP, TIF · max 200 MB</div>
        </div>
        """, unsafe_allow_html=True)
        return

    pil_img = Image.open(io.BytesIO(uploaded.read()))
    raw_arr, batch = preprocess_image(pil_img)
    probs = model.predict(batch, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_cls = CLASSES[pred_idx]
    confidence = float(probs[pred_idx])
    color = severity_color(danger_score(pred_cls, confidence))

    col_img, col_res = st.columns([1, 1.15], gap="large")
    with col_img:
        st.markdown("<div class='card'>"
                    "<div class='card-title'>Input scan</div>", unsafe_allow_html=True)
        st.image(pil_img, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_res:
        pill_label = "NO TUMOR" if pred_cls == "notumor" else "TUMOR DETECTED"
        st.markdown(f"""
        <div class='pred-card' style='--accent:{color}'>
            <span class='pred-pill'>{pill_label}</span>
            <div class='pred-label'>{pred_cls.upper()}</div>
            <div class='pred-desc'>{CLASS_INFO[pred_cls]}</div>
            <div class='confidence-row'>
                <div class='v'>{confidence*100:.2f}%</div>
                <div class='l'>model confidence</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        bars_html = "<div class='card' style='margin-top:1rem'>"
        bars_html += "<div class='card-title'>Class probabilities</div>"
        for i in np.argsort(probs)[::-1]:
            c = severity_color(danger_score(CLASSES[i], float(probs[i])))
            bars_html += (
                f"<div class='prob-row'>"
                f"<div class='prob-head'><span class='n'>{CLASSES[i]}</span>"
                f"<span class='v'>{probs[i]*100:.2f}%</span></div>"
                f"<div class='prob-track'>"
                f"<div class='prob-fill' style='--w:{probs[i]*100:.2f}%;--c:{c}'></div>"
                f"</div></div>")
        bars_html += "</div>"
        st.markdown(bars_html, unsafe_allow_html=True)

    if show_cam:
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown("<div class='card'>"
                    "<div class='card-title'>Grad-CAM · where the model looked</div>",
                    unsafe_allow_html=True)
        try:
            cam_model = build_cam_model(model_path)
            heatmap = gradcam_heatmap(cam_model, raw_arr, pred_idx)
            overlay = overlay_heatmap(raw_arr, heatmap)
            g1, g2 = st.columns(2)
            with g1:
                st.image(raw_arr.astype("uint8"),
                         caption="Resized input (128×128)",
                         use_container_width=True)
            with g2:
                st.image(overlay,
                         caption=f"Activation overlay — predicted: {pred_cls}",
                         use_container_width=True, clamp=True)
            st.markdown("<div class='dim' style='margin-top:.6rem'>"
                        "Warm regions most influenced the prediction. Qualitative "
                        "check only — not a performance metric.</div>",
                        unsafe_allow_html=True)
        except Exception as e:
            st.info(f"Grad-CAM unavailable for this model: {e}")
        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================== DASHBOARD =====
def page_dashboard():
    st.markdown("""
    <div class='hero'>
        <span class='eyebrow'>Model performance · held-out test set</span>
        <h1>Model Dashboard</h1>
        <p>Real test-set metrics for the deployed classifier — measured once,
        on the held-out <code style='background:rgba(255,255,255,.12);
        padding:.05rem .35rem;border-radius:6px;color:#fff'>Testing/</code> split.</p>
    </div>
    """, unsafe_allow_html=True)

    if metrics is None:
        st.warning("No `model_metrics.json` found. Run `python train_model.py` "
                   "to train the model and generate metrics.")
        return

    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        (c1, f"{metrics['accuracy']*100:.2f}%",  "Test accuracy", "overall"),
        (c2, f"{metrics['macro_f1']*100:.2f}%",  "Macro F1",      "balanced across classes"),
        (c3, f"{metrics['n_test']:,}",           "Test images",   "held-out"),
        (c4, f"{metrics['n_train']:,}",          "Train images",  "with 20% validation"),
    ]
    for col, val, label, sub in kpis:
        col.markdown(
            f"<div class='kpi'><div class='l'>{label}</div>"
            f"<div class='v'>{val}</div>"
            f"<div class='s'>{sub}</div></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.markdown("<div class='card'>"
                    "<div class='card-title'>Per-class performance</div>",
                    unsafe_allow_html=True)
        rows = metrics["per_class"]
        st.dataframe({
            "Class":     list(rows.keys()),
            "Precision": [f"{rows[c]['precision']*100:.1f}%" for c in rows],
            "Recall":    [f"{rows[c]['recall']*100:.1f}%"    for c in rows],
            "F1":        [f"{rows[c]['f1']*100:.1f}%"        for c in rows],
            "Support":   [rows[c]["support"] for c in rows],
        }, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        bars = "<div class='card'><div class='card-title'>Per-class recall</div>"
        for c in rows:
            col = CLASS_COLOR[c]
            r = rows[c]["recall"] * 100
            bars += (f"<div class='prob-row'>"
                     f"<div class='prob-head'><span class='n'>{c}</span>"
                     f"<span class='v'>{r:.1f}%</span></div>"
                     f"<div class='prob-track'>"
                     f"<div class='prob-fill' style='--w:{r:.2f}%;--c:{col}'></div>"
                     f"</div></div>")
        bars += "</div>"
        st.markdown(bars, unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>"
                    "<div class='card-title'>Confusion matrix</div>",
                    unsafe_allow_html=True)
        cm_arr = np.array(metrics["confusion_matrix"])
        classes = metrics["classes"]
        plt.rcParams.update({"font.family": "DejaVu Sans"})
        fig, ax = plt.subplots(figsize=(4.8, 4.3))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        im = ax.imshow(cm_arr, cmap="Purples")
        ax.set_xticks(range(len(classes)), classes, rotation=35, ha="right",
                      color="#0b1424")
        ax.set_yticks(range(len(classes)), classes, color="#0b1424")
        ax.set_xlabel("Predicted", color="#5b6577", fontsize=10)
        ax.set_ylabel("True",      color="#5b6577", fontsize=10)
        for s in ax.spines.values():
            s.set_color("#e6eaf2")
        ax.tick_params(colors="#5b6577")
        thresh = cm_arr.max() / 2
        for i in range(cm_arr.shape[0]):
            for j in range(cm_arr.shape[1]):
                ax.text(j, i, int(cm_arr[i, j]), ha="center", va="center",
                        color="white" if cm_arr[i, j] > thresh else "#1e1b4b",
                        fontweight="bold", fontsize=11)
        fig.colorbar(im, fraction=0.046, pad=0.04)
        fig.tight_layout()
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        f"<div class='dim' style='margin-top:1rem;text-align:right'>"
        f"Trained at {metrics.get('trained_at','?')} · "
        f"training took {metrics.get('train_seconds','?')}s · "
        f"metrics computed once on the held-out Testing/ set.</div>",
        unsafe_allow_html=True)


# ================================================================== ABOUT =====
def page_about():
    st.markdown("""
    <div class='hero'>
        <span class='eyebrow'>About</span>
        <h1>Method · Data · Disclaimer</h1>
        <p>Transfer learning on ResNet50 with a custom classification head and
        Grad-CAM visual explainability for 4-class brain MRI classification.</p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(3, gap="large")
    cards = [
        ("🧬", "Architecture",
         "ResNet50 backbone (ImageNet weights), frozen for feature extraction, "
         "with a dense classification head trained on extracted features."),
        ("📂", "Data",
         "Brain Tumor MRI Dataset — 4 balanced classes (glioma, meningioma, "
         "notumor, pituitary), ~5.7k train / ~1.3k held-out test."),
        ("🔍", "Explainability",
         "Grad-CAM uses the gradient of the predicted class with respect to the "
         "last convolutional feature map to highlight the regions that mattered."),
    ]
    for col, (icon, title, body) in zip(cols, cards):
        col.markdown(
            f"<div class='card' style='height:100%'>"
            f"<div style='font-size:1.6rem'>{icon}</div>"
            f"<div style='font-weight:700;font-size:1.05rem;margin:.5rem 0 .35rem'>{title}</div>"
            f"<div class='muted' style='line-height:1.55'>{body}</div></div>",
            unsafe_allow_html=True)

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card' style='border-color:#fecaca;background:linear-gradient(180deg,#fff 0%,#fef2f2 100%)'>
        <div style='display:flex;gap:1rem;align-items:flex-start'>
            <div style='font-size:1.6rem'>⚠️</div>
            <div>
                <div style='font-weight:700;font-size:1.05rem;color:#991b1b'>Disclaimer</div>
                <div class='muted' style='margin-top:.3rem;color:#7f1d1d;line-height:1.55'>
                    Research and educational use only. This is <b>not</b> a medical
                    device and must not be used for real diagnosis. Always consult
                    a qualified radiologist.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ===================================================================== run ====
if page.startswith("🔬"):
    page_analysis()
elif page.startswith("📊"):
    page_dashboard()
else:
    page_about()
