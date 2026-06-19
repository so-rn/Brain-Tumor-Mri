"""
PixelMind — Neural Brain MRI Atlas
==================================
Clean, dark, professional Streamlit dashboard for a 4-class brain-MRI classifier
(glioma · meningioma · notumor · pituitary). Upload a scan for an instant
prediction with Grad-CAM explainability, plus an analytics deck driven by the
training notebook's metrics.json.

Run:  streamlit run app.py
"""

from __future__ import annotations

import io
import json
import pathlib
from contextlib import contextmanager
from typing import Any

import matplotlib.cm as mcm
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess

# ─────────────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────────────
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_DIMS = (128, 128)
MODEL_DEFAULT = "ResNet50_finetuned.keras"
METRICS_PATH = pathlib.Path("metrics.json")

CLASS_INFO = {
    "glioma":     "Tumor arising from glial cells of the brain or spine.",
    "meningioma": "Tumor of the meninges — the membranes around brain & cord.",
    "notumor":    "No tumor detected in this scan.",
    "pituitary":  "Tumor in the pituitary gland region at the base of the brain.",
}

# one accent per class — distinct but anchored in the green/teal medical palette
ACCENT = {
    "glioma":     "#059669",   # emerald
    "meningioma": "#0891b2",   # cyan
    "notumor":    "#16a34a",   # healthy green
    "pituitary":  "#d97706",   # amber (warm contrast)
}

# chart ramp (kept these names for compatibility; values are now the green theme)
INDIGO = "#10b981"   # mint-emerald
VIOLET = "#059669"   # emerald
PINK = "#0d9488"     # teal

st.set_page_config(
    page_title="PixelMind — Brain MRI Atlas",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  Style — calm dark, restrained motion. Cards = NATIVE bordered containers.
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Sora:wght@500;600;700&display=swap');
:root{
  --bg:#eef5f0;
  --panel:rgba(255,255,255,0.78);
  --panel-border:rgba(6,95,70,0.14);
  --panel-border-hover:rgba(5,150,105,0.55);
  --text:#0f2e22;
  --mute:#5c7468;
  --faint:#8aa499;
  --indigo:#34d399;
  --violet:#059669;
  --pink:#0d9488;
}

#MainMenu, footer, header[data-testid="stHeader"]{visibility:hidden; height:0;}

.stApp{
  background:
    radial-gradient(1100px 560px at 78% -12%, rgba(16,185,129,0.16), transparent 60%),
    radial-gradient(900px 500px at 8% 8%, rgba(13,148,136,0.12), transparent 55%),
    linear-gradient(180deg, #f1f8f4 0%, #e9f4ee 45%, #f4faf6 100%);
  background-attachment: fixed;
  color: var(--text);
}
/* slow drifting aurora — subtle motion across the whole app */
.stApp::before{
  content:""; position:fixed; inset:-20%; z-index:0; pointer-events:none;
  background:
    radial-gradient(620px 620px at 20% 30%, rgba(16,185,129,0.12), transparent 60%),
    radial-gradient(560px 560px at 80% 70%, rgba(13,148,136,0.10), transparent 60%);
  filter:blur(8px); opacity:.7;
  animation:aurora 26s ease-in-out infinite;
}
.stApp > *{ position:relative; z-index:1; }
@keyframes aurora{
  0%,100%{ transform:translate3d(0,0,0) scale(1); }
  33%    { transform:translate3d(3%,-2%,0) scale(1.04); }
  66%    { transform:translate3d(-2%,3%,0) scale(1.02); }
}

.block-container{ padding-top:1.4rem !important; padding-bottom:4rem; max-width:1320px; }

h1,h2,h3,h4,h5{ font-family:'Sora',sans-serif !important; color:var(--text) !important; letter-spacing:-.01em; }
body,p,span,label,.stMarkdown,div{ font-family:'Inter',sans-serif; }
.mute{ color:var(--mute); }

/* Hero */
.hero{ animation:rise .55s ease backwards; margin-bottom:.4rem; }
.hero-kicker{
  display:inline-flex; align-items:center; gap:8px;
  font-family:'Sora'; font-size:.72rem; letter-spacing:.22em; text-transform:uppercase;
  color:var(--violet); font-weight:600;
}
.dot{ width:7px;height:7px;border-radius:50%;background:var(--pink); box-shadow:0 0 10px var(--pink); }
.hero-title{
  font-family:'Sora'; font-weight:700; font-size:2.7rem; line-height:1.05;
  margin:.5rem 0 .5rem;
  background:linear-gradient(110deg,#065f46,#059669 32%,#0d9488 52%,#059669 72%,#065f46);
  background-size:220% auto;
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
  animation:nameSweep 9s ease-in-out infinite;
}
.hero-sub{ color:var(--mute); font-size:.98rem; max-width:720px; line-height:1.6; }

/* KPI cards — self-contained HTML, no widgets inside (safe) */
.kpi-grid{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:1.4rem 0 .4rem; }
.kpi{
  position:relative; overflow:hidden;
  background:var(--panel); border:1px solid var(--panel-border); border-radius:16px;
  padding:1.1rem 1.2rem 1.2rem; animation:rise .5s ease backwards;
  transition:transform .2s ease, border-color .2s ease;
}
.kpi:hover{ transform:translateY(-3px); border-color:var(--panel-border-hover); }
.kpi:nth-child(2){animation-delay:.06s} .kpi:nth-child(3){animation-delay:.12s} .kpi:nth-child(4){animation-delay:.18s}
.kpi::before{ content:""; position:absolute; left:0; top:0; bottom:0; width:3px;
  background:linear-gradient(var(--indigo),var(--pink)); }
.kpi-label{ font-size:.66rem; letter-spacing:.13em; text-transform:uppercase; color:var(--mute); font-weight:600; }
.kpi-value{ font-family:'Sora'; font-weight:700; font-size:2.05rem; line-height:1; margin:.45rem 0 .3rem; color:#0b3b2c; }
.kpi-sub{ font-size:.76rem; color:var(--faint); }

/* ── NATIVE bordered container = the real card (wraps widgets correctly) ── */
div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"]{
  background:var(--panel);
  border:1px solid var(--panel-border) !important;
  border-radius:18px !important;
  padding:1.3rem 1.45rem !important;
  backdrop-filter:blur(8px);
  transition:border-color .2s ease, transform .2s ease, box-shadow .2s ease;
  animation:rise .5s ease backwards;
}
div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"]:hover{
  border-color:var(--panel-border-hover) !important;
  box-shadow:0 14px 40px rgba(5,150,105,0.14);
}

.panel-eyebrow{ font-size:.66rem; letter-spacing:.2em; text-transform:uppercase; color:var(--violet); font-weight:600; }
.panel-title{ font-family:'Sora'; font-weight:600; font-size:1.18rem; margin:.25rem 0 1rem;
  display:flex; align-items:center; gap:9px; }
.panel-title .pt-dot{ width:7px;height:7px;border-radius:50%;
  background:linear-gradient(var(--violet),var(--pink)); box-shadow:0 0 9px var(--violet); }

/* Prediction banner */
.pred{
  background:linear-gradient(120deg, rgba(16,185,129,.16), rgba(13,148,136,.10));
  border:1px solid var(--panel-border); border-radius:16px; padding:1.4rem 1.6rem;
}
.pred-eyebrow{ font-family:'Sora'; font-size:.68rem; letter-spacing:.2em; text-transform:uppercase; color:var(--mute); }
.pred-value{ font-family:'Sora'; font-weight:700; font-size:2.8rem; line-height:1; margin:.35rem 0 .3rem;
  background:linear-gradient(110deg,#065f46,#059669,#0d9488);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text; }
.pred-desc{ color:var(--mute); font-size:.92rem; }
.pred-chip{ display:inline-block; margin-top:1rem; padding:.4rem .95rem; border-radius:999px;
  background:rgba(5,150,105,.10); border:1px solid var(--panel-border); color:#065f46;
  font-family:'Sora'; font-weight:600; font-size:.9rem; }

/* probability bars */
.pb{ margin:.7rem 0; }
.pb-row{ display:flex; justify-content:space-between; font-size:.86rem; margin-bottom:.32rem; }
.pb-row .nm{ text-transform:capitalize; color:var(--text); font-weight:500; }
.pb-row .vl{ font-family:'Sora'; font-weight:600; color:var(--mute); }
.pb-track{ height:8px; border-radius:999px; background:rgba(6,95,70,.08); overflow:hidden; }
.pb-fill{ height:100%; border-radius:999px;
  background:linear-gradient(90deg,var(--indigo),var(--violet),var(--pink));
  animation:grow .8s cubic-bezier(.22,1,.36,1) backwards; }
@keyframes grow{ from{ width:0 !important; } }

/* Sidebar */
section[data-testid="stSidebar"]{
  background:rgba(255,255,255,.78) !important;
  backdrop-filter:blur(20px);
  border-right:1px solid var(--panel-border);
}
section[data-testid="stSidebar"] *{ color:var(--text); }

.brand{ display:flex; align-items:center; gap:13px; animation:fadeR .6s ease backwards; }
.brand-logo{ width:46px;height:46px;border-radius:14px;flex:none; display:grid; place-items:center;
  background:linear-gradient(135deg,var(--indigo),var(--violet),var(--pink)); background-size:180% 180%;
  animation:logoGlow 3.4s ease-in-out infinite, logoHue 8s ease infinite; }
.brand-logo svg{ width:25px;height:25px; stroke:#fff; fill:none;
  filter:drop-shadow(0 1px 4px rgba(0,0,0,.4)); }
.brand-name{ font-family:'Sora'; font-weight:700; font-size:1.85rem; line-height:1; letter-spacing:-.02em;
  background:linear-gradient(110deg,#065f46,#059669 35%,#0d9488 55%,#059669 75%,#065f46);
  background-size:220% auto;
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
  filter:drop-shadow(0 2px 12px rgba(5,150,105,.35));
  animation:nameSweep 6s linear infinite; }
.brand-ver{ font-size:.62rem; letter-spacing:.2em; text-transform:uppercase; color:var(--faint); margin-top:5px; }
@keyframes nameSweep{ to{ background-position:220% center; } }
@keyframes logoGlow{ 0%,100%{ box-shadow:0 5px 16px rgba(16,185,129,.42); } 50%{ box-shadow:0 8px 30px rgba(13,148,136,.65); } }
@keyframes logoHue{ 0%,100%{ background-position:0% 50%; } 50%{ background-position:100% 50%; } }

.side-h{ font-size:.64rem; letter-spacing:.24em; text-transform:uppercase; color:var(--violet); font-weight:600; margin:.2rem 0 .6rem; }

/* nav radio → flat mask icons, no dot */
section[data-testid="stSidebar"] div[role="radiogroup"]{ gap:.28rem !important; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label{
  position:relative; padding:.58rem .8rem .58rem 2.6rem !important; border-radius:12px; cursor:pointer;
  transition:background .2s ease, transform .15s ease; animation:fadeR .45s ease backwards;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(2){animation-delay:.05s}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(3){animation-delay:.10s}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(4){animation-delay:.15s}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(5){animation-delay:.20s}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover{ background:rgba(6,95,70,.05); transform:translateX(2px); }
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child{ display:none !important; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label p{ color:var(--mute) !important; font-weight:500; transition:color .15s; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover p,
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p{ color:var(--text) !important; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label::before{
  content:""; position:absolute; left:.85rem; top:50%; transform:translateY(-50%);
  width:17px;height:17px; background:var(--mute);
  -webkit-mask-repeat:no-repeat;mask-repeat:no-repeat;
  -webkit-mask-position:center;mask-position:center;
  -webkit-mask-size:contain;mask-size:contain; transition:background .2s ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked){
  background:linear-gradient(120deg,rgba(16,185,129,.20),rgba(13,148,136,.12));
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked)::before{ background:var(--pink); }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked)::after{
  content:""; position:absolute; left:0; top:20%; bottom:20%; width:3px; border-radius:0 3px 3px 0;
  background:linear-gradient(var(--violet),var(--pink));
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(1)::before{ -webkit-mask-image:var(--ic-grid);mask-image:var(--ic-grid); }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(2)::before{ -webkit-mask-image:var(--ic-scan);mask-image:var(--ic-scan); }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(3)::before{ -webkit-mask-image:var(--ic-chart);mask-image:var(--ic-chart); }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(4)::before{ -webkit-mask-image:var(--ic-flask);mask-image:var(--ic-flask); }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(5)::before{ -webkit-mask-image:var(--ic-info);mask-image:var(--ic-info); }
:root{
  --ic-grid:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='3' width='7' height='7' rx='1.5'/%3E%3Crect x='14' y='3' width='7' height='7' rx='1.5'/%3E%3Crect x='3' y='14' width='7' height='7' rx='1.5'/%3E%3Crect x='14' y='14' width='7' height='7' rx='1.5'/%3E%3C/svg%3E");
  --ic-scan:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M22 12h-4l-3 9L9 3l-3 9H2'/%3E%3C/svg%3E");
  --ic-chart:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='12' y1='20' x2='12' y2='10'/%3E%3Cline x1='18' y1='20' x2='18' y2='4'/%3E%3Cline x1='6' y1='20' x2='6' y2='14'/%3E%3C/svg%3E");
  --ic-flask:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M9 2v6l-5.4 9.3A2 2 0 0 0 5.3 21h13.4a2 2 0 0 0 1.7-3.7L15 8V2'/%3E%3Cpath d='M8 2h8'/%3E%3Cpath d='M7 15h10'/%3E%3C/svg%3E");
  --ic-info:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='10'/%3E%3Cline x1='12' y1='16' x2='12' y2='11'/%3E%3Cline x1='12' y1='8' x2='12.01' y2='8'/%3E%3C/svg%3E");
}

/* tech stack */
.tech{ display:flex; align-items:center; gap:7px; padding:.32rem .55rem; margin-bottom:.28rem;
  background:var(--panel); border:1px solid var(--panel-border); border-radius:8px;
  font-size:.68rem; color:var(--mute); animation:fadeR .5s ease backwards;
  transition:transform .18s ease, color .18s ease, border-color .18s ease; }
.tech:hover{ transform:translateX(3px); color:var(--text); border-color:var(--panel-border-hover); }
.tech svg{ width:12px;height:12px;flex:none; stroke:var(--violet); fill:none; }
.tech b{ color:var(--text); font-weight:600; }

.credit{ margin-top:1.2rem; display:flex; align-items:center; gap:8px; font-family:'Sora';
  font-size:.84rem; color:var(--mute); }
.credit b{ background:linear-gradient(110deg,#059669,#0d9488);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text; font-weight:700; }
.credit svg{ width:14px;height:14px; stroke:var(--pink); fill:none; }

.disclaimer{ margin-top:1.2rem; padding:.7rem .9rem; border-radius:11px;
  background:rgba(217,119,6,.10); border:1px solid rgba(217,119,6,.30); color:#b45309; font-size:.8rem; }

[data-testid="stFileUploader"] section{
  background:var(--panel) !important; border:1.5px dashed rgba(6,95,70,.22) !important; border-radius:14px;
}
.stButton>button, .stDownloadButton>button{
  background:linear-gradient(120deg,var(--indigo),var(--violet)); color:#fff !important; border:none;
  border-radius:11px; font-weight:600; }
/* slider — sleek thumb + gradient fill */
div[data-testid="stSlider"]{ padding-top:.1rem; }
div[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"]{
  background:#fff !important; border:2px solid var(--violet) !important; height:17px !important; width:17px !important;
  box-shadow:0 0 0 4px rgba(5,150,105,.20), 0 2px 8px rgba(6,95,70,.25) !important;
  transition:box-shadow .2s ease, transform .15s ease !important;
}
div[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"]:hover{
  box-shadow:0 0 0 6px rgba(5,150,105,.28), 0 3px 12px rgba(6,95,70,.32) !important; transform:scale(1.08); }
div[data-testid="stSlider"] [data-baseweb="slider"] > div > div{ background:rgba(6,95,70,.12) !important; height:6px !important; }
div[data-testid="stSlider"] [data-baseweb="slider"] > div > div > div{
  background:linear-gradient(90deg,var(--indigo),var(--violet),var(--pink)) !important; }
div[data-testid="stSlider"] [data-testid="stThumbValue"]{ color:var(--violet) !important; font-family:'Sora'; font-weight:600; background:transparent !important; }
[data-testid="stImage"] img{ border-radius:13px; border:1px solid var(--panel-border); }

@keyframes rise{ from{opacity:0; transform:translateY(10px);} to{opacity:1; transform:translateY(0);} }
@keyframes fadeR{ from{opacity:0; transform:translateX(-10px);} to{opacity:1; transform:translateX(0);} }

.foot{ text-align:center; color:var(--faint); font-size:.74rem; padding:3rem 0 .5rem;
  font-family:'Sora'; letter-spacing:.16em; }
</style>
"""
# Inject via st.html (NOT st.markdown) — markdown truncates a <style> block at the
# first blank line, which silently dropped ~95% of the CSS in earlier versions.
st.html(CSS)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_metrics() -> dict[str, Any] | None:
    if not METRICS_PATH.exists():
        return None
    with METRICS_PATH.open() as f:
        return json.load(f)

METRICS = load_metrics()


def find_models() -> list[str]:
    """Auto-detect .keras model files next to the app (resolves the symlink too).
    Returns a sorted list; always includes the default if it exists on disk."""
    found = {str(p) for p in pathlib.Path(".").glob("*.keras")}
    if pathlib.Path(MODEL_DEFAULT).exists():
        found.add(MODEL_DEFAULT)
    return sorted(found) or [MODEL_DEFAULT]


@contextmanager
def panel(title: str, eyebrow: str = ""):
    """A real card: native bordered container that correctly wraps widgets."""
    box = st.container(border=True)
    with box:
        head = ""
        if eyebrow:
            head += f"<div class='panel-eyebrow'>{eyebrow}</div>"
        head += f"<div class='panel-title'><span class='pt-dot'></span>{title}</div>"
        st.markdown(head, unsafe_allow_html=True)
        yield


def style_fig(fig: go.Figure, height: int | None = None) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#234a3c", size=13),
        margin=dict(l=8, r=8, t=10, b=8),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="rgba(6,95,70,.18)",
                        font=dict(family="Inter", color="#0f2e22")),
    )
    fig.update_xaxes(gridcolor="rgba(6,95,70,0.08)", zerolinecolor="rgba(6,95,70,0.14)",
                     tickfont=dict(color="#5c7468"), title_font=dict(color="#5c7468"))
    fig.update_yaxes(gridcolor="rgba(6,95,70,0.08)", zerolinecolor="rgba(6,95,70,0.14)",
                     tickfont=dict(color="#5c7468"), title_font=dict(color="#5c7468"))
    if height:
        fig.update_layout(height=height)
    return fig

PLOT_CFG = {"displayModeBar": False, "responsive": True}


# ─────────────────────────────────────────────────────────────────────────────
#  Model + Grad-CAM
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model(path: str):
    return tf.keras.models.load_model(path, compile=False)

def _find_backbone(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    raise ValueError("No nested backbone Model found.")

def _find_last_conv(base):
    for layer in reversed(base.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found.")

@st.cache_resource(show_spinner=False)
def build_cam_model(_path: str):
    model = load_model(_path)
    base = _find_backbone(model)
    last_conv = _find_last_conv(base)
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

def preprocess_image(pil_img: Image.Image):
    img = pil_img.convert("RGB").resize(IMG_DIMS)
    arr = np.asarray(img, dtype="float32")
    return arr, np.expand_dims(arr, 0)

def gradcam_heatmap(cam_model, raw_batch, pred_index):
    pre = resnet_preprocess(tf.identity(raw_batch))
    with tf.GradientTape() as tape:
        conv_out, preds = cam_model(pre)
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.squeeze(conv_out[0] @ pooled[..., tf.newaxis])
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return tf.image.resize(heatmap[..., tf.newaxis], IMG_DIMS).numpy().squeeze()

def overlay_heatmap(raw_arr, heatmap, alpha=0.45):
    colored = mcm.inferno(heatmap)[..., :3]
    base = raw_arr / 255.0
    return np.clip((1 - alpha) * base + alpha * colored, 0, 1)


def hot_region(heatmap, thr=0.5):
    """Stats for the high-activation region (heatmap is normalised to [0,1])."""
    mask = heatmap >= thr
    area = float(mask.mean())
    peak = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)  # (row, col)
    if mask.any():
        ys, xs = np.where(mask)
        centroid = (float(xs.mean()), float(ys.mean()))            # (x, y)
    else:
        centroid = None
    return mask, area, peak, centroid


def draw_contour(img_float, mask, color=(0.20, 1.0, 0.85)):
    """Outline the boolean mask boundary on an RGB float image (no cv2/scipy)."""
    m = mask
    edge = np.zeros_like(m, dtype=bool)
    edge[:-1, :] |= m[:-1, :] != m[1:, :]; edge[1:, :] |= m[:-1, :] != m[1:, :]
    edge[:, :-1] |= m[:, :-1] != m[:, 1:]; edge[:, 1:] |= m[:, :-1] != m[:, 1:]
    thick = edge.copy()                                            # 1px dilation
    thick[:-1, :] |= edge[1:, :]; thick[1:, :] |= edge[:-1, :]
    thick[:, :-1] |= edge[:, 1:]; thick[:, 1:] |= edge[:, :-1]
    out = img_float.copy()
    out[thick] = color
    return np.clip(out, 0, 1)


# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────────────────────
_ICON = {
    "model":    "<path d='m12 2 9 5-9 5-9-5 9-5Z'/><path d='m3 12 9 5 9-5'/><path d='m3 17 9 5 9-5'/>",
    "engine":   "<rect x='4' y='4' width='16' height='16' rx='2'/><rect x='9' y='9' width='6' height='6'/><path d='M9 2v2M15 2v2M9 20v2M15 20v2M2 9h2M2 15h2M20 9h2M20 15h2'/>",
    "cam":      "<path d='M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7Z'/><circle cx='12' cy='12' r='3'/>",
    "chart":    "<path d='M3 3v18h18'/><path d='m19 9-5 5-4-4-3 3'/>",
    "bolt":     "<path d='M13 2 3 14h9l-1 8 10-12h-9l1-8Z'/>",
    "ensemble": "<rect x='9' y='9' width='13' height='13' rx='2'/><path d='M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1'/>",
}
def _svg(key: str) -> str:
    return (f"<svg viewBox='0 0 24 24' stroke-width='2' stroke-linecap='round' "
            f"stroke-linejoin='round'>{_ICON[key]}</svg>")

with st.sidebar:
    st.markdown(
        "<div class='brand'>"
        "<div class='brand-logo'><svg viewBox='0 0 24 24' stroke-width='2' stroke-linecap='round' "
        "stroke-linejoin='round'><path d='M12 5a3 3 0 0 0-3 3 3 3 0 0 0-1 5.8V18a2 2 0 0 0 4 0'/>"
        "<path d='M12 5a3 3 0 0 1 3 3 3 3 0 0 1 1 5.8V18a2 2 0 0 1-4 0'/></svg></div>"
        "<div><div class='brand-name'>PixelMind</div><div class='brand-ver'>v1.0 · MRI Atlas</div></div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    page = st.radio("Navigate", ["Overview", "Diagnose", "Analytics", "Model Lab", "About"],
                    label_visibility="collapsed")

    st.markdown("<hr style='border-color:rgba(6,95,70,.12); margin:1.3rem 0 1rem;'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='side-h'>Models &amp; Tech</div>"
        f"<div class='tech' style='animation-delay:.04s'>{_svg('model')}<span><b>ResNet50</b> · fine-tuned</span></div>"
        f"<div class='tech' style='animation-delay:.09s'>{_svg('engine')}<span><b>TensorFlow</b> / Keras</span></div>"
        f"<div class='tech' style='animation-delay:.14s'>{_svg('cam')}<span><b>Grad-CAM</b> explainability</span></div>"
        f"<div class='tech' style='animation-delay:.19s'>{_svg('ensemble')}<span><b>Ensemble</b> + TTA ×4</span></div>"
        f"<div class='tech' style='animation-delay:.24s'>{_svg('chart')}<span><b>Plotly</b> analytics</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='credit'><svg viewBox='0 0 24 24' stroke-width='2' stroke-linecap='round' "
        "stroke-linejoin='round'><path d='M12 3l2.1 5.5L20 9.3l-4.3 3.7L17 19l-5-3.2L7 19l1.3-6L4 9.3l5.9-.8Z'/></svg>"
        "Built by <b>Karzan</b></div>",
        unsafe_allow_html=True,
    )

    st.markdown("<hr style='border-color:rgba(6,95,70,.12); margin:1.2rem 0 .9rem;'>", unsafe_allow_html=True)
    st.markdown("<div class='side-h'>Active model</div>", unsafe_allow_html=True)
    _models = find_models()
    model_path = st.selectbox(
        "Active model", _models,
        index=_models.index(MODEL_DEFAULT) if MODEL_DEFAULT in _models else 0,
        format_func=lambda p: pathlib.Path(p).stem.replace("_", " "),
        label_visibility="collapsed",
    )
    if len(_models) == 1:
        st.caption("Only one model found. Train & save more backbones in the "
                   "notebook and they'll appear here automatically.")
    st.markdown(
        "<div class='disclaimer'>⚠️ Research / educational use only — not a medical device.</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Hero
# ─────────────────────────────────────────────────────────────────────────────
final_name = (METRICS or {}).get("final_model_name", "ResNet50 (fine-tuned)")
st.markdown(
    f"""
    <div class="hero">
      <div class="hero-kicker"><span class="dot"></span>PixelMind · {final_name}</div>
      <div class="hero-title">See the brain. Read the signal.</div>
      <div class="hero-sub">A four-class brain-MRI classifier with Grad-CAM explainability and a live
      analytics deck — confusion matrix, ROC curves and the full experiment leaderboard, straight from
      the training notebook.</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Pages
# ─────────────────────────────────────────────────────────────────────────────
def kpi_row(m: dict):
    tm, ds = m["test_metrics"], m["dataset"]
    cards = [
        ("Test Accuracy", f"{tm['accuracy']*100:.2f}%",
         f"95% CI {tm['accuracy_ci95'][0]*100:.1f}–{tm['accuracy_ci95'][1]*100:.1f}%"),
        ("Macro F1", f"{tm['macro_f1']*100:.2f}%",
         f"95% CI {tm['macro_f1_ci95'][0]*100:.1f}–{tm['macro_f1_ci95'][1]*100:.1f}%"),
        ("Macro AUC", f"{tm['macro_auc']*100:.2f}%", "one-vs-rest · test"),
        ("Test / Train", f"{ds['test_total']:,} · {ds['train_total']:,}", f"val {ds['val_total']:,}"),
    ]
    html = "<div class='kpi-grid'>"
    for label, value, sub in cards:
        html += (f"<div class='kpi'><div class='kpi-label'>{label}</div>"
                 f"<div class='kpi-value'>{value}</div><div class='kpi-sub'>{sub}</div></div>")
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_overview():
    if METRICS is None:
        st.warning("`metrics.json` not found — run notebook cell **12b** to populate the dashboard.")
        return
    kpi_row(METRICS)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1.2, 1], gap="medium")
    with c1:
        with panel("Precision · Recall · F1", "Per-class quality"):
            pc = METRICS["per_class"]; classes = list(pc.keys())
            fig = go.Figure()
            for (key, label), color in zip([("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")],
                                           [INDIGO, VIOLET, PINK]):
                fig.add_bar(x=classes, y=[pc[c][key] for c in classes], name=label,
                            marker=dict(color=color, line=dict(width=0)),
                            hovertemplate="<b>%{x}</b><br>" + label + ": %{y:.3f}<extra></extra>")
            fig.update_layout(barmode="group", bargap=.28, yaxis=dict(range=[.85, 1.0], tickformat=".0%"),
                              legend=dict(orientation="h", y=1.12, x=1, xanchor="right"))
            st.plotly_chart(style_fig(fig, 360), use_container_width=True, config=PLOT_CFG)
    with c2:
        with panel("Test set composition", "Dataset"):
            ds = METRICS["dataset"]
            donut = go.Figure(go.Pie(
                labels=list(ds["test_per_class"].keys()), values=list(ds["test_per_class"].values()),
                hole=.64, marker=dict(colors=[ACCENT[c] for c in CLASSES], line=dict(color="#ffffff", width=2)),
                textinfo="label+percent", textfont=dict(family="Inter", size=12, color="#234a3c"),
                hovertemplate="<b>%{label}</b><br>%{value} scans (%{percent})<extra></extra>"))
            donut.add_annotation(text=f"<b>{ds['test_total']:,}</b><br><span style='font-size:11px;color:#5c7468'>scans</span>",
                                 showarrow=False, font=dict(family="Sora", size=20, color="#0b3b2c"))
            donut.update_layout(showlegend=False)
            st.plotly_chart(style_fig(donut, 360), use_container_width=True, config=PLOT_CFG)


def render_diagnose():
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Could not load model from **{model_path}**. Download `ResNet50_finetuned.keras` "
                 f"from the training notebook and place it next to this app.\n\nDetails: {e}")
        return

    with panel("Drop a brain MRI scan", "Inference"):
        uploaded = st.file_uploader("Upload MRI scan",
                                    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
                                    label_visibility="collapsed")

    if uploaded is None:
        st.info("⬆️ Upload an MRI scan to get a prediction. Supported: PNG · JPG · BMP · TIFF")
        return

    pil_img = Image.open(io.BytesIO(uploaded.read()))
    raw_arr, batch = preprocess_image(pil_img)
    # This saved model does NOT bake in preprocessing, so apply ResNet preprocess_input
    # before predicting. (Feeding raw [0,255] collapses everything to "notumor".)
    # Use a copy — `batch` stays raw for Grad-CAM, which preprocesses internally.
    probs = model.predict(resnet_preprocess(np.copy(batch)), verbose=0)[0]
    pred_idx = int(np.argmax(probs)); pred_cls = CLASSES[pred_idx]; conf = float(probs[pred_idx])

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1.15], gap="medium")
    with c1:
        with panel("Scan preview", "Input"):
            st.image(pil_img, use_container_width=True)
    with c2:
        st.markdown(
            f"<div class='pred'><div class='pred-eyebrow'>Predicted class</div>"
            f"<div class='pred-value'>{pred_cls.upper()}</div>"
            f"<div class='pred-desc'>{CLASS_INFO[pred_cls]}</div>"
            f"<div class='pred-chip'>Confidence · {conf*100:.2f}%</div></div>",
            unsafe_allow_html=True,
        )
        bars = "<div class='pb'>"
        for i in np.argsort(probs)[::-1]:
            pct = probs[i] * 100
            bars += (f"<div class='pb-row'><span class='nm'>{CLASSES[i]}</span>"
                     f"<span class='vl'>{pct:.2f}%</span></div>"
                     f"<div class='pb-track'><div class='pb-fill' style='width:{pct:.2f}%'></div></div>")
        bars += "</div>"
        st.markdown(bars, unsafe_allow_html=True)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    g1, g2 = st.columns([1.4, 1], gap="medium")
    with g1:
        with panel("Grad-CAM — where the model looked", "Explainability"):
            try:
                cam_model = build_cam_model(model_path)
                heatmap = gradcam_heatmap(cam_model, batch, pred_idx)
                s1, s2 = st.columns(2)
                alpha = s1.slider("Overlay opacity", 0.10, 0.90, 0.45, 0.05)
                thr = s2.slider("Hot-region threshold", 0.30, 0.80, 0.50, 0.05)
                overlay = overlay_heatmap(raw_arr, heatmap, alpha=alpha)
                mask, area, peak, centroid = hot_region(heatmap, thr)
                overlay_c = draw_contour(overlay, mask)
                i1, i2 = st.columns(2)
                with i1:
                    st.image(raw_arr.astype("uint8"), caption="Input (128×128)", use_container_width=True)
                with i2:
                    st.image(overlay_c, caption=f"Grad-CAM + contour — {pred_cls}",
                             use_container_width=True, clamp=True)
                r1, r2, r3 = st.columns(3)
                r1.metric("Hot-region area", f"{area*100:.1f}%")
                r2.metric("Peak intensity", f"({peak[1]}, {peak[0]})")
                r3.metric("Focus centroid",
                          f"({centroid[0]:.0f}, {centroid[1]:.0f})" if centroid else "—")
                st.caption("Cyan contour outlines where activation ≥ threshold. "
                           "Area = share of the scan the model focused on. Coordinates in 128×128 px. "
                           "Qualitative check, not a clinical metric.")
            except Exception as e:
                st.info(f"Grad-CAM unavailable for this model: {e}")
    with g2:
        with panel("Confidence radar", "Distribution"):
            radar = go.Figure(go.Scatterpolar(
                r=list(probs) + [probs[0]], theta=CLASSES + [CLASSES[0]], fill="toself",
                fillcolor="rgba(16,185,129,0.20)", line=dict(color=VIOLET, width=2),
                marker=dict(color=PINK, size=7), hovertemplate="<b>%{theta}</b><br>%{r:.3f}<extra></extra>"))
            radar.update_layout(showlegend=False, polar=dict(bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(range=[0, 1], gridcolor="rgba(6,95,70,.12)",
                                tickfont=dict(color="#5c7468", size=10), tickformat=".0%"),
                angularaxis=dict(gridcolor="rgba(6,95,70,.12)",
                                 tickfont=dict(color="#234a3c", family="Sora", size=11))))
            st.plotly_chart(style_fig(radar, 360), use_container_width=True, config=PLOT_CFG)


def render_analytics():
    if METRICS is None:
        st.warning("Run notebook cell **12b** to generate `metrics.json` for analytics.")
        return

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with panel("Where errors live", "Confusion"):
            cm = np.asarray(METRICS["confusion_matrix"], dtype=int)
            cmn = cm / cm.sum(axis=1, keepdims=True)
            txt = [[f"<b>{cm[i, j]}</b><br><span style='font-size:10px;opacity:.7'>{cmn[i, j]*100:.1f}%</span>"
                    for j in range(len(CLASSES))] for i in range(len(CLASSES))]
            hm = go.Figure(go.Heatmap(
                z=cmn, x=CLASSES, y=CLASSES,
                colorscale=[[0, "rgba(16,185,129,.05)"], [.5, "rgba(16,185,129,.42)"], [1, "rgba(5,120,87,.92)"]],
                text=txt, texttemplate="%{text}", textfont=dict(family="Sora", color="#0a3325", size=13),
                customdata=cm, hovertemplate="True <b>%{y}</b> · Pred <b>%{x}</b><br>%{customdata} scans<extra></extra>",
                showscale=False))
            hm.update_layout(xaxis=dict(title="Predicted", side="bottom"),
                             yaxis=dict(title="True", autorange="reversed"))
            st.plotly_chart(style_fig(hm, 400), use_container_width=True, config=PLOT_CFG)
    with c2:
        with panel("ROC · class by class", "Separability"):
            roc = METRICS["roc_curves"]
            rf = go.Figure()
            for cls, color in ACCENT.items():
                d = roc[cls]
                rf.add_scatter(x=d["fpr"], y=d["tpr"], mode="lines", name=f"{cls} · {d['auc']:.4f}",
                               line=dict(color=color, width=2.5, shape="spline"),
                               hovertemplate=f"<b>{cls}</b><br>FPR %{{x:.3f}} · TPR %{{y:.3f}}<extra></extra>")
            rf.add_scatter(x=[0, 1], y=[0, 1], mode="lines",
                           line=dict(color="rgba(6,95,70,.28)", width=1, dash="dot"),
                           showlegend=False, hoverinfo="skip")
            rf.update_layout(xaxis=dict(title="False Positive Rate", range=[-.02, 1.02]),
                             yaxis=dict(title="True Positive Rate", range=[0, 1.02]),
                             legend=dict(orientation="h", y=-.28, x=.5, xanchor="center"))
            st.plotly_chart(style_fig(rf, 400), use_container_width=True, config=PLOT_CFG)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    with panel("Recall per class — what would be missed", "Clinical sensitivity"):
        pc = METRICS["per_class"]
        bf = go.Figure(go.Bar(
            y=list(pc.keys()), x=[pc[c]["recall"] for c in pc], orientation="h",
            marker=dict(color=[ACCENT[c] for c in pc]),
            text=[f"{pc[c]['recall']*100:.2f}%" for c in pc], textposition="outside",
            textfont=dict(family="Sora", color="#234a3c"),
            hovertemplate="<b>%{y}</b><br>Recall %{x:.3f}<extra></extra>"))
        bf.update_layout(xaxis=dict(range=[.9, 1.03], tickformat=".0%"), yaxis=dict(autorange="reversed"))
        st.plotly_chart(style_fig(bf, 280), use_container_width=True, config=PLOT_CFG)
        st.caption("Missing a tumor (low recall) is far costlier than a false alarm — we report it first.")


def render_model_lab():
    if METRICS is None or not METRICS.get("leaderboard"):
        st.warning("No leaderboard yet — run notebook cell **12b** to populate the experiment history.")
        return
    df = pd.DataFrame(METRICS["leaderboard"]).sort_values("Val Macro-F1", ascending=False).reset_index(drop=True)

    c1, c2 = st.columns([1.05, 1], gap="medium")
    with c1:
        with panel("Every model tried", "Leaderboard"):
            show = df[["Model", "Val Accuracy", "Val Macro-F1", "Val Macro AUC", "Params", "Train Time (s)"]].copy()
            show["Params"] = (show["Params"] / 1e6).round(2).astype(str) + "M"
            st.dataframe(show, hide_index=True, use_container_width=True, height=330,
                column_config={
                    "Val Accuracy":  st.column_config.ProgressColumn(format="%.4f", min_value=.6, max_value=1.0),
                    "Val Macro-F1":  st.column_config.ProgressColumn(format="%.4f", min_value=.6, max_value=1.0),
                    "Val Macro AUC": st.column_config.ProgressColumn(format="%.4f", min_value=.6, max_value=1.0),
                })
    with c2:
        with panel("Capacity vs quality", "Pareto"):
            sc = go.Figure()
            sc.add_scatter(x=df["Params"] / 1e6, y=df["Val Macro-F1"], mode="markers+text",
                text=df["Model"], textposition="top center",
                textfont=dict(family="Inter", size=9, color="#5c7468"),
                marker=dict(size=np.sqrt(df["Train Time (s)"]) * 1.7, color=df["Val Macro-F1"],
                            colorscale=[[0, INDIGO], [.5, VIOLET], [1, PINK]], showscale=False,
                            line=dict(color="rgba(6,95,70,.25)", width=1)),
                hovertemplate="<b>%{text}</b><br>%{x:.2f}M params · F1 %{y:.4f}<extra></extra>")
            sc.update_layout(xaxis=dict(title="Parameters (M)", type="log"),
                             yaxis=dict(title="Val Macro-F1", tickformat=".3f"))
            st.plotly_chart(style_fig(sc, 330), use_container_width=True, config=PLOT_CFG)


def render_about():
    with panel("How PixelMind works", "About"):
        st.markdown(
            """
PixelMind is the inference + analytics surface for a four-class brain-MRI classifier
(**glioma · meningioma · notumor · pituitary**) trained in the companion notebook.

The notebook trains a baseline CNN, an augmented CNN, and three transfer backbones
(ResNet50, EfficientNetB0, DenseNet121), then fine-tunes all three with the top 120 layers
unfrozen at `lr = 5e-5` while keeping BatchNorm in inference mode. The final predictor is
chosen on **validation Macro-F1** — the test set is touched once at the end, with TTA ×4 and
95% bootstrap confidence intervals.

Dashboard numbers load from `metrics.json`, regenerated by running notebook cell **12b**.
Re-train → re-run that cell → reload here.
            """
        )
        a, b, c = st.columns(3)
        a.metric("Input", "128×128 RGB")
        b.metric("Explainability", "Grad-CAM")
        c.metric("Honesty", "Bootstrap CI 95%")


# ─────────────────────────────────────────────────────────────────────────────
#  Route
# ─────────────────────────────────────────────────────────────────────────────
if page == "Overview":
    render_overview()
elif page == "Diagnose":
    render_diagnose()
elif page == "Analytics":
    render_analytics()
elif page == "Model Lab":
    render_model_lab()
else:
    render_about()

st.markdown("<div class='foot'>PIXELMIND · STREAMLIT · TENSORFLOW · PLOTLY · BUILT BY KARZAN</div>",
            unsafe_allow_html=True)
