# Brain Tumor MRI Classifier — Streamlit App

A simple web app: upload a brain MRI image and the fine-tuned **ResNet50** model
predicts one of `glioma`, `meningioma`, `notumor`, `pituitary`, with a **Grad-CAM**
heatmap showing where the model looked.

## 1. Export the model from the notebook (Colab)

After the notebook finishes training, the fine-tuned ResNet50 is already saved as
`ResNet50_finetuned.keras` (by the `ModelCheckpoint` callback). Download it:

```python
from google.colab import files
files.download('ResNet50_finetuned.keras')
```

Put that file in the **same folder as `app.py`**.

## 2. Install & run (on your computer)

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser. Upload an MRI scan and you get the prediction,
per-class probabilities, and the Grad-CAM overlay.

## Notes

- The model contains its own preprocessing, so the app just resizes images to
  128×128 — no manual normalization.
- If the model file has a different name/location, set the path in the sidebar.
- **Research/educational use only.** This is not a medical device and must not be
  used for real diagnosis.
