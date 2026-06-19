# Brain Tumor MRI Classifier

Multi-class classification of brain MRI scans into **glioma**, **meningioma**,
**notumor**, and **pituitary**, using transfer learning with a fine-tuned
ResNet50, plus a Streamlit app for interactive inference with Grad-CAM
explainability.

## Results

| Metric        | Test score |
| ------------- | ---------- |
| Accuracy      | 0.9863     |
| Macro-F1      | 0.9856     |
| Macro AUC     | 0.9995     |

Final predictor: **ResNet50 (fine-tuned)**, selected purely on the validation set.
The test set was evaluated only once. 95% bootstrap confidence intervals are
reported in the notebook.

## Method

1. **Split** — `Training/` is split 80/20 into train/validation (`seed=42`); the
   original `Testing/` folder is the held-out test set.
2. **Golden rule** — no decision is made by looking at the test set; it is touched
   exactly once, at the end.
3. **Selection metric** — Macro-F1 on validation. Clinical metric — per-class recall.
4. **Pipeline** — baseline CNN → CNN + augmentation → compare ResNet50 /
   EfficientNetB0 / DenseNet121 (frozen) → fine-tune (BatchNorm frozen) →
   ensemble + TTA + bootstrap CIs → single test evaluation → Grad-CAM.

See `Brain_Tumor_Detector_Solution_HighAccuracy.ipynb` for the full pipeline.

## App

A Streamlit app (`app.py`) lets you upload an MRI scan and get the predicted class,
per-class probabilities, and a Grad-CAM heatmap.

```bash
pip install -r requirements.txt
# place ResNet50_finetuned.keras (exported from the notebook) next to app.py
streamlit run app.py
```

See `README_app.md` for details.

## Dataset

Brain Tumor MRI Dataset (4 classes). The model file and dataset are not committed
(see `.gitignore`) — export the trained model from the notebook before running the app.

## Disclaimer

Research and educational use only. This is **not** a medical device and must not be
used for real diagnosis.
