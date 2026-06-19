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

A professional, multi-page Streamlit app (`app.py`):

- **Analysis** — upload an MRI scan → predicted class, per-class probabilities
  (bars coloured by tumor risk: green → yellow → orange → red), and a Grad-CAM
  heatmap.
- **Model Dashboard** — real test-set metrics (accuracy, macro-F1, per-class
  scores, confusion matrix) from `model_metrics.json`.
- **About** — method, data, disclaimer.

The trained model (`ResNet50_finetuned.keras`) and its metrics
(`model_metrics.json`) are **included in this repo**, so the app runs right away
— no training or dataset download required:

```bash
pip install -r requirements.txt
./run_app.sh            # recommended launcher (sets the protobuf env var)
# or: streamlit run app.py
```

To **retrain from scratch** (downloads the dataset, rewrites model + metrics):

```bash
gdown 1bXBSfKDaItFigHa5QfcnyTXADG2wlWJj -O brain_tumor.zip && unzip -q -o brain_tumor.zip
python train_model.py
```

`train_model.py` reproduces a deployable model on CPU in a few minutes
(~93% test accuracy). See `README_app.md` for details.

## Dataset

Brain Tumor MRI Dataset (4 classes). The **dataset is not committed** (see
`.gitignore`) — download it with the `gdown` command above only if you want to
retrain. The trained model needed to run the app is already in the repo.

## Disclaimer

Research and educational use only. This is **not** a medical device and must not be
used for real diagnosis.
