"""Train the Brain Tumor MRI classifier and export everything the app needs.

Strategy (CPU-friendly): the ResNet50 backbone is frozen, so we run it over the
data ONCE to extract pooled features, then train a small classifier head on
those features (seconds instead of minutes per epoch). Finally we assemble the
full model the Streamlit app expects — raw [0,255] input -> ResNet50 backbone ->
GAP -> head -> 4-class softmax — and evaluate it on the held-out Testing/ set.

Outputs (project root):
  ResNet50_finetuned.keras   the model app.py loads
  model_metrics.json         accuracy / macro-F1 / per-class / confusion matrix

Preprocessing contract: the saved model does NOT preprocess internally. The app
applies tensorflow.keras.applications.resnet.preprocess_input before predict
(same as the Grad-CAM path already does).

Run:  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 train_model.py
"""
import os
import json
import time
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from sklearn.metrics import (confusion_matrix, f1_score, recall_score,
                             precision_score, accuracy_score)

SEED = 42
tf.keras.utils.set_random_seed(SEED)

IMG_DIMS = (128, 128)
IMG_SHAPE = IMG_DIMS + (3,)
BATCH = 32
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASSES)
TRAIN_DIR = "Training"
TEST_DIR = "Testing"

t0 = time.time()
print("TF", tf.__version__, "| GPUs:", tf.config.list_physical_devices("GPU"))

# --- data ---------------------------------------------------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, validation_split=0.2, subset="training", class_names=CLASSES,
    seed=SEED, image_size=IMG_DIMS, batch_size=BATCH)
val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, validation_split=0.2, subset="validation", class_names=CLASSES,
    seed=SEED, image_size=IMG_DIMS, batch_size=BATCH)
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, class_names=CLASSES, seed=SEED, image_size=IMG_DIMS,
    batch_size=BATCH, shuffle=False)

# --- frozen backbone + feature extractor -------------------------------------
backbone = ResNet50(include_top=False, weights="imagenet", input_shape=IMG_SHAPE)
backbone.trainable = False

fin = layers.Input(IMG_SHAPE)
fx = preprocess_input(fin)              # raw [0,255] -> caffe-preprocessed
fx = backbone(fx, training=False)
fx = layers.GlobalAveragePooling2D()(fx)
feat_extractor = Model(fin, fx, name="feat_extractor")


def extract(ds, name):
    xs, ys = [], []
    for i, (imgs, labels) in enumerate(ds):
        xs.append(feat_extractor.predict(imgs, verbose=0))
        ys.append(labels.numpy())
        if i % 20 == 0:
            print(f"  [{name}] batch {i}", flush=True)
    return np.concatenate(xs), np.concatenate(ys)


print("Extracting features (frozen ResNet50, one pass)...")
Xtr, ytr = extract(train_ds, "train")
Xva, yva = extract(val_ds, "val")
Xte, yte = extract(test_ds, "test")
print(f"features: train{Xtr.shape} val{Xva.shape} test{Xte.shape}  "
      f"({time.time()-t0:.0f}s)")

# --- classifier head ----------------------------------------------------------
head = Sequential([
    layers.Input((Xtr.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation="softmax"),
], name="head")
head.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
             loss="sparse_categorical_crossentropy", metrics=["accuracy"])
es = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6,
                                      restore_best_weights=True)
head.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=40, batch_size=64,
         callbacks=[es], verbose=2)

# --- assemble full model the app loads ---------------------------------------
inp = layers.Input(IMG_SHAPE)                 # raw [0,255]; app applies preprocess
feat = backbone(inp)                          # nested Model -> find_backbone()
feat = layers.GlobalAveragePooling2D()(feat)
out = head(feat)
model = Model(inp, out, name="ResNet50_finetuned")
model.save("ResNet50_finetuned.keras")
print("saved ResNet50_finetuned.keras")

# --- evaluate on held-out test set -------------------------------------------
yprob = head.predict(Xte, verbose=0)
ypred = yprob.argmax(1)
acc = accuracy_score(yte, ypred)
macro_f1 = f1_score(yte, ypred, average="macro")
per_recall = recall_score(yte, ypred, average=None, labels=range(NUM_CLASSES))
per_prec = precision_score(yte, ypred, average=None, labels=range(NUM_CLASSES))
per_f1 = f1_score(yte, ypred, average=None, labels=range(NUM_CLASSES))
cm = confusion_matrix(yte, ypred, labels=range(NUM_CLASSES))

metrics = {
    "classes": CLASSES,
    "n_train": int(len(ytr)), "n_val": int(len(yva)), "n_test": int(len(yte)),
    "accuracy": float(acc),
    "macro_f1": float(macro_f1),
    "per_class": {
        CLASSES[i]: {"precision": float(per_prec[i]),
                     "recall": float(per_recall[i]),
                     "f1": float(per_f1[i]),
                     "support": int((yte == i).sum())}
        for i in range(NUM_CLASSES)
    },
    "confusion_matrix": cm.tolist(),
    "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "train_seconds": round(time.time() - t0, 1),
}
with open("model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nTEST accuracy={acc:.4f}  macro_f1={macro_f1:.4f}  "
      f"({time.time()-t0:.0f}s total)")
print("wrote model_metrics.json")
