# inference_exp1_baseline.py
# Inference code for Experiment 1 (Baseline)
# Loads: examples/exp1_baseline_model.keras + examples/exp1_baseline_preprocess.pkl
# Uses: the SAME one-hot columns + SAME scaler + SAME threshold

import os
import pickle
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt

# -----------------------------
# Config (change only paths if needed)
# -----------------------------
MODEL_PATH = "examples/exp1_baseline_model.keras"
PREP_PATH  = "examples/exp1_baseline_preprocess.pkl"

# Put the file you want to predict here:
# - If you want to evaluate like test: use the SAME cleaned file.
DATA_PATH = "data/data_clean.csv"

OUT_DIR = "data/output_exp1_infer"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# Load preprocess objects
# -----------------------------
with open(PREP_PATH, "rb") as f:
    prep = pickle.load(f)

TARGET = prep["target"]
cat_cols = prep["cat_cols"]
feature_columns = prep["feature_columns"]
scaler = prep["scaler"]
threshold = prep.get("threshold", 0.50)

print("Loaded preprocess:")
print(" - target:", TARGET)
print(" - threshold:", threshold)
print(" - #features:", len(feature_columns))

# -----------------------------
# Load model
# -----------------------------
print("\nLoading model...")
model = keras.models.load_model(MODEL_PATH)
print("Model loaded:", MODEL_PATH)

# -----------------------------
# Load data
# -----------------------------
print("\nLoading data for inference...")
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)

# If target exists, we can evaluate. If not, we only predict.
has_target = TARGET in df.columns

if has_target:
    y_true_raw = df[TARGET]
    X = df.drop(columns=[TARGET])
else:
    X = df.copy()

# -----------------------------
# One-hot encode using SAME columns as training
# -----------------------------
# (Important) get_dummies might create different columns if categories differ,
# so we align with training columns using reindex.
X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# Align columns to training feature_columns
X_enc = X_enc.reindex(columns=feature_columns, fill_value=0)

# -----------------------------
# Scale using SAME scaler (fit on training only)
# -----------------------------
X_scaled = scaler.transform(X_enc)

# -----------------------------
# Predict
# -----------------------------
proba = model.predict(X_scaled, verbose=0).flatten()
pred = (proba >= threshold).astype(int)

# Save predictions CSV (always useful evidence)
out_csv = os.path.join(OUT_DIR, "predictions_exp1.csv")
results = X.copy()
results["pred_proba"] = proba
results["pred_label"] = pred
if has_target:
    results["true_label"] = y_true_raw.values
results.to_csv(out_csv, index=False)
print("\nSaved predictions:", out_csv)

# -----------------------------
# If ground-truth exists: evaluate + confusion matrix
# -----------------------------
if has_target:
    # If y is "Yes/No" or strings, convert to 0/1 (same as training idea)
    # In your training you used LabelEncoder. For baseline report, usually ProdTaken is already 0/1.
    # We'll handle both cases safely:
    try:
        y_true = y_true_raw.astype(int).values
    except:
        # fallback: map common strings
        y_true = y_true_raw.map({"yes": 1, "no": 0, "Yes": 1, "No": 0}).astype(int).values

    acc = accuracy_score(y_true, pred)
    prec = precision_score(y_true, pred, zero_division=0)
    rec = recall_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, proba)
    except:
        auc = float("nan")

    print("\n" + "=" * 60)
    print("INFERENCE (EXP1 BASELINE) - RESULTS")
    print("=" * 60)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, pred, target_names=["0", "1"]))

    cm = confusion_matrix(y_true, pred)
    print("\nConfusion Matrix:\n", cm)

    # Save confusion matrix image (evidence)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix - Exp1 Baseline (INFERENCE)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    cm_path = os.path.join(OUT_DIR, "confusion_matrix_exp1_infer.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print("\nSaved confusion matrix:", cm_path)

else:
    print("\nNo target column found in file, so I only saved predictions (no evaluation).")