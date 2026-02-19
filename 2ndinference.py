import os, pickle
import numpy as np
import pandas as pd

from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt

MODEL_PATH = "examples/exp2_improved_model.keras"
PREP_PATH  = "examples/exp2_improved_preprocess.pkl"
DATA_PATH  = "data/data_clean.csv"
OUT_DIR    = "data/output_exp2"

os.makedirs(OUT_DIR, exist_ok=True)

with open(PREP_PATH, "rb") as f:
    prep = pickle.load(f)

model = keras.models.load_model(MODEL_PATH)

df = pd.read_csv(DATA_PATH)
TARGET = prep["target"]

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_enc = pd.get_dummies(X, columns=prep["cat_cols"], drop_first=True)
X_enc = X_enc.reindex(columns=prep["feature_columns"], fill_value=0)

X_scaled = prep["scaler"].transform(X_enc)
y_enc = prep["label_encoder"].transform(y)

proba = model.predict(X_scaled, verbose=0).flatten()
threshold = prep["threshold"]
pred = (proba >= threshold).astype(int)

acc = accuracy_score(y_enc, pred)
prec = precision_score(y_enc, pred, zero_division=0)
rec = recall_score(y_enc, pred, zero_division=0)
f1 = f1_score(y_enc, pred, zero_division=0)
auc = roc_auc_score(y_enc, proba)

print("\n" + "="*60)
print("INFERENCE (EXP2 IMPROVED)")
print("="*60)
print(f"Threshold: {threshold:.2f}")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"AUC      : {auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_enc, pred, target_names=["0","1"]))

cm = confusion_matrix(y_enc, pred)
print("\nConfusion Matrix:\n", cm)

plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix - Exp2 Improved (INFER)")
plt.xlabel("Predicted")
plt.ylabel("True")
for (i, j), v in np.ndenumerate(cm):
    plt.text(j, i, str(v), ha="center", va="center")
path = os.path.join(OUT_DIR, "confusion_matrix_exp2_infer.png")
plt.tight_layout()
plt.savefig(path, dpi=300)
plt.close()
print("Saved:", path)