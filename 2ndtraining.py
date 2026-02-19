import os, pickle
import numpy as np
import pandas as pd

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt

DATA_PATH = "data/data_clean.csv"
TARGET = "ProdTaken"

OUT_DIR = "data/output_exp2"
MODEL_PATH = "examples/exp2_improved_model.keras"
PREP_PATH = "examples/exp2_improved_preprocess.pkl"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("examples", exist_ok=True)

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=[TARGET])
y = df[TARGET]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=True)
feature_columns = X_enc.columns.tolist()

# Split first: Train+Val 75%, Test 25%
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_enc, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
)

# Split again: Train 80% of trainval, Val 20% of trainval
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval
)

print(f"Split sizes -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# Class weights (imbalanced)
classes = np.unique(y_train)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
print("Class weights:", class_weight)

# Model (slightly stronger)
model = keras.Sequential([
    keras.layers.Input(shape=(X_train_s.shape[1],)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc")]
)

history = model.fit(
    X_train_s, y_train,
    validation_data=(X_val_s, y_val),
    epochs=200,
    batch_size=32,
    verbose=2,
    class_weight=class_weight,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1)
    ]
)

# -----------------------------
# THRESHOLD TUNING on VAL (maximize F1)
# -----------------------------
val_proba = model.predict(X_val_s, verbose=0).flatten()

thresholds = np.linspace(0.05, 0.95, 91)
best_t, best_f1 = 0.50, -1

for t in thresholds:
    val_pred = (val_proba >= t).astype(int)
    f1 = f1_score(y_val, val_pred, zero_division=0)
    if f1 > best_f1:
        best_f1, best_t = f1, t

print("\nBest threshold on VAL:", round(best_t, 2))
print("VAL F1:", round(best_f1, 4))

# Plot threshold vs F1 (good evidence)
f1_list = []
for t in thresholds:
    f1_list.append(f1_score(y_val, (val_proba >= t).astype(int), zero_division=0))

plt.figure()
plt.plot(thresholds, f1_list)
plt.title("Threshold vs F1 (Validation)")
plt.xlabel("Threshold")
plt.ylabel("F1")
thr_plot = os.path.join(OUT_DIR, "threshold_vs_f1.png")
plt.savefig(thr_plot, dpi=300, bbox_inches="tight")
plt.close()
print("Saved:", thr_plot)

# -----------------------------
# TEST evaluation using best threshold
# -----------------------------
test_proba = model.predict(X_test_s, verbose=0).flatten()
test_pred = (test_proba >= best_t).astype(int)

acc = accuracy_score(y_test, test_pred)
prec = precision_score(y_test, test_pred, zero_division=0)
rec = recall_score(y_test, test_pred, zero_division=0)
f1 = f1_score(y_test, test_pred, zero_division=0)
auc = roc_auc_score(y_test, test_proba)

print("\n" + "="*60)
print("EXPERIMENT 2 (IMPROVED) - TEST RESULTS")
print("="*60)
print(f"Threshold: {best_t:.2f}")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"AUC      : {auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, test_pred, target_names=["0", "1"]))

cm = confusion_matrix(y_test, test_pred)
print("\nConfusion Matrix:\n", cm)

# Save confusion matrix evidence
plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix - Exp2 Improved (TEST)")
plt.xlabel("Predicted")
plt.ylabel("True")
for (i, j), v in np.ndenumerate(cm):
    plt.text(j, i, str(v), ha="center", va="center")
cm_path = os.path.join(OUT_DIR, "confusion_matrix_exp2.png")
plt.tight_layout()
plt.savefig(cm_path, dpi=300)
plt.close()
print("Saved:", cm_path)

# Save model + preprocess
model.save(MODEL_PATH)

prep = {
    "target": TARGET,
    "label_encoder": label_encoder,
    "cat_cols": cat_cols,
    "num_cols": num_cols,
    "feature_columns": feature_columns,
    "scaler": scaler,
    "threshold": float(best_t)
}

with open(PREP_PATH, "wb") as f:
    pickle.dump(prep, f)

print("Saved model:", MODEL_PATH)
print("Saved preprocess:", PREP_PATH)