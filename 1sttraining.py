import os
import pickle
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------
DATA_PATH = "data/data_clean.csv"   # <-- change to your cleaned dataset path
TARGET = "ProdTaken"

OUT_DIR = "data/output_exp1"
MODEL_PATH = "examples/exp1_baseline_model.keras"
PREP_PATH = "examples/exp1_baseline_preprocess.pkl"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("examples", exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print("Missing values:", int(df.isna().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))
print("\nTarget distribution:")
print(df[TARGET].value_counts())
print(df[TARGET].value_counts(normalize=True))

# -----------------------------
# Separate X, y
# -----------------------------
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Encode target to 0/1 if needed
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Identify categorical/numeric columns (evidence)
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
print("\nCategorical columns:", cat_cols)
print("Numeric columns:", num_cols)

# -----------------------------
# Encode categorical features (one-hot)
# -----------------------------
X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=True)
feature_columns = X_enc.columns.tolist()

print("\nFeature count before encoding:", X.shape[1])
print("Feature count after encoding:", X_enc.shape[1])

# -----------------------------
# Split: Train (75%) / Test (25%) stratified
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_enc, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
)
print(f"\nSplit sizes -> Train: {len(X_train)}, Test: {len(X_test)}")

# -----------------------------
# Scale numeric inputs (fit on train only!)
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Build baseline model
# -----------------------------
model = keras.Sequential([
    keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation="sigmoid")
])

# -----------------------------
# Compile (baseline: built-in BCE)
# -----------------------------
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc")]
)

model.summary()

# -----------------------------
# Train (simple: validation_split inside training)
# -----------------------------
history = model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=2,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        )
    ]
)

# -----------------------------
# Evaluate on test with default threshold = 0.50
# -----------------------------
test_proba = model.predict(X_test_scaled, verbose=0).flatten()
test_pred = (test_proba >= 0.50).astype(int)

acc = accuracy_score(y_test, test_pred)
prec = precision_score(y_test, test_pred, zero_division=0)
rec = recall_score(y_test, test_pred, zero_division=0)
f1 = f1_score(y_test, test_pred, zero_division=0)
auc = roc_auc_score(y_test, test_proba)

print("\n" + "="*60)
print("EXPERIMENT 1 (BASELINE) - TEST RESULTS")
print("="*60)
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"AUC      : {auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, test_pred, target_names=["0", "1"]))

cm = confusion_matrix(y_test, test_pred)
print("\nConfusion Matrix:\n", cm)

# Save confusion matrix image (evidence)
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix - Exp1 Baseline (TEST)")
plt.xlabel("Predicted")
plt.ylabel("True")
for (i, j), v in np.ndenumerate(cm):
    plt.text(j, i, str(v), ha="center", va="center")
cm_path = os.path.join(OUT_DIR, "confusion_matrix_exp1.png")
plt.tight_layout()
plt.savefig(cm_path, dpi=300)
plt.close()
print("\nSaved:", cm_path)

# Save model
model.save(MODEL_PATH)
print("Saved model:", MODEL_PATH)

# Save preprocess for inference
prep = {
    "target": TARGET,
    "label_encoder": label_encoder,
    "cat_cols": cat_cols,
    "num_cols": num_cols,
    "feature_columns": feature_columns,
    "scaler": scaler,
    "threshold": 0.50
}
with open(PREP_PATH, "wb") as f:
    pickle.dump(prep, f)
print("Saved preprocess:", PREP_PATH)