import os
import pickle
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Config
# ----------------------------
MODEL_PATH = "examples/bank_classification_model.h5"
PREPROCESS_PATH = "examples/preprocess.pkl"

CANDIDATE_TEST_PATHS = [
    "data/class5data.csv",
    "data/class5data_fixed.csv",
    "class5data.csv",
    "data/bank-additional.csv",
    "data/bank-additional-full.csv",
]

OUTPUT_DIR = "data/output"
THRESHOLD = 0.3  # keep your old threshold for similar output


# ----------------------------
# Data cleaning (same as training)
# ----------------------------
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.drop_duplicates()
    df = df.replace({"unknown": np.nan, "Unknown": np.nan, "UNKNOWN": np.nan})

    if "Gender" in df.columns:
        df["Gender"] = (
            df["Gender"]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.lower()
            .replace({"fe male": "female"})
        )

    # Fix MaritalStatus wording: "single" -> "unmarried" (only if column exists)
    if "MaritalStatus" in df.columns:
        df["MaritalStatus"] = (
            df["MaritalStatus"]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.lower()
            .replace({"single": "unmarried"})
        )

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in df.columns if c not in cat_cols]

    for c in cat_cols:
        if df[c].isna().sum() > 0:
            mode_val = df[c].mode(dropna=True)
            fill_val = mode_val.iloc[0] if len(mode_val) else "missing"
            df[c] = df[c].fillna(fill_val)

    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="ignore")
        if df[c].isna().sum() > 0 and pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(df[c].median())

    return df


def pick_test_path() -> str:
    for p in CANDIDATE_TEST_PATHS:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Could not find test dataset. Tried: " + ", ".join(CANDIDATE_TEST_PATHS))


# ----------------------------
# Setup output dir
# ----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Load model + preprocess artifacts
# ----------------------------
print("Loading trained model...")
model = keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")
model.summary()

with open(PREPROCESS_PATH, "rb") as f:
    artifacts = pickle.load(f)

target_col = artifacts["target_col"]
label_classes = artifacts["label_classes"]
feature_columns = artifacts["feature_columns"]
scaler = artifacts["scaler"]

print(f"\nUsing target column: {target_col}")

# ----------------------------
# Load + clean test data
# ----------------------------
print("\nLoading test data...")
test_path = pick_test_path()
print(f"Using test dataset: {test_path}")

try:
    df_raw = pd.read_csv(test_path, sep=";")
    if df_raw.shape[1] == 1:
        df_raw = pd.read_csv(test_path)
except Exception:
    df_raw = pd.read_csv(test_path)

df_test = clean_dataframe(df_raw)

print(f"Test dataset shape: {df_test.shape}")

if target_col not in df_test.columns:
    raise ValueError(f"Target column '{target_col}' not found in test data. Columns: {df_test.columns.tolist()}")

print(f"\nTarget distribution:\n{df_test[target_col].value_counts()}")

X_test = df_test.drop(target_col, axis=1)
y_test = df_test[target_col]

# Encode y using saved classes
# We recreate encoder behavior with same class order
class_to_int = {c: i for i, c in enumerate(label_classes)}
y_test_encoded = y_test.map(class_to_int).astype(int).values

# One-hot encode X and align columns to training
categorical_columns = X_test.select_dtypes(include=["object"]).columns.tolist()
print(f"\nCategorical columns: {categorical_columns}")

X_test_encoded = pd.get_dummies(X_test, columns=categorical_columns, drop_first=True)

# Align with training columns
X_test_encoded = X_test_encoded.reindex(columns=feature_columns, fill_value=0)

print(f"Features after encoding (aligned): {X_test_encoded.shape[1]}")

# Scale using training scaler
X_test_scaled = scaler.transform(X_test_encoded)

# Predict
print("\nMaking predictions...")
y_pred_proba = model.predict(X_test_scaled, verbose=0)
y_pred = (y_pred_proba > THRESHOLD).astype(int).flatten()

print("\n" + "="*60)
print("INFERENCE RESULTS")
print("="*60)

accuracy = accuracy_score(y_test_encoded, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

try:
    auc_score = roc_auc_score(y_test_encoded, y_pred_proba)
    print(f"AUC Score: {auc_score:.4f}")
except Exception:
    print("AUC Score: Could not calculate")

print("\nClassification Report:")
print(classification_report(y_test_encoded, y_pred, target_names=[str(c) for c in label_classes]))

cm = confusion_matrix(y_test_encoded, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Plot confusion matrix (same outputs)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[str(c) for c in label_classes],
            yticklabels=[str(c) for c in label_classes],
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - Classification Model', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)

tn, fp, fn, tp = cm.ravel()
stats_text = f'True Negatives: {tn}\nFalse Positives: {fp}\nFalse Negatives: {fn}\nTrue Positives: {tp}'
stats_text += f'\n\nAccuracy: {accuracy:.4f}'
plt.text(2.5, 0.5, stats_text, fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
         verticalalignment='center')

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, "confusion_matrix.jpg")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nConfusion matrix plot saved to: {output_path}")

plt.figure(figsize=(10, 8))
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens',
            xticklabels=[str(c) for c in label_classes],
            yticklabels=[str(c) for c in label_classes],
            cbar_kws={'label': 'Percentage'})
plt.title('Normalized Confusion Matrix - Classification Model', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)

plt.tight_layout()
output_path_normalized = os.path.join(OUTPUT_DIR, "confusion_matrix_normalized.jpg")
plt.savefig(output_path_normalized, dpi=300, bbox_inches='tight')
print(f"Normalized confusion matrix plot saved to: {output_path_normalized}")

# Save predictions to CSV
results_df = df_test.copy()
# map predicted int back to class label
int_to_class = {i: c for i, c in enumerate(label_classes)}
results_df['predicted_label'] = pd.Series(y_pred).map(int_to_class)
results_df['prediction_probability'] = y_pred_proba.flatten()
results_df['true_label'] = y_test
results_df['correct_prediction'] = (results_df['true_label'].astype(str) == results_df['predicted_label'].astype(str))

output_csv_path = os.path.join(OUTPUT_DIR, "predictions.csv")
results_df.to_csv(output_csv_path, index=False)
print(f"Predictions saved to: {output_csv_path}")

print("\n" + "="*60)
print("Inference completed successfully!")
print("="*60)
