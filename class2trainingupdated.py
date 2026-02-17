import os
import pickle
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------
# Config (change if needed)
# ----------------------------
# Try these paths in order (so it works even if your file is in a different folder)
CANDIDATE_DATA_PATHS = [
    "data/class5data.csv",
    "data/class5data_fixed.csv",
    "class5data.csv",
    "data/bank-additional-full.csv",
]

# If your dataset target column is different, change it here.
# (Auto-detect: uses 'y' if exists, else 'ProdTaken' if exists.)
TARGET_COL = None

# Where to save model + preprocessing artifacts
MODEL_PATH = "examples/bank_classification_model.h5"
PREPROCESS_PATH = "examples/preprocess.pkl"


# ----------------------------
# Data cleaning (5 requirements)
# ----------------------------
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # 1) Trim column names (remove extra spaces)
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # 2) Remove duplicate rows
    df = df.drop_duplicates()

    # 3) Replace "unknown" -> NaN (missing)
    df = df.replace({"unknown": np.nan, "Unknown": np.nan, "UNKNOWN": np.nan})

    # 4) Fix Gender typo: "fe male" -> "female" (only if Gender column exists)
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

    # 5) Fill missing values
    #    - categorical (text): fill with mode
    #    - numeric: fill with median
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in df.columns if c not in cat_cols]

    for c in cat_cols:
        if df[c].isna().sum() > 0:
            mode_val = df[c].mode(dropna=True)
            fill_val = mode_val.iloc[0] if len(mode_val) else "missing"
            df[c] = df[c].fillna(fill_val)

    for c in num_cols:
        # coerce numeric if needed
        df[c] = pd.to_numeric(df[c], errors="ignore")
        if df[c].isna().sum() > 0 and pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(df[c].median())

    return df


def pick_data_path() -> str:
    for p in CANDIDATE_DATA_PATHS:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "Could not find dataset. Tried: " + ", ".join(CANDIDATE_DATA_PATHS)
    )


# ----------------------------
# Load + Clean
# ----------------------------
print("Loading data...")
data_path = pick_data_path()
print(f"Using dataset: {data_path}")

# Auto sep detection (some files use ';', some use ',')
try:
    df_raw = pd.read_csv(data_path, sep=";")
    if df_raw.shape[1] == 1:
        df_raw = pd.read_csv(data_path)  # fallback
except Exception:
    df_raw = pd.read_csv(data_path)

print(f"Original shape: {df_raw.shape}")

df = clean_dataframe(df_raw)
print(f"Cleaned shape : {df.shape}")
print(f"\nColumn names: {df.columns.tolist()}")

# Auto-detect target column
if TARGET_COL is None:
    if "y" in df.columns:
        TARGET_COL = "y"
    elif "ProdTaken" in df.columns:
        TARGET_COL = "ProdTaken"
    else:
        raise ValueError("Target column not found. Set TARGET_COL manually.")

print(f"\nTarget column: {TARGET_COL}")
print(f"\nTarget distribution:\n{df[TARGET_COL].value_counts()}")

# Separate features and target
X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# One-hot encode categorical features
categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
print(f"\nCategorical columns: {categorical_columns}")

X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
feature_columns = X_encoded.columns.tolist()
print(f"Features after encoding: {X_encoded.shape[1]}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set size: {X_train_scaled.shape[0]}")
print(f"Testing set size : {X_test_scaled.shape[0]}")

# ----------------------------
# Build model (same style output)
# ----------------------------
print("\nBuilding neural network model...")
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

model.summary()

# Train
print("\nTraining the model...")
history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]
)

# Evaluate
print("\nEvaluating the model...")
test_loss, test_accuracy, test_auc = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test AUC: {test_auc:.4f}")

y_pred_proba = model.predict(X_test_scaled, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=[str(c) for c in label_encoder.classes_]))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
os.makedirs("examples", exist_ok=True)
model.save(MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")

# Save preprocessing artifacts for inference (so inference output stays consistent)
artifacts = {
    "target_col": TARGET_COL,
    "label_classes": label_encoder.classes_,
    "feature_columns": feature_columns,
    "scaler": scaler,
}
with open(PREPROCESS_PATH, "wb") as f:
    pickle.dump(artifacts, f)

print(f"Preprocess artifacts saved to {PREPROCESS_PATH}")
