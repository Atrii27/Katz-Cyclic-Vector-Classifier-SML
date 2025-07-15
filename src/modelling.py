# === src/modelling.py ===

import pandas as pd
import numpy as np
import os
import sys
import joblib
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# === Base Path for Deployment ===
BASE_DIR = "C:/Users/DELL/Desktop/SRM2025IISERM/katz_cyclic_vector_ml"

# Fix import path
sys.path.append(BASE_DIR)

from src.features import load_dataset, add_matrix_features, get_feature_target_split, normalize_features
from src.utils import make_dataframe_arrow_safe

# === CLI Argument for Dataset Path ===
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default=os.path.join(BASE_DIR, "data", "raw", "samples_n3.csv"), help="Path to input CSV file")
args = parser.parse_args()
csv_path = args.input

# === Load & Preprocess ===
print("📦 Loading dataset...")
df = load_dataset(csv_path)

print(f"📊 Dataset shape: {df.shape}")
print(df['cyclic'].value_counts())

# Feature engineering
df = add_matrix_features(df, n=3)
df = make_dataframe_arrow_safe(df)

# Feature/Target split
print("🧪 Splitting data...")
X, y = get_feature_target_split(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# === Train Model ===
print("🚀 Training Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Evaluation ===
print("📈 Evaluating model...")
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# === Save Artifacts ===
models_dir = os.path.join(BASE_DIR, "models")
os.makedirs(models_dir, exist_ok=True)

model_path = os.path.join(models_dir, "trained_model.pkl")
joblib.dump(model, model_path)
print(f"✅ Model saved to {model_path}")

metrics_path = os.path.join(models_dir, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump({
        "accuracy": acc,
        "f1_score": f1,
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }, f, indent=2)

print(f"📊 Metrics saved to {metrics_path}")

# === Save Confusion Matrix Plot ===
figures_dir = os.path.join(BASE_DIR, "outputs", "figures")
os.makedirs(figures_dir, exist_ok=True)

cm_path = os.path.join(figures_dir, "confusion_matrix.png")
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(cm_path)
plt.close()
print(f"✅ Confusion matrix saved to {cm_path}")

# === Save Logs ===
logs_dir = os.path.join(BASE_DIR, "outputs", "logs")
os.makedirs(logs_dir, exist_ok=True)

log_path = os.path.join(logs_dir, "train_summary.txt")
with open(log_path, "w") as f:
    f.write(f"Dataset: {csv_path}\n")
    f.write(f"Samples: {df.shape[0]}\n")
    f.write(f"Features: {X.shape[1]}\n")
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(cm))

print(f"📝 Training summary saved to {log_path}")

# === Final Summary ===
print("\n📋 Final Evaluation:")
print(json.dumps({
    "accuracy": acc,
    "f1_score": f1,
    "confusion_matrix": cm.tolist(),
}, indent=2))
