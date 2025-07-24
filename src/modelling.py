# === src/modelling.py ===

import pandas as pd  # Data manipulation
import numpy as np   # Numerical operations
import os            # OS path operations
import sys           # System-specific parameters
import joblib        # Model serialization
import argparse      # Command-line argument parsing
import matplotlib.pyplot as plt  # Plotting
import seaborn as sns            # Statistical data visualization
import json         # JSON file operations

from sklearn.ensemble import RandomForestClassifier  # ML model
from sklearn.model_selection import train_test_split # Data splitting
from sklearn.metrics import (                        # Evaluation metrics
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

# === Base Path for Deployment ===
BASE_DIR = "C:/Users/DELL/Desktop/SRM2025IISERM/katz_cyclic_vector_ml"

# Fix import path for local modules
sys.path.append(BASE_DIR)

# Import custom feature engineering and utility functions
from src.features import load_dataset, add_matrix_features, get_feature_target_split, normalize_features
from src.utils import make_dataframe_arrow_safe

# === CLI Argument for Dataset Path ===
parser = argparse.ArgumentParser()  # Create argument parser
parser.add_argument("--input", type=str, default=os.path.join(BASE_DIR, "data", "raw", "samples_n3.csv"), help="Path to input CSV file")
args = parser.parse_args()          # Parse arguments
csv_path = args.input               # Get dataset path

# === Load & Preprocess ===
print("📦 Loading dataset...")
df = load_dataset(csv_path)         # Load dataset

print(f"📊 Dataset shape: {df.shape}")
print(df['cyclic'].value_counts())  # Show class distribution

# Feature engineering
df = add_matrix_features(df, n=3)   # Add matrix-based features
df = make_dataframe_arrow_safe(df)  # Ensure DataFrame is Arrow-safe

# Feature/Target split
print("🧪 Splitting data...")
X, y = get_feature_target_split(df) # Split into features and target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)  # Train/test split

# === Train Model ===
print("🚀 Training Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42)  # Initialize model
model.fit(X_train, y_train)                                        # Train model

# === Evaluation ===
print("📈 Evaluating model...")
y_pred = model.predict(X_test)                 # Predict classes
y_proba = model.predict_proba(X_test)[:, 1]    # Predict probabilities for class 1

acc = accuracy_score(y_test, y_pred)           # Accuracy
f1 = f1_score(y_test, y_pred)                  # F1 score
cm = confusion_matrix(y_test, y_pred)          # Confusion matrix
report = classification_report(y_test, y_pred, output_dict=True)  # Classification report as dict
auc_score = roc_auc_score(y_test, y_proba)     # ROC AUC score

# === Save Artifacts ===
models_dir = os.path.join(BASE_DIR, "models")
os.makedirs(models_dir, exist_ok=True)         # Ensure models directory exists

model_path = os.path.join(models_dir, "trained_model.pkl")
joblib.dump(model, model_path)                 # Save trained model
print(f"✅ Model saved to {model_path}")

metrics_path = os.path.join(models_dir, "metrics.json")
with open(metrics_path, "w") as f:             # Save metrics as JSON
    json.dump({
        "accuracy": acc,
        "f1_score": f1,
        "auc_score": auc_score,
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }, f, indent=2)
print(f"📊 Metrics saved to {metrics_path}")

# === Save Confusion Matrix Plot ===
figures_dir = os.path.join(BASE_DIR, "outputs", "figures")
os.makedirs(figures_dir, exist_ok=True)        # Ensure figures directory exists

cm_path = os.path.join(figures_dir, "confusion_matrix.png")
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")  # Plot confusion matrix
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(cm_path)                           # Save plot
plt.close()
print(f"✅ Confusion matrix saved to {cm_path}")

# === Save ROC Curve ===
fpr, tpr, _ = roc_curve(y_test, y_proba)       # Compute ROC curve
roc_path = os.path.join(figures_dir, "roc_curve.png")
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(roc_path)                          # Save ROC curve plot
plt.close()
print(f"✅ ROC curve saved to {roc_path}")

# === Save Logs ===
logs_dir = os.path.join(BASE_DIR, "outputs", "logs")
os.makedirs(logs_dir, exist_ok=True)           # Ensure logs directory exists

log_path = os.path.join(logs_dir, "train_summary.txt")
with open(log_path, "w") as f:                 # Write training summary log
    f.write(f"Dataset: {csv_path}\n")
    f.write(f"Samples: {df.shape[0]}\n")
    f.write(f"Features: {X.shape[1]}\n")
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"AUC Score: {auc_score:.4f}\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(cm))
print(f"📝 Training summary saved to {log_path}")

# === Final Summary ===
print("\n📋 Final Evaluation:")
print(json.dumps({
    "accuracy": acc,
    "f1_score": f1,
    "auc_score": auc_score,
    "confusion_matrix": cm.tolist(),
},indent=2))