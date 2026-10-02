import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# Load dataset
df = pd.read_csv("data/bank-full.csv", sep=";")

# Remove leakage feature
df = df.drop(columns=["duration"])

# Prepare X and y
X = df.drop(columns=["y"])
y = df["y"].map({"no": 0, "yes": 1})


# Split data
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


# Load trained model
model = joblib.load("models/model.joblib")

print("Model loaded successfully!")


# Predict probability on validation set
y_val_proba = model.predict_proba(X_val)[:, 1]


# PR-AUC
pr_auc = average_precision_score(
    y_val,
    y_val_proba
)

print(f"\nValidation PR-AUC: {pr_auc:.4f}")


# Threshold evaluation
thresholds = [0.30, 0.50, 0.70]

print("\nValidation threshold evaluation:")

for threshold in thresholds:

    y_val_pred = (
        y_val_proba >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        y_val_pred,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        y_val_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        y_val_pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y_val,
        y_val_pred
    )

    print(f"\nThreshold: {threshold}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)


# ==================================================
# FINAL TEST EVALUATION
# ==================================================

print("\n" + "=" * 50)
print("FINAL TEST EVALUATION")
print("=" * 50)

# Predict probability on test set
y_test_proba = model.predict_proba(X_test)[:, 1]

# Test PR-AUC
test_pr_auc = average_precision_score(
    y_test,
    y_test_proba
)

print(f"\nTest PR-AUC: {test_pr_auc:.4f}")


# Evaluate at threshold 0.30
threshold = 0.30

y_test_pred = (
    y_test_proba >= threshold
).astype(int)

test_precision = precision_score(
    y_test,
    y_test_pred,
    zero_division=0
)

test_recall = recall_score(
    y_test,
    y_test_pred,
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    y_test_pred,
    zero_division=0
)

test_cm = confusion_matrix(
    y_test,
    y_test_pred
)

print(f"\nThreshold: {threshold}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall:    {test_recall:.4f}")
print(f"F1:        {test_f1:.4f}")

print("Confusion Matrix:")
print(test_cm)


# ==================================================
# ERROR ANALYSIS
# ==================================================

# ... giữ nguyên toàn bộ phần Error Analysis của bro ...


# ==================================================
# SAVE FINAL TEST EVALUATION
# ==================================================

import json
from pathlib import Path

# Get confusion matrix values from FINAL TEST
test_tn, test_fp, test_fn, test_tp = test_cm.ravel()

evaluation = {
    "threshold": 0.30,
    "pr_auc": round(float(test_pr_auc), 4),
    "precision": round(float(test_precision), 4),
    "recall": round(float(test_recall), 4),
    "f1": round(float(test_f1), 4),
    "tn": int(test_tn),
    "fp": int(test_fp),
    "fn": int(test_fn),
    "tp": int(test_tp)
}

Path("reports").mkdir(exist_ok=True)

with open("reports/evaluation.json", "w", encoding="utf-8") as f:
    json.dump(evaluation, f, indent=4)

print("\nEvaluation saved to reports/evaluation.json")