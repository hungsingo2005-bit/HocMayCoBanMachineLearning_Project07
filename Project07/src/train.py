import numpy as np
import pandas as pd

import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score
)

from features import preprocessor



PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "bank-full.csv"

df = pd.read_csv(DATA_PATH, sep=";")

# Remove leakage column
df = df.drop(columns=["duration"])
# Separate features and target
X = df.drop(columns=["y"])
y = df["y"].map({"no": 0, "yes": 1})
# Split Train / Validation / Test
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
# Build model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])
# Train
model.fit(X_train, y_train)

# Save trained model
Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/model.joblib")

print("Model trained successfully!")
print("Model saved to models/model.joblib")


# Predict probability
y_val_proba = model.predict_proba(X_val)[:, 1]

print("\nFirst 10 predicted probabilities:")
print(y_val_proba[:10])

# PR-AUC
pr_auc = average_precision_score(y_val, y_val_proba)

print(f"\nPR-AUC: {pr_auc:.4f}")

# Threshold evaluation
thresholds = [0.30, 0.50, 0.70]

print("\nThreshold evaluation:")

for threshold in thresholds:
    y_val_pred = (y_val_proba >= threshold).astype(int)

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

    cm = confusion_matrix(y_val, y_val_pred)

    print(f"\nThreshold: {threshold}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)


# Random baseline with the same contact rate as threshold 0.30

threshold = 0.30

y_model_pred = (y_val_proba >= threshold).astype(int)

n_contacts = y_model_pred.sum()

rng = np.random.default_rng(42)

random_indices = rng.choice(
    len(y_val),
    size=n_contacts,
    replace=False
)

y_random = np.zeros(len(y_val), dtype=int)
y_random[random_indices] = 1

random_precision = precision_score(
    y_val,
    y_random,
    zero_division=0
)

random_recall = recall_score(
    y_val,
    y_random,
    zero_division=0
)

random_f1 = f1_score(
    y_val,
    y_random,
    zero_division=0
)

print("\nRandom Baseline:")
print(f"Number of contacts: {n_contacts}")
print(f"Contact rate: {n_contacts / len(y_val):.4f}")
print(f"Precision: {random_precision:.4f}")
print(f"Recall:    {random_recall:.4f}")
print(f"F1:        {random_f1:.4f}")


# Baseline: all negative
y_baseline = np.zeros(len(y_val), dtype=int)

baseline_precision = precision_score(
    y_val,
    y_baseline,
    zero_division=0
)

baseline_recall = recall_score(
    y_val,
    y_baseline,
    zero_division=0
)

baseline_f1 = f1_score(
    y_val,
    y_baseline,
    zero_division=0
)

print("\nBaseline - All Negative:")
print(f"Precision: {baseline_precision:.4f}")
print(f"Recall:    {baseline_recall:.4f}")
print(f"F1:        {baseline_f1:.4f}")
