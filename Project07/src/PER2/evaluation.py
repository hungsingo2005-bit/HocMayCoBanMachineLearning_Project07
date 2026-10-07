import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    precision_recall_curve
)

from paths import (
    DATA_FILE,
    MODEL_FILE,
    EVALUATION_FILE
)


# ==================================================
# CONFIG
# ==================================================

RANDOM_STATE = 42

# Tìm threshold từ 0.10 đến 0.90
# với bước 0.01
THRESHOLDS = np.arange(
    0.10,
    0.91,
    0.01
)


# ==================================================
# REPORT DIRECTORIES
# ==================================================

REPORTS_DIR = EVALUATION_FILE.parent
FIGURES_DIR = REPORTS_DIR / "figures"

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# LOAD DATA
# ==================================================

df = pd.read_csv(
    DATA_FILE,
    sep=";"
)


# ==================================================
# LEAKAGE REMOVAL
# ==================================================

# duration là leakage feature
# vì chỉ biết được sau/trong quá trình gọi điện
df = df.drop(
    columns=["duration"]
)


# ==================================================
# PREPARE X AND Y
# ==================================================

X = df.drop(
    columns=["y"]
)

y = df["y"].map(
    {
        "no": 0,
        "yes": 1
    }
)


# ==================================================
# SPLIT
# ==================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp
)


# ==================================================
# LOAD MODEL
# ==================================================

model = joblib.load(
    MODEL_FILE
)


print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print("\nModel loaded successfully.")


# ==================================================
# VALIDATION PROBABILITY
# ==================================================

y_val_proba = model.predict_proba(
    X_val
)[:, 1]


# ==================================================
# VALIDATION PR-AUC
# ==================================================

val_pr_auc = average_precision_score(
    y_val,
    y_val_proba
)


print(
    f"\nValidation PR-AUC: {val_pr_auc:.4f}"
)


# ==================================================
# BASELINE - ALL NEGATIVE
# ==================================================

y_baseline = pd.Series(
    0,
    index=y_val.index
)


baseline_accuracy = accuracy_score(
    y_val,
    y_baseline
)

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

baseline_cm = confusion_matrix(
    y_val,
    y_baseline
)


print("\n" + "=" * 60)
print("BASELINE - ALL NEGATIVE")
print("=" * 60)

print(
    f"Accuracy:  {baseline_accuracy:.4f}"
)

print(
    f"Precision: {baseline_precision:.4f}"
)

print(
    f"Recall:    {baseline_recall:.4f}"
)

print(
    f"F1:        {baseline_f1:.4f}"
)

print("\nConfusion Matrix:")
print(baseline_cm)


# ==================================================
# THRESHOLD EXPERIMENT
# ==================================================

threshold_results = []


for threshold in THRESHOLDS:

    y_val_pred = (
        y_val_proba >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_val,
        y_val_pred
    )

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

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        y_val_pred
    ).ravel()

    threshold_results.append(
        {
            "threshold": round(float(threshold), 2),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp
        }
    )


threshold_df = pd.DataFrame(
    threshold_results
)


print("\n" + "=" * 60)
print("THRESHOLD COMPARISON - VALIDATION")
print("=" * 60)

print(
    threshold_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# Save threshold experiment
threshold_df.to_csv(
    REPORTS_DIR / "threshold_results.csv",
    index=False
)


# ==================================================
# SELECT BEST THRESHOLD USING VALIDATION ONLY
# ==================================================

# Chọn threshold có F1 cao nhất
best_row = threshold_df.loc[
    threshold_df["f1"].idxmax()
]

selected_threshold = float(
    best_row["threshold"]
)


print("\n" + "=" * 60)
print("BEST THRESHOLD")
print("=" * 60)

print(
    f"Selected threshold: {selected_threshold:.2f}"
)

print(
    f"Validation Accuracy: "
    f"{best_row['accuracy']:.4f}"
)

print(
    f"Validation Precision: "
    f"{best_row['precision']:.4f}"
)

print(
    f"Validation Recall: "
    f"{best_row['recall']:.4f}"
)

print(
    f"Validation F1: "
    f"{best_row['f1']:.4f}"
)

print("\nValidation Confusion Matrix:")

print(
    int(best_row["tn"]),
    int(best_row["fp"]),
    int(best_row["fn"]),
    int(best_row["tp"])
)


# ==================================================
# PRECISION-RECALL CURVE
# ==================================================

precision_curve, recall_curve, curve_thresholds = (
    precision_recall_curve(
        y_val,
        y_val_proba
    )
)


plt.figure(
    figsize=(8, 6)
)

plt.plot(
    recall_curve,
    precision_curve
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title(
    "Precision-Recall Curve - Validation"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "precision_recall_curve.png",
    dpi=300
)

plt.close()


# ==================================================
# FINAL TEST EVALUATION
# ==================================================

print("\n" + "=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)


# Predict probability on test
y_test_proba = model.predict_proba(
    X_test
)[:, 1]


# Test PR-AUC
test_pr_auc = average_precision_score(
    y_test,
    y_test_proba
)


# IMPORTANT:
# Threshold was selected using validation only.
# Do NOT select threshold using test.

y_test_pred = (
    y_test_proba >= selected_threshold
).astype(int)


test_accuracy = accuracy_score(
    y_test,
    y_test_pred
)

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


test_tn, test_fp, test_fn, test_tp = (
    test_cm.ravel()
)


print(
    f"\nSelected threshold: "
    f"{selected_threshold:.2f}"
)

print(
    f"Test PR-AUC:  {test_pr_auc:.4f}"
)

print(
    f"Test Accuracy: {test_accuracy:.4f}"
)

print(
    f"Test Precision: {test_precision:.4f}"
)

print(
    f"Test Recall:    {test_recall:.4f}"
)

print(
    f"Test F1:        {test_f1:.4f}"
)

print("\nTest Confusion Matrix:")
print(test_cm)


# ==================================================
# ERROR ANALYSIS
# ==================================================

error_df = X_test.copy()

error_df["actual"] = y_test.values

error_df["predicted"] = y_test_pred

error_df["probability"] = y_test_proba


# ==================================================
# CLASSIFY ERROR TYPE
# ==================================================

error_df["error_type"] = "TN"


error_df.loc[
    (error_df["actual"] == 0)
    & (error_df["predicted"] == 1),
    "error_type"
] = "FP"


error_df.loc[
    (error_df["actual"] == 1)
    & (error_df["predicted"] == 0),
    "error_type"
] = "FN"


error_df.loc[
    (error_df["actual"] == 1)
    & (error_df["predicted"] == 1),
    "error_type"
] = "TP"


# ==================================================
# ERROR COUNTS
# ==================================================

error_counts = (
    error_df["error_type"]
    .value_counts()
    .reindex(
        ["TN", "FP", "FN", "TP"],
        fill_value=0
    )
)


print("\n" + "=" * 60)
print("ERROR COUNTS")
print("=" * 60)

print(error_counts)


# ==================================================
# AGE ANALYSIS
# ==================================================

def age_group(age):

    if age <= 30:
        return "18-30"

    if age <= 40:
        return "31-40"

    if age <= 50:
        return "41-50"

    if age <= 60:
        return "51-60"

    return "61+"


error_df["age_group"] = (
    error_df["age"]
    .apply(age_group)
)


age_analysis = (
    error_df
    .groupby("age_group")
    .agg(
        samples=("actual", "size"),

        positive_actual=(
            "actual",
            "sum"
        ),

        positive_predicted=(
            "predicted",
            "sum"
        ),

        FP=(
            "error_type",
            lambda x: (x == "FP").sum()
        ),

        FN=(
            "error_type",
            lambda x: (x == "FN").sum()
        ),

        TP=(
            "error_type",
            lambda x: (x == "TP").sum()
        ),

        TN=(
            "error_type",
            lambda x: (x == "TN").sum()
        )
    )
    .reset_index()
)


print("\n" + "=" * 60)
print("ERROR ANALYSIS BY AGE")
print("=" * 60)

print(age_analysis)


age_analysis.to_csv(
    REPORTS_DIR / "error_analysis_age.csv",
    index=False
)


# ==================================================
# JOB ANALYSIS
# ==================================================

job_analysis = (
    error_df
    .groupby("job")
    .agg(
        samples=("actual", "size"),

        positive_actual=(
            "actual",
            "sum"
        ),

        positive_predicted=(
            "predicted",
            "sum"
        ),

        FP=(
            "error_type",
            lambda x: (x == "FP").sum()
        ),

        FN=(
            "error_type",
            lambda x: (x == "FN").sum()
        ),

        TP=(
            "error_type",
            lambda x: (x == "TP").sum()
        ),

        TN=(
            "error_type",
            lambda x: (x == "TN").sum()
        )
    )
    .reset_index()
)


print("\n" + "=" * 60)
print("ERROR ANALYSIS BY JOB")
print("=" * 60)

print(job_analysis)


job_analysis.to_csv(
    REPORTS_DIR / "error_analysis_job.csv",
    index=False
)


# ==================================================
# SAVE FINAL EVALUATION
# ==================================================

evaluation = {

    "validation": {

        "pr_auc": round(
            float(val_pr_auc),
            4
        ),

        "selected_threshold": round(
            selected_threshold,
            2
        ),

        "accuracy": round(
            float(best_row["accuracy"]),
            4
        ),

        "precision": round(
            float(best_row["precision"]),
            4
        ),

        "recall": round(
            float(best_row["recall"]),
            4
        ),

        "f1": round(
            float(best_row["f1"]),
            4
        )
    },

    "test": {

        "pr_auc": round(
            float(test_pr_auc),
            4
        ),

        "accuracy": round(
            float(test_accuracy),
            4
        ),

        "precision": round(
            float(test_precision),
            4
        ),

        "recall": round(
            float(test_recall),
            4
        ),

        "f1": round(
            float(test_f1),
            4
        ),

        "tn": int(test_tn),

        "fp": int(test_fp),

        "fn": int(test_fn),

        "tp": int(test_tp)
    },

    "baseline": {

        "accuracy": round(
            float(baseline_accuracy),
            4
        ),

        "precision": round(
            float(baseline_precision),
            4
        ),

        "recall": round(
            float(baseline_recall),
            4
        ),

        "f1": round(
            float(baseline_f1),
            4
        )
    }
}


# ==================================================
# SAVE JSON
# ==================================================

with open(
    EVALUATION_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        evaluation,
        f,
        indent=4
    )


print(
    f"\nEvaluation saved to:"
    f"\n{EVALUATION_FILE}"
)

print(
    "\nEvaluation completed successfully."
)