import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from paths import DATA_FILE


# ==================================================
# CONFIG
# ==================================================

RANDOM_STATE = 42

REPORTS_DIR = DATA_FILE.parent.parent / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

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

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicates:")
print(df.duplicated().sum())


# ==================================================
# TARGET DISTRIBUTION
# ==================================================

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

target_counts = df["y"].value_counts()

target_percent = (
    df["y"]
    .value_counts(normalize=True)
    .mul(100)
)

print("\nCounts:")
print(target_counts)

print("\nPercentage:")
print(target_percent)


# Target distribution plot

plt.figure(figsize=(7, 5))

target_counts.plot(
    kind="bar"
)

plt.title("Phân bố biến mục tiêu")
plt.xlabel("Kết quả đăng ký tiền gửi")
plt.ylabel("Số lượng")

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "target_distribution.png",
    dpi=300
)

plt.close()


# ==================================================
# NUMERICAL DISTRIBUTION
# ==================================================

numeric_columns = df.select_dtypes(
    include="number"
).columns.tolist()

print("\n" + "=" * 60)
print("NUMERICAL STATISTICS")
print("=" * 60)

print(
    df[numeric_columns]
    .describe()
    .T
)


# Age distribution

plt.figure(figsize=(8, 5))

df["age"].plot(
    kind="hist",
    bins=30
)

plt.title("Phân bố độ tuổi khách hàng")
plt.xlabel("Tuổi")
plt.ylabel("Số lượng")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "age_distribution.png",
    dpi=300
)

plt.close()


# Balance distribution

plt.figure(figsize=(8, 5))

df["balance"].plot(
    kind="hist",
    bins=50
)

plt.title("Phân bố số dư tài khoản")
plt.xlabel("Số dư")
plt.ylabel("Số lượng")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "balance_distribution.png",
    dpi=300
)

plt.close()


# ==================================================
# TARGET BY AGE
# ==================================================

df["age_group"] = pd.cut(
    df["age"],
    bins=[17, 30, 40, 50, 60, 100],
    labels=[
        "18-30",
        "31-40",
        "41-50",
        "51-60",
        "61+"
    ]
)

age_target = (
    df.groupby(
        "age_group",
        observed=False
    )["y"]
    .apply(
        lambda x: (x == "yes").mean() * 100
    )
)

print("\n" + "=" * 60)
print("SUBSCRIPTION RATE BY AGE")
print("=" * 60)

print(age_target)


plt.figure(figsize=(8, 5))

age_target.plot(
    kind="bar"
)

plt.title("Tỷ lệ đăng ký theo nhóm tuổi")
plt.xlabel("Nhóm tuổi")
plt.ylabel("Tỷ lệ đăng ký (%)")

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "target_by_age.png",
    dpi=300
)

plt.close()


# ==================================================
# TARGET BY JOB
# ==================================================

job_target = (
    df.groupby("job")["y"]
    .apply(
        lambda x: (x == "yes").mean() * 100
    )
    .sort_values(
        ascending=False
    )
)

print("\n" + "=" * 60)
print("SUBSCRIPTION RATE BY JOB")
print("=" * 60)

print(job_target)


plt.figure(figsize=(10, 6))

job_target.plot(
    kind="bar"
)

plt.title("Tỷ lệ đăng ký theo nghề nghiệp")
plt.xlabel("Nghề nghiệp")
plt.ylabel("Tỷ lệ đăng ký (%)")

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "target_by_job.png",
    dpi=300
)

plt.close()


# ==================================================
# LEAKAGE AUDIT
# ==================================================

print("\n" + "=" * 60)
print("LEAKAGE AUDIT")
print("=" * 60)

print(
    "duration exists:",
    "duration" in df.columns
)

print(
    "y exists:",
    "y" in df.columns
)


# Remove duration

df_model = df.drop(
    columns=["duration"]
)

X = df_model.drop(
    columns=["y"]
)

y = df_model["y"].map(
    {
        "no": 0,
        "yes": 1
    }
)


print("\nAfter removing duration:")
print("X shape:", X.shape)
print("y shape:", y.shape)

print(
    "duration in X:",
    "duration" in X.columns
)

print(
    "y in X:",
    "y" in X.columns
)


# ==================================================
# SPLIT AUDIT
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


print("\n" + "=" * 60)
print("SPLIT AUDIT")
print("=" * 60)

print(
    f"Train:      {len(X_train):,} "
    f"({len(X_train) / len(X) * 100:.2f}%)"
)

print(
    f"Validation: {len(X_val):,} "
    f"({len(X_val) / len(X) * 100:.2f}%)"
)

print(
    f"Test:       {len(X_test):,} "
    f"({len(X_test) / len(X) * 100:.2f}%)"
)


print("\nPositive rate:")

print(
    f"Train:      {y_train.mean() * 100:.2f}%"
)

print(
    f"Validation: {y_val.mean() * 100:.2f}%"
)

print(
    f"Test:       {y_test.mean() * 100:.2f}%"
)


# ==================================================
# SAVE EDA SUMMARY
# ==================================================

summary = pd.DataFrame(
    {
        "dataset": [
            "train",
            "validation",
            "test"
        ],
        "samples": [
            len(y_train),
            len(y_val),
            len(y_test)
        ],
        "positive": [
            int(y_train.sum()),
            int(y_val.sum()),
            int(y_test.sum())
        ],
        "negative": [
            int((y_train == 0).sum()),
            int((y_val == 0).sum()),
            int((y_test == 0).sum())
        ],
        "positive_rate": [
            y_train.mean(),
            y_val.mean(),
            y_test.mean()
        ]
    }
)

summary.to_csv(
    REPORTS_DIR / "split_summary.csv",
    index=False
)


print("\nEDA completed successfully.")

print(
    f"Figures saved to: {FIGURES_DIR}"
)