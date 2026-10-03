import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "bank-full.csv"

df = pd.read_csv(DATA_PATH, sep=";")

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())


print("\nTarget distribution:")
print(df["y"].value_counts())
print("\nTarget percentage:")
print(df["y"].value_counts(normalize=True) * 100)


print("\nUnknown values:")
for col in df.select_dtypes(include="str"):
    count = (df[col] == "unknown").sum()
    if count > 0:
        print(f"{col}: {count}")


print("\nCategorical values:")
for col in df.select_dtypes(include="str"):
    print(f"\n{col}:")
    print(df[col].value_counts())


print("\nNumerical statistics:")
print(df.select_dtypes(include="number").describe().T)


# =========================
# Prepare X and y
# =========================
# Remove duration because it causes data leakage
df = df.drop(columns=["duration"])
# Separate features and target
X = df.drop(columns=["y"])
y = df["y"].map({"no": 0, "yes": 1})
print("\nAfter removing duration:")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("\nTarget after encoding:")
print(y.value_counts())


# =========================
# Split Train / Validation / Test
# =========================
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
print("\nDataset split:")
print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)


print("\nClass distribution after split:")

print("\nTrain:")
print(y_train.value_counts())
print(y_train.value_counts(normalize=True) * 100)

print("\nValidation:")
print(y_val.value_counts())
print(y_val.value_counts(normalize=True) * 100)

print("\nTest:")
print(y_test.value_counts())
print(y_test.value_counts(normalize=True) * 100)