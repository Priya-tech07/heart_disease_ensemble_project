import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# -----------------------------
# Paths
# -----------------------------
DATA_DIR = "data"
OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

KAGGLE_PATH = os.path.join(DATA_DIR, "heart.csv")
UCI_PATH = os.path.join(DATA_DIR, "heart_disease_cleveland.csv")

# -----------------------------
# Load datasets
# -----------------------------
kaggle_df = pd.read_csv(KAGGLE_PATH)
uci_df = pd.read_csv(UCI_PATH)

print("✅ Loaded datasets successfully!")
print("Kaggle shape:", kaggle_df.shape)
print("UCI shape:", uci_df.shape)

# -----------------------------
# Basic cleaning function
# -----------------------------
def clean_dataset(df, name="dataset"):
    df = df.copy()

    # Remove duplicates
    before_dup = df.shape[0]
    df.drop_duplicates(inplace=True)
    after_dup = df.shape[0]

    print(f"\n🔹 {name} duplicates removed: {before_dup - after_dup}")

    # Handle missing values
    # Numeric -> median, Categorical -> mode
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ["int64", "float64"]:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)

    return df

# -----------------------------
# Clean datasets
# -----------------------------
kaggle_df = clean_dataset(kaggle_df, "Kaggle heart.csv")
uci_df = clean_dataset(uci_df, "UCI Cleveland")

# -----------------------------
# Align columns (keep only common)
# -----------------------------
common_cols = list(set(kaggle_df.columns).intersection(set(uci_df.columns)))
common_cols = sorted(common_cols)

kaggle_df = kaggle_df[common_cols]
uci_df = uci_df[common_cols]

print("\n✅ Common columns:", common_cols)

# -----------------------------
# Ensure target column is correct
# -----------------------------
if "target" not in common_cols:
    raise ValueError("❌ target column not found in datasets!")

# In some UCI datasets target may have values 0-4
# Convert: 0 -> 0, 1/2/3/4 -> 1
uci_df["target"] = uci_df["target"].apply(lambda x: 0 if int(x) == 0 else 1)
kaggle_df["target"] = kaggle_df["target"].apply(lambda x: 0 if int(x) == 0 else 1)

# -----------------------------
# Merge datasets
# -----------------------------
merged_df = pd.concat([kaggle_df, uci_df], axis=0, ignore_index=True)

# Shuffle dataset
merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\n✅ Merged dataset shape:", merged_df.shape)

# Save merged dataset
merged_path = os.path.join(OUTPUT_DIR, "merged_heart_dataset.csv")
merged_df.to_csv(merged_path, index=False)
print("✅ Saved merged dataset to:", merged_path)

# -----------------------------
# Split features and target
# -----------------------------
X = merged_df.drop("target", axis=1)
y = merged_df["target"]

# Train-test split 70/30
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

print("\n✅ Train shape:", X_train.shape)
print("✅ Test shape:", X_test.shape)

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
scaler_path = os.path.join(OUTPUT_DIR, "scaler.pkl")
joblib.dump(scaler, scaler_path)
print("✅ Saved scaler to:", scaler_path)

# Save train-test files
np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train_scaled)
np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test_scaled)
np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train.to_numpy())
np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test.to_numpy())

print("\n🎉 Preprocessing complete! Train-test data saved in data/processed/")
