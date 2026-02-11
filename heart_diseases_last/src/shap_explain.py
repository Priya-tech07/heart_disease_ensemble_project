import numpy as np
import pandas as pd
import shap
import joblib
import os
import matplotlib.pyplot as plt

# -----------------------------
# Paths
# -----------------------------
DATA_DIR = "data/processed"
MODEL_DIR = "models"
SHAP_DIR = "reports/shap"
os.makedirs(SHAP_DIR, exist_ok=True)

# Load merged dataset (for column names)
merged_df = pd.read_csv(os.path.join(DATA_DIR, "merged_heart_dataset.csv"))
feature_cols = [c for c in merged_df.columns if c != "target"]

# Load scaled data
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))

# Convert scaled arrays back to DataFrame with feature names
X_train_df = pd.DataFrame(X_train, columns=feature_cols)
X_test_df  = pd.DataFrame(X_test, columns=feature_cols)

# Load XGBoost model (best for SHAP)
xgb = joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl"))

print("✅ Loaded XGBoost model + datasets for SHAP")

# -----------------------------
# SHAP Explainer
# -----------------------------
explainer = shap.Explainer(xgb, X_train_df)
shap_values = explainer(X_test_df)

print("✅ SHAP values computed successfully!")

# -----------------------------
# 1) Global Explanation: Summary Plot
# -----------------------------
plt.figure()
shap.summary_plot(shap_values, X_test_df, show=False)
plt.title("SHAP Summary Plot (Global Feature Importance)")
plt.tight_layout()
summary_path = os.path.join(SHAP_DIR, "shap_summary_plot.png")
plt.savefig(summary_path, dpi=300)
plt.close()

print("✅ Saved:", summary_path)

# -----------------------------
# 2) Global Explanation: Bar Plot
# -----------------------------
plt.figure()
shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False)
plt.title("SHAP Feature Importance (Bar)")
plt.tight_layout()
bar_path = os.path.join(SHAP_DIR, "shap_feature_importance_bar.png")
plt.savefig(bar_path, dpi=300)
plt.close()

print("✅ Saved:", bar_path)

# -----------------------------
# 3) Local Explanation: Waterfall Plot (for one sample)
# -----------------------------
index = 0  # change index to test different patients
true_label = y_test[index]

plt.figure()
shap.plots.waterfall(shap_values[index], show=False)
plt.title(f"SHAP Waterfall Plot (Patient {index}) | True Label: {true_label}")
waterfall_path = os.path.join(SHAP_DIR, f"shap_waterfall_patient_{index}.png")
plt.savefig(waterfall_path, dpi=300, bbox_inches="tight")
plt.close()

print("✅ Saved:", waterfall_path)

# -----------------------------
# Print top features for patient
# -----------------------------
patient_shap = pd.Series(shap_values[index].values, index=feature_cols)
top_positive = patient_shap.sort_values(ascending=False).head(5)
top_negative = patient_shap.sort_values().head(5)

print("\n================ Patient-Level SHAP Explanation ================")
print(f"Patient Index: {index}")
print("✅ True Label:", true_label)

print("\n🔺 Top 5 Features Increasing Risk:")
print(top_positive)

print("\n🔻 Top 5 Features Decreasing Risk:")
print(top_negative)

print("\n🎉 SHAP Explainability completed! Check reports/shap/")
