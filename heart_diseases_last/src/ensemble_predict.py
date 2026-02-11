import numpy as np
import joblib
import os

# -----------------------------
# Paths
# -----------------------------
MODEL_DIR = "models"
DATA_DIR = "data/processed"

# Load trained models
lr  = joblib.load(os.path.join(MODEL_DIR, "logistic_regression.pkl"))
svm = joblib.load(os.path.join(MODEL_DIR, "svm.pkl"))
rf  = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
xgb = joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl"))

models = {
    "Logistic Regression": lr,
    "SVM": svm,
    "Random Forest": rf,
    "XGBoost": xgb
}

# Load test data just for demo prediction
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))


# -----------------------------
# Risk category function
# -----------------------------
def get_risk_category(score):
    if score <= 30:
        return "LOW"
    elif score <= 70:
        return "MEDIUM"
    else:
        return "HIGH"


# -----------------------------
# Ensemble prediction for one sample
# -----------------------------
def ensemble_predict(sample_x):
    preds = {}
    probs = {}

    for name, model in models.items():
        pred = int(model.predict(sample_x.reshape(1, -1))[0])
        prob = float(model.predict_proba(sample_x.reshape(1, -1))[0][1])

        preds[name] = pred
        probs[name] = prob

    # Voting
    votes = list(preds.values())
    final_pred = 1 if sum(votes) >= 2 else 0   # majority voting

    # Confidence %
    agree_count = votes.count(final_pred)
    confidence = (agree_count / len(votes)) * 100

    # Risk Score (0-100)
    avg_prob = np.mean(list(probs.values()))
    risk_score = avg_prob * 100
    risk_category = get_risk_category(risk_score)

    return final_pred, confidence, risk_score, risk_category, preds, probs


# -----------------------------
# Demo: Run prediction on sample test row
# -----------------------------
index = 0  # you can change this index
sample_x = X_test[index]
true_y = y_test[index]

final_pred, confidence, risk_score, risk_category, preds, probs = ensemble_predict(sample_x)

print("\n================ Ensemble Prediction Output ================")
print("✅ True label:", true_y)
print("✅ Final Prediction:", "HEART DISEASE ✅" if final_pred == 1 else "NO DISEASE ✅")
print("✅ Confidence (%):", round(confidence, 2))
print("✅ Risk Score (0-100):", round(risk_score, 2))
print("✅ Risk Category:", risk_category)

print("\n--- Individual Model Predictions ---")
for k, v in preds.items():
    print(f"{k}: {v}")

print("\n--- Individual Model Probabilities ---")
for k, v in probs.items():
    print(f"{k}: {round(v, 4)}")
