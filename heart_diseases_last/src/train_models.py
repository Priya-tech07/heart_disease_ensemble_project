import numpy as np
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from xgboost import XGBClassifier


# -----------------------------
# Paths
# -----------------------------
DATA_DIR = "data/processed"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load processed train-test
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))

print("✅ Loaded processed train/test data")
print("Train:", X_train.shape, " Test:", X_test.shape)


# -----------------------------
# Utility function
# -----------------------------
def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)

    # ROC-AUC needs probability scores
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # for SVM if probability=False, fallback to decision_function
        y_proba = model.decision_function(X_test)

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n================ {model_name} ================")
    print("✅ Accuracy:", round(acc * 100, 2), "%")
    print("✅ ROC-AUC :", round(roc, 4))
    print("✅ Confusion Matrix:\n", cm)
    print("\n✅ Classification Report:\n", classification_report(y_test, y_pred))


# -----------------------------
# 1) Logistic Regression
# -----------------------------
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
joblib.dump(lr, os.path.join(MODEL_DIR, "logistic_regression.pkl"))
evaluate_model(lr, X_test, y_test, "Logistic Regression")


# -----------------------------
# 2) SVM
# -----------------------------
svm = SVC(kernel="rbf", probability=True, random_state=42)
svm.fit(X_train, y_train)
joblib.dump(svm, os.path.join(MODEL_DIR, "svm.pkl"))
evaluate_model(svm, X_test, y_test, "SVM (RBF)")


# -----------------------------
# 3) Random Forest
# -----------------------------
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    max_depth=None
)
rf.fit(X_train, y_train)
joblib.dump(rf, os.path.join(MODEL_DIR, "random_forest.pkl"))
evaluate_model(rf, X_test, y_test, "Random Forest")


# -----------------------------
# 4) XGBoost
# -----------------------------
xgb = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=42
)
xgb.fit(X_train, y_train)
joblib.dump(xgb, os.path.join(MODEL_DIR, "xgboost.pkl"))
evaluate_model(xgb, X_test, y_test, "XGBoost")


print("\n🎉 All models trained and saved in /models folder!")
