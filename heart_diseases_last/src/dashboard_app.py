import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt

from validation_engine import validate_patient_input
from recommendations import get_recommendations
from pdf_report import generate_pdf_report
from risk_meter import generate_risk_meter

# -----------------------------
# Paths
# -----------------------------
# -----------------------------
# Dynamic Base Path (Cloud Safe)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
SHAP_DIR = os.path.join(PROJECT_ROOT, "reports", "shap")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")

os.makedirs(SHAP_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# -----------------------------
# Load dataset for feature columns
# -----------------------------
merged_df = pd.read_csv(os.path.join(DATA_DIR, "merged_heart_dataset.csv"))
feature_cols = [c for c in merged_df.columns if c != "target"]

# -----------------------------
# Load scaler + models
# -----------------------------
scaler = joblib.load(os.path.join(DATA_DIR, "scaler.pkl"))

lr = joblib.load(os.path.join(MODEL_DIR, "logistic_regression.pkl"))
svm = joblib.load(os.path.join(MODEL_DIR, "svm.pkl"))
rf = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
xgb = joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl"))

models = {
    "Logistic Regression": lr,
    "SVM": svm,
    "Random Forest": rf,
    "XGBoost": xgb
}

# SHAP background dataset
X_train_scaled = np.load(os.path.join(DATA_DIR, "X_train.npy"))
X_train_df = pd.DataFrame(X_train_scaled, columns=feature_cols)

# -----------------------------
# Session state init
# -----------------------------
if "pred_done" not in st.session_state:
    st.session_state.pred_done = False

if "pred_result" not in st.session_state:
    st.session_state.pred_result = {}

if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None


# -----------------------------
# Utility functions
# -----------------------------
def get_risk_category(score):
    if score <= 30:
        return "LOW"
    elif score <= 70:
        return "MEDIUM"
    else:
        return "HIGH"


def ensemble_predict(sample_x_scaled):
    preds = {}
    probs = {}

    for name, model in models.items():
        pred = int(model.predict(sample_x_scaled)[0])
        prob = float(model.predict_proba(sample_x_scaled)[0][1])
        preds[name] = pred
        probs[name] = prob

    votes = list(preds.values())
    final_pred = 1 if sum(votes) >= 2 else 0

    agree_count = votes.count(final_pred)
    confidence = (agree_count / len(votes)) * 100

    avg_prob = np.mean(list(probs.values()))
    risk_score = avg_prob * 100
    risk_category = get_risk_category(risk_score)

    return final_pred, confidence, risk_score, risk_category, preds, probs


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Heart Disease Prediction Dashboard", layout="wide")

st.markdown("## ❤️ Heart Disease Prediction using Ensemble Learning")
st.caption("Risk Score | Confidence Voting | SHAP Explainability | Recommendations | Input Validation | PDF Report")

st.sidebar.header("🧾 Enter Patient Details")

patient = {}
patient["age"] = st.sidebar.number_input("Age", min_value=1, max_value=120, value=45)
patient["sex"] = st.sidebar.number_input("Sex (0=Female, 1=Male)", min_value=0, max_value=1, value=1)
patient["cp"] = st.sidebar.number_input("Chest Pain Type (cp)", min_value=0, max_value=3, value=1)
patient["trestbps"] = st.sidebar.number_input("Resting Blood Pressure (trestbps)", min_value=60, max_value=250, value=130)
patient["chol"] = st.sidebar.number_input("Cholesterol (chol)", min_value=80, max_value=700, value=250)
patient["fbs"] = st.sidebar.number_input("Fasting Blood Sugar > 120 (fbs)", min_value=0, max_value=1, value=0)
patient["restecg"] = st.sidebar.number_input("Resting ECG (restecg)", min_value=0, max_value=2, value=1)
patient["thalach"] = st.sidebar.number_input("Max Heart Rate (thalach)", min_value=40, max_value=250, value=150)
patient["exang"] = st.sidebar.number_input("Exercise Induced Angina (exang)", min_value=0, max_value=1, value=0)
patient["oldpeak"] = st.sidebar.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
patient["slope"] = st.sidebar.number_input("Slope", min_value=0, max_value=2, value=1)
patient["ca"] = st.sidebar.number_input("Number of major vessels (ca)", min_value=0, max_value=4, value=0)
patient["thal"] = st.sidebar.number_input("Thal", min_value=0, max_value=3, value=2)

predict_btn = st.sidebar.button("✅ Predict")


# -----------------------------
# Prediction action
# -----------------------------
if predict_btn:
    st.session_state.pdf_bytes = None

    # ✅ Validation Engine
    valid, warnings = validate_patient_input(patient)
    if not valid:
        st.error("⚠️ Input validation failed. Please correct the values.")
        for w in warnings:
            st.warning(w)
        st.stop()

    # Prepare input
    input_df = pd.DataFrame([patient], columns=feature_cols)
    input_scaled = scaler.transform(input_df)

    # Ensemble prediction
    final_pred, confidence, risk_score, risk_category, preds, probs = ensemble_predict(input_scaled)

    # SHAP local explanation
    patient_scaled_df = pd.DataFrame(input_scaled, columns=feature_cols)
    explainer = shap.Explainer(xgb, X_train_df)
    local_shap_values = explainer(patient_scaled_df)

    # Save local waterfall plot image (needed for PDF)
    shap.plots.waterfall(local_shap_values[0], show=False)
    local_plot_path = os.path.join(SHAP_DIR, "local_shap_waterfall.png")
    plt.savefig(local_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Extract top SHAP features
    patient_shap = pd.Series(local_shap_values[0].values, index=feature_cols)
    top_positive = patient_shap.sort_values(ascending=False).head(5)
    top_negative = patient_shap.sort_values().head(5)

    # Recommendations
    top_risk_features = list(top_positive.index[:5])
    recs = get_recommendations(risk_score, risk_category, top_risk_features)

    # Store in session state
    st.session_state.pred_done = True
    st.session_state.pred_result = {
        "patient": patient,
        "final_pred": final_pred,
        "confidence": confidence,
        "risk_score": risk_score,
        "risk_category": risk_category,
        "preds": preds,
        "probs": probs,
        "local_plot_path": local_plot_path,
        "top_positive": top_positive,
        "top_negative": top_negative,
        "recs": recs
    }


# -----------------------------
# Show results
# -----------------------------
if st.session_state.pred_done:
    res = st.session_state.pred_result

    col1, col2, col3 = st.columns(3)
    col1.metric("Final Prediction", "HEART DISEASE ✅" if res["final_pred"] == 1 else "NO DISEASE ✅")
    col2.metric("Ensemble Confidence (%)", f"{res['confidence']:.2f}")
    col3.metric("Risk Score (0–100)", f"{res['risk_score']:.2f} ({res['risk_category']})")

    st.divider()

    # Model voting table
    st.subheader("📌 Individual Model Voting")
    model_table = pd.DataFrame({
        "Model": list(res["preds"].keys()),
        "Prediction (0/1)": list(res["preds"].values()),
        "Probability": [round(v, 4) for v in res["probs"].values()]
    })
    st.dataframe(model_table, use_container_width=True)

    st.divider()

    # SHAP section
    st.subheader("🧠 Explainable AI (SHAP)")

    shap_summary_img = os.path.join(SHAP_DIR, "shap_summary_plot.png")
    shap_bar_img = os.path.join(SHAP_DIR, "shap_feature_importance_bar.png")

    colA, colB = st.columns(2)

    if os.path.exists(shap_summary_img):
        colA.image(shap_summary_img, caption="SHAP Summary Plot (Global)", use_container_width=True)
    else:
        colA.warning("Global SHAP summary plot not found. Run shap_explain.py first.")

    if os.path.exists(shap_bar_img):
        colB.image(shap_bar_img, caption="SHAP Feature Importance (Global Bar)", use_container_width=True)
    else:
        colB.warning("Global SHAP bar plot not found. Run shap_explain.py first.")

    st.divider()

    # ✅ Dynamic local SHAP waterfall
    st.subheader("🧠 Local SHAP Waterfall (Dynamic per Patient)")
    st.image(res["local_plot_path"], caption="Local SHAP Waterfall Plot", use_container_width=True)

    st.write("🔺 Top 5 Features Increasing Risk")
    st.dataframe(res["top_positive"])

    st.write("🔻 Top 5 Features Decreasing Risk")
    st.dataframe(res["top_negative"])

    st.divider()

    # Recommendations
    st.subheader("✅ Personalized Recommendations")
    for i, r in enumerate(res["recs"], 1):
        st.write(f"{i}. {r}")

    st.divider()

    # ✅ PDF Report section
    st.subheader("📄 PDF Report (with Risk Meter + SHAP Images)")

    if st.button("⬇️ Generate PDF Report"):
        prediction_text = "HEART DISEASE DETECTED" if res["final_pred"] == 1 else "NO HEART DISEASE DETECTED"

        # Generate risk meter image
        risk_meter_img = generate_risk_meter(res["risk_score"], save_path=os.path.join(REPORT_DIR, "risk_meter.png"))

        pdf_path = generate_pdf_report(
            patient_details=res["patient"],
            final_prediction=prediction_text,
            confidence=res["confidence"],
            risk_score=res["risk_score"],
            risk_category=res["risk_category"],
            top_positive=list(res["top_positive"].index),
            top_negative=list(res["top_negative"].index),
            recommendations=res["recs"],
            save_path=os.path.join(REPORT_DIR, "heart_disease_report.pdf"),
            shap_summary_img=shap_summary_img,
            shap_bar_img=shap_bar_img,
            shap_local_img=res["local_plot_path"],
            risk_meter_img=risk_meter_img
        )

        with open(pdf_path, "rb") as f:
            st.session_state.pdf_bytes = f.read()

        st.success("✅ PDF generated successfully. Download enabled below.")

    if st.session_state.pdf_bytes is not None:
        st.download_button(
            "✅ Download PDF Report",
            data=st.session_state.pdf_bytes,
            file_name="heart_disease_report.pdf",
            mime="application/pdf"
        )

else:
    st.info("👈 Enter patient details and click ✅ Predict")
