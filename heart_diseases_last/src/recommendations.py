def get_recommendations(risk_score, risk_category, top_risk_features):
    rec = []

    # -----------------------------
    # Risk-level recommendations
    # -----------------------------
    if risk_category == "LOW":
        rec.append("Maintain a healthy lifestyle with balanced diet and regular exercise.")
        rec.append("Do routine health checkups once per year.")
        rec.append("Avoid smoking and manage stress properly.")

    elif risk_category == "MEDIUM":
        rec.append("Monitor blood pressure and cholesterol regularly (every 3–6 months).")
        rec.append("Follow cardio-friendly diet (low salt, low saturated fat).")
        rec.append("Do daily exercise (walking/cycling) at least 30 minutes.")
        rec.append("Consult a doctor if symptoms like chest pain or breathlessness occur.")

    elif risk_category == "HIGH":
        rec.append("Immediate medical consultation is recommended.")
        rec.append("Get ECG / Stress Test done for further confirmation.")
        rec.append("Strictly control BP, cholesterol and sugar levels.")
        rec.append("Avoid heavy physical stress until doctor approval.")

    # -----------------------------
    # SHAP Feature-based recommendations
    # -----------------------------
    feature_map = {
        "chol": [
            "Reduce oily/fast foods and increase fiber intake (fruits, vegetables).",
            "Consider a Lipid Profile test if cholesterol is high."
        ],
        "trestbps": [
            "Reduce salt intake and monitor blood pressure daily.",
            "Practice stress control (yoga/meditation)."
        ],
        "fbs": [
            "Control sugar intake and check blood glucose levels.",
            "Consider HbA1c test for diabetes screening."
        ],
        "thalach": [
            "Improve cardiovascular fitness with gradual exercise.",
            "Avoid sudden heavy workouts; consult doctor if fatigue occurs."
        ],
        "exang": [
            "Avoid intense exertion if exercise-induced angina is present.",
            "Consult cardiologist for stress test evaluation."
        ],
        "oldpeak": [
            "High ST depression may indicate risk; ECG monitoring recommended.",
            "Seek professional consultation for heart function assessment."
        ],
        "ca": [
            "Coronary vessel condition impacts risk; doctor evaluation recommended.",
            "Follow strict cardiac lifestyle changes and monitoring."
        ],
        "cp": [
            "If chest pain type indicates risk, consult doctor immediately.",
            "Do regular heart checkups (ECG + BP monitoring)."
        ],
        "age": [
            "As age increases, regular heart screening is recommended.",
            "Maintain healthy BMI, sleep, and stress control."
        ]
    }

    for feat in top_risk_features:
        if feat in feature_map:
            rec.extend(feature_map[feat])

    # Remove duplicates while preserving order
    final_rec = []
    for r in rec:
        if r not in final_rec:
            final_rec.append(r)

    return final_rec


# -----------------------------
# Demo Test
# -----------------------------
if __name__ == "__main__":
    # sample values (you can connect this with ensemble_predict output)
    risk_score = 60.67
    risk_category = "MEDIUM"
    top_risk_features = ["slope", "age", "ca", "thalach", "chol"]

    recs = get_recommendations(risk_score, risk_category, top_risk_features)

    print("\n================ Personalized Recommendations ================")
    print("✅ Risk Score:", risk_score)
    print("✅ Risk Category:", risk_category)
    print("✅ Top Risk Features:", top_risk_features)

    print("\n📌 Recommendations:")
    for i, r in enumerate(recs, 1):
        print(f"{i}. {r}")
