def validate_patient_input(patient):
    """
    patient: dict of input values
    returns: (is_valid, warnings_list)
    """

    warnings = []

    # Basic checks
    age = patient.get("age", None)
    trestbps = patient.get("trestbps", None)
    chol = patient.get("chol", None)
    thalach = patient.get("thalach", None)
    oldpeak = patient.get("oldpeak", None)

    # ---- Age validation ----
    if age is None:
        warnings.append("Age is missing.")
    else:
        if age < 1 or age > 120:
            warnings.append("Age is unrealistic. Please enter valid age (1–120).")

    # ---- BP validation ----
    if trestbps is None:
        warnings.append("Resting Blood Pressure (trestbps) is missing.")
    else:
        if trestbps < 60 or trestbps > 250:
            warnings.append("trestbps seems abnormal (expected 60–250 mmHg).")

    # ---- Cholesterol validation ----
    if chol is None:
        warnings.append("Cholesterol (chol) is missing.")
    else:
        if chol < 80 or chol > 700:
            warnings.append("chol seems abnormal (expected 80–700 mg/dl).")

    # ---- Heart rate validation ----
    if thalach is None:
        warnings.append("Max Heart Rate (thalach) is missing.")
    else:
        if thalach < 40 or thalach > 250:
            warnings.append("thalach seems abnormal (expected 40–250 bpm).")

    # ---- Oldpeak validation ----
    if oldpeak is None:
        warnings.append("Oldpeak is missing.")
    else:
        if oldpeak < 0 or oldpeak > 10:
            warnings.append("oldpeak seems abnormal (expected 0–10).")

    # ---- Contradiction rules (high-tech) ----
    if age is not None and chol is not None:
        if age < 25 and chol > 500:
            warnings.append("Contradiction: Age is very low but cholesterol is extremely high.")

    if age is not None and trestbps is not None:
        if age < 25 and trestbps > 200:
            warnings.append("Contradiction: Age is low but BP is extremely high.")

    is_valid = len(warnings) == 0
    return is_valid, warnings


# -----------------------------
# Demo Test
# -----------------------------
if __name__ == "__main__":
    # Example patient input
    sample_patient = {
        "age": 20,
        "trestbps": 240,
        "chol": 600,
        "thalach": 120,
        "oldpeak": 2.5
    }

    valid, warnings = validate_patient_input(sample_patient)

    print("\n================ Input Validation Engine ================")
    print("✅ Input Valid:", valid)

    if not valid:
        print("\n⚠️ Warnings Found:")
        for w in warnings:
            print("-", w)
    else:
        print("✅ No validation issues found.")
