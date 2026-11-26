import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# ==========================================================
# LOAD TRAINED MODEL
# ==========================================================
model = joblib.load("diabetes_model2_v2.pkl")

# ==========================================================
# HbA1c Approximation Formula
# HbA1c ≈ (Average Glucose + 46.7) / 28.7
# ==========================================================
def glucose_to_hba1c(avg_glucose):
    return (avg_glucose + 46.7) / 28.7


# ==========================================================
# Digital Twin Simulation Function
# ==========================================================
def simulate_digital_twin(patient, days=90, scenario="no_intervention"):

    glucose = patient["FastingGlucose"]
    bmi = patient["BMI"]
    activity = patient["Activity"]

    glucose_history = []
    hba1c_history = []

    for day in range(days):

        # -------------------------------
        # Natural disease progression
        # -------------------------------
        # higher BMI increases glucose
        bmi_effect = (bmi - 25) * 0.05

        # activity reduces glucose
        activity_effect = -(activity / 1000) * 0.8

        daily_change = bmi_effect + activity_effect

        # -------------------------------
        # Scenario interventions
        # -------------------------------
        if scenario == "lifestyle":
            # lifestyle reduces glucose gradually
            daily_change -= 0.15
            bmi -= 0.003  # BMI slowly reduces

        elif scenario == "medication":
            # medication reduces glucose faster
            daily_change -= 0.30

        # -------------------------------
        # Add random biological noise
        # -------------------------------
        noise = np.random.normal(0, 0.8)
        glucose = glucose + daily_change + noise

        # clamp glucose
        if glucose < 60:
            glucose = 60

        glucose_history.append(glucose)

        # HbA1c calculated from average glucose so far
        avg_glucose = np.mean(glucose_history)
        hba1c = glucose_to_hba1c(avg_glucose)

        hba1c_history.append(hba1c)

    return glucose_history, hba1c_history


# ==========================================================
# PATIENT INPUT (Example Patient)
# ==========================================================
patient = {
    "Age": 45,
    "BMI": 32,
    "FastingGlucose": 140,
    "SystolicBP": 140,
    "DiastolicBP": 90,
    "LDL": 140,
    "HDL": 38,
    "Triglycerides": 210,
    "Smoking": 1,
    "Activity": 100
}

# ==========================================================
# MODEL PREDICTION FOR THIS PATIENT
# ==========================================================
features = [
    "Age", "BMI", "FastingGlucose",
    "SystolicBP", "DiastolicBP",
    "LDL", "HDL", "Triglycerides",
    "Smoking", "Activity"
]

X_patient = pd.DataFrame([patient])[features]

predicted_class = model.predict(X_patient)[0]

class_labels = {
    0: "Normal",
    1: "Prediabetes",
    2: "Diabetes"
}

print("\n===============================")
print(" DIGITAL TWIN PATIENT PREDICTION")
print("===============================")
print("Predicted Diabetes Risk:", class_labels[predicted_class])


# ==========================================================
# SIMULATE THREE SCENARIOS
# ==========================================================
days = 90

glucose_no, hba1c_no = simulate_digital_twin(patient, days, "no_intervention")
glucose_life, hba1c_life = simulate_digital_twin(patient, days, "lifestyle")
glucose_med, hba1c_med = simulate_digital_twin(patient, days, "medication")

time = np.arange(days)


# ==========================================================
# PLOT 1: GLUCOSE TREND
# ==========================================================
plt.figure(figsize=(10,5))
plt.plot(time, glucose_no, label="No Intervention")
plt.plot(time, glucose_life, label="Lifestyle Improvement")
plt.plot(time, glucose_med, label="Medication (Metformin Effect)")

plt.axhline(126, linestyle="--", label="Diabetes Threshold (126 mg/dL)")
plt.title("Digital Twin Simulation: Fasting Glucose Trend (90 Days)")
plt.xlabel("Days")
plt.ylabel("Fasting Glucose (mg/dL)")
plt.legend()
plt.tight_layout()
plt.savefig("digital_twin_glucose_trend.png", dpi=300)
plt.show()

print("✅ Saved: digital_twin_glucose_trend.png")


# ==========================================================
# PLOT 2: HbA1c TREND
# ==========================================================
plt.figure(figsize=(10,5))
plt.plot(time, hba1c_no, label="No Intervention")
plt.plot(time, hba1c_life, label="Lifestyle Improvement")
plt.plot(time, hba1c_med, label="Medication (Metformin Effect)")

plt.axhline(5.7, linestyle="--", label="Prediabetes Threshold (5.7%)")
plt.axhline(6.5, linestyle="--", label="Diabetes Threshold (6.5%)")

plt.title("Digital Twin Simulation: HbA1c Trend (90 Days)")
plt.xlabel("Days")
plt.ylabel("HbA1c (%)")
plt.legend()
plt.tight_layout()
plt.savefig("digital_twin_hba1c_trend.png", dpi=300)
plt.show()

print("✅ Saved: digital_twin_hba1c_trend.png")


# ==========================================================
# SAVE FULL SIMULATION RESULTS
# ==========================================================
results = pd.DataFrame({
    "Day": time,
    "Glucose_NoIntervention": glucose_no,
    "Glucose_Lifestyle": glucose_life,
    "Glucose_Medication": glucose_med,
    "HbA1c_NoIntervention": hba1c_no,
    "HbA1c_Lifestyle": hba1c_life,
    "HbA1c_Medication": hba1c_med
})

results.to_csv("digital_twin_simulation_results.csv", index=False)

print("✅ Saved: digital_twin_simulation_results.csv")

print("\n🎉 Digital Twin Simulation Completed Successfully!")