import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="DiabeCare AI: Clinical Risk Assessment System",
    layout="wide"
)

# ==========================================================
# THEME TOGGLE (DARK/LIGHT)
# ==========================================================
st.sidebar.header("⚙ Settings")
theme_mode = st.sidebar.toggle("🌙 Dark Mode", value=True)

if theme_mode:
    st.markdown("""
        <style>
        .stApp {
            background-color: #0e1117;
            color: white;
        }
        h1,h2,h3,h4,h5,h6,p,label,div {
            color: white !important;
        }
        section[data-testid="stSidebar"] {
            background-color: #161b22;
        }
        .stMetric {
            background-color: #1f2630;
            padding: 15px;
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
        <style>
        .stApp {
            background-color: white;
            color: black;
        }
        h1,h2,h3,h4,h5,h6,p,label,div {
            color: black !important;
        }
        section[data-testid="stSidebar"] {
            background-color: #f0f2f6;
        }
        .stMetric {
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

# ==========================================================
# LOAD MODEL (XGBOOST JSON)
# ==========================================================
model = xgb.XGBClassifier()
model.load_model("xgb_model.json")

# ==========================================================
# TITLE
# ==========================================================
st.title("🩺 DiabeCare AI: Clinical Risk Assessment System")
st.write("Enter patient data to predict diabetes risk and simulate future trends.")

# ==========================================================
# INPUT PANEL (ONLY REQUIRED INPUTS)
# ==========================================================
st.sidebar.header("🧾 Patient Input Panel")

age = st.sidebar.number_input("Age (years)", min_value=1, max_value=120, value=35)

bmi = st.sidebar.number_input("BMI (kg/m²)", min_value=10.0, max_value=60.0, value=27.0)

glucose = st.sidebar.number_input(
    "Fasting Blood Glucose (mg/dL)",
    min_value=50,
    max_value=400,
    value=110
)

activity = st.sidebar.number_input(
    "Physical Activity (minutes/week)",
    min_value=0,
    max_value=10000,
    value=500
)

# ==========================================================
# CREATE INPUT DATAFRAME (MODEL NEEDS FULL FEATURES)
# ==========================================================
patient = pd.DataFrame([{
    "Age": age,
    "BMI": bmi,
    "FastingGlucose": glucose,

    # Default clinical values (fixed, because UI only needs 4 inputs)
    "SystolicBP": 120,
    "DiastolicBP": 80,
    "LDL": 110,
    "HDL": 50,
    "Triglycerides": 150,
    "Smoking": 0,
    "Activity": activity
}])

# ==========================================================
# MODEL PREDICTION
# ==========================================================
prediction = int(model.predict(patient)[0])
probabilities = model.predict_proba(patient)[0]

risk_labels = {
    0: "Normal",
    1: "Prediabetes",
    2: "Diabetes"
}
risk = risk_labels[prediction]

# ==========================================================
# HbA1c ESTIMATION (Clinical Approximation Formula)
# HbA1c ≈ (Avg Glucose + 46.7) / 28.7
# ==========================================================
predicted_hba1c = (glucose + 46.7) / 28.7

# ==========================================================
# OUTPUT PANEL
# ==========================================================
st.header("📊 Output Panel")

col1, col2 = st.columns(2)

with col1:
    st.metric("Predicted HbA1c (%)", f"{predicted_hba1c:.2f}")

with col2:
    st.metric("Predicted Risk Level", risk)

# ==========================================================
# PROBABILITY DISPLAY
# ==========================================================
st.subheader("📌 Prediction Confidence")

prob_df = pd.DataFrame({
    "Class": ["Normal", "Prediabetes", "Diabetes"],
    "Probability": probabilities
})

st.bar_chart(prob_df.set_index("Class"))

# ==========================================================
# DIGITAL TWIN SIMULATION
# ==========================================================
st.header("📈 Graph Panel: Trend Visualization (90 Days Simulation)")

days = 90

def simulate(glucose, bmi, activity, scenario):
    glucose_values = []
    hba1c_values = []

    current_glucose = glucose
    current_bmi = bmi

    for day in range(days):

        bmi_effect = (current_bmi - 25) * 0.05
        activity_effect = -(activity / 1000) * 0.8

        daily_change = bmi_effect + activity_effect

        # --------------------------------------------------
        # INTERVENTION SCENARIOS
        # --------------------------------------------------
        if scenario == "Lifestyle":
            daily_change -= 0.20
            current_bmi -= 0.005

        elif scenario == "Medication":
            daily_change -= 0.35

        # Random biological noise
        noise = np.random.normal(0, 0.8)

        current_glucose += daily_change + noise

        if current_glucose < 60:
            current_glucose = 60

        glucose_values.append(current_glucose)

        avg_glucose = np.mean(glucose_values)
        hba1c = (avg_glucose + 46.7) / 28.7
        hba1c_values.append(hba1c)

    return glucose_values, hba1c_values

# ==========================================================
# RUN ALL SCENARIOS
# ==========================================================
g_none, h_none = simulate(glucose, bmi, activity, "None")
g_life, h_life = simulate(glucose, bmi, activity, "Lifestyle")
g_med, h_med = simulate(glucose, bmi, activity, "Medication")

# ==========================================================
# GLUCOSE GRAPH
# ==========================================================
st.subheader("🩸 Glucose Trend (mg/dL)")

fig1, ax1 = plt.subplots(figsize=(10, 4))

ax1.plot(g_none, label="No Intervention")
ax1.plot(g_life, label="Lifestyle Improvement")
ax1.plot(g_med, label="Medication")

ax1.axhline(126, linestyle="--", label="Diabetes Cutoff (126 mg/dL)")

ax1.set_xlabel("Days")
ax1.set_ylabel("Glucose (mg/dL)")
ax1.legend()

st.pyplot(fig1)

# ==========================================================
# HbA1c GRAPH
# ==========================================================
st.subheader("🧪 HbA1c Trend (%)")

fig2, ax2 = plt.subplots(figsize=(10, 4))

ax2.plot(h_none, label="No Intervention")
ax2.plot(h_life, label="Lifestyle Improvement")
ax2.plot(h_med, label="Medication")

ax2.axhline(5.7, linestyle="--", label="Prediabetes Cutoff (5.7%)")
ax2.axhline(6.5, linestyle="--", label="Diabetes Cutoff (6.5%)")

ax2.set_xlabel("Days")
ax2.set_ylabel("HbA1c (%)")
ax2.legend()

st.pyplot(fig2)

# ==========================================================
# FINAL MESSAGE
# ==========================================================
st.success("✅ DiabeCare AI Completed Successfully")
