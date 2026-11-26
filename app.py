import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="Diabetes Digital Twin",
    layout="centered"
)

# ==========================================================
# LOAD MODEL
# ==========================================================
model = joblib.load("diabetes_model2_v2.pkl")

# ==========================================================
# TITLE
# ==========================================================
st.title("🩺 Diabetes Digital Twin")
st.write("AI-Based Diabetes Risk Prediction System")

# ==========================================================
# INPUT PANEL
# ==========================================================
st.header("📥 Patient Input")

age = st.slider("Age", 18, 90, 35)

bmi = st.slider("BMI", 15.0, 50.0, 28.0)

glucose = st.slider("Fasting Glucose", 70, 300, 120)

activity = st.slider("Physical Activity", 0, 5000, 1000)

# ==========================================================
# DEFAULT VALUES FOR HIDDEN FEATURES
# ==========================================================
patient = pd.DataFrame([{
    "Age": age,
    "BMI": bmi,
    "FastingGlucose": glucose,
    "SystolicBP": 120,
    "DiastolicBP": 80,
    "LDL": 100,
    "HDL": 50,
    "Triglycerides": 150,
    "Smoking": 0,
    "Activity": activity
}])

# ==========================================================
# PREDICTION
# ==========================================================
prediction = model.predict(patient)[0]

risk_labels = {
    0: "Normal",
    1: "Prediabetes",
    2: "Diabetes"
}

risk = risk_labels[prediction]

# ==========================================================
# PREDICTED HBA1C
# ==========================================================
predicted_hba1c = (glucose + 46.7) / 28.7

# ==========================================================
# OUTPUT PANEL
# ==========================================================
st.header("📊 Prediction Output")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Predicted HbA1c",
        f"{predicted_hba1c:.2f}%"
    )

with col2:
    st.metric(
        "Risk Level",
        risk
    )

# ==========================================================
# DIGITAL TWIN SIMULATION
# ==========================================================
st.header("📈 Trend Visualization")

days = 90

glucose_values = []

current_glucose = glucose

for day in range(days):

    bmi_effect = (bmi - 25) * 0.05

    activity_effect = -(activity / 1000) * 0.8

    noise = np.random.normal(0, 0.8)

    current_glucose += bmi_effect + activity_effect + noise

    if current_glucose < 60:
        current_glucose = 60

    glucose_values.append(current_glucose)

# ==========================================================
# GRAPH
# ==========================================================
fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(glucose_values)

ax.set_xlabel("Days")

ax.set_ylabel("Glucose Level")

ax.set_title("Digital Twin Glucose Trend")

st.pyplot(fig)

# ==========================================================
# FOOTER
# ==========================================================
st.success("Simulation Completed Successfully")