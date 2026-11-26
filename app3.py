import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import xgboost as xgb
import joblib

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="DiabeCare AI: Clinical Risk Assessment System",
    page_icon="🩺",
    layout="wide"
)

# ==========================================================
# THEME TOGGLE (DARK/LIGHT)
# ==========================================================
st.sidebar.header("⚙ Settings")
theme_mode = st.sidebar.toggle("🌙 Dark Mode", value=True)

if theme_mode:
    # DARK MODE CSS
    st.markdown("""
        <style>
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        section[data-testid="stSidebar"] {
            background-color: #161B22;
        }
        .metric-card {
            background-color: #21262D;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #30363D;
        }
        </style>
    """, unsafe_allow_html=True)
else:
    # LIGHT MODE CSS
    st.markdown("""
        <style>
        .stApp {
            background-color: #FFFFFF;
            color: #000000;
        }
        section[data-testid="stSidebar"] {
            background-color: #F0F2F6;
        }
        .metric-card {
            background-color: #F8F9FA;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #E0E0E0;
        }
        </style>
    """, unsafe_allow_html=True)

# ==========================================================
# LOAD MODEL (XGBOOST JSON WITH JOBLIB FALLBACK)
# ==========================================================
@st.cache_resource
def load_prediction_model():
    if os.path.exists("xgb_model.json"):
        m = xgb.XGBClassifier()
        m.load_model("xgb_model.json")
        return m
    return joblib.load("diabetes_model2_v2.pkl")

model = load_prediction_model()

# ==========================================================
# TITLE
# ==========================================================
st.title("🩺 DiabeCare AI — Clinical Risk Assessment System")
st.write("Enter patient data to predict diabetes risk and simulate future trends.")

# ==========================================================
# INPUT PANEL (ONLY REQUIRED INPUTS)
# ==========================================================
st.sidebar.header("🧾 Patient Input Panel")

age = st.sidebar.number_input("Age (years)", min_value=1, max_value=120, value=35)

bmi = st.sidebar.number_input(
    "BMI (kg/m²)",
    min_value=10.0,
    max_value=60.0,
    value=24.5,
    help="Underweight: <18.5 | Normal: 18.5-24.9 | Overweight: 25-29.9 | Obese: ≥30"
)

glucose = st.sidebar.number_input(
    "Fasting Blood Glucose (mg/dL)",
    min_value=50,
    max_value=300,
    value=95,
    help="Normal: <100 | Prediabetes: 100-125 | Diabetes: ≥126"
)

activity = st.sidebar.number_input(
    "Physical Activity (minutes/week)",
    min_value=0,
    max_value=1000,
    value=150,
    help="Recommended: ≥150 mins/week"
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
    "PhysicalActivity": activity
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
risk = risk_labels.get(prediction, "Unknown")

# ==========================================================
# HbA1c ESTIMATION (Clinical Approximation Formula)
# HbA1c ≈ (Avg Glucose + 46.7) / 28.7
# ==========================================================
predicted_hba1c = (glucose + 46.7) / 28.7

# ==========================================================
# OUTPUT PANEL
# ==========================================================
st.header("📊 Output Panel: Clinical Assessment")

col1, col2 = st.columns(2)

with col1:
    st.metric("Estimated HbA1c (%)", f"{predicted_hba1c:.1f}%")

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
# DIGITAL TWIN SIMULATION FUNCTION
# ==========================================================
st.header("📈 Graph Panel: Trend Visualization (90 Days Simulation)")

def simulate(base_glucose, bmi, activity, mode):
    days = 90
    t = np.arange(days)
    
    if mode == "No Intervention":
        trend = base_glucose + (0.05 * t) + (bmi * 0.02)
    elif mode == "Lifestyle Improvement":
        trend = base_glucose - (0.08 * t) - (activity * 0.01)
    elif mode == "Medication":
        trend = base_glucose - (0.15 * t)
    
    # Clip realistic biological limits
    trend = np.clip(trend, 70, 300)
    hba1c_trend = (trend + 46.7) / 28.7
    return trend, hba1c_trend

# Run simulations
g_none, h_none = simulate(glucose, bmi, activity, "No Intervention")
g_life, h_life = simulate(glucose, bmi, activity, "Lifestyle Improvement")
g_med, h_med = simulate(glucose, bmi, activity, "Medication")

# ==========================================================
# GLUCOSE GRAPH
# ==========================================================
st.subheader("🩸 Glucose Trend (mg/dL)")

fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(g_none, label="No Intervention", color="red")
ax1.plot(g_life, label="Lifestyle Improvement", color="green")
ax1.plot(g_med, label="Medication", color="blue")
ax1.set_xlabel("Days")
ax1.set_ylabel("Glucose (mg/dL)")
ax1.legend()
ax1.grid(True, linestyle="--", alpha=0.5)

st.pyplot(fig1)

# ==========================================================
# HbA1c GRAPH
# ==========================================================
st.subheader("🧪 HbA1c Trend (%)")

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(h_none, label="No Intervention", color="red")
ax2.plot(h_life, label="Lifestyle Improvement", color="green")
ax2.plot(h_med, label="Medication", color="blue")
ax2.set_xlabel("Days")
ax2.set_ylabel("HbA1c (%)")
ax2.legend()
ax2.grid(True, linestyle="--", alpha=0.5)

st.pyplot(fig2)

# ==========================================================
# FINAL MESSAGE
# ==========================================================
st.success("✅ DiabeCare AI: Clinical Risk Assessment System Completed Successfully")
