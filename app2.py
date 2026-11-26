import io
import os
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================
st.set_page_config(
    page_title="DiabeCare AI: Clinical Decision Support & Digital Twin",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# EXACT COLOR PALETTE CONSTANTS
# ==========================================================
# - Primary Accent (Deep Clinical Teal): #0F5C6E
# - Primary Hover: #0B4654
# - Main Background (Warm Alabaster): #FAF9F6
# - Sidebar / Secondary Background (Soft Stone): #F0EFEA
# - Card / Container Surface: #FFFFFF
# - Border / Divider: #E2E0D8
# - Primary Text (Charcoal Ink): #1E242B
# - Muted / Secondary Text: #64748B
# - Low Risk / Normal (Sage Teal): #2A7B6F
# - Borderline / Prediabetes (Warm Ochre): #D98A2B
# - High Risk / Diabetic (Terracotta Red): #C84B31

COLOR_PRIMARY = "#0F5C6E"
COLOR_PRIMARY_HOVER = "#0B4654"
COLOR_BG_MAIN = "#FAF9F6"
COLOR_BG_SIDEBAR = "#F0EFEA"
COLOR_SURFACE_CARD = "#FFFFFF"
COLOR_BORDER = "#E2E0D8"
COLOR_TEXT_MAIN = "#1E242B"
COLOR_TEXT_MUTED = "#64748B"
COLOR_RISK_NORMAL = "#2A7B6F"
COLOR_RISK_PREDIAB = "#D98A2B"
COLOR_RISK_DIAB = "#C84B31"

# ==========================================================
# CUSTOM CSS OVERRIDES (KILLING DEFAULT STREAMLIT LOOK)
# ==========================================================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global Typography & Background */
    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    .stApp {{
        background-color: {COLOR_BG_MAIN} !important;
        color: {COLOR_TEXT_MAIN} !important;
    }}

    /* Sidebar Overrides */
    section[data-testid="stSidebar"] {{
        background-color: {COLOR_BG_SIDEBAR} !important;
        border-right: 1px solid {COLOR_BORDER} !important;
    }}

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div {{
        color: {COLOR_TEXT_MAIN} !important;
    }}

    /* Hero Header */
    .hero-container {{
        background: linear-gradient(135deg, {COLOR_SURFACE_CARD} 0%, {COLOR_BG_SIDEBAR} 100%);
        border: 1px solid {COLOR_BORDER};
        padding: 22px 28px;
        border-radius: 14px;
        margin-bottom: 22px;
        box-shadow: 0 4px 16px rgba(30, 36, 43, 0.04);
    }}

    .hero-title {{
        font-size: 1.85rem;
        font-weight: 800;
        color: {COLOR_PRIMARY} !important;
        margin: 0;
        letter-spacing: -0.5px;
    }}

    .hero-subtitle {{
        color: {COLOR_TEXT_MUTED} !important;
        font-size: 0.95rem;
        font-weight: 400;
        margin: 6px 0 0 0;
    }}

    .status-badge {{
        display: inline-flex;
        align-items: center;
        background: rgba(42, 123, 111, 0.12);
        color: {COLOR_RISK_NORMAL} !important;
        border: 1px solid {COLOR_RISK_NORMAL};
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        float: right;
    }}

    /* Card Containers */
    .card {{
        background: {COLOR_SURFACE_CARD};
        border: 1px solid {COLOR_BORDER};
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(30, 36, 43, 0.04);
        margin-bottom: 18px;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }}

    .card:hover {{
        border-color: {COLOR_PRIMARY};
        box-shadow: 0 6px 16px rgba(15, 92, 110, 0.08);
    }}

    .card-title {{
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: {COLOR_TEXT_MUTED} !important;
        font-weight: 700;
        margin-bottom: 8px;
    }}

    .card-value {{
        font-size: 1.9rem;
        font-weight: 800;
        color: {COLOR_TEXT_MAIN} !important;
    }}

    .card-footer {{
        font-size: 0.78rem;
        color: {COLOR_TEXT_MUTED} !important;
        margin-top: 8px;
    }}

    /* Clinical Badges */
    .badge-normal {{
        background: rgba(42, 123, 111, 0.12);
        color: {COLOR_RISK_NORMAL} !important;
        border: 1.5px solid {COLOR_RISK_NORMAL};
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }}

    .badge-prediabetes {{
        background: rgba(217, 138, 43, 0.12);
        color: {COLOR_RISK_PREDIAB} !important;
        border: 1.5px solid {COLOR_RISK_PREDIAB};
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }}

    .badge-diabetes {{
        background: rgba(200, 75, 49, 0.12);
        color: {COLOR_RISK_DIAB} !important;
        border: 1.5px solid {COLOR_RISK_DIAB};
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }}

    /* Delta Impact Box */
    .delta-box {{
        background: {COLOR_SURFACE_CARD};
        border: 1px solid {COLOR_BORDER};
        border-radius: 10px;
        padding: 16px 18px;
        margin-top: 6px;
    }}

    /* Streamlit Button & Download Customization */
    .stDownloadButton > button,
    button[kind="primary"] {{
        background-color: {COLOR_PRIMARY} !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        transition: background-color 0.2s ease, transform 0.1s ease !important;
    }}

    .stDownloadButton > button:hover,
    button[kind="primary"]:hover {{
        background-color: {COLOR_PRIMARY_HOVER} !important;
        color: #FFFFFF !important;
        transform: translateY(-1px);
    }}

    /* Tab Customization */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        border-bottom: 2px solid {COLOR_BORDER} !important;
    }}

    .stTabs [data-baseweb="tab"] {{
        color: {COLOR_TEXT_MUTED} !important;
        font-weight: 600;
        padding: 10px 18px;
        border-radius: 8px 8px 0 0;
    }}

    .stTabs [aria-selected="true"] {{
        color: {COLOR_PRIMARY} !important;
        border-bottom: 3px solid {COLOR_PRIMARY} !important;
        background-color: transparent !important;
    }}

    /* Sliders & Accents */
    div[data-baseweb="slider"] div {{
        color: {COLOR_PRIMARY} !important;
    }}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD MODEL & SCALER
# ==========================================================
@st.cache_resource
def load_ml_assets():
    model = xgb.XGBClassifier()
    model.load_model("xgb_model.json")
    
    scaler = None
    if os.path.exists("scaler.pkl"):
        scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_ml_assets()

# ==========================================================
# HERO HEADER
# ==========================================================
st.markdown(f"""
<div class="hero-container">
    <span class="status-badge">● CLINICAL ENGINE ACTIVE</span>
    <h1 class="hero-title">🩺 DiabeCare AI: Clinical Decision Support & Digital Twin</h1>
    <p class="hero-subtitle">CDC NHANES-Calibrated Extreme Gradient Boosting (XGBoost) Model with 90-Day Metabolic Twin Progression Simulation</p>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR: PATIENT INPUTS & DEMO PERSONA PRESETS
# ==========================================================
st.sidebar.markdown(f'<div style="font-size:1.1rem; font-weight:700; color:{COLOR_PRIMARY}; margin-bottom:12px;">🧪 Patient Profile & Presets</div>', unsafe_allow_html=True)

preset_choice = st.sidebar.selectbox(
    "⚡ Quick-Load Clinical Personas",
    options=[
        "Custom Patient Input",
        "🟢 Persona 1: Healthy Young Adult",
        "🟡 Persona 2: Borderline Prediabetic",
        "🔴 Persona 3: High-Risk Diabetic"
    ],
    index=0
)

# Populate values based on selection
if preset_choice == "🟢 Persona 1: Healthy Young Adult":
    def_name = "Emily Watson"
    def_age, def_bmi, def_gluc, def_act = 25, 21.5, 85, 1800
    def_sys, def_dia, def_ldl, def_hdl, def_trig, def_smk = 115, 75, 90, 60, 100, 3
elif preset_choice == "🟡 Persona 2: Borderline Prediabetic":
    def_name = "Michael Vance"
    def_age, def_bmi, def_gluc, def_act = 45, 28.5, 112, 400
    def_sys, def_dia, def_ldl, def_hdl, def_trig, def_smk = 125, 82, 120, 48, 160, 3
elif preset_choice == "🔴 Persona 3: High-Risk Diabetic":
    def_name = "Robert Jenkins"
    def_age, def_bmi, def_gluc, def_act = 60, 34.0, 175, 100
    def_sys, def_dia, def_ldl, def_hdl, def_trig, def_smk = 142, 92, 155, 36, 220, 1
else:
    def_name = "Jane Doe"
    def_age, def_bmi, def_gluc, def_act = 38, 27.2, 110, 500
    def_sys, def_dia, def_ldl, def_hdl, def_trig, def_smk = 120, 80, 110, 50, 150, 3

patient_name = st.sidebar.text_input("👤 Patient Full Name", value=def_name, placeholder="Enter patient full name...")

st.sidebar.markdown("---")
st.sidebar.markdown(f'<div style="font-size:0.9rem; font-weight:700; color:{COLOR_TEXT_MAIN}; margin-bottom:6px;">📋 Core Physiological Parameters</div>', unsafe_allow_html=True)

age = st.sidebar.slider("Age (years)", min_value=18, max_value=90, value=def_age)
bmi = st.sidebar.slider("Body Mass Index (BMI, kg/m²)", min_value=15.0, max_value=50.0, value=float(def_bmi), step=0.1)
glucose = st.sidebar.slider("Fasting Blood Glucose (mg/dL)", min_value=60, max_value=300, value=def_gluc)
activity = st.sidebar.slider("Physical Activity (minutes/week)", min_value=0, max_value=3000, value=def_act, step=50)

with st.sidebar.expander("🔬 Comprehensive Lab & Lipid Panel", expanded=False):
    systolic = st.number_input("Systolic Blood Pressure (mmHg)", 80, 220, def_sys)
    diastolic = st.number_input("Diastolic Blood Pressure (mmHg)", 50, 140, def_dia)
    ldl = st.number_input("LDL Cholesterol (mg/dL)", 40, 300, def_ldl)
    hdl = st.number_input("HDL Cholesterol (mg/dL)", 20, 120, def_hdl)
    triglycerides = st.number_input("Triglycerides (mg/dL)", 40, 500, def_trig)
    smoking_label = st.selectbox(
        "Smoking Status (CDC SMQ040)",
        ["Non-Smoker (Not at all)", "Occasional smoker", "Daily Smoker"],
        index=0 if def_smk == 3 else (1 if def_smk == 2 else 2)
    )
    smoking_map = {"Non-Smoker (Not at all)": 3, "Occasional smoker": 2, "Daily Smoker": 1}
    smoking = smoking_map[smoking_label]

# ==========================================================
# INFERENCE PIPELINE (WITH FEATURE SCALING)
# ==========================================================
patient_df = pd.DataFrame([{
    "Age": age,
    "BMI": bmi,
    "FastingGlucose": glucose,
    "SystolicBP": systolic,
    "DiastolicBP": diastolic,
    "LDL": ldl,
    "HDL": hdl,
    "Triglycerides": triglycerides,
    "Smoking": smoking,
    "Activity": activity
}])

if scaler is not None:
    patient_scaled = pd.DataFrame(scaler.transform(patient_df), columns=patient_df.columns)
else:
    patient_scaled = patient_df

prediction = int(model.predict(patient_scaled)[0])
probabilities = model.predict_proba(patient_scaled)[0]

risk_labels = {0: "Normal", 1: "Prediabetes", 2: "Diabetes"}
risk_name = risk_labels[prediction]
pred_prob = probabilities[prediction] * 100

# Estimated HbA1c based on Nathan clinical relation
predicted_hba1c = (glucose + 46.7) / 28.7
today_str = datetime.date.today().strftime("%B %d, %Y")

# Active Patient Consultation Banner
st.markdown(f"""
<div style="background:{COLOR_SURFACE_CARD}; border:1px solid {COLOR_BORDER}; border-radius:12px; padding:14px 22px; margin-bottom:18px; display:flex; justify-content:space-between; align-items:center; box-shadow: 0 2px 8px rgba(30, 36, 43, 0.03);">
    <div>
        <span style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.8px; color:{COLOR_TEXT_MUTED}; font-weight:700;">Active Clinical Evaluation</span>
        <div style="font-size:1.25rem; font-weight:800; color:{COLOR_TEXT_MAIN}; margin-top:2px;">
            👤 {patient_name} <span style="font-size:0.88rem; font-weight:500; color:{COLOR_TEXT_MUTED}; margin-left:8px;">(Age: {age} yrs | BMI: {bmi:.1f} kg/m² | Fasting Glucose: {glucose} mg/dL)</span>
        </div>
    </div>
    <div style="text-align:right;">
        <span style="font-size:0.75rem; color:{COLOR_TEXT_MUTED}; font-weight:600; text-transform:uppercase;">Evaluation Date</span>
        <div style="font-size:0.92rem; font-weight:700; color:{COLOR_PRIMARY};">{today_str}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# TOP ROW: CLINICAL ASSESSMENT METRICS
# ==========================================================
col_risk, col_gauge, col_hba1c = st.columns([1.15, 1.35, 1.0])

with col_risk:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Diagnostic Risk Tier</div>', unsafe_allow_html=True)
    
    if prediction == 0:
        st.markdown('<span class="badge-normal">🟢 NORMAL (LOW RISK)</span>', unsafe_allow_html=True)
        risk_desc = "Biomarkers indicate healthy glycemic regulation and metabolic balance."
    elif prediction == 1:
        st.markdown('<span class="badge-prediabetes">🟡 PREDIABETES (MODERATE RISK)</span>', unsafe_allow_html=True)
        risk_desc = "Impaired fasting glucose observed. Lifestyle and dietary intervention indicated."
    else:
        st.markdown('<span class="badge-diabetes">🔴 DIABETES (HIGH CLINICAL RISK)</span>', unsafe_allow_html=True)
        risk_desc = "Elevated glycemic biomarkers detected. Clinical evaluation & pharmacotherapy indicated."
        
    st.markdown(f"<p style='color:{COLOR_TEXT_MUTED}; font-size:0.88rem; margin-top:12px; line-height:1.4;'>{risk_desc}</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='card-footer'>Confidence: <strong style='color:{COLOR_TEXT_MAIN};'>{pred_prob:.1f}%</strong> | Model: Calibrated XGBoost</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_gauge:
    st.markdown('<div class="card" style="padding:10px 15px;">', unsafe_allow_html=True)
    
    severity_score = (probabilities[0] * 15) + (probabilities[1] * 50) + (probabilities[2] * 90)
    
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=severity_score,
        number={'suffix': "/100", 'font': {'size': 22, 'color': COLOR_TEXT_MAIN, 'family': 'Inter'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': COLOR_TEXT_MUTED},
            'bar': {'color': COLOR_PRIMARY, 'thickness': 0.28},
            'bgcolor': COLOR_BG_SIDEBAR,
            'borderwidth': 0,
            'steps': [
                {'range': [0, 33], 'color': "rgba(42, 123, 111, 0.22)"},
                {'range': [33, 66], 'color': "rgba(217, 138, 43, 0.22)"},
                {'range': [66, 100], 'color': "rgba(200, 75, 49, 0.22)"}
            ],
            'threshold': {
                'line': {'color': COLOR_RISK_DIAB, 'width': 3},
                'thickness': 0.75,
                'value': 66.0
            }
        }
    ))
    fig_gauge.update_layout(
        title={'text': "Metabolic Risk Gauge", 'font': {'size': 13, 'color': COLOR_TEXT_MUTED, 'family': 'Inter'}},
        margin=dict(l=25, r=25, t=35, b=12),
        height=165,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_gauge, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_hba1c:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Estimated HbA1c</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card-value">{predicted_hba1c:.2f}%</div>', unsafe_allow_html=True)
    
    if predicted_hba1c < 5.7:
        tag_color = COLOR_RISK_NORMAL
        tag_text = "Optimal Range (< 5.7%)"
    elif predicted_hba1c < 6.5:
        tag_color = COLOR_RISK_PREDIAB
        tag_text = "Elevated Range (5.7 - 6.4%)"
    else:
        tag_color = COLOR_RISK_DIAB
        tag_text = "Diabetic Range (≥ 6.5%)"
        
    st.markdown(f"<span style='color:{tag_color}; font-size:0.85rem; font-weight:700;'>{tag_text}</span>", unsafe_allow_html=True)
    st.markdown(f"<div class='card-footer'>Nathan et al.: (Avg Glucose + 46.7) / 28.7</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================================
# PROBABILITY DISTRIBUTION BREAKDOWN
# ==========================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">Class Probability Distribution</div>', unsafe_allow_html=True)

fig_bar = go.Figure()
classes = ["Normal", "Prediabetes", "Diabetes"]
colors_list = [COLOR_RISK_NORMAL, COLOR_RISK_PREDIAB, COLOR_RISK_DIAB]

for c, p, col in zip(classes, probabilities, colors_list):
    fig_bar.add_trace(go.Bar(
        y=["Probability"],
        x=[p * 100],
        name=c,
        orientation='h',
        marker=dict(color=col),
        text=f"<b>{c}</b>: {p*100:.1f}%",
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(color="#ffffff", size=13, family="Inter")
    ))

fig_bar.update_layout(
    barmode='stack',
    height=54,
    margin=dict(l=5, r=5, t=5, b=5),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    showlegend=False,
    xaxis=dict(showgrid=False, showticklabels=False, range=[0, 100]),
    yaxis=dict(showgrid=False, showticklabels=False)
)
st.plotly_chart(fig_bar, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ==========================================================
# 90-DAY DIGITAL TWIN SIMULATION ENGINE
# ==========================================================
st.markdown(f"""
<div class="hero-container" style="margin-top:10px; margin-bottom:15px; padding:16px 22px;">
    <h2 style="font-size:1.3rem; font-weight:700; margin:0; color:{COLOR_PRIMARY};">
        📈 Digital Twin: 90-Day Glycemic Progression Simulation
    </h2>
    <p style="color:{COLOR_TEXT_MUTED}; font-size:0.86rem; margin:4px 0 0 0;">
        In-silico metabolic simulation modeling natural progression versus targeted lifestyle and pharmacotherapy interventions.
    </p>
</div>
""", unsafe_allow_html=True)

days = 90
np.random.seed(42)

def simulate_twin(start_glucose, start_bmi, start_activity, scenario):
    glucose_hist = []
    hba1c_hist = []
    curr_gluc = start_glucose
    curr_bmi = start_bmi
    
    for day in range(days):
        bmi_effect = (curr_bmi - 25) * 0.05
        activity_effect = -(start_activity / 1000) * 0.8
        daily_change = bmi_effect + activity_effect
        
        if scenario == "Lifestyle":
            daily_change -= 0.20
            curr_bmi -= 0.005
        elif scenario == "Medication":
            daily_change -= 0.35
            
        noise = np.random.normal(0, 0.7)
        curr_gluc += daily_change + noise
        
        if curr_gluc < 60:
            curr_gluc = 60
            
        glucose_hist.append(curr_gluc)
        
        avg_gluc = np.mean(glucose_hist)
        hba1c_hist.append((avg_gluc + 46.7) / 28.7)
        
    return glucose_hist, hba1c_hist

g_none, h_none = simulate_twin(glucose, bmi, activity, "None")
g_life, h_life = simulate_twin(glucose, bmi, activity, "Lifestyle")
g_med, h_med = simulate_twin(glucose, bmi, activity, "Medication")

days_x = list(range(1, days + 1))

tab_gluc, tab_hba1c = st.tabs(["🩸 Fasting Glucose Trajectory (mg/dL)", "🧪 HbA1c Progression (%)"])

with tab_gluc:
    fig_g = go.Figure()
    
    # ADA Zone Shading
    fig_g.add_hrect(y0=60, y1=100, fillcolor="rgba(42, 123, 111, 0.06)", line_width=0, layer="below")
    fig_g.add_hrect(y0=100, y1=126, fillcolor="rgba(217, 138, 43, 0.06)", line_width=0, layer="below")
    fig_g.add_hrect(y0=126, y1=320, fillcolor="rgba(200, 75, 49, 0.06)", line_width=0, layer="below")
    
    # Traces
    fig_g.add_trace(go.Scatter(
        x=days_x, y=g_none, mode='lines', name='No Intervention (Natural Progression)',
        line=dict(color=COLOR_RISK_DIAB, width=3),
        hovertemplate="Day %{x}<br>Glucose: %{y:.1f} mg/dL<extra>No Intervention</extra>"
    ))
    fig_g.add_trace(go.Scatter(
        x=days_x, y=g_life, mode='lines', name='Lifestyle Intervention',
        line=dict(color=COLOR_RISK_PREDIAB, width=3),
        hovertemplate="Day %{x}<br>Glucose: %{y:.1f} mg/dL<extra>Lifestyle</extra>"
    ))
    fig_g.add_trace(go.Scatter(
        x=days_x, y=g_med, mode='lines', name='Pharmacotherapy (Medication)',
        line=dict(color=COLOR_RISK_NORMAL, width=3),
        hovertemplate="Day %{x}<br>Glucose: %{y:.1f} mg/dL<extra>Medication</extra>"
    ))
    
    # Threshold Annotation Lines
    fig_g.add_hline(y=126, line_dash="dash", line_color=COLOR_RISK_DIAB, annotation_text="Diabetes Diagnostic Threshold (126 mg/dL)", annotation_position="top left", annotation_font_color=COLOR_RISK_DIAB)
    fig_g.add_hline(y=100, line_dash="dot", line_color=COLOR_RISK_PREDIAB, annotation_text="Impaired Fasting Glucose (100 mg/dL)", annotation_position="bottom left", annotation_font_color=COLOR_RISK_PREDIAB)
    
    fig_g.update_layout(
        height=390,
        margin=dict(l=30, r=20, t=30, b=30),
        paper_bgcolor=COLOR_SURFACE_CARD,
        plot_bgcolor=COLOR_BG_MAIN,
        font=dict(family="Inter", color=COLOR_TEXT_MAIN),
        xaxis=dict(title="Simulation Timeline (Days)", gridcolor=COLOR_BORDER, color=COLOR_TEXT_MUTED),
        yaxis=dict(title="Fasting Blood Glucose (mg/dL)", gridcolor=COLOR_BORDER, color=COLOR_TEXT_MUTED),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=COLOR_TEXT_MAIN))
    )
    st.plotly_chart(fig_g, use_container_width=True)

with tab_hba1c:
    fig_h = go.Figure()
    
    # ADA Zone Shading
    fig_h.add_hrect(y0=4.0, y1=5.7, fillcolor="rgba(42, 123, 111, 0.06)", line_width=0, layer="below")
    fig_h.add_hrect(y0=5.7, y1=6.5, fillcolor="rgba(217, 138, 43, 0.06)", line_width=0, layer="below")
    fig_h.add_hrect(y0=6.5, y1=14.0, fillcolor="rgba(200, 75, 49, 0.06)", line_width=0, layer="below")
    
    fig_h.add_trace(go.Scatter(
        x=days_x, y=h_none, mode='lines', name='No Intervention',
        line=dict(color=COLOR_RISK_DIAB, width=3),
        hovertemplate="Day %{x}<br>HbA1c: %{y:.2f}%<extra>No Intervention</extra>"
    ))
    fig_h.add_trace(go.Scatter(
        x=days_x, y=h_life, mode='lines', name='Lifestyle Intervention',
        line=dict(color=COLOR_RISK_PREDIAB, width=3),
        hovertemplate="Day %{x}<br>HbA1c: %{y:.2f}%<extra>Lifestyle</extra>"
    ))
    fig_h.add_trace(go.Scatter(
        x=days_x, y=h_med, mode='lines', name='Pharmacotherapy (Medication)',
        line=dict(color=COLOR_RISK_NORMAL, width=3),
        hovertemplate="Day %{x}<br>HbA1c: %{y:.2f}%<extra>Medication</extra>"
    ))
    
    fig_h.add_hline(y=6.5, line_dash="dash", line_color=COLOR_RISK_DIAB, annotation_text="Diabetes Cutoff (6.5%)", annotation_position="top left", annotation_font_color=COLOR_RISK_DIAB)
    fig_h.add_hline(y=5.7, line_dash="dot", line_color=COLOR_RISK_PREDIAB, annotation_text="Prediabetes Cutoff (5.7%)", annotation_position="bottom left", annotation_font_color=COLOR_RISK_PREDIAB)
    
    fig_h.update_layout(
        height=390,
        margin=dict(l=30, r=20, t=30, b=30),
        paper_bgcolor=COLOR_SURFACE_CARD,
        plot_bgcolor=COLOR_BG_MAIN,
        font=dict(family="Inter", color=COLOR_TEXT_MAIN),
        xaxis=dict(title="Simulation Timeline (Days)", gridcolor=COLOR_BORDER, color=COLOR_TEXT_MUTED),
        yaxis=dict(title="Estimated HbA1c (%)", gridcolor=COLOR_BORDER, color=COLOR_TEXT_MUTED),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=COLOR_TEXT_MAIN))
    )
    st.plotly_chart(fig_h, use_container_width=True)

# ==========================================================
# 90-DAY COMPARATIVE DELTA SUMMARY
# ==========================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">📊 90-Day Delta Summary (Intervention Efficacy)</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

delta_none = g_none[-1] - glucose
delta_life = g_life[-1] - glucose
delta_med = g_med[-1] - glucose

with c1:
    sign = "+" if delta_none >= 0 else ""
    st.markdown(f"""
    <div class="delta-box" style="border-left: 4px solid {COLOR_RISK_DIAB};">
        <span style="color:{COLOR_RISK_DIAB}; font-weight:700; font-size:0.9rem;">🔴 Natural Progression</span>
        <div style="font-size:1.6rem; font-weight:800; color:{COLOR_TEXT_MAIN}; margin:6px 0;">{g_none[-1]:.1f} mg/dL</div>
        <div style="font-size:0.82rem; color:{COLOR_TEXT_MUTED};">90-Day Net Shift: <strong style="color:{COLOR_RISK_DIAB};">{sign}{delta_none:.1f} mg/dL</strong></div>
        <div style="font-size:0.78rem; color:{COLOR_TEXT_MUTED}; margin-top:4px;">Projected HbA1c: <strong>{h_none[-1]:.2f}%</strong></div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    sign = "+" if delta_life >= 0 else ""
    st.markdown(f"""
    <div class="delta-box" style="border-left: 4px solid {COLOR_RISK_PREDIAB};">
        <span style="color:{COLOR_RISK_PREDIAB}; font-weight:700; font-size:0.9rem;">🟡 Lifestyle Intervention</span>
        <div style="font-size:1.6rem; font-weight:800; color:{COLOR_TEXT_MAIN}; margin:6px 0;">{g_life[-1]:.1f} mg/dL</div>
        <div style="font-size:0.82rem; color:{COLOR_TEXT_MUTED};">90-Day Net Shift: <strong style="color:{COLOR_RISK_PREDIAB};">{sign}{delta_life:.1f} mg/dL</strong></div>
        <div style="font-size:0.78rem; color:{COLOR_TEXT_MUTED}; margin-top:4px;">Projected HbA1c: <strong>{h_life[-1]:.2f}%</strong></div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    sign = "+" if delta_med >= 0 else ""
    st.markdown(f"""
    <div class="delta-box" style="border-left: 4px solid {COLOR_RISK_NORMAL};">
        <span style="color:{COLOR_RISK_NORMAL}; font-weight:700; font-size:0.9rem;">🟢 Pharmacotherapy</span>
        <div style="font-size:1.6rem; font-weight:800; color:{COLOR_TEXT_MAIN}; margin:6px 0;">{g_med[-1]:.1f} mg/dL</div>
        <div style="font-size:0.82rem; color:{COLOR_TEXT_MUTED};">90-Day Net Shift: <strong style="color:{COLOR_RISK_NORMAL};">{sign}{delta_med:.1f} mg/dL</strong></div>
        <div style="font-size:0.78rem; color:{COLOR_TEXT_MUTED}; margin-top:4px;">Projected HbA1c: <strong>{h_med[-1]:.2f}%</strong></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ==========================================================
# IN-APP CONSULTATION REPORT GENERATOR (PDF)
# ==========================================================
def generate_pdf_report(p_name, p_date, p_data, p_class, p_probs, p_hba1c, g_none_end, g_life_end, g_med_end):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(name="TitleStyle", parent=styles["Title"], fontSize=18, textColor=colors.HexColor(COLOR_PRIMARY), alignment=TA_CENTER, spaceAfter=8)
    sub_style = ParagraphStyle(name="SubStyle", parent=styles["Normal"], fontSize=10, textColor=colors.HexColor(COLOR_TEXT_MUTED), alignment=TA_CENTER, spaceAfter=18)
    h2_style = ParagraphStyle(name="H2Style", parent=styles["Heading2"], fontSize=12, textColor=colors.HexColor(COLOR_PRIMARY), spaceBefore=12, spaceAfter=6)
    body_style = ParagraphStyle(name="BodyStyle", parent=styles["BodyText"], fontSize=9, textColor=colors.HexColor(COLOR_TEXT_MAIN), leading=13)
    
    story = []
    
    story.append(Paragraph("DiabeCare AI: Clinical Decision Support Consultation Report", title_style))
    story.append(Paragraph(f"Confidential Patient Assessment &bull; Patient: <b>{p_name}</b> &bull; Date: {p_date}", sub_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("1. Patient Baseline Physiological Markers", h2_style))
    patient_table_data = [
        ["Patient Full Name", f"<b>{p_name}</b>", "Consultation Date", f"<b>{p_date}</b>"],
        ["Age", f"{p_data['Age']} years", "Systolic Blood Pressure", f"{p_data['SystolicBP']} mmHg"],
        ["BMI", f"{p_data['BMI']:.1f} kg/m²", "Diastolic Blood Pressure", f"{p_data['DiastolicBP']} mmHg"],
        ["Fasting Blood Glucose", f"{p_data['FastingGlucose']} mg/dL", "Physical Activity", f"{p_data['Activity']} min/wk"],
        ["Estimated Baseline HbA1c", f"{p_hba1c:.2f}%", "Smoking Status", "Active" if p_data['Smoking'] == 1 else "Non-smoker"]
    ]
    t1 = Table(patient_table_data, colWidths=[130, 120, 130, 120])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor(COLOR_BG_SIDEBAR)),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor(COLOR_TEXT_MAIN)),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor(COLOR_BORDER)),
        ('FONTSIZE', (0,0), (-1,-1), 8)
    ]))
    story.append(t1)
    story.append(Spacer(1, 12))
    
    story.append(Paragraph("2. Machine Learning Risk Classification", h2_style))
    risk_text = ["Normal (Low Risk)", "Prediabetes (Moderate Risk)", "Diabetes (High Clinical Risk)"][p_class]
    diag_table_data = [
        ["Patient Evaluated", f"<b>{p_name}</b>"],
        ["Predicted Diagnostic Tier", f"<b>{risk_text}</b>"],
        ["Prediction Confidence", f"<b>{p_probs[p_class]*100:.1f}%</b>"],
        ["Estimated Baseline HbA1c", f"<b>{p_hba1c:.2f}%</b> (ADA Diagnostic Threshold: ≥ 6.5%)"],
        ["Inference Algorithm", "Extreme Gradient Boosting (XGBoost) calibrated on CDC NHANES"]
    ]
    t2 = Table(diag_table_data, colWidths=[180, 320])
    t2.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor(COLOR_BORDER)),
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor(COLOR_BG_SIDEBAR)),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('FONTSIZE', (0,0), (-1,-1), 8)
    ]))
    story.append(t2)
    story.append(Spacer(1, 12))
    
    story.append(Paragraph("3. 90-Day Digital Twin Metabolic Forecast", h2_style))
    twin_table_data = [
        ["Scenario", "Baseline Glucose", "Day-90 Projected Glucose", "Clinical Prognosis"],
        ["No Intervention", f"{p_data['FastingGlucose']} mg/dL", f"{g_none_end:.1f} mg/dL", "Gradual metabolic decline & progressive glycemic drift."],
        ["Lifestyle Modification", f"{p_data['FastingGlucose']} mg/dL", f"{g_life_end:.1f} mg/dL", "Dietary & physical activity changes stabilize biomarkers."],
        ["Pharmacotherapy", f"{p_data['FastingGlucose']} mg/dL", f"{g_med_end:.1f} mg/dL", "Medication regimen achieves robust glycemic control."]
    ]
    t3 = Table(twin_table_data, colWidths=[110, 85, 115, 190])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor(COLOR_BG_SIDEBAR)),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor(COLOR_TEXT_MAIN)),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor(COLOR_BORDER)),
        ('FONTSIZE', (0,0), (-1,-1), 8)
    ]))
    story.append(t3)
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("Attending Physician Review & Signature:", h2_style))
    story.append(Spacer(1, 25))
    story.append(Paragraph(f"Patient: <b>{p_name}</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Physician Signature: __________________________ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Date: {p_date}", body_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

pdf_bytes = generate_pdf_report(
    patient_name,
    today_str,
    patient_df.iloc[0].to_dict(),
    prediction,
    probabilities,
    predicted_hba1c,
    g_none[-1],
    g_life[-1],
    g_med[-1]
)

clean_name = "".join(c for c in patient_name if c.isalnum() or c in (' ', '_', '-')).strip().replace(' ', '_')
pdf_filename = f"DiabeCare_Report_{clean_name}.pdf" if clean_name else f"DiabeCare_Report_Age{age}_Gluc{glucose}.pdf"

st.download_button(
    label=f"📄 Download Official Consultation Report for {patient_name} (PDF)",
    data=pdf_bytes,
    file_name=pdf_filename,
    mime="application/pdf",
    use_container_width=True
)

st.markdown(f"<p style='text-align:center; color:{COLOR_TEXT_MUTED}; font-size:0.75rem; margin-top:20px;'>"
            f"DiabeCare AI Decision Support System • Academic & Clinical Research • Dr. Sir Hajji Prof. Rizwan Lab"
            f"</p>", unsafe_allow_html=True)
