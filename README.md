# 🩺 DiabeCare AI — Clinical Risk Assessment & Digital Twin System

An AI-powered healthcare application that predicts diabetes risk levels and simulates future glucose & HbA1c trends using Machine Learning and Digital Twin Technology.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://diabecare-ai-hfnkum47fqwj6wnyjsvlbt.streamlit.app/)

---

# 🚀 Live Demo

🌐 **Streamlit Cloud Deployment:**  
[https://diabecare-ai-hfnkum47fqwj6wnyjsvlbt.streamlit.app/](https://diabecare-ai-hfnkum47fqwj6wnyjsvlbt.streamlit.app/)

---

# 📌 Project Overview

DiabeCare AI is an intelligent clinical decision-support and digital twin platform built using:

- **Machine Learning**: Multi-class XGBoost Classifier trained on CDC NHANES data.
- **Digital Twin Simulation**: 90-day predictive trajectory modeling under distinct clinical scenarios.
- **Clinical Data Pipeline**: Robust ETL processing of multi-cycle CDC NHANES laboratory and examination datasets.
- **Interactive Web Interface**: Streamlit application with custom medical UI, dark/light themes, and automated PDF report export.

The system predicts:
- ✅ **Diabetes Risk Level**: Normal, Prediabetes, or Diabetes with class confidence probabilities.
- ✅ **Estimated HbA1c**: Derived from clinical correlation models.
- ✅ **90-Day Glucose Trajectories**: Forecasting under No Intervention, Lifestyle Improvement, and Medication.
- ✅ **90-Day HbA1c Progression**: Longitudinal metabolic trajectory visualization.

---

# 🎯 Key Features

### 1. 🤖 AI-Based Diabetes Prediction
Predicts patient metabolic status (**Normal**, **Prediabetes**, **Diabetes**) using patient biomarkers (Age, BMI, Fasting Glucose, Blood Pressure, Lipid Profile, and Physical Activity).

### 2. 📈 Digital Twin Simulation (90 Days)
Simulates patient health progression across three scenarios:
- **No Intervention**: Baseline disease progression with continuous metabolic decline.
- **Lifestyle Intervention**: Moderate physical activity & dietary changes yielding gradual glucose stabilization.
- **Pharmacotherapy (Medication)**: Active clinical treatment driving rapid glucose and HbA1c normalization.

### 3. 🎨 Interactive Dashboard & Themes
- Dual theme support (Dark Mode and Medical Light Mode).
- Dynamic interactive metric cards and gauge charts.
- Side-by-side comparative simulation graphs.

### 4. 📄 Clinical PDF Report Generation
- Direct export of comprehensive, patient-ready clinical summary reports via ReportLab.

---

# 🧠 Machine Learning Model & Performance

## 🔹 XGBoost Classifier (Production Model)

- **Dataset**: CDC NHANES Multi-Cycle Dataset (12,122 records).
- **Validation Strategy**: Stratified 5-Fold Cross Validation.
- **Class Balancing**: SMOTE (Synthetic Minority Over-sampling Technique).

### Performance Metrics

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 81% |
| **Macro F1-Score** | 79% |
| **ROC-AUC** | 0.88+ |

### Best Hyperparameters

```python
{
    'colsample_bytree': 1.0,
    'learning_rate': 0.1,
    'max_depth': 7,
    'n_estimators': 300,
    'subsample': 0.8
}
```

---

# 📂 Project Structure & File Inventory

| File / Folder | Purpose | Status |
| :--- | :--- | :--- |
| `app2.py` | **Full Production App**: Comprehensive Streamlit interface with custom medical UI, Plotly charts, risk gauges, and PDF export. | ✅ Production Ready |
| `app3.py` | **Cloud Streamlit App**: Lightweight Streamlit deployment optimized for Streamlit Cloud and low-memory runtimes. | ✅ Deployed |
| `app.py` | Baseline minimal prototype Streamlit app. | ✅ Functional |
| `xgb_model.json` | Portable JSON format XGBoost Classifier model for production deployment. | ✅ Generated |
| `scaler.pkl` | Standard scaler parameters corresponding to the trained model. | ✅ Available |
| `diabetes_model2_v2.pkl` | Scikit-Learn/Imblearn Pipeline (Scaler + SMOTE + XGBoost). | ✅ Available |
| `convert_model.py` | Utility to extract XGBoost JSON and Scaler from pipeline. | ✅ Functional |
| `model_v3_Aplus.py` | Model training script with SMOTE and 5-Fold CV. | ✅ Functional |
| `digital_twin_simulation.py` | Standalone CLI script that simulates 90-day progression and exports CSV/PNG trends. | ✅ Functional |
| `dataset_builder_final.py` | ETL script that parses and merges raw CDC NHANES `.xpt` files into a single cleaned CSV. | ✅ Functional |
| `final_nhanes_dataset.csv` | Cleaned and merged NHANES dataset (12,122 rows, 17 clinical variables). | ✅ Present |
| `visualize_poster.py` | Generates poster-ready plots (Confusion Matrix, ROC Curve, Feature Importance). | ✅ Functional |
| `make_pdf.py` | Generates project technical summary report PDF using ReportLab. | ✅ Functional |
| `nhanes/` | Raw CDC NHANES `.xpt` data files across 3 examination cycles. | ✅ 34 files |
| `screenshots/` | Dashboard screenshots and visual evaluations. | ✅ Present |
| `run_app.bat` | One-click Windows launch script. | ✅ Ready |
| `requirements.txt` | Complete project dependencies. | ✅ Complete |

---

# 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/ShahjahanVighio/DiabeCare-AI.git
cd DiabeCare-AI
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Launch Application

**Option A — One-Click Launcher (Windows):**
Double-click `run_app.bat`.

**Option B — Streamlit Command:**
```bash
# Run full clinical app:
streamlit run app2.py

# Or run lightweight cloud app:
streamlit run app3.py
```

---

# ⚙️ Dependencies (`requirements.txt`)

- `streamlit`
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `joblib`
- `xgboost`
- `imbalanced-learn`
- `reportlab`
- `plotly`

---

# 📜 License & Acknowledgments

- **Dataset**: Centers for Disease Control and Prevention (CDC) - National Health and Nutrition Examination Survey (NHANES).
- **Developers**: DiabeCare AI Research & Engineering Team.
