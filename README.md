# 🩺 DiabeCare AI — Clinical Risk Assessment System

An AI-powered healthcare application that predicts diabetes risk levels and simulates future glucose & HbA1c trends using Machine Learning and Digital Twin Technology.

---

# 🚀 Live Demo

🌐 Streamlit App:  
https://diabecare-ai-hfnkum47fqwj6wnyjsvlbt.streamlit.app/

---

# 📌 Project Overview

DiabeCare AI is an intelligent diabetes risk assessment system built using:

- Machine Learning
- Digital Twin Simulation
- Clinical Data Analysis
- Interactive Streamlit Dashboard

The system predicts:

✅ Diabetes Risk Level  
✅ Estimated HbA1c  
✅ Future Glucose Trends  
✅ Future HbA1c Trends  

using patient health information.

---

# 🎯 Features

## ✅ AI-Based Diabetes Prediction

Predicts whether a patient is:

- Normal
- Prediabetic
- Diabetic

using trained ML models.

---

## ✅ Digital Twin Simulation

Simulates patient health progression for:

- No Intervention
- Lifestyle Improvement
- Medication

over a 90-day period.

---

## ✅ Interactive Dashboard

### Input Panel
- Age
- BMI
- Fasting Blood Glucose
- Physical Activity

### Output Panel
- Predicted HbA1c
- Diabetes Risk Level

### Graph Panel
- Glucose Trend Visualization
- HbA1c Trend Visualization

---

## ✅ Dark / Light Mode

User-friendly UI with theme toggle support.

---

# 🧠 Machine Learning Models

## 🔹 XGBoost Classifier (Production Model)

Used as the primary deployed model.

### Performance

| Metric | Score |
|---|---|
| Accuracy | 81% |
| Macro F1-Score | 79% |

### Best Parameters

```python
{
 'colsample_bytree': 1.0,
 'learning_rate': 0.1,
 'max_depth': 7,
 'n_estimators': 200,
 'subsample': 0.8
}
```

---

## 🔹 Random Forest Classifier

Used for comparison and validation.

---

# 📂 Dataset Information

## Source

NHANES (National Health and Nutrition Examination Survey)

### Data Years Used

- NHANES 2017–2018
- NHANES 2017–March 2020
- NHANES 2021–2023

---

# 📊 Features Used

| Feature | Description |
|---|---|
| Age | Patient age |
| BMI | Body Mass Index |
| FastingGlucose | Blood glucose level |
| Activity | Physical activity |
| Blood Pressure | Systolic & Diastolic |
| HDL / LDL | Cholesterol levels |
| Triglycerides | Fat levels |
| Smoking | Smoking status |

---

# 🛠 Tech Stack

## Frontend
- Streamlit

## Backend & ML
- Python
- Scikit-Learn
- XGBoost
- Pandas
- NumPy

## Visualization
- Matplotlib

## Deployment
- Streamlit Community Cloud

---

# 📁 Project Structure

```bash
DiabeCare-AI/
│
├── app3.py
├── xgb_model.json
├── requirements.txt
├── runtime.txt
├── pyproject.toml
└── README.md
```

---

# ⚙ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/diabecare-ai.git
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Run Application

```bash
streamlit run app3.py
```

---

# 🌐 Deployment

This project is deployed using:

✅ Streamlit Community Cloud

---

# 📈 Digital Twin Simulation

The system simulates future health conditions using:

- Glucose progression
- BMI effects
- Physical activity impact
- Medication impact

over 90 days.

---

# 📷 Dashboard Preview

Add screenshots here after deployment.

---

# 🔮 Future Improvements

- IoT wearable integration
- Doctor recommendation system
- Real-time monitoring
- Blockchain-based health records
- Mobile application

---

# 👨‍💻 Team Members


- Shahjahan Vighio
- Muhammad Muteeb


---

# 📜 License

This project is developed for educational and research purposes only.

---

# ⭐ Support

If you like this project:

⭐ Star the repository  
🍴 Fork the project  
🩺 Share with others
