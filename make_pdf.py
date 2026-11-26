from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch


# ==========================================================
# PDF FILE NAME
# ==========================================================
output_file = "Diabetes_Digital_Twin_Project_Report.pdf"

doc = SimpleDocTemplate(output_file, pagesize=A4,
                        rightMargin=40, leftMargin=40,
                        topMargin=50, bottomMargin=50)

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    name="TitleStyle",
    parent=styles["Title"],
    alignment=TA_CENTER,
    fontSize=18,
    spaceAfter=20
)

heading_style = ParagraphStyle(
    name="HeadingStyle",
    parent=styles["Heading2"],
    fontSize=13,
    spaceAfter=10
)

text_style = ParagraphStyle(
    name="TextStyle",
    parent=styles["BodyText"],
    fontSize=11,
    leading=15,
    spaceAfter=10
)

story = []


# ==========================================================
# TITLE
# ==========================================================
story.append(Paragraph("Digital Twin Model for Diabetes Mellitus (NHANES Based)", title_style))
story.append(Paragraph("Poster-Ready Technical Project Summary", styles["Italic"]))
story.append(Spacer(1, 15))


# ==========================================================
# CONTENT
# ==========================================================
sections = [
    ("Project Title",
     "Digital Twin Model for Diabetes Mellitus Risk Prediction and Progression Simulation using NHANES Data."),

    ("Aim / Objective",
     """To develop an AI-based Digital Twin System that predicts diabetes risk (Normal / Prediabetes / Diabetes),
     estimates HbA1c level, simulates future glucose and HbA1c progression, and compares intervention scenarios
     (No Intervention, Lifestyle Improvement, Medication)."""),

    ("Project Concept (Digital Twin)",
     """A Digital Twin is a virtual representation of a patient that mimics current health state using clinical parameters,
     predicts disease status using machine learning, simulates future progression using mathematical models, and evaluates
     the effect of interventions. This helps in personalized healthcare decision-making."""),

    ("Dataset Used (NHANES)",
     """Dataset was created using CDC NHANES (National Health and Nutrition Examination Survey)."""),

    ("NHANES Cycles Used",
     """• NHANES 2017–2018<br/>
        • NHANES 2017–March 2020 Pre-Pandemic<br/>
        • NHANES 2021–2023"""),

    ("NHANES Data Files Used",
     """• DEMO: Age, Gender<br/>
        • BMX: BMI<br/>
        • GLU: Fasting Blood Glucose<br/>
        • GHB: HbA1c<br/>
        • BPX/BPXO: Blood Pressure<br/>
        • SMQ: Smoking Status<br/>
        • PAQ: Physical Activity<br/>
        • TCHOL: Total Cholesterol<br/>
        • HDL: HDL Cholesterol<br/>
        • TRIGLY: Triglycerides<br/>
        • DIQ: Family History of Diabetes<br/>
        • INS: Insulin"""),

    ("Dataset Building Method",
     """Each NHANES cycle was merged using SEQN (unique patient ID). All cycles were combined into one dataset.
     Missing values were handled using median/mode imputation. LDL was calculated using the Friedewald approximation:
     LDL = TotalCholesterol − HDL − (Triglycerides / 5)."""),

    ("Final Features in Dataset",
     """• Age<br/>
        • Gender<br/>
        • BMI<br/>
        • Fasting Glucose<br/>
        • Systolic Blood Pressure<br/>
        • Diastolic Blood Pressure<br/>
        • Total Cholesterol<br/>
        • HDL<br/>
        • Triglycerides<br/>
        • LDL (calculated)<br/>
        • Smoking Status<br/>
        • Physical Activity<br/>
        • Family History<br/>
        • Insulin"""),

    ("Target Variable (Output Class)",
     """Diabetes classification created using HbA1c criteria:<br/>
        • HbA1c < 5.7 → Normal (0)<br/>
        • HbA1c 5.7–6.4 → Prediabetes (1)<br/>
        • HbA1c ≥ 6.5 → Diabetes (2)"""),

    ("Tech Stack Used",
     """Programming Language: Python 3.12<br/>
        Libraries: pandas, numpy, matplotlib, scikit-learn, xgboost, joblib<br/>
        Dashboard: Streamlit<br/>
        Hyperparameter tuning: GridSearchCV, Stratified K-Fold"""),

    ("Machine Learning Methodology",
     """Data preprocessing included missing value handling, feature selection, and train-test splitting with stratification.
     Models tested were Random Forest and XGBoost. Hyperparameter tuning was performed using GridSearchCV.
     Evaluation metrics included Accuracy, Precision, Recall, F1-score, and Confusion Matrix."""),

    ("Best Model Selected",
     """Random Forest model was selected for deployment with overall accuracy around 81%.
     Performance was strong for Normal and Diabetes classes, while Prediabetes classification showed moderate overlap."""),

    ("Digital Twin Simulation Module",
     """After prediction, system simulates patient progression for 90 days. Daily glucose changes are influenced by BMI,
     activity, and biological variation. HbA1c is estimated using the clinical approximation formula:
     HbA1c = (Glucose + 46.7) / 28.7."""),

    ("Intervention Scenarios (Simulation)",
     """Three scenarios are simulated:<br/>
        1) No Intervention: glucose gradually increases.<br/>
        2) Lifestyle Improvement: glucose decreases slowly and BMI slightly improves.<br/>
        3) Medication: glucose decreases faster compared to lifestyle."""),

    ("User Interface (Streamlit Dashboard)",
     """Input Panel:<br/>
        • Age (years)<br/>
        • BMI (kg/m²)<br/>
        • Fasting Glucose (mg/dL)<br/>
        • Physical Activity (minutes/week)<br/><br/>
        Output Panel:<br/>
        • Predicted HbA1c (%)<br/>
        • Risk Classification<br/><br/>
        Graph Panel:<br/>
        • Glucose Trend Graph (90 days)<br/>
        • HbA1c Trend Graph (90 days)<br/>
        • Scenario comparison visualization"""),

    ("Practical Importance",
     """This system supports early diabetes screening, identifies prediabetes patients, predicts risk progression,
     and educates patients on lifestyle versus medication impact. It can assist pharmacists in diabetes counseling."""),

    ("Limitations",
     """NHANES dataset is cross-sectional, not longitudinal. Simulation is based on simplified clinical assumptions.
     The model predicts risk category and does not replace clinical diagnosis."""),

    ("Future Improvements",
     """Include diet variables, medication-specific effects, explainable AI (SHAP), mobile deployment, and use of
     longitudinal datasets for improved progression modeling."""),

    ("Conclusion",
     """This project demonstrates that diabetes risk can be predicted using NHANES clinical data, machine learning can
     classify diabetes stages with ~81% accuracy, and a digital twin simulation can visualize future trends and intervention impact."""
     )
]


for heading, text in sections:
    story.append(Paragraph(heading, heading_style))
    story.append(Paragraph(text, text_style))
    story.append(Spacer(1, 10))


# ==========================================================
# BUILD PDF
# ==========================================================
doc.build(story)

print("PDF Created Successfully:", output_file)