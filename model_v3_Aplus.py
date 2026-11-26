import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


# ==========================================================
# LOAD DATASET
# ==========================================================
df = pd.read_csv("final_nhanes_dataset.csv")

# ==========================================================
# FEATURES (IMPORTANT: NO HbA1c leakage)
# ==========================================================
features = [
    "Age",
    "BMI",
    "FastingGlucose",
    "SystolicBP",
    "DiastolicBP",
    "LDL",
    "HDL",
    "Triglycerides",
    "Smoking",
    "Activity"
]

X = df[features]
y = df["DiabetesClass"]

# ==========================================================
# TRAIN / TEST SPLIT
# ==========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================================
# CROSS VALIDATION SETUP
# ==========================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ==========================================================
# PIPELINE (SMOTE + SCALING + MODEL)
# IMPORTANT: SMOTE INSIDE PIPELINE (NO LEAKAGE)
# ==========================================================
pipeline = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("model", XGBClassifier(eval_metric="mlogloss", use_label_encoder=False))
])


# ==========================================================
# HYPERPARAMETER GRID (XGBOOST)
# ==========================================================
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [3, 5, 7],
    "model__learning_rate": [0.01, 0.1],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.8, 1.0]
}


# ==========================================================
# GRID SEARCH CV
# ==========================================================
grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=2
)

print("\n🚀 Training XGBoost with GridSearchCV...")
grid.fit(X_train, y_train)

print("\n✅ Best Parameters:")
print(grid.best_params_)


# ==========================================================
# BEST MODEL EVALUATION
# ==========================================================
best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)

print("\n===== FINAL RESULTS (XGBoost) =====")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ==========================================================
# RANDOM FOREST COMPARISON (BASELINE)
# ==========================================================
rf_pipeline = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("model", RandomForestClassifier())
])

rf_params = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [5, 10, None]
}

rf_grid = GridSearchCV(
    rf_pipeline,
    rf_params,
    cv=cv,
    scoring="f1_macro",
    n_jobs=-1
)

print("\n🚀 Training Random Forest...")
rf_grid.fit(X_train, y_train)

rf_pred = rf_grid.predict(X_test)

print("\n===== RANDOM FOREST RESULTS =====")
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))


# ==========================================================
# SAVE MODEL
# ==========================================================
import joblib
joblib.dump(best_model, "diabetes_model2_v2.pkl")

print("\n💾 Model saved as diabetes_model2_v2.pkl")