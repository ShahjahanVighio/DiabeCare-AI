import joblib

# Load trained pipeline/model
model = joblib.load("diabetes_model2_v2.pkl")

# If pipeline exists, extract scaler and xgboost model
if hasattr(model, "named_steps"):
    if "scaler" in model.named_steps:
        joblib.dump(model.named_steps["scaler"], "scaler.pkl")
        print("✅ Scaler saved as scaler.pkl")
    xgb_model = model.named_steps.get("model", model)
else:
    xgb_model = model

# Save in JSON format
xgb_model.save_model("xgb_model.json")

print("✅ Model converted & saved as xgb_model.json")