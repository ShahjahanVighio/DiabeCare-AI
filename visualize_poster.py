import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# ==========================================================
# LOAD DATA + MODEL
# ==========================================================
df = pd.read_csv("final_nhanes_dataset.csv")
model = joblib.load("diabetes_model2_v2.pkl")

# ==========================================================
# FEATURES USED IN TRAINING
# ==========================================================
features = [
    "Age", "BMI", "FastingGlucose",
    "SystolicBP", "DiastolicBP",
    "LDL", "HDL", "Triglycerides",
    "Smoking", "Activity"
]

X = df[features]
y = df["DiabetesClass"]

# ==========================================================
# SPLIT DATA (same as training)
# ==========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================================
# PREDICTION
# ==========================================================
y_pred = model.predict(X_test)

# ==========================================================
# 1) CLASS DISTRIBUTION PLOT
# ==========================================================
plt.figure(figsize=(6,4))
df["DiabetesClass"].value_counts().sort_index().plot(kind="bar")
plt.xticks([0,1,2], ["Normal", "Prediabetes", "Diabetes"], rotation=0)
plt.title("Class Distribution in NHANES Dataset")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.savefig("poster_class_distribution.png", dpi=300)
plt.show()

print("✅ Saved: poster_class_distribution.png")

# ==========================================================
# 2) CONFUSION MATRIX PLOT
# ==========================================================
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["Normal","Prediabetes","Diabetes"])

fig, ax = plt.subplots(figsize=(6,5))
disp.plot(ax=ax, cmap="Blues", values_format="d")
plt.title("Confusion Matrix (Diabetes Classification)")
plt.tight_layout()
plt.savefig("poster_confusion_matrix.png", dpi=300)
plt.show()

print("✅ Saved: poster_confusion_matrix.png")

# ==========================================================
# 3) ROC CURVE (MULTICLASS)
# ==========================================================
y_test_bin = label_binarize(y_test, classes=[0,1,2])
y_prob = model.predict_proba(X_test)

plt.figure(figsize=(7,6))

for i, label in enumerate(["Normal", "Prediabetes", "Diabetes"]):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], "k--")
plt.title("ROC Curve (Multiclass Diabetes Prediction)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("poster_roc_curve.png", dpi=300)
plt.show()

print("✅ Saved: poster_roc_curve.png")

# ==========================================================
# 4) FEATURE IMPORTANCE (if supported)
# ==========================================================
try:
    model_step = model.named_steps["model"]

    if hasattr(model_step, "feature_importances_"):
        importances = model_step.feature_importances_
        imp_df = pd.DataFrame({"Feature": features, "Importance": importances})
        imp_df = imp_df.sort_values(by="Importance", ascending=False)

        plt.figure(figsize=(9,5))
        plt.bar(imp_df["Feature"], imp_df["Importance"])
        plt.xticks(rotation=45)
        plt.title("Feature Importance (Model Explanation)")
        plt.ylabel("Importance Score")
        plt.tight_layout()
        plt.savefig("poster_feature_importance.png", dpi=300)
        plt.show()

        print("✅ Saved: poster_feature_importance.png")

    else:
        print("⚠ Feature importance not available in this model.")

except Exception as e:
    print("⚠ Feature importance plot error:", e)

# ==========================================================
# 5) CORRELATION HEATMAP (poster style)
# ==========================================================
corr_data = df[features + ["DiabetesClass"]].corr()

plt.figure(figsize=(10,7))
plt.imshow(corr_data, cmap="coolwarm", interpolation="nearest")
plt.colorbar()

plt.xticks(range(len(corr_data.columns)), corr_data.columns, rotation=90)
plt.yticks(range(len(corr_data.columns)), corr_data.columns)

plt.title("Correlation Heatmap of Clinical Variables")
plt.tight_layout()
plt.savefig("poster_correlation_heatmap.png", dpi=300)
plt.show()

print("✅ Saved: poster_correlation_heatmap.png")

print("\n🎉 All poster images generated successfully!")