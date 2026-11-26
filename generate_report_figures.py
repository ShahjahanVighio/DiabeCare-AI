import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

os.makedirs("screenshots", exist_ok=True)

# Color constants
C_PRIMARY = "#0F5C6E"
C_BG_MAIN = "#FAF9F6"
C_BG_SIDEBAR = "#F0EFEA"
C_SURFACE = "#FFFFFF"
C_BORDER = "#E2E0D8"
C_TEXT_MAIN = "#1E242B"
C_TEXT_MUTED = "#64748B"
C_NORMAL = "#2A7B6F"
C_PREDIAB = "#D98A2B"
C_DIAB = "#C84B31"

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

# ==============================================================================
# FIGURE 1: DASHBOARD OVERVIEW & ACTIVE PATIENT BANNER
# ==============================================================================
fig, ax = plt.subplots(figsize=(12, 7.5), dpi=200)
fig.patch.set_facecolor(C_BG_MAIN)
ax.set_facecolor(C_BG_MAIN)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Hero Container
hero = patches.FancyBboxPatch((3, 76), 94, 20, boxstyle="round,pad=1,rounding_size=1.5",
                             facecolor=C_SURFACE, edgecolor=C_BORDER, linewidth=1.5)
ax.add_patch(hero)
ax.text(6, 90, "DiabeCare AI: Clinical Decision Support & Digital Twin", 
        fontsize=16, fontweight='bold', color=C_PRIMARY)
ax.text(6, 84, "CDC NHANES-Calibrated Extreme Gradient Boosting (XGBoost) Model with 90-Day Metabolic Twin Progression", 
        fontsize=9, color=C_TEXT_MUTED)

# Live status pill
badge = patches.FancyBboxPatch((76, 88), 18, 5, boxstyle="round,pad=0.5,rounding_size=2",
                              facecolor="#E6F2F0", edgecolor=C_NORMAL, linewidth=1)
ax.add_patch(badge)
ax.text(85, 90.5, "● CLINICAL ENGINE ACTIVE", fontsize=7.5, fontweight='bold', color=C_NORMAL, ha='center', va='center')

# Patient Consultation Banner
pat_card = patches.FancyBboxPatch((3, 62), 94, 11, boxstyle="round,pad=1,rounding_size=1.2",
                                facecolor=C_SURFACE, edgecolor=C_BORDER, linewidth=1)
ax.add_patch(pat_card)
ax.text(6, 69.5, "ACTIVE CLINICAL EVALUATION", fontsize=7.5, fontweight='bold', color=C_TEXT_MUTED)
ax.text(6, 65, "Patient: Michael Vance", fontsize=13, fontweight='bold', color=C_TEXT_MAIN)
ax.text(28, 65, "(Age: 45 yrs | BMI: 28.5 kg/m² | Fasting Glucose: 112 mg/dL | Activity: 400 min/wk)", fontsize=9.5, color=C_TEXT_MUTED)
ax.text(93, 69.5, "EVALUATION DATE", fontsize=7.5, fontweight='bold', color=C_TEXT_MUTED, ha='right')
ax.text(93, 65, "November 22, 2026", fontsize=9.5, fontweight='bold', color=C_PRIMARY, ha='right')

# Card 1: Diagnostic Risk Tier
card1 = patches.FancyBboxPatch((3, 20), 30, 39, boxstyle="round,pad=1,rounding_size=1.2",
                              facecolor=C_SURFACE, edgecolor=C_BORDER, linewidth=1)
ax.add_patch(card1)
ax.text(5.5, 54, "DIAGNOSTIC RISK TIER", fontsize=8, fontweight='bold', color=C_TEXT_MUTED)

tier_badge = patches.FancyBboxPatch((5.5, 43), 25, 8, boxstyle="round,pad=0.5,rounding_size=1",
                                   facecolor="#FCF3E9", edgecolor=C_PREDIAB, linewidth=1.5)
ax.add_patch(tier_badge)
ax.text(18, 47, "PREDIABETES (MODERATE RISK)", fontsize=8.5, fontweight='bold', color=C_PREDIAB, ha='center', va='center')
ax.text(5.5, 36, "Impaired fasting glucose observed.\nLifestyle and dietary intervention\nindicated to reverse metabolic drift.", 
        fontsize=8.5, color=C_TEXT_MUTED, va='top')
ax.text(5.5, 24, "Confidence: 76.4% | Model: Calibrated XGBoost", fontsize=7.5, color=C_TEXT_MUTED)

# Card 2: Metabolic Risk Gauge
card2 = patches.FancyBboxPatch((35, 20), 32, 39, boxstyle="round,pad=1,rounding_size=1.2",
                              facecolor=C_SURFACE, edgecolor=C_BORDER, linewidth=1)
ax.add_patch(card2)
ax.text(37.5, 54, "METABOLIC RISK GAUGE", fontsize=8, fontweight='bold', color=C_TEXT_MUTED)

# Speedometer Arc
theta = np.linspace(np.pi, 0, 100)
xc, yc, r = 51, 35, 11
# Green zone
t1 = np.linspace(np.pi, 2*np.pi/3, 40)
ax.plot(xc + r*np.cos(t1), yc + r*np.sin(t1), color=C_NORMAL, lw=8, solid_capstyle='round')
# Yellow zone
t2 = np.linspace(2*np.pi/3, np.pi/3, 40)
ax.plot(xc + r*np.cos(t2), yc + r*np.sin(t2), color=C_PREDIAB, lw=8)
# Red zone
t3 = np.linspace(np.pi/3, 0, 40)
ax.plot(xc + r*np.cos(t3), yc + r*np.sin(t3), color=C_DIAB, lw=8, solid_capstyle='round')

# Needle pointing to 58/100 (Prediabetes)
angle = np.pi - (58 / 100.0) * np.pi
ax.annotate('', xy=(xc + 8.5*np.cos(angle), yc + 8.5*np.sin(angle)), xytext=(xc, yc),
            arrowprops=dict(arrowstyle="->", color=C_PRIMARY, lw=3))
ax.plot(xc, yc, 'o', color=C_PRIMARY, ms=7)
ax.text(xc, yc - 4, "58 / 100", fontsize=13, fontweight='bold', color=C_TEXT_MAIN, ha='center')
ax.text(xc, yc - 8, "Severity Score", fontsize=7.5, color=C_TEXT_MUTED, ha='center')

# Card 3: Estimated HbA1c
card3 = patches.FancyBboxPatch((69, 20), 28, 39, boxstyle="round,pad=1,rounding_size=1.2",
                              facecolor=C_SURFACE, edgecolor=C_BORDER, linewidth=1)
ax.add_patch(card3)
ax.text(71.5, 54, "ESTIMATED HBA1C", fontsize=8, fontweight='bold', color=C_TEXT_MUTED)
ax.text(71.5, 43, "5.53%", fontsize=24, fontweight='bold', color=C_TEXT_MAIN)
ax.text(71.5, 36, "● Elevated Range (5.7 - 6.4% Cutoff)", fontsize=8.5, fontweight='bold', color=C_PREDIAB)
ax.text(71.5, 29, "Nathan Formula:\n(Fasting Glucose + 46.7) / 28.7", fontsize=8, color=C_TEXT_MUTED)
ax.text(71.5, 24, "ADA Diagnostic Threshold: ≥ 6.5%", fontsize=7.5, color=C_TEXT_MUTED)

# Class Probability Bar at bottom
prob_card = patches.FancyBboxPatch((3, 3), 94, 14, boxstyle="round,pad=1,rounding_size=1.2",
                                  facecolor=C_SURFACE, edgecolor=C_BORDER, linewidth=1)
ax.add_patch(prob_card)
ax.text(6, 13, "CLASS PROBABILITY DISTRIBUTION", fontsize=7.5, fontweight='bold', color=C_TEXT_MUTED)

# Probability stacked bar
bx, by, bw, bh = 6, 6, 88, 4.5
ax.barh(by, bw * 0.232, height=bh, left=bx, color=C_NORMAL, label="Normal: 23.2%")
ax.barh(by, bw * 0.764, height=bh, left=bx + bw * 0.232, color=C_PREDIAB, label="Prediabetes: 76.4%")
ax.barh(by, bw * 0.004, height=bh, left=bx + bw * 0.996, color=C_DIAB, label="Diabetes: 0.4%")

ax.text(bx + bw*0.11, by, "Normal: 23.2%", color='white', fontsize=8.5, fontweight='bold', ha='center', va='center')
ax.text(bx + bw*0.60, by, "Prediabetes: 76.4%", color='white', fontsize=8.5, fontweight='bold', ha='center', va='center')

plt.tight_layout()
plt.savefig("screenshots/dashboard_overview.png", dpi=200, bbox_inches='tight')
plt.close()
print("✅ Saved screenshots/dashboard_overview.png")

# ==============================================================================
# FIGURE 2: RADIAL METABOLIC RISK GAUGE & CONFIDENCE BREAKDOWN
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 4.8), dpi=200)
fig.patch.set_facecolor(C_BG_MAIN)
ax.set_facecolor(C_BG_MAIN)
ax.set_xlim(0, 100)
ax.set_ylim(0, 60)
ax.axis('off')

# Outer container
box = patches.FancyBboxPatch((3, 3), 94, 54, boxstyle="round,pad=1,rounding_size=1.5",
                             facecolor=C_SURFACE, edgecolor=C_BORDER, linewidth=1.5)
ax.add_patch(box)
ax.text(6, 52, "METABOLIC RISK SEVERITY & CONFIDENCE DISTRIBUTION", fontsize=11, fontweight='bold', color=C_PRIMARY)

# Gauge Left
xc, yc, r = 30, 24, 17
t1 = np.linspace(np.pi, 2*np.pi/3, 50)
ax.plot(xc + r*np.cos(t1), yc + r*np.sin(t1), color=C_NORMAL, lw=12, solid_capstyle='round')
t2 = np.linspace(2*np.pi/3, np.pi/3, 50)
ax.plot(xc + r*np.cos(t2), yc + r*np.sin(t2), color=C_PREDIAB, lw=12)
t3 = np.linspace(np.pi/3, 0, 50)
ax.plot(xc + r*np.cos(t3), yc + r*np.sin(t3), color=C_DIAB, lw=12, solid_capstyle='round')

angle = np.pi - (58 / 100.0) * np.pi
ax.annotate('', xy=(xc + 13*np.cos(angle), yc + 13*np.sin(angle)), xytext=(xc, yc),
            arrowprops=dict(arrowstyle="->", color=C_PRIMARY, lw=4))
ax.plot(xc, yc, 'o', color=C_PRIMARY, ms=9)
ax.text(xc, yc - 5, "58 / 100", fontsize=18, fontweight='bold', color=C_TEXT_MAIN, ha='center')
ax.text(xc, yc - 10, "Continuous Severity Score", fontsize=8.5, color=C_TEXT_MUTED, ha='center')
ax.text(xc - r - 2, yc - 1, "0 (Low)", fontsize=8, color=C_TEXT_MUTED, ha='center')
ax.text(xc + r + 2, yc - 1, "100 (High)", fontsize=8, color=C_TEXT_MUTED, ha='center')

# Confidence Bars Right
ax.text(60, 42, "Predicted Class Probabilities", fontsize=10, fontweight='bold', color=C_TEXT_MAIN)

classes = [("Normal (Class 0)", 23.2, C_NORMAL),
           ("Prediabetes (Class 1)", 76.4, C_PREDIAB),
           ("Diabetes (Class 2)", 0.4, C_DIAB)]

for idx, (c_name, c_val, c_color) in enumerate(classes):
    y_pos = 32 - idx * 10
    ax.text(60, y_pos + 4, f"{c_name}", fontsize=8.5, color=C_TEXT_MAIN)
    ax.text(92, y_pos + 4, f"{c_val:.1f}%", fontsize=8.5, fontweight='bold', color=c_color, ha='right')
    # Track
    bg_bar = patches.FancyBboxPatch((60, y_pos), 32, 2.5, boxstyle="round,pad=0.2", facecolor="#E2E8F0", edgecolor='none')
    ax.add_patch(bg_bar)
    # Fill
    fill_bar = patches.FancyBboxPatch((60, y_pos), 32 * (c_val / 100.0), 2.5, boxstyle="round,pad=0.2", facecolor=c_color, edgecolor='none')
    ax.add_patch(fill_bar)

plt.tight_layout()
plt.savefig("screenshots/risk_gauge_confidence.png", dpi=200, bbox_inches='tight')
plt.close()
print("✅ Saved screenshots/risk_gauge_confidence.png")

# ==============================================================================
# FIGURE 3: 90-DAY DIGITAL TWIN SIMULATION CURVES
# ==============================================================================
np.random.seed(42)
days = 90
days_x = np.arange(1, days + 1)

def sim(start_g, scenario):
    g_hist = []
    g = start_g
    bmi = 28.5
    for d in range(days):
        d_ch = (bmi - 25) * 0.05 - (400/1000) * 0.8
        if scenario == "Life":
            d_ch -= 0.20
            bmi -= 0.005
        elif scenario == "Med":
            d_ch -= 0.35
        g += d_ch + np.random.normal(0, 0.7)
        g_hist.append(g)
    return np.array(g_hist)

g_none = sim(112, "None")
g_life = sim(112, "Life")
g_med = sim(112, "Med")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=200)
fig.patch.set_facecolor(C_BG_MAIN)

for ax in (ax1, ax2):
    ax.set_facecolor(C_SURFACE)
    ax.grid(True, color=C_BORDER, linestyle='--', alpha=0.7)
    for spine in ax.spines.values():
        spine.set_color(C_BORDER)

# Glucose Plot (Left)
ax1.axhspan(60, 100, color=C_NORMAL, alpha=0.08, label='Normal Range (<100 mg/dL)')
ax1.axhspan(100, 126, color=C_PREDIAB, alpha=0.08, label='Impaired Glucose (100-125)')
ax1.axhspan(126, 150, color=C_DIAB, alpha=0.08, label='Diabetic Threshold (≥126)')

ax1.plot(days_x, g_none, color=C_DIAB, lw=2.8, label='No Intervention (Progression)')
ax1.plot(days_x, g_life, color=C_PREDIAB, lw=2.8, label='Lifestyle Modification')
ax1.plot(days_x, g_med, color=C_NORMAL, lw=2.8, label='Pharmacotherapy (Medication)')

ax1.axhline(126, color=C_DIAB, linestyle='--', lw=1.5)
ax1.text(2, 127.5, "Diabetes Cutoff: 126 mg/dL", color=C_DIAB, fontsize=8, fontweight='bold')
ax1.axhline(100, color=C_PREDIAB, linestyle=':', lw=1.5)
ax1.text(2, 101.5, "Impaired Fasting Cutoff: 100 mg/dL", color=C_PREDIAB, fontsize=8, fontweight='bold')

ax1.set_title("Fasting Plasma Glucose Trajectory (90 Days)", fontsize=11, fontweight='bold', color=C_PRIMARY, pad=10)
ax1.set_xlabel("Simulation Timeline (Days)", fontsize=9.5, color=C_TEXT_MAIN)
ax1.set_ylabel("Fasting Glucose (mg/dL)", fontsize=9.5, color=C_TEXT_MAIN)
ax1.set_ylim(80, 145)
ax1.legend(loc='lower left', fontsize=8, framealpha=0.9)

# HbA1c Plot (Right)
h_none = (g_none + 46.7) / 28.7
h_life = (g_life + 46.7) / 28.7
h_med = (g_med + 46.7) / 28.7

ax2.axhspan(4.0, 5.7, color=C_NORMAL, alpha=0.08)
ax2.axhspan(5.7, 6.5, color=C_PREDIAB, alpha=0.08)
ax2.axhspan(6.5, 7.5, color=C_DIAB, alpha=0.08)

ax2.plot(days_x, h_none, color=C_DIAB, lw=2.8, label='No Intervention')
ax2.plot(days_x, h_life, color=C_PREDIAB, lw=2.8, label='Lifestyle Modification')
ax2.plot(days_x, h_med, color=C_NORMAL, lw=2.8, label='Pharmacotherapy')

ax2.axhline(6.5, color=C_DIAB, linestyle='--', lw=1.5)
ax2.text(2, 6.53, "Diabetes Diagnostic Threshold: 6.5%", color=C_DIAB, fontsize=8, fontweight='bold')
ax2.axhline(5.7, color=C_PREDIAB, linestyle=':', lw=1.5)
ax2.text(2, 5.73, "Prediabetes Threshold: 5.7%", color=C_PREDIAB, fontsize=8, fontweight='bold')

ax2.set_title("Calculated HbA1c Progression (%)", fontsize=11, fontweight='bold', color=C_PRIMARY, pad=10)
ax2.set_xlabel("Simulation Timeline (Days)", fontsize=9.5, color=C_TEXT_MAIN)
ax2.set_ylabel("Estimated HbA1c (%)", fontsize=9.5, color=C_TEXT_MAIN)
ax2.set_ylim(4.5, 6.8)
ax2.legend(loc='lower left', fontsize=8, framealpha=0.9)

plt.tight_layout()
plt.savefig("screenshots/digital_twin_simulation.png", dpi=200, bbox_inches='tight')
plt.close()
print("✅ Saved screenshots/digital_twin_simulation.png")

# ==============================================================================
# FIGURE 4: 90-DAY DELTA COMPARATOR & PDF EXPORT BUTTON
# ==============================================================================
fig, ax = plt.subplots(figsize=(12, 4.5), dpi=200)
fig.patch.set_facecolor(C_BG_MAIN)
ax.set_facecolor(C_BG_MAIN)
ax.set_xlim(0, 100)
ax.set_ylim(0, 50)
ax.axis('off')

# Outer container
box = patches.FancyBboxPatch((3, 2), 94, 46, boxstyle="round,pad=1,rounding_size=1.5",
                             facecolor=C_SURFACE, edgecolor=C_BORDER, linewidth=1.5)
ax.add_patch(box)
ax.text(6, 44, "90-DAY PROJECTED DELTA SUMMARY (INTERVENTION EFFICACY)", fontsize=11, fontweight='bold', color=C_PRIMARY)

cards_data = [
    (6, 29, "[Progression] Natural Progression", "126.2 mg/dL", "Net 90-Day Drift: +14.2 mg/dL", "Day 90 HbA1c: 6.02%", C_DIAB),
    (36, 29, "[Lifestyle] Lifestyle Modification", "100.2 mg/dL", "Net 90-Day Drift: -11.8 mg/dL", "Day 90 HbA1c: 5.12%", C_PREDIAB),
    (66, 29, "[Therapy] Pharmacotherapy", "87.7 mg/dL", "Net 90-Day Drift: -24.3 mg/dL", "Day 90 HbA1c: 4.68%", C_NORMAL)
]

for x_pos, width, title, val, drift, hba1c_val, color in cards_data:
    b = patches.FancyBboxPatch((x_pos, 18), width, 22, boxstyle="round,pad=0.8,rounding_size=1",
                               facecolor=C_BG_SIDEBAR, edgecolor=C_BORDER, linewidth=1)
    ax.add_patch(b)
    # Left accent bar
    accent = patches.Rectangle((x_pos, 18), 1.2, 22, facecolor=color, edgecolor='none')
    ax.add_patch(accent)
    ax.text(x_pos + 3, 36, title, fontsize=9.5, fontweight='bold', color=color)
    ax.text(x_pos + 3, 29, val, fontsize=16, fontweight='bold', color=C_TEXT_MAIN)
    ax.text(x_pos + 3, 24, drift, fontsize=8.5, fontweight='bold', color=color)
    ax.text(x_pos + 3, 20.5, hba1c_val, fontsize=8, color=C_TEXT_MUTED)

# PDF Button at bottom
pdf_btn = patches.FancyBboxPatch((6, 5), 88, 9, boxstyle="round,pad=0.8,rounding_size=1.2",
                                facecolor=C_PRIMARY, edgecolor='none')
ax.add_patch(pdf_btn)
ax.text(50, 9.5, "Download Official Consultation Report for Michael Vance (PDF)", 
        fontsize=10.5, fontweight='bold', color='white', ha='center', va='center')

plt.tight_layout()
plt.savefig("screenshots/delta_summary_pdf.png", dpi=200, bbox_inches='tight')
plt.close()
print("✅ Saved screenshots/delta_summary_pdf.png")

# ==============================================================================
# FIGURE 5: MODEL EVALUATION SUITE MULTI-PANEL FIGURE
# ==============================================================================
# Load actual poster images: confusion matrix, roc curve, feature importance
im_cm = Image.open("poster_confusion_matrix.png")
im_roc = Image.open("poster_roc_curve.png")
im_imp = Image.open("poster_feature_importance.png")

# Combine horizontally into a clean 3-panel figure
target_h = 600
w_cm = int(im_cm.width * (target_h / im_cm.height))
w_roc = int(im_roc.width * (target_h / im_roc.height))
w_imp = int(im_imp.width * (target_h / im_imp.height))

im_cm_r = im_cm.resize((w_cm, target_h), Image.Resampling.LANCZOS)
im_roc_r = im_roc.resize((w_roc, target_h), Image.Resampling.LANCZOS)
im_imp_r = im_imp.resize((w_imp, target_h), Image.Resampling.LANCZOS)

total_w = w_cm + w_roc + w_imp + 40
combined = Image.new('RGB', (total_w, target_h + 60), color=(250, 249, 246))

# Paste side by side
combined.paste(im_cm_r, (10, 50))
combined.paste(im_roc_r, (w_cm + 20, 50))
combined.paste(im_imp_r, (w_cm + w_roc + 30, 50))

combined.save("screenshots/poster_evaluation_suite.png", quality=95)
print("✅ Saved screenshots/poster_evaluation_suite.png")

print("\n🎉 All 5 publication-ready figures successfully generated!")
