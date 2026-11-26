import pandas as pd
import numpy as np

def get_existing_column(df, possible_cols):
    for col in possible_cols:
        if col in df.columns:
            return col
    return None


# ==========================================================
# Function for normal cycles (suffix J, L)
# ==========================================================
def load_cycle(path, suffix):

    demo   = pd.read_sas(f"{path}/DEMO_{suffix}.XPT")
    bmx    = pd.read_sas(f"{path}/BMX_{suffix}.XPT")
    glu    = pd.read_sas(f"{path}/GLU_{suffix}.XPT")
    ghb    = pd.read_sas(f"{path}/GHB_{suffix}.XPT")
    bpx    = pd.read_sas(f"{path}/BPX_{suffix}.XPT")
    smq    = pd.read_sas(f"{path}/SMQ_{suffix}.XPT")
    paq    = pd.read_sas(f"{path}/PAQ_{suffix}.XPT")
    tchol  = pd.read_sas(f"{path}/TCHOL_{suffix}.XPT")
    hdl    = pd.read_sas(f"{path}/HDL_{suffix}.XPT")
    trigly = pd.read_sas(f"{path}/TRIGLY_{suffix}.XPT")
    diq    = pd.read_sas(f"{path}/DIQ_{suffix}.XPT")
    ins    = pd.read_sas(f"{path}/INS_{suffix}.XPT")

    demo = demo[["SEQN", "RIDAGEYR", "RIAGENDR"]]
    bmx  = bmx[["SEQN", "BMXBMI", "BMXWAIST"]]
    glu  = glu[["SEQN", "LBXGLU"]]
    ghb  = ghb[["SEQN", "LBXGH"]]
    bpx  = bpx[["SEQN", "BPXSY1", "BPXDI1"]]
    smq  = smq[["SEQN", "SMQ040"]]
    paq  = paq[["SEQN", "PAD680"]]
    tchol = tchol[["SEQN", "LBXTC"]]
    hdl   = hdl[["SEQN", "LBDHDD"]]
    trigly = trigly[["SEQN", "LBXTR"]]
    diq   = diq[["SEQN", "DIQ170"]]

    # -------- INSULIN SAFE LOADING ----------
    insulin_col = get_existing_column(ins, ["LBXINSI", "LBXIN", "LBXINS", "LBDINSI", "INS"])
    if insulin_col is None:
        ins = ins[["SEQN"]]
        ins["Insulin"] = np.nan
    else:
        ins = ins[["SEQN", insulin_col]]
        ins.rename(columns={insulin_col: "Insulin"}, inplace=True)

    # Merge
    df = demo.merge(bmx, on="SEQN", how="inner")
    df = df.merge(glu, on="SEQN", how="inner")
    df = df.merge(ghb, on="SEQN", how="inner")

    df = df.merge(bpx, on="SEQN", how="left")
    df = df.merge(smq, on="SEQN", how="left")
    df = df.merge(paq, on="SEQN", how="left")

    df = df.merge(tchol, on="SEQN", how="left")
    df = df.merge(hdl, on="SEQN", how="left")
    df = df.merge(trigly, on="SEQN", how="left")

    df = df.merge(diq, on="SEQN", how="left")
    df = df.merge(ins, on="SEQN", how="left")

    return df


# ==========================================================
# Function for Pre-Pandemic cycle (P_ prefix)
# ==========================================================
def load_prepandemic(path):

    demo   = pd.read_sas(f"{path}/P_DEMO.XPT")
    bmx    = pd.read_sas(f"{path}/P_BMX.XPT")
    glu    = pd.read_sas(f"{path}/P_GLU.XPT")
    ghb    = pd.read_sas(f"{path}/P_GHB.XPT")
    bpx    = pd.read_sas(f"{path}/P_BPX.XPT")
    smq    = pd.read_sas(f"{path}/P_SMQ.XPT")
    paq    = pd.read_sas(f"{path}/P_PAQ.XPT")
    tchol  = pd.read_sas(f"{path}/P_TCHOL.XPT")
    hdl    = pd.read_sas(f"{path}/P_HDL.XPT")
    trigly = pd.read_sas(f"{path}/P_TRIGLY.XPT")
    diq    = pd.read_sas(f"{path}/P_DIQ.XPT")
    ins    = pd.read_sas(f"{path}/P_INS.XPT")

    demo = demo[["SEQN", "RIDAGEYR", "RIAGENDR"]]
    bmx  = bmx[["SEQN", "BMXBMI", "BMXWAIST"]]
    glu  = glu[["SEQN", "LBXGLU"]]
    ghb  = ghb[["SEQN", "LBXGH"]]
    bpx  = bpx[["SEQN", "BPXSY1", "BPXDI1"]]
    smq  = smq[["SEQN", "SMQ040"]]
    paq  = paq[["SEQN", "PAD680"]]
    tchol = tchol[["SEQN", "LBXTC"]]
    hdl   = hdl[["SEQN", "LBDHDD"]]
    trigly = trigly[["SEQN", "LBXTR"]]
    diq   = diq[["SEQN", "DIQ170"]]

    # -------- INSULIN SAFE LOADING ----------
    insulin_col = get_existing_column(ins, ["LBXINSI", "LBXIN", "LBXINS", "LBDINSI", "INS"])
    if insulin_col is None:
        ins = ins[["SEQN"]]
        ins["Insulin"] = np.nan
    else:
        ins = ins[["SEQN", insulin_col]]
        ins.rename(columns={insulin_col: "Insulin"}, inplace=True)

    df = demo.merge(bmx, on="SEQN", how="inner")
    df = df.merge(glu, on="SEQN", how="inner")
    df = df.merge(ghb, on="SEQN", how="inner")

    df = df.merge(bpx, on="SEQN", how="left")
    df = df.merge(smq, on="SEQN", how="left")
    df = df.merge(paq, on="SEQN", how="left")

    df = df.merge(tchol, on="SEQN", how="left")
    df = df.merge(hdl, on="SEQN", how="left")
    df = df.merge(trigly, on="SEQN", how="left")

    df = df.merge(diq, on="SEQN", how="left")
    df = df.merge(ins, on="SEQN", how="left")

    return df


# ==========================================================
# Load all cycles
# ==========================================================
print("Loading 2017-2018...")
df1 = load_cycle("nhanes/2017_2018", "J")

print("Loading 2017-March 2020 Pre-Pandemic...")
df2 = load_prepandemic("nhanes/prepandemic")

print("Loading 2021-2023...")
df3 = load_cycle("nhanes/2021_2023", "L")

df = pd.concat([df1, df2, df3], ignore_index=True)

print("Combined shape:", df.shape)

# ==========================================================
# Rename columns
# ==========================================================
df.rename(columns={
    "RIDAGEYR": "Age",
    "RIAGENDR": "Gender",
    "BMXBMI": "BMI",
    "BMXWAIST": "Waist",
    "LBXGLU": "FastingGlucose",
    "LBXGH": "HbA1c",
    "BPXSY1": "SystolicBP",
    "BPXDI1": "DiastolicBP",
    "SMQ040": "Smoking",
    "PAD680": "PhysicalActivity",
    "LBXTC": "TotalCholesterol",
    "LBDHDD": "HDL",
    "LBXTR": "Triglycerides",
    "DIQ170": "FamilyHistory"
}, inplace=True)

df["Gender"] = df["Gender"].map({1: "Male", 2: "Female"})

# ==========================================================
# LDL Calculation
# ==========================================================
df["LDL"] = df["TotalCholesterol"] - df["HDL"] - (df["Triglycerides"] / 5)

# ==========================================================
# Cleaning
# ==========================================================
df = df.dropna(subset=["Age", "BMI", "FastingGlucose", "HbA1c"])

optional_cols = [
    "Waist",
    "SystolicBP",
    "DiastolicBP",
    "TotalCholesterol",
    "HDL",
    "Triglycerides",
    "LDL",
    "Smoking",
    "PhysicalActivity",
    "FamilyHistory",
    "Insulin"
]

for col in optional_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# ==========================================================
# Target class
# ==========================================================
def classify_diabetes(hba1c):
    if hba1c < 5.7:
        return 0
    elif hba1c < 6.5:
        return 1
    else:
        return 2

df["DiabetesClass"] = df["HbA1c"].apply(classify_diabetes)

# ==========================================================
# Save dataset
# ==========================================================
df.to_csv("final_nhanes_dataset_v3.csv", index=False)

print("\n✅ Dataset created successfully!")
print("Saved as final_nhanes_dataset_v3.csv")
print(df.head())
print("Final shape:", df.shape)