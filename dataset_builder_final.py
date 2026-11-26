import pandas as pd
import numpy as np

# ==========================================================
# SAFE COLUMN PICKER
# ==========================================================
def pick_col(df, options):
    for c in options:
        if c in df.columns:
            return c
    return None


# ==========================================================
# SAFE FEATURE EXTRACTOR
# ==========================================================
def safe_extract(df, cols, new_name, demo=None):
    col = pick_col(df, cols)
    if col:
        return df[["SEQN", col]].rename(columns={col: new_name})
    else:
        if demo is not None:
            return pd.DataFrame({"SEQN": demo["SEQN"], new_name: np.nan})
        return pd.DataFrame({"SEQN": df["SEQN"], new_name: np.nan})


# ==========================================================
# BP PROCESSOR
# ==========================================================
def process_bpx(bpx):
    sys_col = pick_col(bpx, ["BPXSY1", "BPXSY", "SYSTOLIC"])
    dia_col = pick_col(bpx, ["BPXDI1", "BPXDI", "DIASTOLIC"])

    out = pd.DataFrame()
    out["SEQN"] = bpx["SEQN"]

    out["SystolicBP"] = bpx[sys_col] if sys_col else np.nan
    out["DiastolicBP"] = bpx[dia_col] if dia_col else np.nan

    return out


# ==========================================================
# LOAD CYCLE
# ==========================================================
def load_cycle(folder, suffix, bpx_prefix):

    demo   = pd.read_sas(f"{folder}/DEMO_{suffix}.xpt")
    bmx    = pd.read_sas(f"{folder}/BMX_{suffix}.xpt")
    glu    = pd.read_sas(f"{folder}/GLU_{suffix}.xpt")
    ghb    = pd.read_sas(f"{folder}/GHB_{suffix}.xpt")
    smq    = pd.read_sas(f"{folder}/SMQ_{suffix}.xpt")
    paq    = pd.read_sas(f"{folder}/PAQ_{suffix}.xpt")
    tchol  = pd.read_sas(f"{folder}/TCHOL_{suffix}.xpt")
    hdl    = pd.read_sas(f"{folder}/HDL_{suffix}.xpt")
    trigly = pd.read_sas(f"{folder}/TRIGLY_{suffix}.xpt")
    diq    = pd.read_sas(f"{folder}/DIQ_{suffix}.xpt")
    ins    = pd.read_sas(f"{folder}/INS_{suffix}.xpt")
    bpx    = pd.read_sas(f"{folder}/{bpx_prefix}_{suffix}.xpt")

    demo = demo[["SEQN", "RIDAGEYR", "RIAGENDR"]]

    bmx   = safe_extract(bmx, ["BMXBMI"], "BMI", demo)
    glu   = safe_extract(glu, ["LBXGLU"], "FastingGlucose", demo)
    ghb   = safe_extract(ghb, ["LBXGH"], "HbA1c", demo)

    smq   = safe_extract(smq, ["SMQ040"], "Smoking", demo)
    paq   = safe_extract(paq, ["PAD680"], "Activity", demo)

    tchol = safe_extract(tchol, ["LBXTC"], "Cholesterol", demo)
    hdl   = safe_extract(hdl, ["LBDHDD"], "HDL", demo)

    # FIXED TRIGLYCERIDES
    trigly = safe_extract(trigly, ["LBXTR", "LBDTRSI"], "Triglycerides", demo)

    bpx = process_bpx(bpx)

    # DIQ
    fam = pick_col(diq, ["DIQ170", "DIQ010"])
    if fam:
        diq = diq[["SEQN", fam]].rename(columns={fam: "FamilyHistory"})
    else:
        diq = pd.DataFrame({"SEQN": demo["SEQN"], "FamilyHistory": np.nan})

    # INS
    ins_col = pick_col(ins, ["LBXINSI", "LBXIN", "LBXINS"])
    if ins_col:
        ins = ins[["SEQN", ins_col]].rename(columns={ins_col: "Insulin"})
    else:
        ins = pd.DataFrame({"SEQN": demo["SEQN"], "Insulin": np.nan})

    # MERGE
    df = demo.merge(bmx, on="SEQN")
    df = df.merge(glu, on="SEQN")
    df = df.merge(ghb, on="SEQN")
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
# PRE-PANDEMIC
# ==========================================================
def load_prepandemic(folder):

    demo   = pd.read_sas(f"{folder}/P_DEMO.xpt")
    bmx    = pd.read_sas(f"{folder}/P_BMX.xpt")
    glu    = pd.read_sas(f"{folder}/P_GLU.xpt")
    ghb    = pd.read_sas(f"{folder}/P_GHB.xpt")
    smq    = pd.read_sas(f"{folder}/P_SMQ.xpt")
    paq    = pd.read_sas(f"{folder}/P_PAQ.xpt")
    tchol  = pd.read_sas(f"{folder}/P_TCHOL.xpt")
    hdl    = pd.read_sas(f"{folder}/P_HDL.xpt")
    trigly = pd.read_sas(f"{folder}/P_TRIGLY.xpt")
    diq    = pd.read_sas(f"{folder}/P_DIQ.xpt")
    ins    = pd.read_sas(f"{folder}/P_INS.xpt")
    bpx    = pd.read_sas(f"{folder}/P_BPXO.xpt")

    demo = demo[["SEQN", "RIDAGEYR", "RIAGENDR"]]

    bmx   = safe_extract(bmx, ["BMXBMI"], "BMI", demo)
    glu   = safe_extract(glu, ["LBXGLU"], "FastingGlucose", demo)
    ghb   = safe_extract(ghb, ["LBXGH"], "HbA1c", demo)

    smq   = safe_extract(smq, ["SMQ040"], "Smoking", demo)
    paq   = safe_extract(paq, ["PAD680"], "Activity", demo)

    tchol = safe_extract(tchol, ["LBXTC"], "Cholesterol", demo)
    hdl   = safe_extract(hdl, ["LBDHDD"], "HDL", demo)

    trigly = safe_extract(trigly, ["LBXTR", "LBDTRSI"], "Triglycerides", demo)

    bpx = process_bpx(bpx)

    fam = pick_col(diq, ["DIQ170", "DIQ010"])
    if fam:
        diq = diq[["SEQN", fam]].rename(columns={fam: "FamilyHistory"})
    else:
        diq = pd.DataFrame({"SEQN": demo["SEQN"], "FamilyHistory": np.nan})

    ins_col = pick_col(ins, ["LBXINSI", "LBXIN", "LBXINS"])
    if ins_col:
        ins = ins[["SEQN", ins_col]].rename(columns={ins_col: "Insulin"})
    else:
        ins = pd.DataFrame({"SEQN": demo["SEQN"], "Insulin": np.nan})

    df = demo.merge(bmx, on="SEQN")
    df = df.merge(glu, on="SEQN")
    df = df.merge(ghb, on="SEQN")
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
# MAIN
# ==========================================================
print("Loading 2017-2018...")
df1 = load_cycle("nhanes/2017_2018", "J", "BPX")

print("Loading Pre-Pandemic...")
df2 = load_prepandemic("nhanes/2017-March 2020 Pre-Pandemic")

print("Loading 2021-2023...")
df3 = load_cycle("nhanes/2021_2023", "L", "BPXO")

df = pd.concat([df1, df2, df3], ignore_index=True)

print("Shape:", df.shape)


# ==========================================================
# CLEANING (FIXED SAFE VERSION)
# ==========================================================
df.rename(columns={
    "RIDAGEYR": "Age",
    "RIAGENDR": "Gender"
}, inplace=True)

df["Gender"] = df["Gender"].map({1: "Male", 2: "Female"})


# -----------------------
# SAFE MISSING VALUE FILL
# -----------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(exclude=[np.number]).columns

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

for c in categorical_cols:
    df[c] = df[c].fillna(df[c].mode()[0] if not df[c].mode().empty else "Unknown")


# ==========================================================
# FEATURE ENGINEERING
# ==========================================================
df["LDL"] = df["Cholesterol"] - df["HDL"] - (df["Triglycerides"] / 5)


# ==========================================================
# TARGET LABEL
# ==========================================================
def label(x):
    if x < 5.7:
        return 0
    elif x < 6.5:
        return 1
    else:
        return 2

df["DiabetesClass"] = df["HbA1c"].apply(label)


# ==========================================================
# DROP INCOMPLETE ROWS
# ==========================================================
df = df.dropna(subset=["Age", "BMI", "FastingGlucose", "HbA1c"])


# ==========================================================
# SAVE FINAL DATASET
# ==========================================================
df.to_csv("final_nhanes_dataset.csv", index=False)

print("DONE ✔ dataset saved as final_nhanes_dataset.csv")