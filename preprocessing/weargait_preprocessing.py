#!/usr/bin/env python3
"""
WearGait-PD Clinical Feature Extraction
========================================
Extracts ALL 65 individual MDS-UPDRS items (Parts I-IV), Hoehn & Yahr,
demographics, and derived features from the WearGait-PD dataset.

This is the PRIMARY cohort with REAL DBS status labels.

Output: data/processed/clinical_features/weargait_clinical.csv
        data/processed/clinical_features/weargait_pd_only.csv  (PD patients only)

Author: Kartic Mishra, Gachon University
"""

import os
import sys
import warnings
import multiprocessing

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Hardware config ──────────────────────────────────────────────────────────
N_CORES = multiprocessing.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(N_CORES)
os.environ["MKL_NUM_THREADS"] = str(N_CORES)

SEED = 42
np.random.seed(SEED)

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "weargait_pd")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "clinical_features")
os.makedirs(OUT_DIR, exist_ok=True)

PD_CSV = os.path.join(RAW_DIR, "PD - Demographic+Clinical - datasetV1.csv")
CTRL_CSV = os.path.join(RAW_DIR, "CONTROLS - Demographic+Clinical - datasetV1.csv")

# ── MDS-UPDRS item definitions ──────────────────────────────────────────────
# Part I: Non-motor Experiences of Daily Living (13 items)
PART1_ITEMS = [f"MDSUPDRS_1-{i}" for i in range(1, 14)]

# Part II: Motor Experiences of Daily Living (13 items)
PART2_ITEMS = [f"MDSUPDRS_2-{i}" for i in range(1, 14)]

# Part III: Motor Examination (33 items with bilateral sub-items)
PART3_ITEMS = [
    "MDSUPDRS_3-1",          # Speech
    "MDSUPDRS_3-2",          # Facial expression
    "MDSUPDRS_3-3-Neck",     # Rigidity - Neck
    "MDSUPDRS_3-3-RUE",      # Rigidity - Right upper extremity
    "MDSUPDRS_3-3-LUE",      # Rigidity - Left upper extremity
    "MDSUPDRS_3-3-RLE",      # Rigidity - Right lower extremity
    "MDSUPDRS_3-3-LLE",      # Rigidity - Left lower extremity
    "MDSUPDRS_3-4-R",        # Finger tapping - Right
    "MDSUPDRS_3-4-L",        # Finger tapping - Left
    "MDSUPDRS_3-5-R",        # Hand movements - Right
    "MDSUPDRS_3-5-L",        # Hand movements - Left
    "MDSUPDRS_3-6-R",        # Pronation-supination - Right
    "MDSUPDRS_3-6-L",        # Pronation-supination - Left
    "MDSUPDRS_3-7-R",        # Toe tapping - Right
    "MDSUPDRS_3-7-L",        # Toe tapping - Left
    "MDSUPDRS_3-8-R",        # Leg agility - Right
    "MDSUPDRS_3-8-L",        # Leg agility - Left
    "MDSUPDRS_3-9",          # Arising from chair
    "MDSUPDRS_3-10",         # Gait
    "MDSUPDRS_3-11",         # Freezing of gait
    "MDSUPDRS_3-12",         # Postural stability
    "MDSUPDRS_3-13",         # Posture
    "MDSUPDRS_3-14",         # Global spontaneity of movement
    "MDSUPDRS_3-15-R",       # Postural tremor - Right
    "MDSUPDRS_3-15-L",       # Postural tremor - Left
    "MDSUPDRS_3-16-R",       # Kinetic tremor - Right
    "MDSUPDRS_3-16-L",       # Kinetic tremor - Left
    "MDSUPDRS_3-17-RUE",     # Rest tremor amplitude - Right upper
    "MDSUPDRS_3-17-LUE",     # Rest tremor amplitude - Left upper
    "MDSUPDRS_3-17-RLE",     # Rest tremor amplitude - Right lower
    "MDSUPDRS_3-17-LLE",     # Rest tremor amplitude - Left lower
    "MDSUPDRS_3-17-LipJaw",  # Rest tremor amplitude - Lip/Jaw
    "MDSUPDRS_3-18",         # Constancy of rest tremor
]

# Part IV: Motor Complications (6 items)
PART4_ITEMS = [f"MDSUPDRS_4-{i}" for i in range(1, 7)]

ALL_UPDRS_ITEMS = PART1_ITEMS + PART2_ITEMS + PART3_ITEMS + PART4_ITEMS

# Bilateral item pairs for asymmetry computation (Right, Left)
BILATERAL_PAIRS = {
    "rigidity_UE":       ("MDSUPDRS_3-3-RUE", "MDSUPDRS_3-3-LUE"),
    "rigidity_LE":       ("MDSUPDRS_3-3-RLE", "MDSUPDRS_3-3-LLE"),
    "finger_tapping":    ("MDSUPDRS_3-4-R",   "MDSUPDRS_3-4-L"),
    "hand_movements":    ("MDSUPDRS_3-5-R",   "MDSUPDRS_3-5-L"),
    "pronation_sup":     ("MDSUPDRS_3-6-R",   "MDSUPDRS_3-6-L"),
    "toe_tapping":       ("MDSUPDRS_3-7-R",   "MDSUPDRS_3-7-L"),
    "leg_agility":       ("MDSUPDRS_3-8-R",   "MDSUPDRS_3-8-L"),
    "postural_tremor":   ("MDSUPDRS_3-15-R",  "MDSUPDRS_3-15-L"),
    "kinetic_tremor":    ("MDSUPDRS_3-16-R",  "MDSUPDRS_3-16-L"),
    "rest_tremor_UE":    ("MDSUPDRS_3-17-RUE","MDSUPDRS_3-17-LUE"),
    "rest_tremor_LE":    ("MDSUPDRS_3-17-RLE","MDSUPDRS_3-17-LLE"),
}

# Motor sub-domain definitions (Part III items grouped by clinical domain)
MOTOR_SUBDOMAINS = {
    "speech": ["MDSUPDRS_3-1"],
    "facial_expression": ["MDSUPDRS_3-2"],
    "rigidity": ["MDSUPDRS_3-3-Neck", "MDSUPDRS_3-3-RUE", "MDSUPDRS_3-3-LUE",
                  "MDSUPDRS_3-3-RLE", "MDSUPDRS_3-3-LLE"],
    "bradykinesia_UE": ["MDSUPDRS_3-4-R", "MDSUPDRS_3-4-L", "MDSUPDRS_3-5-R",
                         "MDSUPDRS_3-5-L", "MDSUPDRS_3-6-R", "MDSUPDRS_3-6-L"],
    "bradykinesia_LE": ["MDSUPDRS_3-7-R", "MDSUPDRS_3-7-L", "MDSUPDRS_3-8-R",
                         "MDSUPDRS_3-8-L"],
    "gait_posture": ["MDSUPDRS_3-9", "MDSUPDRS_3-10", "MDSUPDRS_3-11",
                      "MDSUPDRS_3-12", "MDSUPDRS_3-13", "MDSUPDRS_3-14"],
    "tremor": ["MDSUPDRS_3-15-R", "MDSUPDRS_3-15-L", "MDSUPDRS_3-16-R",
               "MDSUPDRS_3-16-L", "MDSUPDRS_3-17-RUE", "MDSUPDRS_3-17-LUE",
               "MDSUPDRS_3-17-RLE", "MDSUPDRS_3-17-LLE", "MDSUPDRS_3-17-LipJaw",
               "MDSUPDRS_3-18"],
}


def load_pd_data():
    """Load PD subject data with proper header handling."""
    df = pd.read_csv(PD_CSV, header=1, low_memory=False, on_bad_lines="skip")
    df["cohort"] = "PD"
    print(f"  PD subjects loaded: {len(df)}")
    return df


def load_control_data():
    """Load control subject data with proper header handling."""
    df = pd.read_csv(CTRL_CSV, header=1, low_memory=False, on_bad_lines="skip")
    df["cohort"] = "Control"
    # Controls CSV has 'Age' instead of 'Age (years)'
    if "Age" in df.columns and "Age (years)" not in df.columns:
        df.rename(columns={"Age": "Age (years)"}, inplace=True)
    print(f"  Control subjects loaded: {len(df)}")
    return df


def clean_numeric(series):
    """Convert a column to numeric, replacing '-' and non-numeric with NaN."""
    return pd.to_numeric(series.replace("-", np.nan), errors="coerce")


def compute_asymmetry(row, r_col, l_col):
    """
    Compute asymmetry index: (R - L) / max(R + L, 1).
    Returns value in [-1, 1]. Positive = right-dominant.
    """
    r = row.get(r_col, np.nan)
    l_val = row.get(l_col, np.nan)
    if pd.isna(r) or pd.isna(l_val):
        return np.nan
    denom = r + l_val
    if denom == 0:
        return 0.0
    return (r - l_val) / denom


def classify_motor_subtype(row):
    """
    Classify PD motor subtype as tremor-dominant (TD) or akinetic-rigid (AR).
    TD: tremor_score / (bradykinesia + rigidity score) >= 1.0
    AR: ratio < 1.0
    Indeterminate if denominator is 0.
    """
    tremor = row.get("subdomain_tremor", np.nan)
    brady = row.get("subdomain_bradykinesia_UE", 0) + row.get("subdomain_bradykinesia_LE", 0)
    rigidity = row.get("subdomain_rigidity", 0)
    denom = brady + rigidity
    if pd.isna(tremor) or denom == 0:
        return "Indeterminate"
    ratio = tremor / denom
    return "Tremor-Dominant" if ratio >= 1.0 else "Akinetic-Rigid"


def process_subjects(df):
    """Extract all features from a DataFrame of subjects."""
    rows = []

    for _, subj in df.iterrows():
        rec = {}

        # ── Subject ID and cohort ────────────────────────────────────────
        rec["subject_id"] = subj["Subject ID"]
        rec["cohort"] = subj["cohort"]

        # ── Demographics ─────────────────────────────────────────────────
        rec["age"] = clean_numeric(pd.Series([subj.get("Age (years)", np.nan)])).iloc[0]
        rec["sex"] = 1 if str(subj.get("Gender", "")).strip().lower() == "male" else (
            0 if str(subj.get("Gender", "")).strip().lower() == "female" else np.nan
        )
        rec["height_in"] = clean_numeric(pd.Series([subj.get("Height (in)", np.nan)])).iloc[0]
        rec["weight_kg"] = clean_numeric(pd.Series([subj.get("Weight (kg)", np.nan)])).iloc[0]

        # BMI (weight_kg / (height_m^2))
        if pd.notna(rec["height_in"]) and pd.notna(rec["weight_kg"]) and rec["height_in"] > 0:
            height_m = rec["height_in"] * 0.0254
            rec["bmi"] = rec["weight_kg"] / (height_m ** 2)
        else:
            rec["bmi"] = np.nan

        # Disease duration (PD only)
        yrs = subj.get("Years since PD diagnosis", np.nan)
        rec["disease_duration_years"] = clean_numeric(pd.Series([yrs])).iloc[0]

        # ── DBS status (target label) ────────────────────────────────────
        dbs_raw = str(subj.get("DBS?", "-")).strip()
        if dbs_raw.lower() == "yes":
            rec["dbs_candidate"] = 1
        elif dbs_raw.lower() == "no":
            rec["dbs_candidate"] = 0
        elif subj["cohort"] == "Control":
            rec["dbs_candidate"] = 0  # Controls are definitionally not DBS candidates
        else:
            rec["dbs_candidate"] = np.nan  # Missing/ambiguous

        # DBS details (for PD subjects with DBS)
        rec["dbs_bilateral"] = 1 if "bilateral" in str(subj.get("Bilateral vs uilateral", "")).lower() else 0
        rec["dbs_electrode"] = str(subj.get("Electrode location(s)", "-")).strip()
        rec["dbs_years_since_surgery"] = clean_numeric(
            pd.Series([subj.get("Years since surgery", np.nan)])
        ).iloc[0]

        # Medication state at assessment
        med_state = str(subj.get("3b", "-")).strip().upper()
        rec["med_state_on"] = 1 if "ON" in med_state and "OFF" not in med_state else (
            0 if "OFF" in med_state else np.nan
        )

        # ── Hoehn & Yahr ────────────────────────────────────────────────
        hy_raw = subj.get("Modified Hoehn & Yahr Score", np.nan)
        rec["hoehn_yahr"] = clean_numeric(pd.Series([hy_raw])).iloc[0]

        # ── All 65 MDS-UPDRS individual items ────────────────────────────
        for item in ALL_UPDRS_ITEMS:
            val = subj.get(item, np.nan)
            rec[item] = clean_numeric(pd.Series([val])).iloc[0]

        # ── Derived: Part totals ─────────────────────────────────────────
        p1_vals = [rec.get(it, np.nan) for it in PART1_ITEMS]
        p2_vals = [rec.get(it, np.nan) for it in PART2_ITEMS]
        p3_vals = [rec.get(it, np.nan) for it in PART3_ITEMS]
        p4_vals = [rec.get(it, np.nan) for it in PART4_ITEMS]

        rec["updrs_part1_total"] = np.nansum(p1_vals) if any(pd.notna(v) for v in p1_vals) else np.nan
        rec["updrs_part2_total"] = np.nansum(p2_vals) if any(pd.notna(v) for v in p2_vals) else np.nan
        rec["updrs_part3_total"] = np.nansum(p3_vals) if any(pd.notna(v) for v in p3_vals) else np.nan
        rec["updrs_part4_total"] = np.nansum(p4_vals) if any(pd.notna(v) for v in p4_vals) else np.nan
        rec["updrs_total"] = np.nansum(
            [rec["updrs_part1_total"], rec["updrs_part2_total"],
             rec["updrs_part3_total"], rec["updrs_part4_total"]]
        )

        # ── Derived: Motor sub-domain scores ─────────────────────────────
        for domain, items in MOTOR_SUBDOMAINS.items():
            vals = [rec.get(it, np.nan) for it in items]
            rec[f"subdomain_{domain}"] = np.nansum(vals) if any(pd.notna(v) for v in vals) else np.nan

        # ── Derived: Bilateral asymmetry indices ─────────────────────────
        for name, (r_col, l_col) in BILATERAL_PAIRS.items():
            r_val = rec.get(r_col, np.nan)
            l_val = rec.get(l_col, np.nan)
            if pd.notna(r_val) and pd.notna(l_val):
                denom = r_val + l_val
                rec[f"asymmetry_{name}"] = (r_val - l_val) / denom if denom != 0 else 0.0
            else:
                rec[f"asymmetry_{name}"] = np.nan

        # Total asymmetry (mean of absolute asymmetry indices)
        asym_vals = [abs(rec[f"asymmetry_{name}"]) for name in BILATERAL_PAIRS
                     if pd.notna(rec.get(f"asymmetry_{name}"))]
        rec["total_asymmetry"] = np.mean(asym_vals) if asym_vals else np.nan

        rows.append(rec)

    result = pd.DataFrame(rows)

    # ── Motor subtype classification ─────────────────────────────────────
    result["motor_subtype"] = result.apply(classify_motor_subtype, axis=1)

    return result


def main():
    print("=" * 70)
    print("WearGait-PD Clinical Feature Extraction")
    print("=" * 70)

    # ── Load data ────────────────────────────────────────────────────────
    print("\n[1/4] Loading raw data...")
    df_pd = load_pd_data()
    df_ctrl = load_control_data()

    # ── Process subjects ─────────────────────────────────────────────────
    print("\n[2/4] Extracting features...")
    feat_pd = process_subjects(df_pd)
    feat_ctrl = process_subjects(df_ctrl)

    # Combine
    feat_all = pd.concat([feat_pd, feat_ctrl], ignore_index=True)

    # ── Summary statistics ───────────────────────────────────────────────
    print("\n[3/4] Summary:")
    print(f"  Total subjects: {len(feat_all)}")
    print(f"    PD: {len(feat_pd)}, Controls: {len(feat_ctrl)}")
    print(f"  DBS status distribution:")
    print(f"    DBS=Yes: {(feat_all['dbs_candidate'] == 1).sum()}")
    print(f"    DBS=No:  {(feat_all['dbs_candidate'] == 0).sum()}")
    print(f"    Missing: {feat_all['dbs_candidate'].isna().sum()}")

    # Count feature columns
    meta_cols = ["subject_id", "cohort", "dbs_candidate", "dbs_bilateral",
                 "dbs_electrode", "dbs_years_since_surgery", "motor_subtype",
                 "med_state_on"]
    feat_cols = [c for c in feat_all.columns if c not in meta_cols]
    print(f"  Feature columns: {len(feat_cols)}")
    print(f"  UPDRS items: {len([c for c in feat_cols if 'MDSUPDRS' in c])}")
    print(f"  Demographics: age, sex, height_in, weight_kg, bmi, disease_duration_years")
    print(f"  Derived: 5 part totals, 7 subdomain scores, 11 asymmetry indices, total_asymmetry")

    # H&Y distribution
    print(f"\n  Hoehn & Yahr distribution (PD only):")
    hy_dist = feat_pd["hoehn_yahr"].value_counts(dropna=False).sort_index()
    for val, cnt in hy_dist.items():
        print(f"    H&Y={val}: {cnt}")

    # Motor subtype distribution
    print(f"\n  Motor subtype distribution (PD only):")
    for st, cnt in feat_pd["motor_subtype"].value_counts().items():
        print(f"    {st}: {cnt}")

    # ── Remove subjects with missing DBS label ───────────────────────────
    n_before = len(feat_all)
    feat_all_clean = feat_all.dropna(subset=["dbs_candidate"]).copy()
    feat_all_clean["dbs_candidate"] = feat_all_clean["dbs_candidate"].astype(int)
    n_dropped = n_before - len(feat_all_clean)
    print(f"\n  Dropped {n_dropped} subjects with missing DBS label")
    print(f"  Final cohort: {len(feat_all_clean)} subjects "
          f"({(feat_all_clean['dbs_candidate'] == 1).sum()} DBS+, "
          f"{(feat_all_clean['dbs_candidate'] == 0).sum()} DBS-)")

    # PD-only subset (for focused DBS analysis without trivial control separation)
    feat_pd_clean = feat_all_clean[feat_all_clean["cohort"] == "PD"].copy()
    print(f"  PD-only cohort: {len(feat_pd_clean)} subjects "
          f"({(feat_pd_clean['dbs_candidate'] == 1).sum()} DBS+, "
          f"{(feat_pd_clean['dbs_candidate'] == 0).sum()} DBS-)")

    # ── Save ─────────────────────────────────────────────────────────────
    print("\n[4/4] Saving...")

    # Full cohort (PD + Controls)
    out_all = os.path.join(OUT_DIR, "weargait_clinical.csv")
    feat_all_clean.to_csv(out_all, index=False)
    print(f"  Saved: {out_all} ({feat_all_clean.shape})")

    # PD-only cohort
    out_pd = os.path.join(OUT_DIR, "weargait_pd_only.csv")
    feat_pd_clean.to_csv(out_pd, index=False)
    print(f"  Saved: {out_pd} ({feat_pd_clean.shape})")

    # Column inventory for documentation
    print("\n" + "=" * 70)
    print("COLUMN INVENTORY (for manuscript Methods section)")
    print("=" * 70)
    print(f"\nTotal columns: {len(feat_all_clean.columns)}")
    print(f"\nMetadata: {meta_cols}")
    print(f"\nDemographics (6): age, sex, height_in, weight_kg, bmi, disease_duration_years")
    print(f"H&Y (1): hoehn_yahr")
    print(f"UPDRS Part I items (13): {PART1_ITEMS}")
    print(f"UPDRS Part II items (13): {PART2_ITEMS}")
    print(f"UPDRS Part III items (33): {PART3_ITEMS}")
    print(f"UPDRS Part IV items (6): {PART4_ITEMS}")
    print(f"Part totals (5): updrs_part1_total..updrs_part4_total, updrs_total")
    print(f"Sub-domain scores (7): {list(MOTOR_SUBDOMAINS.keys())}")
    print(f"Asymmetry indices (12): {list(BILATERAL_PAIRS.keys())} + total_asymmetry")
    print(f"\nRaw feature count (excl. metadata): {len(feat_cols)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
