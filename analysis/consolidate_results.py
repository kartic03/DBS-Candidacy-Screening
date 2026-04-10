#!/usr/bin/env python3
"""
Consolidate All Results Tables — Dated Snapshot
================================================
Creates a clean, dated snapshot of ALL result tables in:
  results/tables/2026-03-20/

This script:
  1. Copies all existing result CSVs
  2. Creates a master summary table (Table 1: All models, all metrics)
  3. Creates a manuscript-ready table set with clean formatting
  4. Generates a table index (manifest) for manuscript reference

Author: Kartic Mishra, Gachon University
"""

import os
import warnings
import multiprocessing
import shutil
from datetime import date

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

N_CORES = multiprocessing.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(N_CORES)
SEED = 42

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES_DIR = os.path.join(PROJECT_ROOT, "results", "tables")
TODAY = date.today().isoformat()  # 2026-03-20
DATED_DIR = os.path.join(TABLES_DIR, TODAY)
os.makedirs(DATED_DIR, exist_ok=True)

FIGURES_DIR = os.path.join(PROJECT_ROOT, "results", "figures")
DATED_FIG_DIR = os.path.join(FIGURES_DIR, TODAY)
os.makedirs(DATED_FIG_DIR, exist_ok=True)


def copy_all_existing():
    """Copy all CSV files from tables/ root to dated subfolder."""
    count = 0
    for f in os.listdir(TABLES_DIR):
        if f.endswith(".csv"):
            src = os.path.join(TABLES_DIR, f)
            dst = os.path.join(DATED_DIR, f)
            shutil.copy2(src, dst)
            count += 1
    print(f"  Copied {count} existing CSV files to {TODAY}/")
    return count


def create_master_summary():
    """
    Table 1 (Manuscript): Master Model Comparison
    Merges clinical, modality, and fusion results into one table.
    """
    print("\n[2/5] Creating master model comparison table...")

    rows = []

    # ── Clinical models ──────────────────────────────────────────────────
    clin_path = os.path.join(TABLES_DIR, "clinical_model_results.csv")
    if os.path.exists(clin_path):
        df = pd.read_csv(clin_path)
        for _, r in df.iterrows():
            rows.append({
                "Model": r.get("Model", ""),
                "Dataset": r.get("Cohort", "WearGait-PD"),
                "Modality": "Clinical",
                "N": 82,
                "Evaluation": r.get("Evaluation", ""),
                "N_features": r.get("N_features", ""),
                "AUC_ROC": r.get("AUC_ROC", np.nan),
                "AUC_CI": f"({r.get('AUC_CI_low', ''):.3f}–{r.get('AUC_CI_high', ''):.3f})" if pd.notna(r.get("AUC_CI_low")) else "",
                "AUC_PR": r.get("AUC_PR", np.nan),
                "Sensitivity": r.get("Sensitivity", np.nan),
                "Specificity": r.get("Specificity", np.nan),
                "F1": r.get("F1", np.nan),
                "MCC": r.get("MCC", np.nan),
                "Brier": r.get("Brier", np.nan),
            })

    # ── Modality models ──────────────────────────────────────────────────
    mod_path = os.path.join(TABLES_DIR, "modality_model_results.csv")
    if os.path.exists(mod_path):
        df = pd.read_csv(mod_path)
        for _, r in df.iterrows():
            modality = "Wearable" if "Wearable" in str(r.get("Dataset", "")) or "PADS" in str(r.get("Dataset", "")) else \
                       "Gait" if "Gait" in str(r.get("Dataset", "")) else \
                       "Voice" if "Voice" in str(r.get("Dataset", "")) else str(r.get("Dataset", ""))
            rows.append({
                "Model": r.get("Model", ""),
                "Dataset": r.get("Dataset", ""),
                "Modality": modality,
                "N": r.get("N", ""),
                "Evaluation": r.get("Evaluation", "5-fold CV"),
                "N_features": r.get("N_features", ""),
                "AUC_ROC": r.get("AUC_ROC", np.nan),
                "AUC_CI": f"({r.get('AUC_CI_low', ''):.3f}–{r.get('AUC_CI_high', ''):.3f})" if pd.notna(r.get("AUC_CI_low")) else "",
                "AUC_PR": r.get("AUC_PR", np.nan),
                "Sensitivity": r.get("Sensitivity", np.nan),
                "Specificity": r.get("Specificity", np.nan),
                "F1": r.get("F1", np.nan),
                "MCC": r.get("MCC", np.nan),
                "Brier": r.get("Brier", np.nan),
            })

    # ── Fusion models ────────────────────────────────────────────────────
    fus_path = os.path.join(TABLES_DIR, "fusion_model_results.csv")
    if os.path.exists(fus_path):
        df = pd.read_csv(fus_path)
        for _, r in df.iterrows():
            rows.append({
                "Model": r.get("Model", ""),
                "Dataset": r.get("Dataset", "GaitPDB"),
                "Modality": "Fusion",
                "N": r.get("N", ""),
                "Evaluation": "5-fold CV",
                "N_features": f"{r.get('Gait_features', '')}+{r.get('Clinical_features', '')}",
                "AUC_ROC": r.get("AUC_ROC", np.nan),
                "AUC_CI": f"({r.get('AUC_CI_low', ''):.3f}–{r.get('AUC_CI_high', ''):.3f})" if pd.notna(r.get("AUC_CI_low")) else "",
                "AUC_PR": r.get("AUC_PR", np.nan),
                "Sensitivity": r.get("Sensitivity", np.nan),
                "Specificity": r.get("Specificity", np.nan),
                "F1": r.get("F1", np.nan),
                "MCC": r.get("MCC", np.nan),
                "Brier": r.get("Brier", np.nan),
            })

    if rows:
        master = pd.DataFrame(rows)
        # Sort: by modality group, then AUC descending
        order = {"Clinical": 0, "Wearable": 1, "Gait": 2, "Voice": 3, "Fusion": 4}
        master["_sort"] = master["Modality"].map(order).fillna(5)
        master = master.sort_values(["_sort", "AUC_ROC"], ascending=[True, False]).drop(columns=["_sort"])

        out = os.path.join(DATED_DIR, "TABLE1_master_model_comparison.csv")
        master.to_csv(out, index=False, float_format="%.3f")
        print(f"  Saved: TABLE1_master_model_comparison.csv ({len(master)} models)")
        return master
    return None


def create_feature_importance_summary():
    """
    Table 2 (Manuscript): Top features per modality from SHAP analysis.
    """
    print("\n[3/5] Creating feature importance summary...")

    all_feats = []
    for fname, modality in [
        ("shap_clinical_importance.csv", "Clinical"),
        ("shap_wearable_importance.csv", "Wearable"),
        ("shap_gait_importance.csv", "Gait"),
    ]:
        path = os.path.join(TABLES_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["modality"] = modality
            # Keep top 10 per modality
            df = df.head(10)
            all_feats.append(df)

    if all_feats:
        combined = pd.concat(all_feats, ignore_index=True)
        out = os.path.join(DATED_DIR, "TABLE2_top_features_per_modality.csv")
        combined.to_csv(out, index=False, float_format="%.4f")
        print(f"  Saved: TABLE2_top_features_per_modality.csv ({len(combined)} features)")
    else:
        print("  No SHAP importance files found — skipping.")


def create_statistical_summary():
    """
    Table 3 (Manuscript): Statistical tests summary.
    """
    print("\n[4/5] Creating statistical summary table...")

    stats_rows = []

    # Mann-Whitney significant features
    mw_path = os.path.join(TABLES_DIR, "mann_whitney_v2.csv")
    if os.path.exists(mw_path):
        mw = pd.read_csv(mw_path)
        n_sig = mw["significant_fdr05"].sum() if "significant_fdr05" in mw.columns else 0
        stats_rows.append({
            "Test": "Mann-Whitney U + BH-FDR",
            "Comparison": "DBS+ vs DBS- (clinical features)",
            "N_tested": len(mw),
            "N_significant": int(n_sig),
            "Key_result": f"{n_sig}/{len(mw)} features significant (FDR < 0.05)",
        })

    # DeLong tests
    dl_path = os.path.join(TABLES_DIR, "delong_within_dataset.csv")
    if os.path.exists(dl_path):
        dl = pd.read_csv(dl_path)
        for _, r in dl.iterrows():
            stats_rows.append({
                "Test": "DeLong test",
                "Comparison": f"{r.get('Model1', '')} vs {r.get('Model2', '')} ({r.get('Dataset', '')})",
                "N_tested": 1,
                "N_significant": 1 if r.get("Significant_0.05", False) else 0,
                "Key_result": f"Z={r.get('Z', ''):.3f}, p={r.get('P_value', ''):.4f}" if pd.notna(r.get("Z")) else str(r.get("Note", "")),
            })

    # Motor subtype
    ms_path = os.path.join(TABLES_DIR, "motor_subtype_v2.csv")
    if os.path.exists(ms_path):
        ms = pd.read_csv(ms_path)
        for _, r in ms.iterrows():
            stats_rows.append({
                "Test": "Motor subtype analysis",
                "Comparison": f"{r.get('subtype', '')} (n={r.get('n', '')})",
                "N_tested": r.get("n", ""),
                "N_significant": "",
                "Key_result": f"DBS+ rate={r.get('dbs_rate', 0):.1%}, UPDRS-III={r.get('mean_updrs_part3', 0):.1f}",
            })

    # Power analysis
    pw_path = os.path.join(TABLES_DIR, "power_analysis.csv")
    if os.path.exists(pw_path):
        pw = pd.read_csv(pw_path)
        for _, r in pw.iterrows():
            stats_rows.append({
                "Test": "Post-hoc power analysis",
                "Comparison": r.get("comparison", ""),
                "N_tested": f"n_pos={r.get('current_n_pos', '')}, n_neg={r.get('current_n_neg', '')}",
                "N_significant": "",
                "Key_result": f"Power={r.get('current_power', 0):.3f} — {r.get('note', '')}",
            })

    # NRI
    nri_path = os.path.join(TABLES_DIR, "nri_results_v2.csv")
    if os.path.exists(nri_path):
        nri = pd.read_csv(nri_path)
        for _, r in nri.iterrows():
            stats_rows.append({
                "Test": "Net Reclassification Improvement",
                "Comparison": r.get("Comparison", ""),
                "N_tested": "",
                "N_significant": "",
                "Key_result": f"NRI={r.get('NRI', ''):.3f} (events={r.get('NRI_events', ''):.3f}, non-events={r.get('NRI_nonevents', ''):.3f})" if pd.notna(r.get("NRI")) else "",
            })

    # Calibration
    cal_path = os.path.join(TABLES_DIR, "calibration_results_v2.csv")
    if os.path.exists(cal_path):
        cal = pd.read_csv(cal_path)
        for _, r in cal.iterrows():
            stats_rows.append({
                "Test": "Calibration (Platt scaling)",
                "Comparison": r.get("Model", ""),
                "N_tested": "",
                "N_significant": "",
                "Key_result": f"Brier: {r.get('Brier_before', ''):.3f}→{r.get('Brier_after_Platt', ''):.3f}, ECE: {r.get('ECE_before', ''):.3f}→{r.get('ECE_after_Platt', ''):.3f}" if pd.notna(r.get("Brier_before")) else "",
            })

    if stats_rows:
        stats_df = pd.DataFrame(stats_rows)
        out = os.path.join(DATED_DIR, "TABLE3_statistical_tests_summary.csv")
        stats_df.to_csv(out, index=False)
        print(f"  Saved: TABLE3_statistical_tests_summary.csv ({len(stats_df)} rows)")


def create_table_index():
    """Create a manifest/index of all tables in the dated directory."""
    print("\n[5/5] Creating table index...")

    files = sorted([f for f in os.listdir(DATED_DIR) if f.endswith(".csv")])
    manifest = []
    for f in files:
        path = os.path.join(DATED_DIR, f)
        df = pd.read_csv(path)
        manifest.append({
            "filename": f,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": ", ".join(df.columns[:8]) + ("..." if len(df.columns) > 8 else ""),
            "size_kb": round(os.path.getsize(path) / 1024, 1),
        })

    manifest_df = pd.DataFrame(manifest)
    out = os.path.join(DATED_DIR, "_TABLE_INDEX.csv")
    manifest_df.to_csv(out, index=False)
    print(f"  Saved: _TABLE_INDEX.csv ({len(manifest_df)} tables)")

    # Also print summary
    print(f"\n  ╔{'═'*60}╗")
    print(f"  ║ {'RESULTS SNAPSHOT: ' + TODAY:^58} ║")
    print(f"  ╠{'═'*60}╣")
    print(f"  ║ {'Total tables:':<40} {len(manifest_df):>16} ║")
    total_kb = manifest_df["size_kb"].sum()
    print(f"  ║ {'Total size:':<40} {total_kb:>13.1f} KB ║")

    # Highlight manuscript tables
    manuscript = [f for f in files if f.startswith("TABLE")]
    print(f"  ║ {'Manuscript tables (TABLE*):':<40} {len(manuscript):>16} ║")
    print(f"  ╚{'═'*60}╝")

    if manuscript:
        print(f"\n  Manuscript-ready tables:")
        for f in manuscript:
            row = manifest_df[manifest_df["filename"] == f].iloc[0]
            print(f"    {f:<50} {row['rows']:>4} rows, {row['columns']:>3} cols")

    return manifest_df


def main():
    print("=" * 70)
    print(f"CONSOLIDATE ALL RESULTS — {TODAY}")
    print("=" * 70)

    # Step 1: Copy existing CSVs
    print("\n[1/5] Copying existing CSV files...")
    copy_all_existing()

    # Step 2: Create master model comparison
    create_master_summary()

    # Step 3: Feature importance summary
    create_feature_importance_summary()

    # Step 4: Statistical tests summary
    create_statistical_summary()

    # Step 5: Table index
    create_table_index()

    print("\n" + "=" * 70)
    print("CONSOLIDATION COMPLETE")
    print("=" * 70)
    print(f"\n  Tables: {DATED_DIR}/")
    print(f"  Figures: {DATED_FIG_DIR}/")


if __name__ == "__main__":
    main()
