#!/usr/bin/env python3
"""
SHAP Analysis (Revised) — Per-Modality Explainability
======================================================
TreeSHAP on XGBoost models for each genuine modality:
  1. Clinical model (WearGait-PD) — real DBS labels
  2. Wearable model (PADS) — PD vs Healthy
  3. Gait model (GaitPDB) — PD vs Control

Output:
  results/tables/shap_clinical_importance.csv
  results/tables/shap_wearable_importance.csv
  results/tables/shap_gait_importance.csv
  results/figures/shap_clinical_beeswarm.png
  results/figures/shap_wearable_beeswarm.png
  results/figures/shap_gait_beeswarm.png

Author: Kartic Mishra, Gachon University
"""

import os
import warnings
import multiprocessing
import pickle

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb

warnings.filterwarnings("ignore")

N_CORES = multiprocessing.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(N_CORES)
SEED = 42
np.random.seed(SEED)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "tables")
FIG_DIR = os.path.join(PROJECT_ROOT, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Figure quality settings
try:
    plt.rcParams['font.family'] = 'Arial'
except Exception:
    plt.rcParams['font.family'] = 'Liberation Sans'
plt.rcParams['font.size'] = 10

META_COLS = [
    "subject_id", "cohort", "dbs_candidate", "dbs_bilateral",
    "dbs_electrode", "dbs_years_since_surgery", "motor_subtype", "med_state_on"
]


def train_xgb_and_explain(X, y, feature_names, dataset_name, label_name,
                           top_k=20, color="#1f77b4"):
    """Train XGBoost, compute TreeSHAP, save beeswarm + importance table."""
    print(f"\n  Training XGBoost for SHAP ({dataset_name})...")

    # Scale + impute on full data (for SHAP visualization)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    imp = SimpleImputer(strategy="median")
    X_clean = imp.fit_transform(X_scaled)

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=len(y[y == 0]) / max(len(y[y == 1]), 1),
        tree_method="hist", device="cuda", random_state=SEED,
        eval_metric="logloss", verbosity=0
    )
    model.fit(X_clean, y)

    # TreeSHAP
    print(f"  Computing TreeSHAP...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_clean)

    # Mean absolute SHAP per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
        "rank": range(1, len(feature_names) + 1),
    }).sort_values("mean_abs_shap", ascending=False)
    importance_df["rank"] = range(1, len(importance_df) + 1)

    # Save importance table
    out_table = os.path.join(RESULTS_DIR, f"shap_{dataset_name}_importance.csv")
    importance_df.to_csv(out_table, index=False)
    print(f"  Saved: {out_table}")

    # Print top features
    print(f"  Top-{min(top_k, 10)} features:")
    for _, row in importance_df.head(10).iterrows():
        print(f"    {int(row['rank']):2d}. {row['feature']}: {row['mean_abs_shap']:.4f}")

    # Beeswarm plot
    print(f"  Generating beeswarm plot...")
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use SHAP's built-in beeswarm with custom settings
    shap_df = pd.DataFrame(X_clean, columns=feature_names)
    explanation = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=X_clean,
        feature_names=feature_names
    )

    plt.sca(ax)
    shap.plots.beeswarm(explanation, max_display=top_k, show=False)
    ax.set_title(f"SHAP Feature Importance — {dataset_name}\n({label_name})",
                 fontsize=12, fontweight="bold", pad=15)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_fig = os.path.join(FIG_DIR, f"shap_{dataset_name}_beeswarm.png")
    fig.savefig(out_fig, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_fig}")

    return importance_df, shap_values


def main():
    print("=" * 70)
    print("SHAP Analysis — Per-Modality Explainability")
    print("=" * 70)

    all_importances = {}

    # ═══════════════════════════════════════════════════════════════════════
    # 1. Clinical Model (WearGait-PD, real DBS labels)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[1/3] Clinical Model (WearGait-PD)...")
    df_pd = pd.read_csv(os.path.join(PROC_DIR, "clinical_features", "weargait_pd_only.csv"))
    feat_cols = [c for c in df_pd.columns if c not in META_COLS
                 and pd.api.types.is_numeric_dtype(df_pd[c])]
    X = df_pd[feat_cols].values.astype(np.float32)
    y = df_pd["dbs_candidate"].values.astype(int)

    imp_clinical, shap_clinical = train_xgb_and_explain(
        X, y, feat_cols, "clinical", "DBS Candidacy (Real Labels, n=82)",
        top_k=20, color="#d62728"
    )
    all_importances["clinical"] = imp_clinical

    # ═══════════════════════════════════════════════════════════════════════
    # 2. Wearable Model (PADS, PD vs Healthy)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[2/3] Wearable Model (PADS)...")
    df_pads = pd.read_csv(os.path.join(PROC_DIR, "wearable_features", "pads_sensor_pd_vs_healthy.csv"))
    pads_meta = ["subject_id", "condition", "age", "gender", "height", "weight",
                 "pd_label", "n_files_processed", "handedness"]
    pads_feats = [c for c in df_pads.columns if c not in pads_meta
                  and pd.api.types.is_numeric_dtype(df_pads[c])]
    X_pads = df_pads[pads_feats].values.astype(np.float32)
    y_pads = df_pads["pd_label"].values.astype(int)

    imp_wearable, _ = train_xgb_and_explain(
        X_pads, y_pads, pads_feats, "wearable", "PD vs Healthy (n=370)",
        top_k=20, color="#1f77b4"
    )
    all_importances["wearable"] = imp_wearable

    # ═══════════════════════════════════════════════════════════════════════
    # 3. Gait Model (GaitPDB, PD vs Control)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[3/3] Gait Model (GaitPDB)...")
    df_gait = pd.read_csv(os.path.join(PROC_DIR, "gait_features", "gaitpdb_sensor_features.csv"))
    gait_meta = ["subject_id", "Study", "Group", "Gender", "pd_label", "sex",
                 "dbs_proxy_hy25", "dbs_proxy_updrs32", "n_single_trials"]
    gait_feats = [c for c in df_gait.columns if c not in gait_meta
                  and pd.api.types.is_numeric_dtype(df_gait[c])]
    X_gait = df_gait[gait_feats].values.astype(np.float32)
    y_gait = df_gait["pd_label"].values.astype(int)

    imp_gait, _ = train_xgb_and_explain(
        X_gait, y_gait, gait_feats, "gait", "PD vs Control (n=165)",
        top_k=20, color="#2ca02c"
    )
    all_importances["gait"] = imp_gait

    # ═══════════════════════════════════════════════════════════════════════
    # Combined modality summary
    # ═══════════════════════════════════════════════════════════════════════
    print("\n  Saving combined modality summary...")
    combined = []
    for modality, imp_df in all_importances.items():
        top10 = imp_df.head(10).copy()
        top10["modality"] = modality
        combined.append(top10)
    df_combined = pd.concat(combined, ignore_index=True)
    df_combined.to_csv(os.path.join(RESULTS_DIR, "shap_combined_top10.csv"), index=False)

    print("\n" + "=" * 70)
    print("SHAP ANALYSIS COMPLETE")
    print("=" * 70)
    for mod, imp_df in all_importances.items():
        top3 = imp_df.head(3)["feature"].tolist()
        print(f"  {mod}: top-3 = {top3}")

    print("\nDone.")


if __name__ == "__main__":
    main()
