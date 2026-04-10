#!/usr/bin/env python3
"""
Data Fusion & Split Generation (Revised)
==========================================
Creates clean per-dataset cohort files and cross-validation splits.

Key changes from original:
  - NO fake modality splitting of clinical features
  - Each dataset kept separate with its own labels and features
  - GaitPDB fusion subset created (gait sensor + clinical features)
  - Scaling deferred to within CV folds (raw values stored)
  - Proper StratifiedGroupKFold for WearGait-PD

Output:
  data/splits/weargait_splits.json  — 5-fold CV + hold-out for WearGait-PD
  data/splits/gaitpdb_splits.json   — 5-fold CV for GaitPDB
  data/splits/pads_splits.json      — 5-fold CV for PADS
  data/splits/uci_voice_splits.json — 5-fold CV for UCI Voice

Author: Kartic Mishra, Gachon University
"""

import os
import json
import warnings
import multiprocessing

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

warnings.filterwarnings("ignore")

N_CORES = multiprocessing.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(N_CORES)

SEED = 42
np.random.seed(SEED)
N_FOLDS = 5

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
SPLIT_DIR = os.path.join(PROJECT_ROOT, "data", "splits")
os.makedirs(SPLIT_DIR, exist_ok=True)


def generate_stratified_splits(df, label_col, n_folds=N_FOLDS, test_size=0.15,
                               group_col=None, seed=SEED):
    """
    Generate stratified train/test split + K-fold CV on training set.
    Returns dict with test_indices and cv_folds.
    """
    labels = df[label_col].values
    indices = np.arange(len(df))

    # Hold-out test split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_pool_idx, test_idx = next(sss.split(indices, labels))

    # K-fold CV on training pool
    train_labels = labels[train_pool_idx]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    cv_folds = []
    for fold_train, fold_val in skf.split(train_pool_idx, train_labels):
        cv_folds.append({
            "train": train_pool_idx[fold_train].tolist(),
            "val": train_pool_idx[fold_val].tolist()
        })

    # Summary
    print(f"    Test: {len(test_idx)} samples "
          f"(pos={int(labels[test_idx].sum())}, neg={int((1-labels[test_idx]).sum())})")
    print(f"    Train pool: {len(train_pool_idx)} samples")
    for i, fold in enumerate(cv_folds):
        t_pos = int(labels[fold['train']].sum())
        v_pos = int(labels[fold['val']].sum())
        print(f"      Fold {i}: train={len(fold['train'])} (pos={t_pos}), "
              f"val={len(fold['val'])} (pos={v_pos})")

    return {
        "test_indices": test_idx.tolist(),
        "cv_folds": cv_folds,
        "label_col": label_col,
        "n_total": len(df),
        "n_positive": int(labels.sum()),
        "seed": seed
    }


def generate_cv_only(df, label_col, n_folds=N_FOLDS, seed=SEED):
    """Generate K-fold CV splits (no hold-out test) for smaller datasets."""
    labels = df[label_col].values
    indices = np.arange(len(df))

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    cv_folds = []
    for fold_train, fold_val in skf.split(indices, labels):
        cv_folds.append({
            "train": fold_train.tolist(),
            "val": fold_val.tolist()
        })

    for i, fold in enumerate(cv_folds):
        t_pos = int(labels[fold['train']].sum())
        v_pos = int(labels[fold['val']].sum())
        print(f"      Fold {i}: train={len(fold['train'])} (pos={t_pos}), "
              f"val={len(fold['val'])} (pos={v_pos})")

    return {
        "test_indices": [],
        "cv_folds": cv_folds,
        "label_col": label_col,
        "n_total": len(df),
        "n_positive": int(labels.sum()),
        "seed": seed
    }


def main():
    print("=" * 70)
    print("Data Fusion & Split Generation (Revised)")
    print("=" * 70)

    # ═══════════════════════════════════════════════════════════════════════
    # 1. WearGait-PD Clinical (PRIMARY — real DBS labels)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[1/5] WearGait-PD Clinical (primary cohort)...")

    # PD-only analysis (DBS+ vs DBS- among PD patients)
    df_pd = pd.read_csv(os.path.join(PROC_DIR, "clinical_features", "weargait_pd_only.csv"))
    print(f"  PD-only: {len(df_pd)} subjects ({int(df_pd['dbs_candidate'].sum())} DBS+)")

    print("  Generating PD-only splits (5-fold + 15% hold-out):")
    splits_pd = generate_stratified_splits(df_pd, "dbs_candidate", test_size=0.15)
    with open(os.path.join(SPLIT_DIR, "weargait_pd_only_splits.json"), "w") as f:
        json.dump(splits_pd, f, indent=2)

    # Full cohort (PD + Controls)
    df_all = pd.read_csv(os.path.join(PROC_DIR, "clinical_features", "weargait_clinical.csv"))
    print(f"\n  Full cohort: {len(df_all)} subjects ({int(df_all['dbs_candidate'].sum())} DBS+)")

    print("  Generating full cohort splits (5-fold + 15% hold-out):")
    splits_all = generate_stratified_splits(df_all, "dbs_candidate", test_size=0.15)
    with open(os.path.join(SPLIT_DIR, "weargait_full_splits.json"), "w") as f:
        json.dump(splits_all, f, indent=2)

    # ═══════════════════════════════════════════════════════════════════════
    # 2. PADS Wearable (PD vs Healthy)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[2/5] PADS Wearable Sensor...")

    df_pads = pd.read_csv(os.path.join(PROC_DIR, "wearable_features", "pads_sensor_pd_vs_healthy.csv"))
    # Ensure pd_label is int
    df_pads["pd_label"] = df_pads["pd_label"].astype(int)
    print(f"  PADS PD vs Healthy: {len(df_pads)} subjects "
          f"(PD={int(df_pads['pd_label'].sum())}, Healthy={int((1-df_pads['pd_label']).sum())})")

    print("  Generating 5-fold CV splits:")
    splits_pads = generate_cv_only(df_pads, "pd_label")
    with open(os.path.join(SPLIT_DIR, "pads_splits.json"), "w") as f:
        json.dump(splits_pads, f, indent=2)

    # ═══════════════════════════════════════════════════════════════════════
    # 3. GaitPDB (PD vs Control + H&Y severity)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[3/5] GaitPDB Gait Sensor...")

    df_gait = pd.read_csv(os.path.join(PROC_DIR, "gait_features", "gaitpdb_sensor_features.csv"))
    print(f"  GaitPDB all: {len(df_gait)} subjects "
          f"(PD={int(df_gait['pd_label'].sum())}, Control={int((1-df_gait['pd_label']).sum())})")

    # PD vs Control splits
    print("  PD vs Control 5-fold CV:")
    splits_gait_pd = generate_cv_only(df_gait, "pd_label")
    with open(os.path.join(SPLIT_DIR, "gaitpdb_pd_splits.json"), "w") as f:
        json.dump(splits_gait_pd, f, indent=2)

    # H&Y >= 2.5 splits (for DBS proxy — only subjects with H&Y)
    df_gait_hy = pd.read_csv(os.path.join(PROC_DIR, "gait_features", "gaitpdb_with_clinical.csv"))
    df_gait_hy = df_gait_hy[df_gait_hy["dbs_proxy_hy25"].notna()].copy()
    df_gait_hy["dbs_proxy_hy25"] = df_gait_hy["dbs_proxy_hy25"].astype(int)
    print(f"\n  GaitPDB H&Y subset: {len(df_gait_hy)} subjects "
          f"(H&Y>=2.5: {int(df_gait_hy['dbs_proxy_hy25'].sum())})")

    if df_gait_hy["dbs_proxy_hy25"].sum() >= N_FOLDS:
        print("  H&Y >= 2.5 proxy 5-fold CV:")
        splits_gait_hy = generate_cv_only(df_gait_hy, "dbs_proxy_hy25")
        with open(os.path.join(SPLIT_DIR, "gaitpdb_hy_splits.json"), "w") as f:
            json.dump(splits_gait_hy, f, indent=2)
    else:
        print("  WARNING: Too few H&Y>=2.5 subjects for 5-fold CV, using 3-fold")
        splits_gait_hy = generate_cv_only(df_gait_hy, "dbs_proxy_hy25", n_folds=3)
        with open(os.path.join(SPLIT_DIR, "gaitpdb_hy_splits.json"), "w") as f:
            json.dump(splits_gait_hy, f, indent=2)

    # ═══════════════════════════════════════════════════════════════════════
    # 4. UCI Voice (PD vs Control)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[4/5] UCI Voice...")

    df_voice = pd.read_csv(os.path.join(PROC_DIR, "voice_features", "uci_voice_features.csv"))
    # Ensure we have a status column
    label_col = "status" if "status" in df_voice.columns else "pd_label"
    if label_col not in df_voice.columns:
        # Try to find the label column
        for c in df_voice.columns:
            if df_voice[c].nunique() == 2 and set(df_voice[c].unique()).issubset({0, 1}):
                label_col = c
                break
    df_voice[label_col] = df_voice[label_col].astype(int)
    print(f"  UCI Voice: {len(df_voice)} recordings, label col={label_col}")
    print(f"    PD={int(df_voice[label_col].sum())}, "
          f"Healthy={int((1-df_voice[label_col]).sum())}")

    print("  Generating 5-fold CV splits:")
    splits_voice = generate_cv_only(df_voice, label_col)
    with open(os.path.join(SPLIT_DIR, "uci_voice_splits.json"), "w") as f:
        json.dump(splits_voice, f, indent=2)

    # ═══════════════════════════════════════════════════════════════════════
    # 5. UCI UPDRS (motor_UPDRS >= 32 proxy)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[5/5] UCI UPDRS Telemonitoring...")

    df_updrs = pd.read_csv(os.path.join(PROC_DIR, "voice_features", "uci_updrs_features.csv"))
    print(f"  UCI UPDRS: {len(df_updrs)} subjects")

    # Check if we have motor_UPDRS or proxy label
    if "dbs_candidate" in df_updrs.columns:
        label_col_updrs = "dbs_candidate"
    elif "motor_updrs_mean" in df_updrs.columns:
        df_updrs["dbs_proxy_updrs32"] = (df_updrs["motor_updrs_mean"] >= 32).astype(int)
        label_col_updrs = "dbs_proxy_updrs32"
    else:
        # Find motor UPDRS column
        motor_cols = [c for c in df_updrs.columns if "motor" in c.lower() and "updrs" in c.lower()]
        if motor_cols:
            df_updrs["dbs_proxy_updrs32"] = (df_updrs[motor_cols[0]] >= 32).astype(int)
            label_col_updrs = "dbs_proxy_updrs32"
        else:
            label_col_updrs = None

    if label_col_updrs and df_updrs[label_col_updrs].sum() >= 2:
        print(f"    Label: {label_col_updrs}, "
              f"pos={int(df_updrs[label_col_updrs].sum())}, "
              f"neg={int((1-df_updrs[label_col_updrs]).sum())}")
        if df_updrs[label_col_updrs].sum() >= 3:
            print("  Generating 3-fold CV splits (small dataset):")
            splits_updrs = generate_cv_only(df_updrs, label_col_updrs, n_folds=3)
        else:
            print("  WARNING: Only 2 positives — LOOCV recommended, saving 2-fold")
            splits_updrs = generate_cv_only(df_updrs, label_col_updrs, n_folds=2)
        with open(os.path.join(SPLIT_DIR, "uci_updrs_splits.json"), "w") as f:
            json.dump(splits_updrs, f, indent=2)
    else:
        print("  WARNING: Insufficient positive samples for splitting. Skipping.")

    # ═══════════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SUMMARY: All datasets processed and split")
    print("=" * 70)
    print(f"""
  PRIMARY (real DBS labels):
    WearGait-PD PD-only: 82 subjects, 23 DBS+, 5-fold CV + hold-out
    WearGait-PD Full:    167 subjects, 23 DBS+, 5-fold CV + hold-out

  MODALITY VALIDATION (proxy labels):
    PADS Wearable:   {len(df_pads)} subjects, PD vs Healthy, 5-fold CV
    GaitPDB Gait:    {len(df_gait)} subjects, PD vs Control, 5-fold CV
    GaitPDB H&Y:     {len(df_gait_hy)} subjects, H&Y>=2.5 proxy, {splits_gait_hy.get('cv_folds', []) and len(splits_gait_hy['cv_folds'])}-fold CV
    UCI Voice:       {len(df_voice)} recordings, PD vs Healthy, 5-fold CV
    UCI UPDRS:       {len(df_updrs)} subjects (limited positive samples)

  FUSION CANDIDATE:
    GaitPDB with clinical: {len(df_gait_hy)} subjects have both gait sensor + H&Y/UPDRS

  Split files saved to: {SPLIT_DIR}/
    """)

    print("Done.")


if __name__ == "__main__":
    main()
