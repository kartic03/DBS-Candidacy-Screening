#!/usr/bin/env python3
"""
GaitPDB Force Plate Feature Extraction
=======================================
Processes raw 100Hz bilateral foot force plate data from PhysioNet GaitPDB.
159 subjects (93 PD, 66 controls), multiple walking trials per subject.

Extracts genuine gait biomarker features:
  - Temporal: Stride time, step time (mean, std, CV)
  - Asymmetry: Step asymmetry index, force asymmetry
  - Force: Peak force, loading rate (bilateral)
  - Variability: Stride time sample entropy
  - FOG index: Power(0.5-3Hz) / Power(3-8Hz)
  - Dual-task cost where applicable

Output: data/processed/gait_features/gaitpdb_sensor_features.csv

Author: Kartic Mishra, Gachon University
"""

import os
import warnings
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# ── Hardware config ──────────────────────────────────────────────────────────
N_CORES = multiprocessing.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(N_CORES)
os.environ["MKL_NUM_THREADS"] = str(N_CORES)

SEED = 42
np.random.seed(SEED)

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAIT_ROOT = os.path.join(
    PROJECT_ROOT, "data", "raw", "gaitpdb",
    "physionet.org", "files", "gaitpdb", "1.0.0"
)
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "gait_features")
os.makedirs(OUT_DIR, exist_ok=True)

FS = 100  # 100 Hz sampling rate


def sample_entropy_simple(sig, m=2, r_factor=0.2):
    """Compute sample entropy for regularity measurement."""
    n = len(sig)
    if n < m + 2:
        return np.nan
    r = r_factor * np.std(sig)
    if r == 0:
        return 0.0

    def _count(template_len):
        templates = np.array([sig[i:i + template_len] for i in range(n - template_len)])
        count = 0
        for i in range(len(templates)):
            dists = np.max(np.abs(templates[i] - templates[i + 1:]), axis=1)
            count += np.sum(dists <= r)
        return count

    A = _count(m + 1)
    B = _count(m)
    if B == 0:
        return np.nan
    return -np.log(A / B + 1e-12)


def compute_fog_index(force_signal, fs):
    """
    Compute Freezing of Gait (FOG) index.
    FOG index = Power(0.5-3Hz) / Power(3-8Hz)
    Higher values indicate more freezing-like gait patterns.
    """
    nperseg = min(len(force_signal), 512)
    if nperseg < 64:
        return np.nan
    freqs, psd = signal.welch(force_signal, fs=fs, nperseg=nperseg)

    freeze_mask = (freqs >= 0.5) & (freqs <= 3.0)
    locomotor_mask = (freqs >= 3.0) & (freqs <= 8.0)

    freeze_power = np.trapz(psd[freeze_mask], freqs[freeze_mask]) if freeze_mask.sum() > 0 else 0
    locomotor_power = np.trapz(psd[locomotor_mask], freqs[locomotor_mask]) if locomotor_mask.sum() > 0 else 1e-12

    return freeze_power / locomotor_power


def detect_heel_strikes(total_force, fs, min_force_threshold=50):
    """
    Detect heel strikes from total vertical ground reaction force.
    Returns indices of heel strike events.
    """
    if len(total_force) < fs:
        return np.array([])

    # Smoothing to reduce noise
    window = max(int(0.05 * fs), 3)  # 50ms window
    if window % 2 == 0:
        window += 1
    smoothed = np.convolve(total_force, np.ones(window) / window, mode='same')

    # Find peaks (heel strikes) with minimum distance of 0.4s (max cadence ~150 steps/min)
    min_distance = int(0.4 * fs)
    peaks, properties = find_peaks(
        smoothed,
        height=min_force_threshold,
        distance=min_distance,
        prominence=min_force_threshold * 0.3
    )

    return peaks


def extract_trial_features(filepath, fs=FS):
    """Extract gait features from a single walking trial."""
    try:
        data = np.loadtxt(filepath, delimiter='\t')
    except Exception:
        return None

    if data.ndim != 2 or data.shape[1] < 19:
        return None

    # Columns: Time(0), 8 left sensors(1-8), 8 right sensors(9-16), TotalL(17), TotalR(18)
    time = data[:, 0]
    total_L = data[:, 17]
    total_R = data[:, 18]
    total_force = total_L + total_R

    feats = {}
    feats["duration_sec"] = time[-1] - time[0]
    feats["n_samples"] = len(data)

    # ── Detect heel strikes (bilateral) ──────────────────────────────────
    hs_L = detect_heel_strikes(total_L, fs)
    hs_R = detect_heel_strikes(total_R, fs)

    # Stride times (time between consecutive heel strikes of same foot)
    if len(hs_L) >= 3:
        stride_L = np.diff(hs_L) / fs
        feats["stride_time_L_mean"] = np.mean(stride_L)
        feats["stride_time_L_std"] = np.std(stride_L)
        feats["stride_time_L_cv"] = np.std(stride_L) / (np.mean(stride_L) + 1e-12)
    else:
        feats["stride_time_L_mean"] = np.nan
        feats["stride_time_L_std"] = np.nan
        feats["stride_time_L_cv"] = np.nan

    if len(hs_R) >= 3:
        stride_R = np.diff(hs_R) / fs
        feats["stride_time_R_mean"] = np.mean(stride_R)
        feats["stride_time_R_std"] = np.std(stride_R)
        feats["stride_time_R_cv"] = np.std(stride_R) / (np.mean(stride_R) + 1e-12)
    else:
        feats["stride_time_R_mean"] = np.nan
        feats["stride_time_R_std"] = np.nan
        feats["stride_time_R_cv"] = np.nan

    # ── Step asymmetry ───────────────────────────────────────────────────
    if pd.notna(feats.get("stride_time_L_mean")) and pd.notna(feats.get("stride_time_R_mean")):
        mean_stride = (feats["stride_time_L_mean"] + feats["stride_time_R_mean"]) / 2
        feats["step_asymmetry_index"] = abs(
            feats["stride_time_L_mean"] - feats["stride_time_R_mean"]
        ) / (mean_stride + 1e-12)

        # Stride time variability (combined)
        feats["stride_time_mean"] = mean_stride
        feats["stride_time_cv"] = (feats["stride_time_L_cv"] + feats["stride_time_R_cv"]) / 2
    else:
        feats["step_asymmetry_index"] = np.nan
        feats["stride_time_mean"] = np.nan
        feats["stride_time_cv"] = np.nan

    # ── Force features ───────────────────────────────────────────────────
    feats["peak_force_L"] = np.max(total_L)
    feats["peak_force_R"] = np.max(total_R)
    feats["mean_force_L"] = np.mean(total_L[total_L > 10])  # Exclude swing phase
    feats["mean_force_R"] = np.mean(total_R[total_R > 10])

    # Force asymmetry
    peak_sum = feats["peak_force_L"] + feats["peak_force_R"]
    if peak_sum > 0:
        feats["force_asymmetry"] = abs(
            feats["peak_force_L"] - feats["peak_force_R"]
        ) / peak_sum
    else:
        feats["force_asymmetry"] = np.nan

    # Loading rate (force rise rate at heel strike)
    loading_rates_L = []
    loading_rates_R = []
    for hs in hs_L[:20]:  # Limit to first 20 strikes
        start = max(0, hs - int(0.05 * fs))
        if hs < len(total_L):
            loading_rates_L.append((total_L[hs] - total_L[start]) / (0.05 + 1e-12))
    for hs in hs_R[:20]:
        start = max(0, hs - int(0.05 * fs))
        if hs < len(total_R):
            loading_rates_R.append((total_R[hs] - total_R[start]) / (0.05 + 1e-12))

    feats["loading_rate_L"] = np.mean(loading_rates_L) if loading_rates_L else np.nan
    feats["loading_rate_R"] = np.mean(loading_rates_R) if loading_rates_R else np.nan

    # ── Cadence ──────────────────────────────────────────────────────────
    total_strides = len(hs_L) + len(hs_R)
    if feats["duration_sec"] > 0:
        feats["cadence_steps_per_min"] = (total_strides / feats["duration_sec"]) * 60
    else:
        feats["cadence_steps_per_min"] = np.nan

    # ── Entropy features ─────────────────────────────────────────────────
    if len(hs_L) >= 10:
        stride_L = np.diff(hs_L) / fs
        feats["stride_entropy_L"] = sample_entropy_simple(stride_L)
    else:
        feats["stride_entropy_L"] = np.nan

    if len(hs_R) >= 10:
        stride_R = np.diff(hs_R) / fs
        feats["stride_entropy_R"] = sample_entropy_simple(stride_R)
    else:
        feats["stride_entropy_R"] = np.nan

    # ── FOG index ────────────────────────────────────────────────────────
    feats["fog_index_L"] = compute_fog_index(total_L, fs)
    feats["fog_index_R"] = compute_fog_index(total_R, fs)
    feats["fog_index_total"] = compute_fog_index(total_force, fs)

    # ── Force variability ────────────────────────────────────────────────
    # Coefficient of variation of peak forces across strides
    if len(hs_L) >= 3:
        peak_forces_L = [total_L[hs] for hs in hs_L if hs < len(total_L)]
        feats["peak_force_cv_L"] = np.std(peak_forces_L) / (np.mean(peak_forces_L) + 1e-12)
    else:
        feats["peak_force_cv_L"] = np.nan

    if len(hs_R) >= 3:
        peak_forces_R = [total_R[hs] for hs in hs_R if hs < len(total_R)]
        feats["peak_force_cv_R"] = np.std(peak_forces_R) / (np.mean(peak_forces_R) + 1e-12)
    else:
        feats["peak_force_cv_R"] = np.nan

    return feats


def load_demographics():
    """Load GaitPDB demographics file."""
    demo_path = os.path.join(GAIT_ROOT, "demographics.txt")
    # Read with flexible parsing (some lines have extra columns)
    rows = []
    with open(demo_path) as f:
        header = f.readline().strip().split('\t')
        for line in f:
            parts = line.strip().split('\t')
            row = {}
            for i, h in enumerate(header):
                if i < len(parts):
                    row[h] = parts[i]
                else:
                    row[h] = None
            rows.append(row)

    df = pd.DataFrame(rows)
    # Clean numeric columns
    for col in ['Age', 'Height', 'Weight', 'HoehnYahr', 'UPDRS', 'UPDRSM', 'TUAG']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'Gender' in df.columns:
        df['Gender'] = pd.to_numeric(df['Gender'], errors='coerce')  # 1=male, 2=female
    if 'Group' in df.columns:
        df['Group'] = pd.to_numeric(df['Group'], errors='coerce')  # 1=PD, 2=Control

    return df


def main():
    print("=" * 70)
    print("GaitPDB Force Plate Feature Extraction")
    print("=" * 70)

    # ── Load demographics ────────────────────────────────────────────────
    print("\n[1/4] Loading demographics...")
    demo = load_demographics()
    print(f"  Subjects in demographics: {len(demo)}")
    print(f"  Group distribution: PD={int((demo['Group'] == 1).sum())}, "
          f"Control={int((demo['Group'] == 2).sum())}")
    print(f"  H&Y available: {demo['HoehnYahr'].notna().sum()}")

    # ── Discover walking trial files ─────────────────────────────────────
    print("\n[2/4] Discovering walking trial files...")
    trial_files = sorted(Path(GAIT_ROOT).glob("*.txt"))
    trial_files = [f for f in trial_files if f.name not in
                   ["demographics.txt", "format.txt"]]
    print(f"  Found {len(trial_files)} walking trial files")

    # Parse subject IDs from filenames (e.g., GaPt03_02.txt → GaPt03)
    trial_info = []
    for tf in trial_files:
        parts = tf.stem.split('_')
        subject_id = parts[0]
        trial_num = int(parts[1]) if len(parts) > 1 else 1
        trial_info.append({
            "filepath": str(tf),
            "subject_id": subject_id,
            "trial_num": trial_num,
            "is_dual_task": trial_num == 10  # Walk 10 = dual-task (serial-7)
        })

    trial_df = pd.DataFrame(trial_info)
    print(f"  Unique subjects in trials: {trial_df['subject_id'].nunique()}")
    print(f"  Dual-task trials: {trial_df['is_dual_task'].sum()}")

    # ── Extract features from all trials ─────────────────────────────────
    print("\n[3/4] Extracting gait features from force plate data...")
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(extract_trial_features)(row["filepath"])
        for _, row in trial_df.iterrows()
    )

    # Combine with trial info
    valid_results = []
    for i, r in enumerate(results):
        if r is not None:
            r.update(trial_info[i])
            valid_results.append(r)

    print(f"  Successfully processed: {len(valid_results)} / {len(trial_files)} trials")

    df_trials = pd.DataFrame(valid_results)

    # ── Aggregate per subject (mean across trials) ───────────────────────
    feat_cols = [c for c in df_trials.columns
                 if c not in ["filepath", "subject_id", "trial_num", "is_dual_task",
                              "n_samples", "duration_sec"]]

    # Separate single-task and dual-task
    df_single = df_trials[~df_trials["is_dual_task"]]
    df_dual = df_trials[df_trials["is_dual_task"]]

    # Aggregate single-task trials per subject
    subject_rows = []
    for sid, grp in df_single.groupby("subject_id"):
        rec = {"subject_id": sid, "n_single_trials": len(grp)}
        for col in feat_cols:
            vals = grp[col].dropna()
            if len(vals) > 0:
                rec[col] = vals.mean()
            else:
                rec[col] = np.nan
        subject_rows.append(rec)

    df_subjects = pd.DataFrame(subject_rows)

    # Add dual-task features if available
    for sid, grp in df_dual.groupby("subject_id"):
        mask = df_subjects["subject_id"] == sid
        if mask.sum() == 0:
            continue
        idx = df_subjects.index[mask][0]
        for col in feat_cols:
            vals = grp[col].dropna()
            if len(vals) > 0:
                df_subjects.loc[idx, f"dual_{col}"] = vals.mean()

                # Dual-task cost
                single_val = df_subjects.loc[idx, col]
                if pd.notna(single_val) and single_val != 0:
                    df_subjects.loc[idx, f"dtc_{col}"] = (
                        (single_val - vals.mean()) / abs(single_val) * 100
                    )

    # ── Merge with demographics ──────────────────────────────────────────
    df_merged = df_subjects.merge(
        demo[["ID", "Study", "Group", "Gender", "Age", "Height", "Weight",
              "HoehnYahr", "UPDRS", "UPDRSM", "TUAG"]],
        left_on="subject_id", right_on="ID", how="left"
    ).drop(columns=["ID"], errors="ignore")

    # Create labels
    df_merged["pd_label"] = (df_merged["Group"] == 1).astype(int)
    df_merged["sex"] = df_merged["Gender"].map({1: 1, 2: 0})  # 1=male, 0=female

    # DBS proxy: H&Y >= 2.5
    df_merged["dbs_proxy_hy25"] = (df_merged["HoehnYahr"] >= 2.5).astype(float)
    df_merged.loc[df_merged["HoehnYahr"].isna(), "dbs_proxy_hy25"] = np.nan

    # DBS proxy: UPDRSM >= 32
    df_merged["dbs_proxy_updrs32"] = (df_merged["UPDRSM"] >= 32).astype(float)
    df_merged.loc[df_merged["UPDRSM"].isna(), "dbs_proxy_updrs32"] = np.nan

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n  Subject summary:")
    print(f"    Total subjects: {len(df_merged)}")
    print(f"    PD: {(df_merged['pd_label'] == 1).sum()}, "
          f"Control: {(df_merged['pd_label'] == 0).sum()}")
    print(f"    H&Y >= 2.5 (DBS proxy): {int(df_merged['dbs_proxy_hy25'].sum())} "
          f"/ {df_merged['dbs_proxy_hy25'].notna().sum()} with H&Y data")
    print(f"    UPDRSM >= 32 (DBS proxy): {int(df_merged['dbs_proxy_updrs32'].sum())} "
          f"/ {df_merged['dbs_proxy_updrs32'].notna().sum()} with UPDRSM data")

    gait_feat_cols = [c for c in df_merged.columns if c not in [
        "subject_id", "Study", "Group", "Gender", "pd_label", "sex",
        "dbs_proxy_hy25", "dbs_proxy_updrs32", "n_single_trials"
    ]]
    print(f"    Feature columns: {len(gait_feat_cols)}")

    # Check key features
    for feat in ["stride_time_cv", "step_asymmetry_index", "fog_index_total",
                 "force_asymmetry", "cadence_steps_per_min"]:
        if feat in df_merged.columns:
            vals = df_merged[feat].dropna()
            print(f"    {feat}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
                  f"non-null={len(vals)}/{len(df_merged)}")

    # ── Save ─────────────────────────────────────────────────────────────
    print("\n[4/4] Saving...")

    out_path = os.path.join(OUT_DIR, "gaitpdb_sensor_features.csv")
    df_merged.to_csv(out_path, index=False)
    print(f"  Saved: {out_path} ({df_merged.shape})")

    # Subset with H&Y data (for fusion with clinical features)
    df_with_hy = df_merged[df_merged["HoehnYahr"].notna()].copy()
    out_hy = os.path.join(OUT_DIR, "gaitpdb_with_clinical.csv")
    df_with_hy.to_csv(out_hy, index=False)
    print(f"  Saved: {out_hy} ({df_with_hy.shape}) — subjects with H&Y for fusion")

    print("\nDone.")


if __name__ == "__main__":
    main()
