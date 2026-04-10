#!/usr/bin/env python3
"""
PADS Smartwatch Sensor Feature Extraction
==========================================
Processes raw 100Hz accelerometer + gyroscope data from Samsung/Apple Watch
for 469 subjects across 11 motor tasks (bilateral wrists).

Extracts genuine wearable biomarker features:
  - Tremor: PSD power in 4-6Hz (resting) and 8-12Hz (action tremor) bands
  - Bradykinesia: Angular velocity metrics, amplitude CV, energy decay
  - Rigidity: Range of motion, log dimensionless jerk, zero-crossing rate
  - Statistical: Mean, std, skewness, kurtosis per axis
  - Entropy: Permutation entropy, sample entropy
  - Frequency: Dominant frequency, spectral entropy, spectral centroid
  - Inter-axis: 6 pairwise correlations

Output: data/processed/wearable_features/pads_sensor_features.csv

Author: Kartic Mishra, Gachon University
"""

import os
import sys
import json
import warnings
import multiprocessing
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
from scipy import signal, stats
from joblib import Parallel, delayed
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Hardware config ──────────────────────────────────────────────────────────
N_CORES = multiprocessing.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(N_CORES)
os.environ["MKL_NUM_THREADS"] = str(N_CORES)

SEED = 42
np.random.seed(SEED)

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PADS_ROOT = os.path.join(
    PROJECT_ROOT, "data", "raw", "pads",
    "physionet.org", "files", "parkinsons-disease-smartwatch", "1.0.0"
)
TS_DIR = os.path.join(PADS_ROOT, "movement", "timeseries")
PATIENT_DIR = os.path.join(PADS_ROOT, "patients")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "wearable_features")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Constants ────────────────────────────────────────────────────────────────
FS = 100  # Sampling rate (~100 Hz verified from data)
WINDOW_SEC = 4  # 4-second windows
WINDOW_SAMPLES = WINDOW_SEC * FS  # 400 samples
OVERLAP = 0.5  # 50% overlap
STEP = int(WINDOW_SAMPLES * (1 - OVERLAP))

TASKS = [
    "CrossArms", "DrinkGlas", "Entrainment", "HoldWeight", "LiftHold",
    "PointFinger", "Relaxed", "RelaxedTask", "StretchHold",
    "TouchIndex", "TouchNose"
]
WRISTS = ["LeftWrist", "RightWrist"]
AXIS_NAMES = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ"]

# Tremor frequency bands (Hz)
RESTING_TREMOR_BAND = (4, 6)
ACTION_TREMOR_BAND = (8, 12)
BRADYKINESIA_BAND = (0.5, 3)


def compute_psd_bandpower(sig, fs, band):
    """Compute power spectral density in a given frequency band using Welch."""
    nperseg = min(len(sig), 256)
    if nperseg < 16:
        return 0.0
    freqs, psd = signal.welch(sig, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if mask.sum() == 0:
        return 0.0
    return np.trapz(psd[mask], freqs[mask])


def spectral_entropy(sig, fs):
    """Compute normalized spectral entropy."""
    nperseg = min(len(sig), 256)
    if nperseg < 16:
        return 0.0
    freqs, psd = signal.welch(sig, fs=fs, nperseg=nperseg)
    psd_norm = psd / (psd.sum() + 1e-12)
    psd_norm = psd_norm[psd_norm > 0]
    return -np.sum(psd_norm * np.log2(psd_norm + 1e-12)) / np.log2(len(psd_norm) + 1)


def dominant_frequency(sig, fs):
    """Find the frequency with maximum power."""
    nperseg = min(len(sig), 256)
    if nperseg < 16:
        return 0.0
    freqs, psd = signal.welch(sig, fs=fs, nperseg=nperseg)
    return freqs[np.argmax(psd)]


def spectral_centroid(sig, fs):
    """Compute spectral centroid (center of mass of spectrum)."""
    nperseg = min(len(sig), 256)
    if nperseg < 16:
        return 0.0
    freqs, psd = signal.welch(sig, fs=fs, nperseg=nperseg)
    total = psd.sum()
    if total == 0:
        return 0.0
    return np.sum(freqs * psd) / total


def log_dimensionless_jerk(sig, fs):
    """
    Compute log dimensionless jerk (LDLJ) — smoothness of movement.
    Lower = smoother movement. Higher = more jerky (rigidity indicator).
    """
    if len(sig) < 3:
        return 0.0
    dt = 1.0 / fs
    jerk = np.diff(sig, n=2) / (dt ** 2)
    duration = len(sig) / fs
    peak = np.max(np.abs(sig)) + 1e-12
    ldlj = -np.log(duration ** 3 / (peak ** 2) * np.sum(jerk ** 2) * dt + 1e-12)
    return ldlj


def zero_crossing_rate(sig):
    """Count zero crossings normalized by signal length."""
    if len(sig) < 2:
        return 0.0
    return np.sum(np.diff(np.sign(sig - np.mean(sig))) != 0) / len(sig)


def permutation_entropy(sig, order=3, delay=1):
    """Compute permutation entropy (complexity measure)."""
    n = len(sig)
    if n < order * delay:
        return 0.0
    n_patterns = n - (order - 1) * delay
    if n_patterns <= 0:
        return 0.0
    patterns = np.zeros((n_patterns, order))
    for i in range(order):
        patterns[:, i] = sig[i * delay: i * delay + n_patterns]
    perms = np.argsort(patterns, axis=1)
    _, counts = np.unique(perms, axis=0, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-12))


def sample_entropy(sig, m=2, r_factor=0.2):
    """Compute sample entropy (regularity/complexity measure)."""
    n = len(sig)
    if n < m + 2:
        return 0.0
    r = r_factor * np.std(sig)
    if r == 0:
        return 0.0

    def _count_matches(template_len):
        templates = np.array([sig[i:i + template_len] for i in range(n - template_len)])
        count = 0
        for i in range(len(templates)):
            dists = np.max(np.abs(templates[i] - templates[i + 1:]), axis=1)
            count += np.sum(dists <= r)
        return count

    A = _count_matches(m + 1)
    B = _count_matches(m)
    if B == 0:
        return 0.0
    return -np.log(A / B + 1e-12)


def extract_window_features(window, fs=FS):
    """
    Extract features from a single 4-second window of 6-axis sensor data.
    window: array of shape (n_samples, 6) — AccX,AccY,AccZ,GyroX,GyroY,GyroZ
    """
    feats = {}

    acc_mag = np.sqrt(window[:, 0] ** 2 + window[:, 1] ** 2 + window[:, 2] ** 2)
    gyro_mag = np.sqrt(window[:, 3] ** 2 + window[:, 4] ** 2 + window[:, 5] ** 2)

    # ── Per-axis statistical features ────────────────────────────────────
    for i, name in enumerate(AXIS_NAMES):
        sig = window[:, i]
        feats[f"{name}_mean"] = np.mean(sig)
        feats[f"{name}_std"] = np.std(sig)
        feats[f"{name}_skew"] = float(stats.skew(sig))
        feats[f"{name}_kurt"] = float(stats.kurtosis(sig))

    # ── Magnitude features ───────────────────────────────────────────────
    feats["acc_mag_mean"] = np.mean(acc_mag)
    feats["acc_mag_std"] = np.std(acc_mag)
    feats["gyro_mag_mean"] = np.mean(gyro_mag)
    feats["gyro_mag_std"] = np.std(gyro_mag)

    # ── Tremor features (PSD band power) ─────────────────────────────────
    feats["tremor_rest_acc"] = compute_psd_bandpower(acc_mag, fs, RESTING_TREMOR_BAND)
    feats["tremor_rest_gyro"] = compute_psd_bandpower(gyro_mag, fs, RESTING_TREMOR_BAND)
    feats["tremor_action_acc"] = compute_psd_bandpower(acc_mag, fs, ACTION_TREMOR_BAND)
    feats["tremor_action_gyro"] = compute_psd_bandpower(gyro_mag, fs, ACTION_TREMOR_BAND)
    total_acc_power = compute_psd_bandpower(acc_mag, fs, (0.5, 25))
    feats["tremor_rest_ratio"] = feats["tremor_rest_acc"] / (total_acc_power + 1e-12)
    feats["tremor_action_ratio"] = feats["tremor_action_acc"] / (total_acc_power + 1e-12)

    # ── Bradykinesia features ────────────────────────────────────────────
    feats["brady_power_acc"] = compute_psd_bandpower(acc_mag, fs, BRADYKINESIA_BAND)
    feats["mean_angular_velocity"] = np.mean(gyro_mag)
    feats["amplitude_cv_acc"] = np.std(acc_mag) / (np.mean(acc_mag) + 1e-12)
    feats["amplitude_cv_gyro"] = np.std(gyro_mag) / (np.mean(gyro_mag) + 1e-12)
    mid = len(acc_mag) // 2
    p1 = np.sum(acc_mag[:mid] ** 2)
    p2 = np.sum(acc_mag[mid:] ** 2)
    feats["energy_decay"] = p2 / (p1 + 1e-12)

    # ── Rigidity features ────────────────────────────────────────────────
    for i, axis in enumerate(["AccX", "AccY", "AccZ"]):
        feats[f"rom_{axis}"] = np.ptp(window[:, i])
    feats["ldlj_acc"] = log_dimensionless_jerk(acc_mag, fs)
    feats["ldlj_gyro"] = log_dimensionless_jerk(gyro_mag, fs)
    feats["zcr_acc"] = zero_crossing_rate(acc_mag)
    feats["zcr_gyro"] = zero_crossing_rate(gyro_mag)

    # ── Frequency features ───────────────────────────────────────────────
    feats["dom_freq_acc"] = dominant_frequency(acc_mag, fs)
    feats["dom_freq_gyro"] = dominant_frequency(gyro_mag, fs)
    feats["spectral_entropy_acc"] = spectral_entropy(acc_mag, fs)
    feats["spectral_entropy_gyro"] = spectral_entropy(gyro_mag, fs)
    feats["spectral_centroid_acc"] = spectral_centroid(acc_mag, fs)
    feats["spectral_centroid_gyro"] = spectral_centroid(gyro_mag, fs)

    # ── Entropy features ─────────────────────────────────────────────────
    feats["perm_entropy_acc"] = permutation_entropy(acc_mag)
    feats["perm_entropy_gyro"] = permutation_entropy(gyro_mag)
    feats["sample_entropy_acc"] = sample_entropy(acc_mag[::2])
    feats["sample_entropy_gyro"] = sample_entropy(gyro_mag[::2])

    # ── Inter-axis correlations ──────────────────────────────────────────
    axis_pairs = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]
    pair_names = ["AccXY", "AccXZ", "AccYZ", "GyroXY", "GyroXZ", "GyroYZ"]
    for (a, b), pname in zip(axis_pairs, pair_names):
        r, _ = stats.pearsonr(window[:, a], window[:, b])
        feats[f"corr_{pname}"] = r if np.isfinite(r) else 0.0

    return feats


def process_single_file(filepath, fs=FS):
    """Load a sensor file and extract windowed features, returning aggregated stats."""
    try:
        data = np.loadtxt(filepath, delimiter=",")
    except Exception:
        return None

    if data.ndim != 2 or data.shape[1] < 7:
        return None

    sensor_data = data[:, 1:7]
    n_samples = len(sensor_data)

    if n_samples < WINDOW_SAMPLES:
        window_features = [extract_window_features(sensor_data, fs)]
    else:
        window_features = []
        for start in range(0, n_samples - WINDOW_SAMPLES + 1, STEP):
            window = sensor_data[start: start + WINDOW_SAMPLES]
            window_features.append(extract_window_features(window, fs))

    if not window_features:
        return None

    df_windows = pd.DataFrame(window_features)
    agg = {}
    for col in df_windows.columns:
        vals = df_windows[col].dropna()
        if len(vals) > 0:
            agg[f"{col}_mean"] = vals.mean()
            agg[f"{col}_std"] = vals.std() if len(vals) > 1 else 0.0

    agg["n_windows"] = len(window_features)
    agg["recording_duration_sec"] = n_samples / fs

    return agg


def process_subject(subject_id, ts_dir, patient_dir):
    """Process all task files for a single subject."""
    patient_file = os.path.join(patient_dir, f"patient_{subject_id}.json")
    if not os.path.exists(patient_file):
        return None

    with open(patient_file) as f:
        meta = json.load(f)

    subject_features = {
        "subject_id": subject_id,
        "condition": meta.get("condition", "Unknown"),
        "age": meta.get("age"),
        "gender": meta.get("gender"),
        "height": meta.get("height"),
        "weight": meta.get("weight"),
        "handedness": meta.get("handedness"),
    }

    cond = meta.get("condition", "").lower()
    if "parkinson" in cond:
        subject_features["pd_label"] = 1
    elif "healthy" in cond:
        subject_features["pd_label"] = 0
    else:
        subject_features["pd_label"] = -1

    n_files_processed = 0
    for task in TASKS:
        for wrist in WRISTS:
            filename = f"{subject_id}_{task}_{wrist}.txt"
            filepath = os.path.join(ts_dir, filename)
            if not os.path.exists(filepath):
                continue

            file_feats = process_single_file(filepath)
            if file_feats is None:
                continue

            prefix = f"{task}_{wrist}"
            for key, val in file_feats.items():
                subject_features[f"{prefix}_{key}"] = val

            n_files_processed += 1

    subject_features["n_files_processed"] = n_files_processed

    if n_files_processed == 0:
        return None

    return subject_features


def aggregate_across_tasks(df):
    """
    Create compact feature set by aggregating task-wrist features.
    Computes resting vs action task aggregates, cross-task variability,
    and bilateral wrist asymmetry.
    """
    resting_tasks = ["Relaxed", "RelaxedTask"]
    action_tasks = ["CrossArms", "DrinkGlas", "Entrainment", "HoldWeight",
                    "LiftHold", "PointFinger", "StretchHold", "TouchIndex", "TouchNose"]

    key_features = [
        "tremor_rest_acc_mean", "tremor_rest_gyro_mean",
        "tremor_action_acc_mean", "tremor_action_gyro_mean",
        "tremor_rest_ratio_mean", "tremor_action_ratio_mean",
        "brady_power_acc_mean", "mean_angular_velocity_mean",
        "amplitude_cv_acc_mean", "amplitude_cv_gyro_mean",
        "energy_decay_mean", "ldlj_acc_mean", "ldlj_gyro_mean",
        "zcr_acc_mean", "zcr_gyro_mean",
        "dom_freq_acc_mean", "dom_freq_gyro_mean",
        "spectral_entropy_acc_mean", "spectral_entropy_gyro_mean",
        "perm_entropy_acc_mean", "perm_entropy_gyro_mean",
        "sample_entropy_acc_mean", "sample_entropy_gyro_mean",
        "acc_mag_mean_mean", "acc_mag_std_mean",
        "gyro_mag_mean_mean", "gyro_mag_std_mean",
        "rom_AccX_mean", "rom_AccY_mean", "rom_AccZ_mean",
    ]

    compact_rows = []
    for _, row in df.iterrows():
        compact = {
            "subject_id": row["subject_id"],
            "condition": row["condition"],
            "age": row["age"],
            "gender": row["gender"],
            "height": row["height"],
            "weight": row["weight"],
            "pd_label": row["pd_label"],
            "n_files_processed": row["n_files_processed"],
        }

        for feat in key_features:
            resting_vals = []
            action_vals = []
            all_vals = []
            for task in TASKS:
                for wrist in WRISTS:
                    col = f"{task}_{wrist}_{feat}"
                    if col in row.index and pd.notna(row[col]):
                        all_vals.append(row[col])
                        if task in resting_tasks:
                            resting_vals.append(row[col])
                        else:
                            action_vals.append(row[col])

            feat_short = feat.replace("_mean", "").replace("_std", "")
            compact[f"rest_{feat_short}"] = np.mean(resting_vals) if resting_vals else np.nan
            compact[f"action_{feat_short}"] = np.mean(action_vals) if action_vals else np.nan
            compact[f"all_{feat_short}"] = np.mean(all_vals) if all_vals else np.nan
            compact[f"all_{feat_short}_var"] = np.var(all_vals) if len(all_vals) > 1 else np.nan

        # Bilateral asymmetry for key features
        asym_features = [
            "tremor_rest_acc_mean", "tremor_action_acc_mean",
            "amplitude_cv_acc_mean", "acc_mag_mean_mean"
        ]
        for feat in asym_features:
            r_vals = []
            l_vals = []
            for task in TASKS:
                r_col = f"{task}_RightWrist_{feat}"
                l_col = f"{task}_LeftWrist_{feat}"
                if r_col in row.index and l_col in row.index:
                    r_val = row[r_col]
                    l_val = row[l_col]
                    if pd.notna(r_val) and pd.notna(l_val):
                        r_vals.append(r_val)
                        l_vals.append(l_val)
            if r_vals and l_vals:
                r_mean = np.mean(r_vals)
                l_mean = np.mean(l_vals)
                denom = abs(r_mean) + abs(l_mean)
                feat_short = feat.replace("_mean", "")
                compact[f"bilateral_asym_{feat_short}"] = (r_mean - l_mean) / denom if denom > 0 else 0.0

        compact_rows.append(compact)

    return pd.DataFrame(compact_rows)


def main():
    print("=" * 70)
    print("PADS Smartwatch Sensor Feature Extraction")
    print("=" * 70)
    print(f"  Sampling rate: {FS} Hz")
    print(f"  Window: {WINDOW_SEC}s ({WINDOW_SAMPLES} samples), {int(OVERLAP * 100)}% overlap")
    print(f"  Tasks: {len(TASKS)}, Wrists: {len(WRISTS)}")

    # ── Discover subjects ────────────────────────────────────────────────
    print("\n[1/4] Discovering subjects...")
    patient_files = sorted(Path(PATIENT_DIR).glob("patient_*.json"))
    subject_ids = [f.stem.replace("patient_", "") for f in patient_files]
    print(f"  Found {len(subject_ids)} subjects")

    # ── Process all subjects in parallel ─────────────────────────────────
    print("\n[2/4] Extracting sensor features (this takes 15-30 min)...")
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(process_subject)(sid, TS_DIR, PATIENT_DIR) for sid in subject_ids
    )

    results = [r for r in results if r is not None]
    print(f"  Successfully processed: {len(results)} / {len(subject_ids)} subjects")

    df_raw = pd.DataFrame(results)
    print(f"  Raw feature matrix: {df_raw.shape}")

    # ── Aggregate across tasks ───────────────────────────────────────────
    print("\n[3/4] Aggregating features across tasks...")
    df_compact = aggregate_across_tasks(df_raw)
    print(f"  Compact feature matrix: {df_compact.shape}")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n  Condition distribution:")
    for cond, cnt in df_compact["condition"].value_counts().items():
        print(f"    {cond}: {cnt}")

    print(f"\n  PD label distribution:")
    print(f"    PD (1): {(df_compact['pd_label'] == 1).sum()}")
    print(f"    Healthy (0): {(df_compact['pd_label'] == 0).sum()}")
    print(f"    Other (-1): {(df_compact['pd_label'] == -1).sum()}")

    meta_cols = ["subject_id", "condition", "age", "gender", "height", "weight",
                 "pd_label", "n_files_processed"]
    feat_cols = [c for c in df_compact.columns if c not in meta_cols]
    print(f"\n  Feature columns: {len(feat_cols)}")

    nan_pcts = df_compact[feat_cols].isna().mean()
    high_nan = nan_pcts[nan_pcts > 0.5]
    print(f"  Features with >50% NaN: {len(high_nan)}")

    # ── Save ─────────────────────────────────────────────────────────────
    print("\n[4/4] Saving...")

    out_full = os.path.join(OUT_DIR, "pads_sensor_features.csv")
    df_compact.to_csv(out_full, index=False)
    print(f"  Saved: {out_full} ({df_compact.shape})")

    df_binary = df_compact[df_compact["pd_label"].isin([0, 1])].copy()
    out_binary = os.path.join(OUT_DIR, "pads_sensor_pd_vs_healthy.csv")
    df_binary.to_csv(out_binary, index=False)
    print(f"  Saved: {out_binary} ({df_binary.shape})")

    out_raw = os.path.join(OUT_DIR, "pads_sensor_raw_all_tasks.csv")
    df_raw.to_csv(out_raw, index=False)
    print(f"  Saved: {out_raw} ({df_raw.shape})")

    print("\nDone.")


if __name__ == "__main__":
    main()
