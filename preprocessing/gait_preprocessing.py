"""
Gait data preprocessing for WearGait-PD and GaitPDB datasets.
Produces standardised feature CSVs for downstream modelling.
"""

# ── Hardware init (must be before any numpy/sklearn import) ──────────────
import os, multiprocessing

N_CORES = multiprocessing.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(N_CORES)
os.environ["MKL_NUM_THREADS"] = str(N_CORES)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_CORES)

# ── Standard imports ─────────────────────────────────────────────────────
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import RobustScaler

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

WEARGAIT_DIR = PROJECT_ROOT / "data" / "raw" / "weargait_pd"
PD_CSV = WEARGAIT_DIR / "PD - Demographic+Clinical - datasetV1.csv"
CTRL_CSV = WEARGAIT_DIR / "CONTROLS - Demographic+Clinical - datasetV1.csv"

GAITPDB_DIR = PROJECT_ROOT / "data" / "raw" / "gaitpdb"

OUT_DIR = PROJECT_ROOT / "data" / "processed" / "gait_features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ───────────────────────────────────────────────────────────────
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

SEED = cfg["model"]["seed"]  # 42
N_JOBS = cfg["training"]["n_jobs"]  # -1
np.random.seed(SEED)

print(f"[gait_preprocessing] Project root : {PROJECT_ROOT}")
print(f"[gait_preprocessing] Seed={SEED}  N_JOBS={N_JOBS}  CPU cores={N_CORES}")


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def _extract_leading_number(val):
    """Extract leading numeric value from a string that may contain annotation text.
    E.g. '2*parationia' -> 2.0, '9' -> 9.0, '-' -> NaN
    """
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    if val_str in ("-", ""):
        return np.nan
    # Try to match a leading number (int or float)
    m = re.match(r"^([0-9]*\.?[0-9]+)", val_str)
    if m:
        return float(m.group(1))
    return np.nan


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from string columns and replace '-' with NaN."""
    # Strip whitespace from all object columns
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip()
    # Replace '-' and empty strings with NaN
    df.replace({"-": np.nan, "": np.nan}, inplace=True)
    return df


def _to_numeric_with_annotations(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Convert columns to numeric, handling annotation text by extracting
    the leading number."""
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(_extract_leading_number)
    return df


def _compute_asymmetry_index(right: pd.Series, left: pd.Series) -> pd.Series:
    """Asymmetry index: |right - left| / (right + left + 1e-6)."""
    return (right - left).abs() / (right + left + 1e-6)


# =====================================================================
# UPDRS COLUMN DEFINITIONS
# =====================================================================

# All Part III sub-item columns present in both PD and Controls files
UPDRS3_ALL_ITEMS = [
    "MDSUPDRS_3-1", "MDSUPDRS_3-2",
    "MDSUPDRS_3-3-Neck", "MDSUPDRS_3-3-RUE", "MDSUPDRS_3-3-LUE",
    "MDSUPDRS_3-3-RLE", "MDSUPDRS_3-3-LLE",
    "MDSUPDRS_3-4-R", "MDSUPDRS_3-4-L",
    "MDSUPDRS_3-5-R", "MDSUPDRS_3-5-L",
    "MDSUPDRS_3-6-R", "MDSUPDRS_3-6-L",
    "MDSUPDRS_3-7-R", "MDSUPDRS_3-7-L",
    "MDSUPDRS_3-8-R", "MDSUPDRS_3-8-L",
    "MDSUPDRS_3-9", "MDSUPDRS_3-10", "MDSUPDRS_3-11",
    "MDSUPDRS_3-12", "MDSUPDRS_3-13", "MDSUPDRS_3-14",
    "MDSUPDRS_3-15-R", "MDSUPDRS_3-15-L",
    "MDSUPDRS_3-16-R", "MDSUPDRS_3-16-L",
    "MDSUPDRS_3-17-RUE", "MDSUPDRS_3-17-LUE",
    "MDSUPDRS_3-17-RLE", "MDSUPDRS_3-17-LLE",
    "MDSUPDRS_3-17-LipJaw",
    "MDSUPDRS_3-18",
]

# Domain sub-scores
TREMOR_ITEMS = [
    "MDSUPDRS_3-15-R", "MDSUPDRS_3-15-L",
    "MDSUPDRS_3-16-R", "MDSUPDRS_3-16-L",
    "MDSUPDRS_3-17-RUE", "MDSUPDRS_3-17-LUE",
    "MDSUPDRS_3-17-RLE", "MDSUPDRS_3-17-LLE",
    "MDSUPDRS_3-17-LipJaw",
    "MDSUPDRS_3-18",
]

RIGIDITY_ITEMS = [
    "MDSUPDRS_3-3-Neck", "MDSUPDRS_3-3-RUE", "MDSUPDRS_3-3-LUE",
    "MDSUPDRS_3-3-RLE", "MDSUPDRS_3-3-LLE",
]

BRADYKINESIA_ITEMS = [
    "MDSUPDRS_3-4-R", "MDSUPDRS_3-4-L",
    "MDSUPDRS_3-5-R", "MDSUPDRS_3-5-L",
    "MDSUPDRS_3-6-R", "MDSUPDRS_3-6-L",
    "MDSUPDRS_3-7-R", "MDSUPDRS_3-7-L",
    "MDSUPDRS_3-8-R", "MDSUPDRS_3-8-L",
    "MDSUPDRS_3-14",
]

GAIT_POSTURE_ITEMS = [
    "MDSUPDRS_3-9", "MDSUPDRS_3-10", "MDSUPDRS_3-11",
    "MDSUPDRS_3-12", "MDSUPDRS_3-13",
]

SPEECH_FACIAL_ITEMS = [
    "MDSUPDRS_3-1", "MDSUPDRS_3-2",
]

# Bilateral pairs for asymmetry indices: (right_col, left_col, label)
BILATERAL_PAIRS = [
    ("MDSUPDRS_3-4-R", "MDSUPDRS_3-4-L", "finger_tapping"),
    ("MDSUPDRS_3-5-R", "MDSUPDRS_3-5-L", "hand_movements"),
    ("MDSUPDRS_3-6-R", "MDSUPDRS_3-6-L", "pronation_supination"),
    ("MDSUPDRS_3-7-R", "MDSUPDRS_3-7-L", "toe_tapping"),
    ("MDSUPDRS_3-8-R", "MDSUPDRS_3-8-L", "leg_agility"),
    ("MDSUPDRS_3-15-R", "MDSUPDRS_3-15-L", "postural_tremor"),
    ("MDSUPDRS_3-16-R", "MDSUPDRS_3-16-L", "kinetic_tremor"),
    ("MDSUPDRS_3-17-RUE", "MDSUPDRS_3-17-LUE", "rest_tremor_UE"),
    ("MDSUPDRS_3-17-RLE", "MDSUPDRS_3-17-LLE", "rest_tremor_LE"),
]


# =====================================================================
# 1. WEARGAIT-PD DATASET
# =====================================================================

def process_weargait_pd() -> pd.DataFrame:
    """Load, clean, extract features, label, and scale the WearGait-PD data."""
    print("\n" + "=" * 70)
    print("[WearGait-PD] Loading data...")
    print("=" * 70)

    # ── 1a. Load CSVs (header on row 2 = header=1) ──────────────────────
    pd_df = pd.read_csv(PD_CSV, header=1, dtype=str)
    ctrl_df = pd.read_csv(CTRL_CSV, header=1, dtype=str)
    print(f"  PD subjects loaded   : {len(pd_df)} rows, {pd_df.shape[1]} cols")
    print(f"  Control subjects loaded: {len(ctrl_df)} rows, {ctrl_df.shape[1]} cols")

    # ── 1b. Clean data ──────────────────────────────────────────────────
    pd_df = _clean_dataframe(pd_df)
    ctrl_df = _clean_dataframe(ctrl_df)

    # Harmonise column names: Controls uses 'Age' while PD uses 'Age (years)'
    if "Age" in ctrl_df.columns and "Age (years)" not in ctrl_df.columns:
        ctrl_df.rename(columns={"Age": "Age (years)"}, inplace=True)

    # ── 1c. Convert UPDRS and numeric columns ───────────────────────────
    numeric_cols = UPDRS3_ALL_ITEMS + ["Modified Hoehn & Yahr Score"]
    # Also add Part I, Part II, Part IV items
    for part_prefix, part_range in [("MDSUPDRS_1-", 14), ("MDSUPDRS_2-", 14),
                                     ("MDSUPDRS_4-", 7)]:
        for i in range(1, part_range):
            col = f"{part_prefix}{i}"
            if col in pd_df.columns or col in ctrl_df.columns:
                numeric_cols.append(col)

    # Demographic numeric cols
    demo_numeric = ["Age (years)", "Height (in)", "Weight (kg)",
                    "Years since PD diagnosis", "Years since surgery"]

    for df in [pd_df, ctrl_df]:
        _to_numeric_with_annotations(df, numeric_cols)
        for col in demo_numeric:
            if col in df.columns:
                df[col] = df[col].apply(_extract_leading_number)

    print("[WearGait-PD] Data cleaned and numeric columns converted.")

    # ── 1d. Feature extraction ──────────────────────────────────────────
    frames = []
    for df, group_label in [(pd_df, "PD"), (ctrl_df, "Control")]:
        feats = pd.DataFrame()
        feats["subject_id"] = df["Subject ID"]
        feats["group"] = group_label

        # UPDRS Part III total score
        available_items = [c for c in UPDRS3_ALL_ITEMS if c in df.columns]
        feats["updrs_iii_total"] = df[available_items].sum(axis=1, min_count=1)

        # Domain sub-scores
        feats["updrs_iii_tremor"] = df[[c for c in TREMOR_ITEMS if c in df.columns]].sum(
            axis=1, min_count=1
        )
        feats["updrs_iii_rigidity"] = df[[c for c in RIGIDITY_ITEMS if c in df.columns]].sum(
            axis=1, min_count=1
        )
        feats["updrs_iii_bradykinesia"] = df[
            [c for c in BRADYKINESIA_ITEMS if c in df.columns]
        ].sum(axis=1, min_count=1)
        feats["updrs_iii_gait_posture"] = df[
            [c for c in GAIT_POSTURE_ITEMS if c in df.columns]
        ].sum(axis=1, min_count=1)
        feats["updrs_iii_speech_facial"] = df[
            [c for c in SPEECH_FACIAL_ITEMS if c in df.columns]
        ].sum(axis=1, min_count=1)

        # Asymmetry indices
        for r_col, l_col, name in BILATERAL_PAIRS:
            if r_col in df.columns and l_col in df.columns:
                feats[f"asymmetry_{name}"] = _compute_asymmetry_index(
                    df[r_col], df[l_col]
                )

        # Hoehn & Yahr score
        feats["hoehn_yahr"] = df["Modified Hoehn & Yahr Score"].values

        # Disease duration (PD only, NaN for controls)
        if "Years since PD diagnosis" in df.columns:
            feats["disease_duration_years"] = df["Years since PD diagnosis"].values
        else:
            feats["disease_duration_years"] = np.nan

        # Demographics
        feats["age"] = df["Age (years)"].values
        feats["sex"] = df["Sex"].map({"Male": 1, "Female": 0}).values

        # BMI: weight_kg / (height_m)^2 ; height is in inches -> metres
        height_m = df["Height (in)"].astype(float, errors="ignore") * 0.0254
        weight_kg = df["Weight (kg)"].astype(float, errors="ignore")
        feats["bmi"] = weight_kg / (height_m ** 2)

        # Carry DBS column for PD labelling
        if "DBS?" in df.columns:
            feats["_dbs_raw"] = df["DBS?"].values
        else:
            feats["_dbs_raw"] = np.nan

        frames.append(feats)

    all_feats = pd.concat(frames, ignore_index=True)
    print(f"[WearGait-PD] Feature extraction complete: {all_feats.shape}")

    # ── 1e. Label engineering ───────────────────────────────────────────
    print("[WearGait-PD] Applying label engineering...")
    all_feats["dbs_candidate"] = np.nan
    all_feats["_exclude"] = False

    # Controls: dbs_candidate = 0
    ctrl_mask = all_feats["group"] == "Control"
    all_feats.loc[ctrl_mask, "dbs_candidate"] = 0

    # PD subjects
    pd_mask = all_feats["group"] == "PD"

    # DBS?=Yes -> dbs_candidate = 1
    dbs_yes = pd_mask & (all_feats["_dbs_raw"] == "Yes")
    all_feats.loc[dbs_yes, "dbs_candidate"] = 1

    # DBS?=No AND UPDRS_III_total < 32 -> dbs_candidate = 0
    dbs_no = pd_mask & (all_feats["_dbs_raw"] == "No")
    low_score = all_feats["updrs_iii_total"] < 32
    all_feats.loc[dbs_no & low_score, "dbs_candidate"] = 0

    # EXCLUDE: DBS?=No AND UPDRS_III_total >= 32 (ambiguous)
    all_feats.loc[dbs_no & ~low_score, "_exclude"] = True

    # EXCLUDE: DBS?=NaN (missing DBS status)
    dbs_missing = pd_mask & (all_feats["_dbs_raw"].isna())
    all_feats.loc[dbs_missing, "_exclude"] = True

    n_excluded = all_feats["_exclude"].sum()
    n_before = len(all_feats)
    all_feats = all_feats[~all_feats["_exclude"]].copy()
    n_after = len(all_feats)

    print(f"  Excluded {n_excluded} subjects ({n_before} -> {n_after})")

    # Add metadata columns
    all_feats["label_type"] = "real"
    all_feats["dataset"] = "weargait_pd"

    # Drop internal helper columns
    all_feats.drop(columns=["_dbs_raw", "_exclude", "group"], inplace=True,
                   errors="ignore")

    # ── 1f. Scaling with RobustScaler ───────────────────────────────────
    # Identify feature columns (exclude metadata/ID/label columns)
    meta_cols = ["subject_id", "dbs_candidate", "label_type", "dataset"]
    feature_cols = [c for c in all_feats.columns if c not in meta_cols]

    print(f"[WearGait-PD] Scaling {len(feature_cols)} features with RobustScaler...")
    scaler = RobustScaler()
    all_feats[feature_cols] = scaler.fit_transform(all_feats[feature_cols])

    # Save scaler
    scaler_path = OUT_DIR / "weargait_robust_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler saved to {scaler_path}")

    # ── 1g. Save ────────────────────────────────────────────────────────
    out_path = OUT_DIR / "weargait_features.csv"
    all_feats.to_csv(out_path, index=False)
    print(f"\n[WearGait-PD] Saved to {out_path}")

    # Summary
    n_total = len(all_feats)
    n_dbs_pos = int((all_feats["dbs_candidate"] == 1).sum())
    n_dbs_neg = int((all_feats["dbs_candidate"] == 0).sum())
    n_features = len(feature_cols)

    print(f"\n{'─' * 50}")
    print(f"  WEARGAIT-PD SUMMARY")
    print(f"{'─' * 50}")
    print(f"  Total subjects (after exclusion) : {n_total}")
    print(f"  DBS+ (candidate=1)               : {n_dbs_pos}")
    print(f"  DBS- (candidate=0)               : {n_dbs_neg}")
    print(f"  Excluded (ambiguous/missing)      : {n_excluded}")
    print(f"  Feature count                     : {n_features}")
    print(f"{'─' * 50}")

    return all_feats


# =====================================================================
# 2. GAITPDB DATASET
# =====================================================================

# Resolve the actual data directory (nested physionet mirror structure)
GAITPDB_DATA_DIR = (
    GAITPDB_DIR / "physionet.org" / "files" / "gaitpdb" / "1.0.0"
)

# Sampling rate for GaitPDB force plate data
GAITPDB_FS = 100  # Hz


def _extract_trial_features(filepath: Path) -> dict | None:
    """Extract gait features from a single GaitPDB force-plate trial file.

    The file is tab-separated with 19 columns (100 Hz):
        Col  0   : time (s)
        Cols 1-8 : left foot VGRF sensors (N)
        Cols 9-16: right foot VGRF sensors (N)
        Col  17  : total left force (N)
        Col  18  : total right force (N)

    Returns a dict of features keyed by name, or None on failure.
    """
    from scipy.signal import find_peaks, welch
    import antropy

    try:
        data = np.loadtxt(filepath, delimiter="\t")
    except Exception as exc:
        print(f"  [WARN] Could not load {filepath.name}: {exc}")
        return None

    if data.ndim != 2 or data.shape[1] < 19:
        return None

    time_col = data[:, 0]
    total_left = data[:, 17]
    total_right = data[:, 18]
    total_force = total_left + total_right

    # ── Heel-strike detection ─────────────────────────────────────────
    # Use find_peaks on total force per foot.  Minimum stride ~0.6 s at
    # 100 Hz => distance=60.  Height threshold = 20% of signal max to
    # reject noise during swing phase.
    min_dist = 60  # samples (~0.6 s)
    left_peaks, _ = find_peaks(
        total_left,
        height=max(np.max(total_left) * 0.20, 50),
        distance=min_dist,
    )
    right_peaks, _ = find_peaks(
        total_right,
        height=max(np.max(total_right) * 0.20, 50),
        distance=min_dist,
    )

    feats: dict = {}

    # ── Stride time (same-foot consecutive heel strikes) ──────────────
    left_stride_times = np.diff(time_col[left_peaks]) if len(left_peaks) > 1 else np.array([])
    right_stride_times = np.diff(time_col[right_peaks]) if len(right_peaks) > 1 else np.array([])
    all_stride_times = np.concatenate([left_stride_times, right_stride_times])

    if len(all_stride_times) > 1:
        feats["stride_time_mean"] = float(np.mean(all_stride_times))
        feats["stride_time_std"] = float(np.std(all_stride_times))
        feats["stride_time_cv"] = feats["stride_time_std"] / (feats["stride_time_mean"] + 1e-6)
    else:
        feats["stride_time_mean"] = np.nan
        feats["stride_time_std"] = np.nan
        feats["stride_time_cv"] = np.nan

    # ── Step asymmetry index ──────────────────────────────────────────
    mean_left_stride = np.mean(left_stride_times) if len(left_stride_times) > 0 else 0.0
    mean_right_stride = np.mean(right_stride_times) if len(right_stride_times) > 0 else 0.0
    feats["step_asymmetry"] = (
        abs(mean_left_stride - mean_right_stride)
        / (mean_left_stride + mean_right_stride + 1e-6)
    )

    # ── Force asymmetry ───────────────────────────────────────────────
    mean_left_force = float(np.mean(total_left))
    mean_right_force = float(np.mean(total_right))
    feats["force_asymmetry"] = (
        abs(mean_left_force - mean_right_force)
        / (mean_left_force + mean_right_force + 1e-6)
    )

    # ── Loading rate (max d/dt of total_force in first 20% of stance) ─
    # Compute across all detected stance phases (combined feet)
    all_peaks = np.sort(np.concatenate([left_peaks, right_peaks]))
    loading_rates = []
    for pk in all_peaks:
        # Stance onset: walk back from peak to where force < 10% of peak
        onset = pk
        thresh = total_force[pk] * 0.10
        while onset > 0 and total_force[onset] > thresh:
            onset -= 1
        stance_len = pk - onset
        if stance_len < 4:
            continue
        end_20pct = onset + max(int(stance_len * 0.20), 2)
        segment = total_force[onset:end_20pct]
        lr = np.max(np.diff(segment)) * GAITPDB_FS  # N/s
        loading_rates.append(lr)
    feats["loading_rate"] = float(np.mean(loading_rates)) if loading_rates else np.nan

    # ── Sample entropy of total force signal ──────────────────────────
    # Downsample to ~500 points for computational efficiency
    ds_factor = max(1, len(total_force) // 500)
    force_ds = total_force[::ds_factor].astype(np.float64)
    try:
        feats["sample_entropy"] = float(antropy.sample_entropy(force_ds))
    except Exception:
        feats["sample_entropy"] = np.nan

    # ── FOG index: power(0.5-3 Hz) / power(3-8 Hz) ───────────────────
    try:
        nperseg = min(256, len(total_force))
        freqs, psd = welch(total_force, fs=GAITPDB_FS, nperseg=nperseg)
        band_freeze = (freqs >= 0.5) & (freqs <= 3.0)
        band_locomotion = (freqs > 3.0) & (freqs <= 8.0)
        power_freeze = np.trapezoid(psd[band_freeze], freqs[band_freeze])
        power_locomotion = np.trapezoid(psd[band_locomotion], freqs[band_locomotion])
        feats["fog_index"] = power_freeze / (power_locomotion + 1e-6)
    except Exception:
        feats["fog_index"] = np.nan

    # ── Mean forces (useful as covariates) ────────────────────────────
    feats["mean_total_force"] = float(np.mean(total_force))

    return feats


def _process_single_subject(
    subject_id: str, trial_files: list[Path]
) -> dict | None:
    """Process all trial files for one subject and aggregate features.

    If multiple trials exist, also computes dual-task cost (DTC) for
    stride time between the first and second trials.
    """
    trial_results = []
    for fpath in sorted(trial_files):
        result = _extract_trial_features(fpath)
        if result is not None:
            trial_results.append(result)

    if not trial_results:
        return None

    # Aggregate across trials (mean of per-trial features)
    agg: dict = {"subject_id": subject_id}
    feature_keys = trial_results[0].keys()
    for key in feature_keys:
        vals = [t[key] for t in trial_results if not np.isnan(t.get(key, np.nan))]
        agg[key] = float(np.mean(vals)) if vals else np.nan

    # ── Dual-task cost ────────────────────────────────────────────────
    # If there are at least 2 trials with valid stride_time_mean, compute
    # DTC = (trial1 - trial2) / trial1 * 100  (single vs dual task proxy)
    stride_vals = [
        t["stride_time_mean"]
        for t in trial_results
        if not np.isnan(t.get("stride_time_mean", np.nan))
    ]
    if len(stride_vals) >= 2:
        single = stride_vals[0]
        dual = stride_vals[1]
        agg["dual_task_cost"] = (single - dual) / (single + 1e-6) * 100.0
    else:
        agg["dual_task_cost"] = np.nan

    agg["n_trials"] = len(trial_results)
    return agg


def process_gaitpdb() -> pd.DataFrame:
    """End-to-end GaitPDB processing pipeline.

    Steps:
    1. Load demographics and discover per-subject trial files
    2. Extract gait features in parallel (stride time, asymmetry, loading
       rate, sample entropy, FOG index, dual-task cost)
    3. Merge with demographics for labelling
    4. Proxy label: dbs_candidate=1 if H&Y >= 2.5 OR UPDRS >= 32
    5. Scale with RobustScaler, save to gaitpdb_features.csv
    """
    print("\n" + "=" * 70)
    print("[GaitPDB] Processing...")
    print("=" * 70)

    if not GAITPDB_DATA_DIR.exists():
        print(f"[GaitPDB] Data directory not found: {GAITPDB_DATA_DIR}")
        return pd.DataFrame()

    # ── Load demographics ─────────────────────────────────────────────
    demo_path = GAITPDB_DATA_DIR / "demographics.txt"
    if not demo_path.exists():
        print(f"[GaitPDB] demographics.txt not found at {demo_path}")
        return pd.DataFrame()

    # Demographics file has variable column counts across studies (Ju has
    # extra Speed columns).  Read with csv module, then take only the core
    # 20 columns that are consistent across all studies.
    import csv as _csv

    with open(demo_path, "r", encoding="latin-1") as fh:
        reader = _csv.reader(fh, delimiter="\t")
        header = next(reader)
        rows = list(reader)

    # Core columns: first 20 (ID through Speed_10)
    n_core = 20
    core_header = header[:n_core]
    core_rows = [row[:n_core] for row in rows if len(row) >= 1 and row[0].strip()]

    demo = pd.DataFrame(core_rows, columns=core_header)
    demo = demo.replace({"NaN": np.nan, "": np.nan})

    # Convert numeric columns from strings
    for col in ["Group", "Subjnum", "Gender", "Age", "Height", "Weight",
                "HoehnYahr", "UPDRS", "UPDRSM", "TUAG"]:
        if col in demo.columns:
            demo[col] = pd.to_numeric(demo[col], errors="coerce")

    # Drop empty rows
    demo = demo.dropna(subset=["ID"]).reset_index(drop=True)
    demo["ID"] = demo["ID"].astype(str).str.strip()
    demo = demo[demo["ID"].str.len() > 0].reset_index(drop=True)

    print(f"  Demographics loaded: {len(demo)} subjects")
    print(f"    PD (Group=1) : {(demo['Group'] == 1).sum()}")
    print(f"    Control (Group=2): {(demo['Group'] == 2).sum()}")

    # ── Discover trial files per subject ──────────────────────────────
    import re as _re

    all_txt = sorted(GAITPDB_DATA_DIR.iterdir())
    subject_files: dict[str, list[Path]] = {}
    trial_pattern = _re.compile(r"^([A-Za-z]+\d+)_(\d+)\.txt$")

    for fpath in all_txt:
        m = trial_pattern.match(fpath.name)
        if m:
            sid = m.group(1)
            subject_files.setdefault(sid, []).append(fpath)

    print(f"  Found {len(subject_files)} subjects with "
          f"{sum(len(v) for v in subject_files.values())} trial files")

    # ── Parallel feature extraction ───────────────────────────────────
    print("[GaitPDB] Extracting features (parallel)...")
    results = joblib.Parallel(n_jobs=N_JOBS, verbose=5)(
        joblib.delayed(_process_single_subject)(sid, files)
        for sid, files in subject_files.items()
    )

    # Filter out failed subjects
    results = [r for r in results if r is not None]
    if not results:
        print("[GaitPDB] No features extracted, aborting.")
        return pd.DataFrame()

    features_df = pd.DataFrame(results)
    print(f"[GaitPDB] Features extracted for {len(features_df)} subjects")

    # ── Merge with demographics ───────────────────────────────────────
    merged = features_df.merge(demo, left_on="subject_id", right_on="ID", how="left")

    # ── Label engineering ─────────────────────────────────────────────
    print("[GaitPDB] Applying proxy label engineering...")

    # Convert to numeric for labelling
    merged["HoehnYahr"] = pd.to_numeric(merged["HoehnYahr"], errors="coerce")
    merged["UPDRS"] = pd.to_numeric(merged["UPDRS"], errors="coerce")
    merged["Group"] = pd.to_numeric(merged["Group"], errors="coerce")

    # dbs_candidate = 1 if H&Y >= 2.5 OR UPDRS >= 32 (PD patients)
    # dbs_candidate = 0 for controls and mild PD
    merged["dbs_candidate"] = 0
    is_pd = merged["Group"] == 1
    hy_severe = merged["HoehnYahr"] >= 2.5
    updrs_severe = merged["UPDRS"] >= 32
    merged.loc[is_pd & (hy_severe | updrs_severe), "dbs_candidate"] = 1

    merged["label_type"] = "proxy"
    merged["dataset"] = "gaitpdb"

    # ── Prepare final feature DataFrame ───────────────────────────────
    # Rename demographics columns for consistency
    rename_map = {
        "Age": "age",
        "Gender": "sex",       # 1=male, 2=female in GaitPDB
        "Height": "height_m",
        "Weight": "weight_kg",
        "HoehnYahr": "hoehn_yahr",
        "UPDRS": "updrs_total",
    }
    merged.rename(columns=rename_map, inplace=True)

    # Recode sex: GaitPDB uses 1=Male, 2=Female -> 1=Male, 0=Female
    merged["sex"] = merged["sex"].map({1: 1, 2: 0})

    # BMI
    height_m = pd.to_numeric(merged["height_m"], errors="coerce")
    weight_kg = pd.to_numeric(merged["weight_kg"], errors="coerce")
    merged["bmi"] = weight_kg / (height_m ** 2)

    # Select and order columns
    meta_cols = ["subject_id", "dbs_candidate", "label_type", "dataset"]
    feature_cols = [
        "stride_time_mean", "stride_time_std", "stride_time_cv",
        "step_asymmetry", "force_asymmetry", "loading_rate",
        "sample_entropy", "fog_index", "mean_total_force",
        "dual_task_cost", "n_trials",
        "age", "sex", "height_m", "weight_kg", "bmi",
        "hoehn_yahr", "updrs_total",
    ]

    # Keep only columns that exist
    feature_cols = [c for c in feature_cols if c in merged.columns]
    final_cols = meta_cols + feature_cols
    final_df = merged[final_cols].copy()

    n_before = len(final_df)
    # Drop rows where all gait features are NaN
    gait_feat_cols = [
        c for c in feature_cols
        if c not in ("age", "sex", "height_m", "weight_kg", "bmi",
                      "hoehn_yahr", "updrs_total", "n_trials")
    ]
    final_df = final_df.dropna(subset=gait_feat_cols, how="all").reset_index(drop=True)
    print(f"  Dropped {n_before - len(final_df)} subjects with no valid gait features")

    # ── Scaling with RobustScaler ─────────────────────────────────────
    scale_cols = [c for c in feature_cols if c not in meta_cols]
    print(f"[GaitPDB] Scaling {len(scale_cols)} features with RobustScaler...")

    scaler = RobustScaler()
    # Only scale non-NaN; RobustScaler handles this per-column
    final_df[scale_cols] = scaler.fit_transform(final_df[scale_cols])

    scaler_path = OUT_DIR / "gaitpdb_robust_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler saved to {scaler_path}")

    # ── Save ──────────────────────────────────────────────────────────
    out_path = OUT_DIR / "gaitpdb_features.csv"
    final_df.to_csv(out_path, index=False)
    print(f"\n[GaitPDB] Saved to {out_path}")

    # ── Summary ───────────────────────────────────────────────────────
    n_total = len(final_df)
    n_dbs_pos = int((final_df["dbs_candidate"] == 1).sum())
    n_dbs_neg = int((final_df["dbs_candidate"] == 0).sum())

    print(f"\n{'─' * 50}")
    print(f"  GAITPDB SUMMARY")
    print(f"{'─' * 50}")
    print(f"  Total subjects                   : {n_total}")
    print(f"  DBS+ (candidate=1, proxy)        : {n_dbs_pos}")
    print(f"  DBS- (candidate=0)               : {n_dbs_neg}")
    print(f"  Feature count                    : {len(scale_cols)}")
    print(f"{'─' * 50}")

    return final_df


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  GAIT PREPROCESSING PIPELINE")
    print("=" * 70)

    # 1. WearGait-PD (real DBS labels)
    weargait_df = process_weargait_pd()

    # 2. GaitPDB (proxy labels from force plate data)
    gaitpdb_df = process_gaitpdb()

    print("\n" + "=" * 70)
    print("  GAIT PREPROCESSING COMPLETE")
    print("=" * 70)
