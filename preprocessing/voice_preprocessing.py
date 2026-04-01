"""
Voice data preprocessing for UCI Parkinsons Voice and UCI UPDRS datasets.
Produces standardised feature CSVs for downstream modelling.
"""

# ── Hardware init (must be before any numpy/sklearn import) ──────────────
import os, multiprocessing

N_CORES = multiprocessing.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(N_CORES)
os.environ["MKL_NUM_THREADS"] = str(N_CORES)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_CORES)

# ── Standard imports ─────────────────────────────────────────────────────
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import skew
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

UCI_VOICE_PATH = PROJECT_ROOT / "data" / "raw" / "uci_voice" / "parkinsons.data"
UCI_UPDRS_PATH = PROJECT_ROOT / "data" / "raw" / "uci_updrs" / "parkinsons_updrs.data"

OUT_DIR = PROJECT_ROOT / "data" / "processed" / "voice_features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ───────────────────────────────────────────────────────────────
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

SEED = cfg["model"]["seed"]  # 42
N_JOBS = cfg["training"]["n_jobs"]  # -1
np.random.seed(SEED)

print(f"[voice_preprocessing] Project root : {PROJECT_ROOT}")
print(f"[voice_preprocessing] Seed={SEED}  N_JOBS={N_JOBS}  CPU cores={N_CORES}")


# =====================================================================
# 1. UCI VOICE DATASET
# =====================================================================
def process_uci_voice() -> pd.DataFrame:
    """Load, transform, and label the UCI Parkinsons Voice dataset."""
    print("\n" + "=" * 60)
    print("[UCI Voice] Loading data ...")
    df = pd.read_csv(UCI_VOICE_PATH)
    print(f"[UCI Voice] Raw shape: {df.shape}")

    # ── Extract subject_id from name (first 3 underscore-separated parts) ─
    df["subject_id"] = df["name"].apply(lambda x: "_".join(x.split("_")[:3]))
    print(f"[UCI Voice] Unique subjects: {df['subject_id'].nunique()}")

    # ── Drop name column ──────────────────────────────────────────────────
    df = df.drop(columns=["name"])

    # ── Identify voice feature columns (everything except status, subject_id)
    voice_cols = [c for c in df.columns if c not in ("status", "subject_id")]
    print(f"[UCI Voice] Voice feature columns ({len(voice_cols)}): {voice_cols[:5]} ...")

    # ── Log1p transform skewed features (skewness > 1.0) ─────────────────
    skewed = []
    for col in voice_cols:
        s = skew(df[col].dropna())
        if abs(s) > 1.0:
            skewed.append(col)
            df[col] = np.log1p(df[col])
    print(f"[UCI Voice] Log1p-transformed {len(skewed)} skewed features: {skewed}")

    # ── Rename label and add metadata ─────────────────────────────────────
    df = df.rename(columns={"status": "pd_status"})
    df["label_type"] = "proxy"
    df["dataset"] = "uci_voice"

    print(f"[UCI Voice] Label distribution:\n{df['pd_status'].value_counts().to_string()}")
    print(f"[UCI Voice] Final shape: {df.shape}")
    return df


# =====================================================================
# 2. UCI UPDRS DATASET
# =====================================================================
def process_uci_updrs() -> pd.DataFrame:
    """Load, aggregate per-subject, compute UPDRS slope, and label."""
    print("\n" + "=" * 60)
    print("[UCI UPDRS] Loading data ...")
    df = pd.read_csv(UCI_UPDRS_PATH)
    print(f"[UCI UPDRS] Raw shape: {df.shape}")
    print(f"[UCI UPDRS] Unique subjects: {df['subject#'].nunique()}")

    # ── Identify voice feature columns ────────────────────────────────────
    meta_cols = ["subject#", "age", "sex", "test_time", "motor_UPDRS", "total_UPDRS"]
    voice_cols = [c for c in df.columns if c not in meta_cols]
    print(f"[UCI UPDRS] Voice feature columns ({len(voice_cols)}): {voice_cols[:5]} ...")

    # ── Per-subject: compute updrs_slope via LinearRegression ─────────────
    def _updrs_slope(grp: pd.DataFrame) -> float:
        X = grp["test_time"].values.reshape(-1, 1)
        y = grp["motor_UPDRS"].values
        if len(X) < 2:
            return 0.0
        model = LinearRegression().fit(X, y)
        return float(model.coef_[0])

    slopes = (
        df.groupby("subject#")
        .apply(_updrs_slope, include_groups=False)
        .rename("updrs_slope")
    )

    # ── Per-subject: aggregate voice features (mean + std) ────────────────
    agg_dict = {}
    for col in voice_cols:
        agg_dict[col] = ["mean", "std"]

    voice_agg = df.groupby("subject#")[voice_cols].agg(agg_dict)
    # Flatten multi-level columns
    voice_agg.columns = [f"{col}_{stat}" for col, stat in voice_agg.columns]
    voice_agg = voice_agg.reset_index()

    # ── Per-subject: demographics + mean UPDRS ────────────────────────────
    demo = (
        df.groupby("subject#")
        .agg(
            age=("age", "first"),
            sex=("sex", "first"),
            motor_UPDRS_mean=("motor_UPDRS", "mean"),
            total_UPDRS_mean=("total_UPDRS", "mean"),
        )
        .reset_index()
    )

    # ── Merge everything ──────────────────────────────────────────────────
    result = demo.merge(voice_agg, on="subject#").merge(
        slopes.reset_index(), on="subject#"
    )

    # ── Label: DBS candidate if motor_UPDRS_mean >= 32 ────────────────────
    result["dbs_candidate"] = (result["motor_UPDRS_mean"] >= 32).astype(int)

    # ── Rename subject column for consistency ─────────────────────────────
    result = result.rename(columns={"subject#": "subject_id"})

    # ── Add metadata ──────────────────────────────────────────────────────
    result["label_type"] = "proxy"
    result["dataset"] = "uci_updrs"

    print(f"[UCI UPDRS] DBS candidate distribution:\n{result['dbs_candidate'].value_counts().to_string()}")
    print(f"[UCI UPDRS] UPDRS slope range: [{slopes.min():.4f}, {slopes.max():.4f}]")
    print(f"[UCI UPDRS] Final shape: {result.shape}")
    return result


# =====================================================================
# 3. SCALING & SAVE
# =====================================================================
def scale_and_save(df: pd.DataFrame, name: str, out_path: Path) -> None:
    """StandardScaler fit on data, save scaler + CSV."""
    # Identify numeric feature columns (exclude metadata / label cols)
    exclude = {
        "subject_id", "pd_status", "dbs_candidate",
        "label_type", "dataset",
        "motor_UPDRS_mean", "total_UPDRS_mean",
    }
    feat_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    scaler = StandardScaler()
    df_out = df.copy()
    df_out[feat_cols] = scaler.fit_transform(df_out[feat_cols])

    # Save scaler
    scaler_path = OUT_DIR / f"{name}_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"[{name}] Scaler saved -> {scaler_path}")

    # Save CSV
    df_out.to_csv(out_path, index=False)
    print(f"[{name}] Features saved -> {out_path}  ({df_out.shape})")


# =====================================================================
# MAIN
# =====================================================================
def main() -> None:
    # ── UCI Voice ─────────────────────────────────────────────────────────
    df_voice = process_uci_voice()
    voice_out = OUT_DIR / "uci_voice_features.csv"
    scale_and_save(df_voice, "uci_voice", voice_out)

    # ── UCI UPDRS ─────────────────────────────────────────────────────────
    df_updrs = process_uci_updrs()
    updrs_out = OUT_DIR / "uci_updrs_features.csv"
    scale_and_save(df_updrs, "uci_updrs", updrs_out)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[SUMMARY]")
    print(f"  UCI Voice : {df_voice.shape[0]} samples, {df_voice.shape[1]} columns")
    print(f"    PD=1: {(df_voice['pd_status'] == 1).sum()}  Healthy=0: {(df_voice['pd_status'] == 0).sum()}")
    print(f"  UCI UPDRS : {df_updrs.shape[0]} subjects, {df_updrs.shape[1]} columns")
    print(f"    DBS candidates: {(df_updrs['dbs_candidate'] == 1).sum()}  Non-candidates: {(df_updrs['dbs_candidate'] == 0).sum()}")
    print(f"  Output dir: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
