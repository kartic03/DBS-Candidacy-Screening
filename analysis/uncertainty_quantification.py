#!/usr/bin/env python3
"""
Uncertainty Quantification via Monte Carlo Dropout (DL) and Bootstrap (XGBoost).

For each test patient, computes:
  - Mean prediction, std (uncertainty), 95% confidence interval
  - Confidence classification: HIGH / MEDIUM / LOW
  - Correlation between uncertainty and misclassification

Outputs:
  results/tables/uncertainty_analysis.csv
  results/figures/uncertainty_analysis.png

JBI DBS Screening Project | Conda env: jbi_dbs | Config: ../config.yaml
"""

# ── Hardware init (MUST be at top) ──────────────────────────────────────────
import os, multiprocessing

N_CORES = multiprocessing.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(N_CORES)
os.environ["MKL_NUM_THREADS"] = str(N_CORES)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_CORES)

# ── GPU setup ───────────────────────────────────────────────────────────────
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# ── Standard imports ────────────────────────────────────────────────────────
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb

warnings.filterwarnings("ignore", category=UserWarning)

# ── Project root & model imports ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.fusion_model import CrossAttentionFusionModel
from models.wearable_encoder import WearableResidualMLP
from models.voice_encoder import VoiceEncoder
from models.gait_encoder import GaitEncoder

# ── Config ──────────────────────────────────────────────────────────────────
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

SEED = CONFIG["model"]["seed"]  # 42
N_FOLDS = CONFIG["training"]["n_folds"]  # 5
FONT_FAMILY = CONFIG["figures"]["font_family"]
DPI = CONFIG["figures"]["dpi"]
N_MC_SAMPLES = 50

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Paths ───────────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "fused"
SPLITS_PATH = PROJECT_ROOT / "data" / "splits" / "primary_splits.json"
CHECKPOINT_DIR = PROJECT_ROOT / "results" / "checkpoints"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Encoder dimensions (from checkpoints) ──────────────────────────────────
WEARABLE_DIM = 4361
VOICE_DIM = 22
GAIT_DIM = 20

# ── Font setup ──────────────────────────────────────────────────────────────
plt.rcParams["font.family"] = FONT_FAMILY
plt.rcParams["font.size"] = CONFIG["figures"]["font_size"]


# ============================================================================
# 1. Data loading
# ============================================================================
def _feature_cols(df: pd.DataFrame):
    """Return feature columns (exclude metadata)."""
    meta = {"subject_id", "dbs_candidate", "label_type", "dataset",
            "condition", "updrs_iii", "hy_stage", "pd_status"}
    cols = [c for c in df.columns if c not in meta]
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


def load_data():
    """Load primary cohort and splits."""
    df = pd.read_csv(DATA_DIR / "primary_cohort.csv")
    with open(SPLITS_PATH, "r") as f:
        splits = json.load(f)
    return df, splits


# ============================================================================
# 2. Build and load DL model
# ============================================================================
def build_fusion_model():
    """Build a CrossAttentionFusionModel with correct encoder dims."""
    w_enc = WearableResidualMLP(input_dim=WEARABLE_DIM, classify=False)
    v_enc = VoiceEncoder(input_dim=VOICE_DIM, classify=False)
    g_enc = GaitEncoder(input_dim=GAIT_DIM, classify=False)
    return CrossAttentionFusionModel(
        wearable_encoder=w_enc,
        voice_encoder=v_enc,
        gait_encoder=g_enc,
    )


def load_fold_model(fold: int):
    """Load a single fold checkpoint."""
    ckpt_path = CHECKPOINT_DIR / f"fusion_model_fold{fold}.pt"
    model = build_fusion_model()
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device)
    return model


# ============================================================================
# 3. MC Dropout inference
# ============================================================================
def mc_dropout_inference(model, w_t, v_t, g_t, n_samples=N_MC_SAMPLES):
    """Run MC Dropout: model.train() mode with torch.no_grad().

    Returns array of shape (n_samples, n_patients) with P(class=1).
    """
    model.train()  # Enable dropout
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = model(w_t, v_t, g_t)
            probs = out["probabilities"][:, 1].cpu().numpy()
            preds.append(probs)
    return np.array(preds)  # (n_samples, n_patients)


def pad_features(X, target_dim):
    """Zero-pad features to match encoder input dim."""
    n = X.shape[0]
    if X.shape[1] < target_dim:
        padded = np.zeros((n, target_dim), dtype=np.float32)
        padded[:, :X.shape[1]] = X
        return padded
    return X


# ============================================================================
# 4. XGBoost bootstrap uncertainty
# ============================================================================
def xgb_bootstrap_uncertainty(X_train, y_train, X_test, n_bootstraps=N_MC_SAMPLES):
    """Train XGBoost on bootstrap resamples and collect predictions.

    Returns array of shape (n_bootstraps, n_test_patients).
    """
    preds = []
    rng = np.random.RandomState(SEED)

    for i in range(n_bootstraps):
        # Bootstrap resample training data
        idx = rng.choice(len(X_train), size=len(X_train), replace=True)
        X_boot = X_train[idx]
        y_boot = y_train[idx]

        # Skip if only one class present
        if len(np.unique(y_boot)) < 2:
            continue

        # Apply SMOTE
        try:
            smote = SMOTE(random_state=SEED + i, k_neighbors=min(3, sum(y_boot == 1) - 1))
            X_sm, y_sm = smote.fit_resample(X_boot, y_boot)
        except Exception:
            X_sm, y_sm = X_boot, y_boot

        clf = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED + i,
            n_jobs=CONFIG["training"]["xgb_n_jobs"],
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )
        clf.fit(X_sm, y_sm)
        prob = clf.predict_proba(X_test)[:, 1]
        preds.append(prob)

    return np.array(preds)


# ============================================================================
# 5. Classify confidence
# ============================================================================
def classify_confidence(std):
    """Classify prediction confidence based on std."""
    if std < 0.05:
        return "HIGH"
    elif std < 0.15:
        return "MEDIUM"
    else:
        return "LOW"


# ============================================================================
# 6. Main analysis
# ============================================================================
def run_uncertainty_analysis():
    """Run full uncertainty quantification analysis."""
    print("=" * 70)
    print("UNCERTAINTY QUANTIFICATION ANALYSIS")
    print("=" * 70)

    # Load data
    df, splits = load_data()
    feat_cols = _feature_cols(df)
    test_idx = splits["test_indices"]
    cv_folds = splits["cv_folds"]

    # All training indices
    all_train_idx = sorted(set(range(len(df))) - set(test_idx))

    # Extract features
    X_all = df[feat_cols].values.astype(np.float32)
    y_all = df["dbs_candidate"].values.astype(np.int64)
    subject_ids = df["subject_id"].values

    X_test = X_all[test_idx]
    y_test = y_all[test_idx]
    test_subjects = subject_ids[test_idx]

    X_train = X_all[all_train_idx]
    y_train = y_all[all_train_idx]

    n_test = len(test_idx)
    print(f"\n[DATA] Primary cohort: {len(df)} patients, {len(feat_cols)} features")
    print(f"[DATA] Train: {len(all_train_idx)}, Test: {n_test}")
    print(f"[DATA] Test class distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")

    # ------------------------------------------------------------------
    # DL: MC Dropout across 5-fold ensemble
    # ------------------------------------------------------------------
    print(f"\n[DL] Running MC Dropout ({N_MC_SAMPLES} samples per fold, {N_FOLDS} folds)...")

    # Prepare tensors: all 20 features go to gait encoder
    # Wearable and voice get zero-padded
    X_test_gait = pad_features(X_test, GAIT_DIM)
    X_test_wearable = np.zeros((n_test, WEARABLE_DIM), dtype=np.float32)
    X_test_voice = np.zeros((n_test, VOICE_DIM), dtype=np.float32)

    # Put all 20 features into gait (already correct size)
    X_test_gait = X_test.copy()  # 20 features = GAIT_DIM

    w_t = torch.as_tensor(X_test_wearable, dtype=torch.float32, device=device)
    v_t = torch.as_tensor(X_test_voice, dtype=torch.float32, device=device)
    g_t = torch.as_tensor(X_test_gait, dtype=torch.float32, device=device)

    # Collect all MC samples across all folds
    all_mc_preds = []  # Will be (n_folds * n_mc_samples, n_test)

    for fold in range(N_FOLDS):
        ckpt_path = CHECKPOINT_DIR / f"fusion_model_fold{fold}.pt"
        if not ckpt_path.exists():
            print(f"  [WARN] Checkpoint fold{fold} not found, skipping")
            continue

        model = load_fold_model(fold)
        mc_preds = mc_dropout_inference(model, w_t, v_t, g_t, N_MC_SAMPLES)
        all_mc_preds.append(mc_preds)
        print(f"  [DL] Fold {fold}: {mc_preds.shape[0]} MC samples collected")

    if all_mc_preds:
        dl_all_preds = np.concatenate(all_mc_preds, axis=0)  # (n_folds*50, n_test)
    else:
        print("[ERROR] No DL checkpoints found!")
        dl_all_preds = np.zeros((1, n_test))

    dl_mean = dl_all_preds.mean(axis=0)
    dl_std = dl_all_preds.std(axis=0)
    dl_ci_low = np.percentile(dl_all_preds, 2.5, axis=0)
    dl_ci_high = np.percentile(dl_all_preds, 97.5, axis=0)

    print(f"\n[DL] MC Dropout predictions collected: {dl_all_preds.shape[0]} total samples")
    print(f"[DL] Mean uncertainty (std): {dl_std.mean():.4f}")

    # ------------------------------------------------------------------
    # XGBoost: Bootstrap uncertainty
    # ------------------------------------------------------------------
    print(f"\n[XGB] Running bootstrap uncertainty ({N_MC_SAMPLES} resamples)...")

    xgb_preds = xgb_bootstrap_uncertainty(X_train, y_train, X_test, N_MC_SAMPLES)

    xgb_mean = xgb_preds.mean(axis=0)
    xgb_std = xgb_preds.std(axis=0)
    xgb_ci_low = np.percentile(xgb_preds, 2.5, axis=0)
    xgb_ci_high = np.percentile(xgb_preds, 97.5, axis=0)

    print(f"[XGB] Bootstrap predictions collected: {xgb_preds.shape[0]} resamples")
    print(f"[XGB] Mean uncertainty (std): {xgb_std.mean():.4f}")

    # ------------------------------------------------------------------
    # Build results dataframe
    # ------------------------------------------------------------------
    results = []
    for i in range(n_test):
        # DL results
        dl_conf = classify_confidence(dl_std[i])
        dl_pred_label = int(dl_mean[i] >= 0.5)
        dl_correct = int(dl_pred_label == y_test[i])

        results.append({
            "subject_id": test_subjects[i],
            "true_label": int(y_test[i]),
            "model": "CrossAttentionFusion_MCDropout",
            "mean_pred": round(float(dl_mean[i]), 4),
            "std_pred": round(float(dl_std[i]), 4),
            "ci_95_low": round(float(dl_ci_low[i]), 4),
            "ci_95_high": round(float(dl_ci_high[i]), 4),
            "confidence_level": dl_conf,
            "correct": dl_correct,
        })

        # XGBoost results
        xgb_conf = classify_confidence(xgb_std[i])
        xgb_pred_label = int(xgb_mean[i] >= 0.5)
        xgb_correct = int(xgb_pred_label == y_test[i])

        results.append({
            "subject_id": test_subjects[i],
            "true_label": int(y_test[i]),
            "model": "XGBoost_Bootstrap",
            "mean_pred": round(float(xgb_mean[i]), 4),
            "std_pred": round(float(xgb_std[i]), 4),
            "ci_95_low": round(float(xgb_ci_low[i]), 4),
            "ci_95_high": round(float(xgb_ci_high[i]), 4),
            "confidence_level": xgb_conf,
            "correct": xgb_correct,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(TABLES_DIR / "uncertainty_analysis.csv", index=False)
    print(f"\n[SAVE] Results saved to {TABLES_DIR / 'uncertainty_analysis.csv'}")

    # ------------------------------------------------------------------
    # Summary: accuracy by confidence zone
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY: Accuracy by Confidence Zone")
    print("=" * 70)

    for model_name in ["CrossAttentionFusion_MCDropout", "XGBoost_Bootstrap"]:
        print(f"\n--- {model_name} ---")
        model_df = results_df[results_df["model"] == model_name]

        for level in ["HIGH", "MEDIUM", "LOW"]:
            zone = model_df[model_df["confidence_level"] == level]
            if len(zone) > 0:
                acc = zone["correct"].mean()
                print(f"  {level:6s} confidence: {len(zone):3d} patients, accuracy = {acc:.4f}")
            else:
                print(f"  {level:6s} confidence:   0 patients")

        # Overall
        overall_acc = model_df["correct"].mean()
        print(f"  {'OVERALL':6s}            : {len(model_df):3d} patients, accuracy = {overall_acc:.4f}")

        # Correlation: uncertainty vs misclassification
        miscls = 1 - model_df["correct"].values
        stds = model_df["std_pred"].values
        if stds.std() > 0 and miscls.std() > 0:
            from scipy.stats import pointbiserialr
            corr, pval = pointbiserialr(miscls, stds)
            print(f"  Uncertainty-Misclassification correlation: r={corr:.4f}, p={pval:.4f}")
        else:
            print(f"  Uncertainty-Misclassification correlation: insufficient variance")

    # ------------------------------------------------------------------
    # Plot: prediction probability vs uncertainty
    # ------------------------------------------------------------------
    print("\n[PLOT] Generating uncertainty analysis figure...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (model_name, short_name, mean_arr, std_arr) in zip(
        axes,
        [
            ("CrossAttentionFusion_MCDropout", "Cross-Attention Fusion\n(MC Dropout)", dl_mean, dl_std),
            ("XGBoost_Bootstrap", "XGBoost\n(Bootstrap)", xgb_mean, xgb_std),
        ],
    ):
        # Color by true label
        colors = ["#2ca02c" if y == 0 else "#d62728" for y in y_test]

        ax.scatter(mean_arr, std_arr, c=colors, alpha=0.7, edgecolors="black",
                   linewidths=0.5, s=80, zorder=3)

        # Confidence bands
        ax.axhspan(0, 0.05, alpha=0.1, color="green", label="HIGH confidence (std < 0.05)")
        ax.axhspan(0.05, 0.15, alpha=0.1, color="orange", label="MEDIUM confidence")
        ax.axhspan(0.15, ax.get_ylim()[1] if ax.get_ylim()[1] > 0.15 else 0.5,
                    alpha=0.1, color="red", label="LOW confidence (std >= 0.15)")

        # Fix y-axis limits after drawing bands
        ymax = max(std_arr.max() * 1.2, 0.2)
        ax.set_ylim(0, ymax)

        # Re-draw bands with correct limits
        ax.clear()
        ax.axhspan(0, 0.05, alpha=0.1, color="green", label="HIGH confidence (std < 0.05)")
        ax.axhspan(0.05, 0.15, alpha=0.1, color="orange", label="MEDIUM confidence")
        ax.axhspan(0.15, ymax, alpha=0.1, color="red", label="LOW confidence (std >= 0.15)")
        ax.axhline(y=0.05, color="green", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.axhline(y=0.15, color="red", linewidth=0.8, linestyle="--", alpha=0.5)

        # Scatter
        for label_val, label_name, color in [(0, "Non-DBS", "#2ca02c"), (1, "DBS Candidate", "#d62728")]:
            mask = y_test == label_val
            ax.scatter(mean_arr[mask], std_arr[mask], c=color, alpha=0.7, edgecolors="black",
                       linewidths=0.5, s=80, zorder=3, label=label_name)

        ax.set_xlabel("Mean Predicted Probability", fontsize=11)
        ax.set_ylabel("Prediction Uncertainty (Std)", fontsize=11)
        ax.set_title(short_name, fontsize=12, fontweight="bold")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0, ymax)
        ax.axvline(x=0.5, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Uncertainty Quantification: Prediction Probability vs. Uncertainty",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "uncertainty_analysis.png", dpi=DPI,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVE] Figure saved to {FIGURES_DIR / 'uncertainty_analysis.png'}")

    print("\n[DONE] Uncertainty quantification analysis complete.")
    return results_df


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    run_uncertainty_analysis()
