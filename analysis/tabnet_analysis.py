#!/usr/bin/env python3
"""
TabNet Training, Evaluation, and Attention Analysis.

1. Train TabNet (pytorch_tabnet) on primary cohort with 5-fold CV + SMOTE
2. Extract attention masks per patient
3. Compare TabNet attention feature ranking vs SHAP top-10 ranking (Spearman)
4. Report AUC, F1, MCC per fold + mean

Outputs:
  results/tables/tabnet_results.csv
  results/tables/tabnet_vs_shap_concordance.csv
  results/figures/tabnet_attention_heatmap.png
  results/figures/tabnet_vs_shap_comparison.png

JBI DBS Screening Project | Conda env: jbi_dbs | Config: ../config.yaml
"""

# ── Hardware init (MUST be at top) ──────────────────────────────────────────
import os, multiprocessing

N_CORES = multiprocessing.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(N_CORES)
os.environ["MKL_NUM_THREADS"] = str(N_CORES)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_CORES)

# ── Imports ─────────────────────────────────────────────────────────────────
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from scipy.stats import spearmanr
import xgboost as xgb
import shap

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Project root & model imports ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.tabnet_model import build_tabnet

# ── Config ──────────────────────────────────────────────────────────────────
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

SEED = CONFIG["model"]["seed"]  # 42
N_FOLDS = CONFIG["training"]["n_folds"]  # 5
FONT_FAMILY = CONFIG["figures"]["font_family"]
DPI = CONFIG["figures"]["dpi"]

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Paths ───────────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "fused"
SPLITS_PATH = PROJECT_ROOT / "data" / "splits" / "primary_splits.json"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

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
# 2. SHAP feature ranking (from XGBoost, for comparison)
# ============================================================================
def compute_shap_ranking(X_train, y_train, X_test, feature_names):
    """Train XGBoost on full training set and compute SHAP feature ranking."""
    print("  [SHAP] Training XGBoost for SHAP comparison...")

    # Apply SMOTE
    try:
        smote = SMOTE(random_state=SEED, k_neighbors=min(5, sum(y_train == 1) - 1))
        X_sm, y_sm = smote.fit_resample(X_train, y_train)
    except Exception:
        X_sm, y_sm = X_train, y_train

    clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=CONFIG["training"]["xgb_n_jobs"],
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0,
    )
    clf.fit(X_sm, y_sm)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)

    # For binary, shap_values might be list
    if isinstance(shap_values, list):
        sv = shap_values[1]  # Class 1
    else:
        sv = shap_values

    mean_abs_shap = np.abs(sv).mean(axis=0)
    ranking = np.argsort(-mean_abs_shap)  # Descending
    return mean_abs_shap, ranking


# ============================================================================
# 3. Main analysis
# ============================================================================
def run_tabnet_analysis():
    """Run full TabNet training, evaluation, and attention analysis."""
    print("=" * 70)
    print("TABNET ANALYSIS")
    print("=" * 70)

    # Load data
    df, splits = load_data()
    feat_cols = _feature_cols(df)
    test_idx = splits["test_indices"]
    cv_folds = splits["cv_folds"]

    X_all = df[feat_cols].values.astype(np.float32)
    y_all = df["dbs_candidate"].values.astype(np.int64)
    subject_ids = df["subject_id"].values

    n_features = len(feat_cols)
    print(f"\n[DATA] Primary cohort: {len(df)} patients, {n_features} features")
    print(f"[DATA] Test set: {len(test_idx)} patients")
    print(f"[DATA] Features: {feat_cols}")

    # ------------------------------------------------------------------
    # 5-fold CV with SMOTE
    # ------------------------------------------------------------------
    fold_results = []
    all_test_attention = []  # Collect attention masks for test patients
    all_fold_models = []

    for fold_idx, fold in enumerate(cv_folds):
        print(f"\n--- Fold {fold_idx} ---")
        train_idx = fold["train"]
        val_idx = fold["val"]

        X_train_fold = X_all[train_idx]
        y_train_fold = y_all[train_idx]
        X_val_fold = X_all[val_idx]
        y_val_fold = y_all[val_idx]

        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold).astype(np.float32)
        X_val_scaled = scaler.transform(X_val_fold).astype(np.float32)

        # SMOTE on training fold
        try:
            n_minority = sum(y_train_fold == 1)
            k = min(5, n_minority - 1) if n_minority > 1 else 1
            smote = SMOTE(random_state=SEED + fold_idx, k_neighbors=k)
            X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train_fold)
        except Exception as e:
            print(f"  [WARN] SMOTE failed ({e}), using original data")
            X_train_sm, y_train_sm = X_train_scaled, y_train_fold

        print(f"  Train: {len(X_train_fold)} -> {len(X_train_sm)} (after SMOTE)")
        print(f"  Val: {len(X_val_fold)}")

        # Build and train TabNet
        clf = build_tabnet(seed=SEED + fold_idx, n_features=n_features)
        clf.fit(
            X_train_sm, y_train_sm,
            eval_set=[(X_val_scaled, y_val_fold)],
            eval_metric=["auc"],
            max_epochs=150,
            patience=20,
            batch_size=min(64, len(X_train_sm)),
            drop_last=False,
        )

        # Predict on validation
        val_proba = clf.predict_proba(X_val_scaled)[:, 1]
        val_pred = (val_proba >= 0.5).astype(int)

        # Metrics
        try:
            auc = roc_auc_score(y_val_fold, val_proba)
        except ValueError:
            auc = np.nan
        f1 = f1_score(y_val_fold, val_pred, zero_division=0)
        mcc = matthews_corrcoef(y_val_fold, val_pred)

        fold_results.append({
            "fold": fold_idx,
            "AUC": round(auc, 4),
            "F1": round(f1, 4),
            "MCC": round(mcc, 4),
        })
        print(f"  AUC={auc:.4f}, F1={f1:.4f}, MCC={mcc:.4f}")

        # Store model and scaler for test inference
        all_fold_models.append((clf, scaler))

    # Mean results
    results_df = pd.DataFrame(fold_results)
    mean_row = {
        "fold": "mean",
        "AUC": round(results_df["AUC"].mean(), 4),
        "F1": round(results_df["F1"].mean(), 4),
        "MCC": round(results_df["MCC"].mean(), 4),
    }
    std_row = {
        "fold": "std",
        "AUC": round(results_df["AUC"].std(), 4),
        "F1": round(results_df["F1"].std(), 4),
        "MCC": round(results_df["MCC"].std(), 4),
    }
    results_df = pd.concat([results_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    results_df.to_csv(TABLES_DIR / "tabnet_results.csv", index=False)
    print(f"\n[SAVE] Results saved to {TABLES_DIR / 'tabnet_results.csv'}")

    print(f"\n[SUMMARY] Mean AUC={mean_row['AUC']:.4f}, F1={mean_row['F1']:.4f}, MCC={mean_row['MCC']:.4f}")

    # ------------------------------------------------------------------
    # Attention masks on test set (using ensemble of fold models)
    # ------------------------------------------------------------------
    print("\n[ATTENTION] Extracting attention masks on test set...")

    X_test = X_all[test_idx]
    y_test = y_all[test_idx]
    test_subjects = subject_ids[test_idx]

    # Collect attention from each fold model
    attention_maps = []
    for fold_idx, (clf, scaler) in enumerate(all_fold_models):
        X_test_scaled = scaler.transform(X_test).astype(np.float32)
        explain_matrix, masks = clf.explain(X_test_scaled)
        attention_maps.append(explain_matrix)  # (n_test, n_features)

    # Average attention across folds
    mean_attention = np.mean(attention_maps, axis=0)  # (n_test, n_features)

    # TabNet feature importance: mean attention per feature
    tabnet_feature_importance = mean_attention.mean(axis=0)  # (n_features,)
    tabnet_ranking = np.argsort(-tabnet_feature_importance)

    print(f"  Attention matrix shape: {mean_attention.shape}")
    print(f"  Top-5 features by TabNet attention:")
    for i in range(min(5, len(tabnet_ranking))):
        fidx = tabnet_ranking[i]
        print(f"    {i+1}. {feat_cols[fidx]}: {tabnet_feature_importance[fidx]:.4f}")

    # ------------------------------------------------------------------
    # SHAP ranking (from XGBoost) for comparison
    # ------------------------------------------------------------------
    print("\n[SHAP] Computing SHAP feature ranking for comparison...")

    all_train_idx = sorted(set(range(len(df))) - set(test_idx))
    X_train_all = X_all[all_train_idx]
    y_train_all = y_all[all_train_idx]

    # Standardize for SHAP
    scaler_shap = StandardScaler()
    X_train_shap = scaler_shap.fit_transform(X_train_all).astype(np.float32)
    X_test_shap = scaler_shap.transform(X_test).astype(np.float32)

    shap_importance, shap_ranking = compute_shap_ranking(
        X_train_shap, y_train_all, X_test_shap, feat_cols
    )

    # ------------------------------------------------------------------
    # Spearman concordance (top-10)
    # ------------------------------------------------------------------
    print("\n[CONCORDANCE] TabNet attention vs SHAP ranking...")

    top_k = min(10, n_features)

    # Create rank arrays for all features
    tabnet_ranks = np.zeros(n_features)
    shap_ranks = np.zeros(n_features)
    for rank, fidx in enumerate(tabnet_ranking):
        tabnet_ranks[fidx] = rank + 1
    for rank, fidx in enumerate(shap_ranking):
        shap_ranks[fidx] = rank + 1

    # Spearman correlation on all feature ranks
    rho_all, pval_all = spearmanr(tabnet_ranks, shap_ranks)

    # Spearman on top-10 intersection
    top10_tabnet = set(tabnet_ranking[:top_k])
    top10_shap = set(shap_ranking[:top_k])
    overlap = top10_tabnet & top10_shap
    n_overlap = len(overlap)

    # Build concordance table
    concordance_rows = []
    for i in range(n_features):
        concordance_rows.append({
            "feature": feat_cols[i],
            "tabnet_attention": round(float(tabnet_feature_importance[i]), 6),
            "tabnet_rank": int(tabnet_ranks[i]),
            "shap_importance": round(float(shap_importance[i]), 6),
            "shap_rank": int(shap_ranks[i]),
            "in_tabnet_top10": feat_cols[i] in [feat_cols[j] for j in tabnet_ranking[:top_k]],
            "in_shap_top10": feat_cols[i] in [feat_cols[j] for j in shap_ranking[:top_k]],
        })

    concordance_df = pd.DataFrame(concordance_rows)
    concordance_df = concordance_df.sort_values("tabnet_rank").reset_index(drop=True)

    # Add summary row
    summary_data = {
        "feature": f"SUMMARY: Spearman rho={rho_all:.4f}, p={pval_all:.4f}, top-10 overlap={n_overlap}/{top_k}",
        "tabnet_attention": np.nan,
        "tabnet_rank": np.nan,
        "shap_importance": np.nan,
        "shap_rank": np.nan,
        "in_tabnet_top10": np.nan,
        "in_shap_top10": np.nan,
    }
    concordance_df = pd.concat([concordance_df, pd.DataFrame([summary_data])], ignore_index=True)
    concordance_df.to_csv(TABLES_DIR / "tabnet_vs_shap_concordance.csv", index=False)
    print(f"[SAVE] Concordance saved to {TABLES_DIR / 'tabnet_vs_shap_concordance.csv'}")

    print(f"\n  Spearman rho (all features): {rho_all:.4f} (p={pval_all:.4f})")
    print(f"  Top-10 overlap: {n_overlap}/{top_k}")

    # ------------------------------------------------------------------
    # Plot 1: Attention heatmap
    # ------------------------------------------------------------------
    print("\n[PLOT] Generating attention heatmap...")

    fig, ax = plt.subplots(figsize=(14, max(6, len(test_idx) * 0.3)))

    # Truncate feature names for display
    display_names = [name[:25] for name in feat_cols]

    im = ax.imshow(mean_attention, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest")
    ax.set_xticks(range(n_features))
    ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(test_subjects)))
    ax.set_yticklabels(test_subjects, fontsize=7)
    ax.set_xlabel("Features", fontsize=11)
    ax.set_ylabel("Test Patients", fontsize=11)
    ax.set_title("TabNet Attention Masks (Test Patients x Features)", fontsize=13, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Attention Weight", fontsize=10)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "tabnet_attention_heatmap.png", dpi=DPI,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVE] Heatmap saved to {FIGURES_DIR / 'tabnet_attention_heatmap.png'}")

    # ------------------------------------------------------------------
    # Plot 2: TabNet attention vs SHAP comparison
    # ------------------------------------------------------------------
    print("[PLOT] Generating TabNet vs SHAP comparison...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Normalize both to [0, 1] for visual comparison
    tabnet_norm = tabnet_feature_importance / tabnet_feature_importance.max() if tabnet_feature_importance.max() > 0 else tabnet_feature_importance
    shap_norm = shap_importance / shap_importance.max() if shap_importance.max() > 0 else shap_importance

    # Sort by TabNet rank
    sort_idx = tabnet_ranking

    # Left panel: side-by-side bar chart
    ax = axes[0]
    x = np.arange(n_features)
    width = 0.35
    bars1 = ax.barh(x - width/2, tabnet_norm[sort_idx], width, label="TabNet Attention",
                    color=CONFIG["figures"]["color_fusion"], alpha=0.8)
    bars2 = ax.barh(x + width/2, shap_norm[sort_idx], width, label="SHAP Importance",
                    color=CONFIG["figures"]["color_proposed"], alpha=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels([feat_cols[i] for i in sort_idx], fontsize=8)
    ax.set_xlabel("Normalized Importance", fontsize=11)
    ax.set_title("Feature Importance:\nTabNet Attention vs SHAP", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.invert_yaxis()

    # Right panel: rank correlation scatter
    ax = axes[1]
    ax.scatter(tabnet_ranks, shap_ranks, s=100, alpha=0.7,
               c=CONFIG["figures"]["color_fusion"], edgecolors="black", linewidths=0.5)

    # Label each point
    for i in range(n_features):
        short = feat_cols[i][:15]
        ax.annotate(short, (tabnet_ranks[i], shap_ranks[i]),
                    fontsize=6, ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points")

    # Perfect agreement line
    ax.plot([0, n_features + 1], [0, n_features + 1], "k--", alpha=0.3, label="Perfect agreement")
    ax.set_xlabel("TabNet Attention Rank", fontsize=11)
    ax.set_ylabel("SHAP Importance Rank", fontsize=11)
    ax.set_title(f"Rank Correlation\n(Spearman rho={rho_all:.3f}, p={pval_all:.3f})",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, n_features + 1)
    ax.set_ylim(0, n_features + 1)
    ax.legend(fontsize=9)

    fig.suptitle("TabNet Attention vs SHAP Feature Importance Comparison",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "tabnet_vs_shap_comparison.png", dpi=DPI,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVE] Comparison saved to {FIGURES_DIR / 'tabnet_vs_shap_comparison.png'}")

    print("\n[DONE] TabNet analysis complete.")
    return results_df, concordance_df


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    run_tabnet_analysis()
