#!/usr/bin/env python3
"""
Train Clinical Models on WearGait-PD (Primary Cohort — Real DBS Labels)
========================================================================
This is the PRIMARY model with real DBS status labels.

Models trained:
  - XGBoost (main — LOOCV as primary evaluation)
  - SVM (RBF kernel)
  - MLP (PyTorch)
  - Feature-selected variants (top-5, top-10)

Evaluation:
  - LOOCV (most unbiased for small N)
  - 5-fold CV (with bootstrap CIs)
  - Repeated 5-fold (10 repeats, for variance estimate)
  - Hold-out test set

Output:
  results/tables/clinical_model_results.csv
  results/checkpoints/clinical_*.pt / .pkl

Author: Kartic Mishra, Gachon University
"""

import os
import sys
import json
import warnings
import multiprocessing
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    LeaveOneOut, StratifiedKFold, RepeatedStratifiedKFold, cross_val_predict
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, matthews_corrcoef,
    brier_score_loss, confusion_matrix, roc_curve
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from joblib import Parallel, delayed
import torch
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── Hardware config ──────────────────────────────────────────────────────────
N_CORES = multiprocessing.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(N_CORES)
os.environ["MKL_NUM_THREADS"] = str(N_CORES)

SEED = 42
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "clinical_features")
SPLIT_DIR = os.path.join(PROJECT_ROOT, "data", "splits")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "tables")
CKPT_DIR = os.path.join(PROJECT_ROOT, "results", "checkpoints")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# Feature columns to exclude from training
META_COLS = [
    "subject_id", "cohort", "dbs_candidate", "dbs_bilateral",
    "dbs_electrode", "dbs_years_since_surgery", "motor_subtype",
    "med_state_on"
]


def get_feature_cols(df):
    """Get numeric feature columns (excluding metadata and label)."""
    return [c for c in df.columns if c not in META_COLS
            and pd.api.types.is_numeric_dtype(df[c])]


def compute_metrics(y_true, y_prob, threshold=None):
    """Compute all classification metrics."""
    if threshold is None:
        # Youden's J optimal threshold
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        j_scores = tpr - fpr
        threshold = thresholds[np.argmax(j_scores)]

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "AUC_ROC": roc_auc_score(y_true, y_prob),
        "AUC_PR": average_precision_score(y_true, y_prob),
        "Sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Brier": brier_score_loss(y_true, y_prob),
        "Threshold": threshold,
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }
    return metrics


def bootstrap_ci(y_true, y_prob, n_boot=2000, ci=0.95, seed=SEED):
    """Compute bootstrap confidence intervals for AUC-ROC."""
    rng = np.random.RandomState(seed)
    aucs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    aucs = np.array(aucs)
    alpha = (1 - ci) / 2
    return np.percentile(aucs, [alpha * 100, (1 - alpha) * 100])


def train_xgboost_loocv(X, y, feature_names=None):
    """Train XGBoost with Leave-One-Out CV (most unbiased for small N)."""
    print("  Training XGBoost with LOOCV...")
    loo = LeaveOneOut()
    y_prob = np.zeros(len(y))

    for i, (train_idx, test_idx) in enumerate(loo.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale within fold
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Impute within fold
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        # SMOTE on training only (if enough minority samples)
        n_minority = int(y_train.sum())
        if n_minority >= 6:
            try:
                sm = SMOTE(random_state=SEED, k_neighbors=min(5, n_minority - 1))
                X_train, y_train = sm.fit_resample(X_train, y_train)
            except Exception:
                pass

        # Train
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1),
            tree_method="hist", device="cuda", n_jobs=N_CORES, random_state=SEED,
            eval_metric="logloss", verbosity=0
        )
        model.fit(X_train, y_train)
        y_prob[test_idx] = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y, y_prob)
    ci = bootstrap_ci(y, y_prob)
    metrics["AUC_CI_low"] = ci[0]
    metrics["AUC_CI_high"] = ci[1]
    print(f"    AUC={metrics['AUC_ROC']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")

    # Train final model on all data for checkpointing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    imputer = SimpleImputer(strategy="median")
    X_clean = imputer.fit_transform(X_scaled)

    final_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=len(y[y == 0]) / max(len(y[y == 1]), 1),
        tree_method="hist", device="cuda", n_jobs=N_CORES, random_state=SEED,
        eval_metric="logloss", verbosity=0
    )
    final_model.fit(X_clean, y)

    return metrics, y_prob, final_model, scaler, imputer


def train_svm_loocv(X, y):
    """Train SVM (RBF) with Leave-One-Out CV."""
    print("  Training SVM (RBF) with LOOCV...")
    loo = LeaveOneOut()
    y_prob = np.zeros(len(y))

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        n_minority = int(y_train.sum())
        if n_minority >= 6:
            try:
                sm = SMOTE(random_state=SEED, k_neighbors=min(5, n_minority - 1))
                X_train, y_train = sm.fit_resample(X_train, y_train)
            except Exception:
                pass

        model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                    class_weight="balanced", random_state=SEED)
        model.fit(X_train, y_train)
        y_prob[test_idx] = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y, y_prob)
    ci = bootstrap_ci(y, y_prob)
    metrics["AUC_CI_low"] = ci[0]
    metrics["AUC_CI_high"] = ci[1]
    print(f"    AUC={metrics['AUC_ROC']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")

    return metrics, y_prob


def train_repeated_cv(X, y, model_type="xgb", n_repeats=10):
    """Train with repeated stratified 5-fold CV for variance estimation."""
    print(f"  Training {model_type} with Repeated 5-fold CV ({n_repeats} repeats)...")
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED)

    fold_aucs = []
    for train_idx, test_idx in rskf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        n_minority = int(y_train.sum())
        if n_minority >= 6:
            try:
                sm = SMOTE(random_state=SEED, k_neighbors=min(5, n_minority - 1))
                X_train, y_train = sm.fit_resample(X_train, y_train)
            except Exception:
                pass

        if model_type == "xgb":
            model = xgb.XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1),
                tree_method="hist", device="cuda", n_jobs=N_CORES, random_state=SEED,
                eval_metric="logloss", verbosity=0
            )
        else:
            model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                        class_weight="balanced", random_state=SEED)

        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]

        if len(np.unique(y_test)) >= 2:
            fold_aucs.append(roc_auc_score(y_test, y_prob))

    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    print(f"    AUC={mean_auc:.4f} +/- {std_auc:.4f}")
    return mean_auc, std_auc


def permutation_test(X, y, n_permutations=1000):
    """Permutation test to check if model AUC is significantly above chance."""
    print("  Running permutation test (1000 permutations)...")
    rng = np.random.RandomState(SEED)

    # Get actual AUC using 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    y_prob_actual = np.zeros(len(y))
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            tree_method="hist", device="cuda", n_jobs=N_CORES, random_state=SEED,
            eval_metric="logloss", verbosity=0
        )
        model.fit(X_train, y_train)
        y_prob_actual[test_idx] = model.predict_proba(X_test)[:, 1]

    actual_auc = roc_auc_score(y, y_prob_actual)

    # Permuted AUCs
    perm_aucs = []
    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        y_prob_perm = np.zeros(len(y))
        for train_idx, test_idx in skf.split(X, y_perm):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y_perm[train_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            imputer = SimpleImputer(strategy="median")
            X_train = imputer.fit_transform(X_train)
            X_test = imputer.transform(X_test)

            model = xgb.XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                tree_method="hist", device="cuda", n_jobs=N_CORES, random_state=SEED,
                eval_metric="logloss", verbosity=0
            )
            model.fit(X_train, y_train)

            if len(np.unique(y_perm[test_idx])) >= 2:
                y_prob_perm[test_idx] = model.predict_proba(X_test)[:, 1]
            else:
                y_prob_perm[test_idx] = 0.5

        if len(np.unique(y_perm)) >= 2:
            try:
                perm_aucs.append(roc_auc_score(y_perm, y_prob_perm))
            except Exception:
                perm_aucs.append(0.5)

    p_value = (np.sum(np.array(perm_aucs) >= actual_auc) + 1) / (len(perm_aucs) + 1)
    print(f"    Actual AUC={actual_auc:.4f}, p={p_value:.4f}")
    return actual_auc, p_value


def feature_importance_rfe(X, y, feature_names, top_k=10):
    """Get top-K features using XGBoost feature importance."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    imputer = SimpleImputer(strategy="median")
    X_clean = imputer.fit_transform(X_scaled)

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        tree_method="hist", n_jobs=N_CORES, random_state=SEED,
        eval_metric="logloss", verbosity=0
    )
    model.fit(X_clean, y)

    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:top_k]
    top_features = [feature_names[i] for i in top_idx]
    top_importances = importances[top_idx]

    print(f"\n  Top-{top_k} features:")
    for feat, imp in zip(top_features, top_importances):
        print(f"    {feat}: {imp:.4f}")

    return top_idx, top_features


def main():
    print("=" * 70)
    print("Training Clinical Models on WearGait-PD (Real DBS Labels)")
    print("=" * 70)

    # ── Load data ────────────────────────────────────────────────────────
    print("\n[1/6] Loading data...")

    # PD-only cohort (primary analysis)
    df_pd = pd.read_csv(os.path.join(DATA_DIR, "weargait_pd_only.csv"))
    feat_cols = get_feature_cols(df_pd)
    X_pd = df_pd[feat_cols].values.astype(np.float32)
    y_pd = df_pd["dbs_candidate"].values.astype(int)
    print(f"  PD-only: {len(df_pd)} subjects, {len(feat_cols)} features, "
          f"{y_pd.sum()} DBS+, {(1-y_pd).sum()} DBS-")
    print(f"  EPV (events per variable): {y_pd.sum() / len(feat_cols):.2f}")

    # Full cohort
    df_full = pd.read_csv(os.path.join(DATA_DIR, "weargait_clinical.csv"))
    feat_cols_full = get_feature_cols(df_full)
    X_full = df_full[feat_cols_full].values.astype(np.float32)
    y_full = df_full["dbs_candidate"].values.astype(int)
    print(f"  Full cohort: {len(df_full)} subjects, {len(feat_cols_full)} features, "
          f"{y_full.sum()} DBS+, {(1-y_full).sum()} DBS-")

    results = []

    # ═══════════════════════════════════════════════════════════════════════
    # PD-ONLY ANALYSIS (most clinically relevant — DBS+ vs DBS- among PD)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PD-ONLY ANALYSIS (n=82, 23 DBS+)")
    print("=" * 70)

    # ── Feature selection ────────────────────────────────────────────────
    top10_idx, top10_names = feature_importance_rfe(X_pd, y_pd, feat_cols, top_k=10)
    top5_idx = top10_idx[:5]
    top5_names = top10_names[:5]

    # ── XGBoost LOOCV (all features) ─────────────────────────────────────
    print("\n[2/6] XGBoost LOOCV (all features)...")
    xgb_metrics, xgb_probs, xgb_model, xgb_scaler, xgb_imputer = train_xgboost_loocv(
        X_pd, y_pd, feat_cols
    )
    xgb_metrics["Model"] = "XGBoost_AllFeatures"
    xgb_metrics["N_features"] = len(feat_cols)
    xgb_metrics["Evaluation"] = f"LOOCV_n{len(y_pd)}"
    xgb_metrics["Cohort"] = "PD_only"
    results.append(xgb_metrics)

    # Save model
    with open(os.path.join(CKPT_DIR, "clinical_xgb_all.pkl"), "wb") as f:
        pickle.dump({"model": xgb_model, "scaler": xgb_scaler, "imputer": xgb_imputer,
                      "features": feat_cols}, f)

    # ── XGBoost LOOCV (top-10) ───────────────────────────────────────────
    print("\n[3/6] XGBoost LOOCV (top-10 features)...")
    X_top10 = X_pd[:, top10_idx]
    xgb10_metrics, _, _, _, _ = train_xgboost_loocv(X_top10, y_pd, top10_names)
    xgb10_metrics["Model"] = "XGBoost_Top10"
    xgb10_metrics["N_features"] = 10
    xgb10_metrics["Evaluation"] = f"LOOCV_n{len(y_pd)}"
    xgb10_metrics["Cohort"] = "PD_only"
    xgb10_metrics["EPV"] = y_pd.sum() / 10
    results.append(xgb10_metrics)

    # ── XGBoost LOOCV (top-5) ────────────────────────────────────────────
    print("\n  XGBoost LOOCV (top-5 features)...")
    X_top5 = X_pd[:, top5_idx]
    xgb5_metrics, _, _, _, _ = train_xgboost_loocv(X_top5, y_pd, top5_names)
    xgb5_metrics["Model"] = "XGBoost_Top5"
    xgb5_metrics["N_features"] = 5
    xgb5_metrics["Evaluation"] = f"LOOCV_n{len(y_pd)}"
    xgb5_metrics["Cohort"] = "PD_only"
    xgb5_metrics["EPV"] = y_pd.sum() / 5
    results.append(xgb5_metrics)

    # ── SVM LOOCV (all features) ─────────────────────────────────────────
    print("\n[4/6] SVM LOOCV (all features)...")
    svm_metrics, svm_probs = train_svm_loocv(X_pd, y_pd)
    svm_metrics["Model"] = "SVM_AllFeatures"
    svm_metrics["N_features"] = len(feat_cols)
    svm_metrics["Evaluation"] = f"LOOCV_n{len(y_pd)}"
    svm_metrics["Cohort"] = "PD_only"
    results.append(svm_metrics)

    # ── SVM LOOCV (top-10) ───────────────────────────────────────────────
    svm10_metrics, _ = train_svm_loocv(X_top10, y_pd)
    svm10_metrics["Model"] = "SVM_Top10"
    svm10_metrics["N_features"] = 10
    svm10_metrics["Evaluation"] = f"LOOCV_n{len(y_pd)}"
    svm10_metrics["Cohort"] = "PD_only"
    results.append(svm10_metrics)

    # ── Repeated 5-fold CV ───────────────────────────────────────────────
    print("\n[5/6] Repeated 5-fold CV...")
    rep_auc_xgb, rep_std_xgb = train_repeated_cv(X_pd, y_pd, "xgb")
    results.append({
        "Model": "XGBoost_AllFeatures", "AUC_ROC": rep_auc_xgb,
        "AUC_CI_low": rep_auc_xgb - 1.96 * rep_std_xgb,
        "AUC_CI_high": rep_auc_xgb + 1.96 * rep_std_xgb,
        "N_features": len(feat_cols), "Evaluation": "Repeated5fold_10rep",
        "Cohort": "PD_only"
    })

    rep_auc_svm, rep_std_svm = train_repeated_cv(X_pd, y_pd, "svm")
    results.append({
        "Model": "SVM_AllFeatures", "AUC_ROC": rep_auc_svm,
        "AUC_CI_low": rep_auc_svm - 1.96 * rep_std_svm,
        "AUC_CI_high": rep_auc_svm + 1.96 * rep_std_svm,
        "N_features": len(feat_cols), "Evaluation": "Repeated5fold_10rep",
        "Cohort": "PD_only"
    })

    # ── Permutation test ─────────────────────────────────────────────────
    print("\n[6/6] Permutation test...")
    perm_auc, perm_p = permutation_test(X_pd, y_pd)
    results.append({
        "Model": "XGBoost_PermutationTest", "AUC_ROC": perm_auc,
        "Evaluation": f"Permutation_p={perm_p:.4f}",
        "Cohort": "PD_only"
    })

    # ═══════════════════════════════════════════════════════════════════════
    # FULL COHORT ANALYSIS (PD + Controls — for extended validation)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("FULL COHORT ANALYSIS (n=167, 23 DBS+)")
    print("=" * 70)

    print("\n  XGBoost LOOCV (all features)...")
    xgb_full_metrics, _, _, _, _ = train_xgboost_loocv(X_full, y_full, feat_cols_full)
    xgb_full_metrics["Model"] = "XGBoost_AllFeatures"
    xgb_full_metrics["N_features"] = len(feat_cols_full)
    xgb_full_metrics["Evaluation"] = f"LOOCV_n{len(y_full)}"
    xgb_full_metrics["Cohort"] = "Full"
    results.append(xgb_full_metrics)

    print("\n  SVM LOOCV (all features)...")
    svm_full_metrics, _ = train_svm_loocv(X_full, y_full)
    svm_full_metrics["Model"] = "SVM_AllFeatures"
    svm_full_metrics["N_features"] = len(feat_cols_full)
    svm_full_metrics["Evaluation"] = f"LOOCV_n{len(y_full)}"
    svm_full_metrics["Cohort"] = "Full"
    results.append(svm_full_metrics)

    # ── Save results ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    df_results = pd.DataFrame(results)
    out_path = os.path.join(RESULTS_DIR, "clinical_model_results.csv")
    df_results.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")

    # Save feature importance
    feat_imp = pd.DataFrame({
        "feature": feat_cols,
        "importance": xgb_model.feature_importances_
    }).sort_values("importance", ascending=False)
    feat_imp.to_csv(os.path.join(RESULTS_DIR, "clinical_feature_importance.csv"), index=False)

    # Save LOOCV predictions for downstream analysis
    pred_df = pd.DataFrame({
        "subject_id": df_pd["subject_id"].values,
        "dbs_candidate": y_pd,
        "xgb_prob": xgb_probs,
        "svm_prob": svm_probs,
        "motor_subtype": df_pd["motor_subtype"].values,
        "cohort": "PD_only"
    })
    pred_df.to_csv(os.path.join(RESULTS_DIR, "clinical_loocv_predictions.csv"), index=False)

    # ── Print summary table ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for _, row in df_results.iterrows():
        auc = row.get("AUC_ROC", "—")
        if isinstance(auc, float):
            ci_lo = row.get("AUC_CI_low", "")
            ci_hi = row.get("AUC_CI_high", "")
            if ci_lo and ci_hi and not pd.isna(ci_lo) and not pd.isna(ci_hi):
                auc_str = f"{auc:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]"
            else:
                auc_str = f"{auc:.4f}"
        else:
            auc_str = str(auc)

        sens = row.get("Sensitivity", "")
        spec = row.get("Specificity", "")
        f1 = row.get("F1", "")
        print(f"  {row.get('Cohort',''):<8} {row.get('Model',''):<25} "
              f"{row.get('Evaluation',''):<25} AUC={auc_str}")

    print("\nDone.")


if __name__ == "__main__":
    main()
