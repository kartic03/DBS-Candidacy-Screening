#!/usr/bin/env python3
"""
Comprehensive Evaluation (Revised)
====================================
Evaluates all models with:
  - DeLong tests (within-dataset model comparisons)
  - Decision Curve Analysis (DCA) for WearGait-PD (mandatory for JBI)
  - Calibration (reliability diagram + Brier + ECE, Platt scaling)
  - NRI vs H&Y-only threshold
  - Motor subtype analysis
  - Fairness (age/sex subgroups with actual ranges)

Also re-runs GaitPDB fusion EXCLUDING H&Y from clinical features (fix tautology).

Output:
  results/tables/evaluation_comprehensive.csv
  results/tables/delong_within_dataset.csv
  results/tables/calibration_results.csv
  results/tables/dca_clinical.csv
  results/tables/fairness_analysis_v2.csv
  results/tables/fusion_fixed_results.csv

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
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, matthews_corrcoef,
    brier_score_loss, confusion_matrix, roc_curve
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import torch
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

N_CORES = multiprocessing.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(N_CORES)
SEED = 42
np.random.seed(SEED)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "tables")
os.makedirs(RESULTS_DIR, exist_ok=True)

META_COLS = [
    "subject_id", "cohort", "dbs_candidate", "dbs_bilateral",
    "dbs_electrode", "dbs_years_since_surgery", "motor_subtype", "med_state_on"
]


def delong_test(y_true, y_prob1, y_prob2):
    """DeLong test for comparing two AUCs. Returns Z-statistic and p-value."""
    n = len(y_true)
    if n < 10:
        return 0.0, 1.0

    auc1 = roc_auc_score(y_true, y_prob1)
    auc2 = roc_auc_score(y_true, y_prob2)

    # Placement values
    pos = y_true == 1
    neg = y_true == 0
    n1 = pos.sum()
    n0 = neg.sum()

    if n1 < 2 or n0 < 2:
        return 0.0, 1.0

    # Variance estimation via DeLong
    def placement_values(y_prob):
        pv_pos = np.array([np.mean(y_prob[neg] < p) + 0.5 * np.mean(y_prob[neg] == p)
                           for p in y_prob[pos]])
        pv_neg = np.array([np.mean(y_prob[pos] > n_val) + 0.5 * np.mean(y_prob[pos] == n_val)
                           for n_val in y_prob[neg]])
        return pv_pos, pv_neg

    pv1_pos, pv1_neg = placement_values(y_prob1)
    pv2_pos, pv2_neg = placement_values(y_prob2)

    s10_1 = np.var(pv1_pos, ddof=1)
    s01_1 = np.var(pv1_neg, ddof=1)
    s10_2 = np.var(pv2_pos, ddof=1)
    s01_2 = np.var(pv2_neg, ddof=1)

    # Covariance
    s10_12 = np.cov(pv1_pos, pv2_pos, ddof=1)[0, 1] if len(pv1_pos) > 1 else 0
    s01_12 = np.cov(pv1_neg, pv2_neg, ddof=1)[0, 1] if len(pv1_neg) > 1 else 0

    var_auc1 = s10_1 / n1 + s01_1 / n0
    var_auc2 = s10_2 / n1 + s01_2 / n0
    cov_auc = s10_12 / n1 + s01_12 / n0

    var_diff = var_auc1 + var_auc2 - 2 * cov_auc
    if var_diff <= 0:
        return 0.0, 1.0

    z = (auc1 - auc2) / np.sqrt(var_diff)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p


def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        avg_confidence = y_prob[mask].mean()
        avg_accuracy = y_true[mask].mean()
        ece += mask.sum() / len(y_true) * abs(avg_accuracy - avg_confidence)
    return ece


def platt_scaling(y_true_cal, y_prob_cal, y_prob_test):
    """Apply Platt scaling (logistic regression on probabilities)."""
    lr = LogisticRegression(random_state=SEED)
    lr.fit(y_prob_cal.reshape(-1, 1), y_true_cal)
    return lr.predict_proba(y_prob_test.reshape(-1, 1))[:, 1]


def decision_curve_analysis(y_true, y_prob, thresholds=None):
    """Compute net benefit for DCA."""
    if thresholds is None:
        thresholds = np.arange(0.01, 0.99, 0.01)

    results = []
    n = len(y_true)
    prevalence = y_true.mean()

    for pt in thresholds:
        y_pred = (y_prob >= pt).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))

        net_benefit = tp / n - fp / n * (pt / (1 - pt + 1e-12))
        net_benefit_all = prevalence - (1 - prevalence) * (pt / (1 - pt + 1e-12))

        results.append({
            "threshold": pt,
            "net_benefit_model": net_benefit,
            "net_benefit_treat_all": net_benefit_all,
            "net_benefit_treat_none": 0.0,
        })

    return pd.DataFrame(results)


def nri_analysis(y_true, y_prob_new, y_prob_old, threshold=0.5):
    """Net Reclassification Improvement."""
    new_class = (y_prob_new >= threshold).astype(int)
    old_class = (y_prob_old >= threshold).astype(int)

    events = y_true == 1
    nonevents = y_true == 0

    # NRI for events (should move UP)
    nri_events = (np.mean(new_class[events] > old_class[events]) -
                  np.mean(new_class[events] < old_class[events]))

    # NRI for nonevents (should move DOWN)
    nri_nonevents = (np.mean(new_class[nonevents] < old_class[nonevents]) -
                     np.mean(new_class[nonevents] > old_class[nonevents]))

    nri = nri_events + nri_nonevents
    return nri, nri_events, nri_nonevents


def main():
    print("=" * 70)
    print("Comprehensive Evaluation (Revised)")
    print("=" * 70)

    # ═══════════════════════════════════════════════════════════════════════
    # 1. Load clinical LOOCV predictions
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[1/7] Loading predictions...")
    pred_path = os.path.join(RESULTS_DIR, "clinical_loocv_predictions.csv")
    df_pred = pd.read_csv(pred_path)
    y_true = df_pred["dbs_candidate"].values
    y_xgb = df_pred["xgb_prob"].values
    y_svm = df_pred["svm_prob"].values

    print(f"  Loaded {len(df_pred)} predictions (pos={y_true.sum()}, neg={(1-y_true).sum()})")

    # ═══════════════════════════════════════════════════════════════════════
    # 2. DeLong Tests (within-dataset)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[2/7] DeLong tests (within-dataset model comparisons)...")
    delong_results = []

    # Clinical: XGBoost vs SVM
    z, p = delong_test(y_true, y_xgb, y_svm)
    auc_xgb = roc_auc_score(y_true, y_xgb)
    auc_svm = roc_auc_score(y_true, y_svm)
    delong_results.append({
        "Dataset": "WearGait-PD", "Model1": "XGBoost_AllFeat",
        "Model2": "SVM_AllFeat", "AUC1": auc_xgb, "AUC2": auc_svm,
        "Z": z, "P_value": p, "Significant_0.05": p < 0.05,
    })
    print(f"  WearGait XGB({auc_xgb:.3f}) vs SVM({auc_svm:.3f}): Z={z:.3f}, p={p:.4f}")

    # Load modality results for within-dataset DeLong
    df_modality = pd.read_csv(os.path.join(RESULTS_DIR, "modality_model_results.csv"))

    # For each dataset, compare XGBoost vs MLP
    for dataset in df_modality["Dataset"].unique():
        sub = df_modality[df_modality["Dataset"] == dataset]
        for label in sub["Label"].unique():
            sub2 = sub[sub["Label"] == label]
            xgb_auc = sub2[sub2["Model"] == "XGBoost"]["AUC_ROC"].values
            mlp_auc = sub2[sub2["Model"] == "MLP"]["AUC_ROC"].values
            if len(xgb_auc) > 0 and len(mlp_auc) > 0:
                delong_results.append({
                    "Dataset": f"{dataset}_{label}",
                    "Model1": "XGBoost", "Model2": "MLP",
                    "AUC1": xgb_auc[0], "AUC2": mlp_auc[0],
                    "Z": np.nan, "P_value": np.nan,
                    "Note": "OOF predictions not stored; AUC comparison only"
                })

    df_delong = pd.DataFrame(delong_results)
    df_delong.to_csv(os.path.join(RESULTS_DIR, "delong_within_dataset.csv"), index=False)
    print(f"  Saved: delong_within_dataset.csv")

    # ═══════════════════════════════════════════════════════════════════════
    # 3. Calibration Analysis
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[3/7] Calibration analysis...")
    cal_results = []

    for name, y_prob in [("XGBoost", y_xgb), ("SVM", y_svm)]:
        brier = brier_score_loss(y_true, y_prob)
        ece = compute_ece(y_true, y_prob)

        # Platt scaling via cross-validation
        loo = LeaveOneOut()
        y_platt = np.zeros(len(y_true))
        for train_idx, test_idx in loo.split(y_true):
            y_platt[test_idx] = platt_scaling(y_true[train_idx], y_prob[train_idx],
                                               y_prob[test_idx])

        brier_platt = brier_score_loss(y_true, y_platt)
        ece_platt = compute_ece(y_true, y_platt)

        cal_results.append({
            "Model": name, "Brier_before": brier, "ECE_before": ece,
            "Brier_after_Platt": brier_platt, "ECE_after_Platt": ece_platt,
        })
        print(f"  {name}: Brier={brier:.4f}→{brier_platt:.4f}, "
              f"ECE={ece:.4f}→{ece_platt:.4f}")

    df_cal = pd.DataFrame(cal_results)
    df_cal.to_csv(os.path.join(RESULTS_DIR, "calibration_results_v2.csv"), index=False)

    # ═══════════════════════════════════════════════════════════════════════
    # 4. Decision Curve Analysis
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[4/7] Decision Curve Analysis...")
    dca_df = decision_curve_analysis(y_true, y_xgb)
    dca_df["model"] = "XGBoost"
    dca_svm = decision_curve_analysis(y_true, y_svm)
    dca_svm["model"] = "SVM"
    dca_all = pd.concat([dca_df, dca_svm], ignore_index=True)
    dca_all.to_csv(os.path.join(RESULTS_DIR, "dca_clinical_v2.csv"), index=False)
    print(f"  Saved: dca_clinical_v2.csv")

    # Show net benefit at key thresholds
    for pt in [0.1, 0.2, 0.3, 0.5]:
        nb = dca_df[dca_df["threshold"].round(2) == pt]["net_benefit_model"].values
        if len(nb) > 0:
            print(f"    XGBoost net benefit at p={pt}: {nb[0]:.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    # 5. NRI vs H&Y-only baseline
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[5/7] NRI analysis...")
    df_pd = pd.read_csv(os.path.join(PROC_DIR, "clinical_features", "weargait_pd_only.csv"))

    # H&Y-only baseline: probability based on H&Y threshold
    hy = df_pd["hoehn_yahr"].values
    hy_prob = np.clip((hy - 1.0) / 3.0, 0, 1)  # Normalize H&Y 1-4 to 0-1
    hy_prob[np.isnan(hy_prob)] = 0.5

    nri, nri_ev, nri_nonev = nri_analysis(y_true, y_xgb, hy_prob, threshold=0.3)
    print(f"  NRI (XGBoost vs H&Y-only): {nri:.4f}")
    print(f"    NRI events: {nri_ev:.4f}, NRI nonevents: {nri_nonev:.4f}")

    nri_results = pd.DataFrame([{
        "Comparison": "XGBoost_Top10 vs HY_Only",
        "NRI": nri, "NRI_events": nri_ev, "NRI_nonevents": nri_nonev,
    }])
    nri_results.to_csv(os.path.join(RESULTS_DIR, "nri_results_v2.csv"), index=False)

    # ═══════════════════════════════════════════════════════════════════════
    # 6. Fairness Analysis (actual age ranges)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[6/7] Fairness analysis (actual demographics)...")
    fairness = []

    # Sex subgroups
    for sex_val, sex_name in [(1, "Male"), (0, "Female")]:
        mask = df_pd["sex"].values == sex_val
        if mask.sum() >= 10 and y_true[mask].sum() >= 2:
            auc = roc_auc_score(y_true[mask], y_xgb[mask])
            fairness.append({
                "Subgroup": f"Sex={sex_name}", "N": int(mask.sum()),
                "N_pos": int(y_true[mask].sum()), "AUC": auc,
            })
            print(f"  Sex={sex_name}: N={mask.sum()}, AUC={auc:.4f}")

    # Age subgroups (actual years)
    age = df_pd["age"].values
    age_valid = ~np.isnan(age)
    if age_valid.sum() > 0:
        age_median = np.nanmedian(age)
        for label, mask in [("Younger (<median)", age < age_median),
                            ("Older (>=median)", age >= age_median)]:
            valid_mask = mask & age_valid
            if valid_mask.sum() >= 10 and y_true[valid_mask].sum() >= 2:
                auc = roc_auc_score(y_true[valid_mask], y_xgb[valid_mask])
                age_range = f"{age[valid_mask].min():.0f}-{age[valid_mask].max():.0f}yr"
                fairness.append({
                    "Subgroup": f"Age {label} ({age_range})",
                    "N": int(valid_mask.sum()),
                    "N_pos": int(y_true[valid_mask].sum()), "AUC": auc,
                })
                print(f"  Age {label}: N={valid_mask.sum()}, AUC={auc:.4f}")

    # Motor subtype
    for subtype in ["Akinetic-Rigid", "Tremor-Dominant"]:
        mask = df_pred["motor_subtype"].values == subtype
        if mask.sum() >= 10 and y_true[mask].sum() >= 2:
            auc = roc_auc_score(y_true[mask], y_xgb[mask])
            fairness.append({
                "Subgroup": f"Subtype={subtype}", "N": int(mask.sum()),
                "N_pos": int(y_true[mask].sum()), "AUC": auc,
            })
            print(f"  Subtype={subtype}: N={mask.sum()}, AUC={auc:.4f}")

    df_fairness = pd.DataFrame(fairness)
    df_fairness.to_csv(os.path.join(RESULTS_DIR, "fairness_analysis_v2.csv"), index=False)

    # ═══════════════════════════════════════════════════════════════════════
    # 7. GaitPDB Fusion Fix: exclude H&Y from clinical features
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[7/7] GaitPDB fusion fix (excluding H&Y from clinical)...")

    df_gait = pd.read_csv(os.path.join(PROC_DIR, "gait_features", "gaitpdb_with_clinical.csv"))
    df_gait = df_gait[df_gait["dbs_proxy_hy25"].notna()].copy()
    df_gait["dbs_proxy_hy25"] = df_gait["dbs_proxy_hy25"].astype(int)

    gait_meta = ["subject_id", "Study", "Group", "Gender", "pd_label", "sex",
                 "dbs_proxy_hy25", "dbs_proxy_updrs32", "n_single_trials",
                 "Age", "Height", "Weight", "HoehnYahr", "UPDRS", "UPDRSM", "TUAG"]
    gait_sensor_cols = [c for c in df_gait.columns if c not in gait_meta
                        and pd.api.types.is_numeric_dtype(df_gait[c])]

    # Clinical features EXCLUDING HoehnYahr (the target variable)
    clinical_cols_fixed = ["Age", "Height", "Weight", "UPDRS", "UPDRSM", "TUAG"]

    for c in clinical_cols_fixed:
        df_gait[c] = pd.to_numeric(df_gait[c], errors="coerce")

    X_gait = df_gait[gait_sensor_cols].values.astype(np.float32)
    X_clinical = df_gait[clinical_cols_fixed].values.astype(np.float32)
    y_hy = df_gait["dbs_proxy_hy25"].values.astype(int)

    print(f"  N={len(y_hy)}, Pos={y_hy.sum()}, Gait_feats={X_gait.shape[1]}, "
          f"Clinical_feats(no H&Y)={X_clinical.shape[1]}")

    # 5-fold CV with XGBoost for each variant
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fusion_results = []

    for variant_name, use_gait, use_clinical in [
        ("GaitOnly_Fixed", True, False),
        ("ClinicalOnly_Fixed", False, True),
        ("Concat_Fixed", True, True),
    ]:
        y_prob_all = np.zeros(len(y_hy))
        fold_aucs = []

        for fold_i, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(y_hy)), y_hy)):
            if use_gait and use_clinical:
                X_train = np.hstack([X_gait[train_idx], X_clinical[train_idx]])
                X_val = np.hstack([X_gait[val_idx], X_clinical[val_idx]])
            elif use_gait:
                X_train, X_val = X_gait[train_idx], X_gait[val_idx]
            else:
                X_train, X_val = X_clinical[train_idx], X_clinical[val_idx]

            y_train = y_hy[train_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

            imputer = SimpleImputer(strategy="median")
            X_train = imputer.fit_transform(X_train)
            X_val = imputer.transform(X_val)

            n_min = int(y_train.sum())
            if n_min >= 3:
                try:
                    sm = SMOTE(random_state=SEED, k_neighbors=min(5, n_min - 1))
                    X_train, y_train = sm.fit_resample(X_train, y_train)
                except Exception:
                    pass

            model = xgb.XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1),
                tree_method="hist", device="cuda", n_jobs=N_CORES,
                random_state=SEED, eval_metric="logloss", verbosity=0
            )
            model.fit(X_train, y_train)
            y_prob_all[val_idx] = model.predict_proba(X_val)[:, 1]

            if len(np.unique(y_hy[val_idx])) >= 2:
                fold_aucs.append(roc_auc_score(y_hy[val_idx], y_prob_all[val_idx]))

        auc = roc_auc_score(y_hy, y_prob_all)
        fusion_results.append({
            "Model": variant_name, "AUC": auc,
            "Mean_fold_AUC": np.mean(fold_aucs), "Std_fold_AUC": np.std(fold_aucs),
        })
        print(f"  {variant_name}: AUC={auc:.4f} (fold mean={np.mean(fold_aucs):.4f})")

    df_fusion_fixed = pd.DataFrame(fusion_results)
    df_fusion_fixed.to_csv(os.path.join(RESULTS_DIR, "fusion_fixed_results.csv"), index=False)

    # ═══════════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"""
  DeLong tests: {len(df_delong)} comparisons saved
  Calibration: XGBoost ECE {cal_results[0]['ECE_before']:.4f} -> {cal_results[0]['ECE_after_Platt']:.4f}
  DCA: net benefit computed for 98 thresholds
  NRI: {nri:.4f} (XGBoost vs H&Y-only)
  Fairness: {len(fairness)} subgroups analyzed
  Fusion fix: H&Y excluded from clinical features
    GaitOnly: AUC={fusion_results[0]['AUC']:.4f}
    ClinicalOnly(no H&Y): AUC={fusion_results[1]['AUC']:.4f}
    Concat(no H&Y): AUC={fusion_results[2]['AUC']:.4f}
    """)

    print("Done.")


if __name__ == "__main__":
    main()
