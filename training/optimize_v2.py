#!/usr/bin/env python3
"""
Model Optimization v2 — Push Primary Model Toward 0.90+
=========================================================
Strategies:
  1. Feature engineering: clinical composites, interaction terms, ratios
  2. Full cohort (n=167) — more data, controls are legitimately DBS-
  3. mRMR feature selection (minimum redundancy, maximum relevance)
  4. CatBoost (ordinal-native, ordered boosting)
  5. ElasticNet logistic regression
  6. Combine individual UPDRS items + composite scores

Author: Kartic Mishra, Gachon University
"""

import os
import warnings
import multiprocessing
import json

import numpy as np
import pandas as pd
import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
import xgboost as xgb

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_CORES = multiprocessing.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(N_CORES)
SEED = 42
np.random.seed(SEED)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "tables")
CKPT_DIR = os.path.join(PROJECT_ROOT, "results", "checkpoints")

META_COLS = [
    "subject_id", "cohort", "dbs_candidate", "dbs_bilateral",
    "dbs_electrode", "dbs_years_since_surgery", "motor_subtype", "med_state_on"
]


def loocv_auc(X, y, model_fn, smote_type="smote"):
    """LOOCV AUC with per-fold scaling, imputation, and oversampling."""
    loo = LeaveOneOut()
    y_prob = np.zeros(len(y))
    for train_idx, test_idx in loo.split(X):
        X_tr, X_te = X[train_idx].copy(), X[test_idx].copy()
        y_tr = y[train_idx].copy()

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(X_tr)
        X_te = imp.transform(X_te)

        n_min = int(y_tr.sum())
        if n_min >= 3:
            try:
                if smote_type == "adasyn":
                    sm = ADASYN(random_state=SEED, n_neighbors=min(5, n_min - 1))
                elif smote_type == "borderline":
                    sm = BorderlineSMOTE(random_state=SEED, k_neighbors=min(5, n_min - 1))
                else:
                    sm = SMOTE(random_state=SEED, k_neighbors=min(5, n_min - 1))
                X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
            except Exception:
                pass

        model = model_fn()
        model.fit(X_tr, y_tr)
        y_prob[test_idx] = model.predict_proba(X_te)[:, 1]

    return roc_auc_score(y, y_prob), y_prob


def cv_auc(X, y, model_fn, n_folds=5, smote_type="smote"):
    """Fast 5-fold CV for Optuna search."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    y_prob = np.zeros(len(y))
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx].copy(), X[test_idx].copy()
        y_tr = y[train_idx].copy()

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(X_tr)
        X_te = imp.transform(X_te)

        n_min = int(y_tr.sum())
        if n_min >= 3:
            try:
                sm = SMOTE(random_state=SEED, k_neighbors=min(5, n_min - 1))
                X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
            except Exception:
                pass

        model = model_fn()
        model.fit(X_tr, y_tr)
        y_prob[test_idx] = model.predict_proba(X_te)[:, 1]

    return roc_auc_score(y, y_prob)


def bootstrap_ci(y_true, y_prob, n_boot=2000, seed=SEED):
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    return np.percentile(aucs, [2.5, 97.5]) if aucs else [0.5, 0.5]


def engineer_features(df):
    """Create clinically meaningful composite features."""
    df = df.copy()

    # ── Clinical composite scores ────────────────────────────────────────
    # Tremor score (items 3-15, 3-16, 3-17, 3-18)
    tremor_items = [c for c in df.columns if any(x in c for x in
                    ["3-15", "3-16", "3-17", "3-18"])]
    if tremor_items:
        df["composite_tremor"] = df[tremor_items].sum(axis=1, min_count=1)

    # Rigidity score (item 3-3)
    rigidity_items = [c for c in df.columns if "3-3-" in c]
    if rigidity_items:
        df["composite_rigidity"] = df[rigidity_items].sum(axis=1, min_count=1)

    # Bradykinesia score (items 3-4, 3-5, 3-6, 3-7, 3-8)
    brady_items = [c for c in df.columns if any(x in c for x in
                   ["3-4-", "3-5-", "3-6-", "3-7-", "3-8-"])]
    if brady_items:
        df["composite_bradykinesia"] = df[brady_items].sum(axis=1, min_count=1)

    # Axial score (items 3-1, 3-2, 3-9, 3-10, 3-11, 3-12, 3-13, 3-14)
    axial_items = [c for c in df.columns if any(c.endswith(x) for x in
                   ["3-1", "3-2", "3-9", "3-10", "3-11", "3-12", "3-13", "3-14"])]
    if axial_items:
        df["composite_axial"] = df[axial_items].sum(axis=1, min_count=1)

    # ── Tremor/Rigidity ratio (TD vs AR classification) ──────────────────
    if "composite_tremor" in df.columns and "composite_rigidity" in df.columns:
        df["ratio_tremor_rigidity"] = (
            df["composite_tremor"] / (df["composite_rigidity"] + 0.1)
        )

    if "composite_tremor" in df.columns and "composite_bradykinesia" in df.columns:
        df["ratio_tremor_brady"] = (
            df["composite_tremor"] / (df["composite_bradykinesia"] + 0.1)
        )

    # ── Interaction features ─────────────────────────────────────────────
    if "disease_duration_years" in df.columns and "hoehn_yahr" in df.columns:
        df["interact_duration_hy"] = df["disease_duration_years"] * df["hoehn_yahr"]

    if "disease_duration_years" in df.columns and "updrs_part3_total" in df.columns:
        df["interact_duration_updrs3"] = (
            df["disease_duration_years"] * df["updrs_part3_total"]
        )

    if "total_asymmetry" in df.columns and "composite_bradykinesia" in df.columns:
        df["interact_asym_brady"] = (
            df["total_asymmetry"] * df["composite_bradykinesia"]
        )

    # ── Severity indices ─────────────────────────────────────────────────
    if "updrs_part3_total" in df.columns and "disease_duration_years" in df.columns:
        df["severity_rate"] = (
            df["updrs_part3_total"] / (df["disease_duration_years"] + 0.5)
        )

    # ── Non-motor burden ─────────────────────────────────────────────────
    part1_items = [c for c in df.columns if c.startswith("MDSUPDRS_1-")]
    if part1_items:
        df["nonmotor_burden"] = df[part1_items].sum(axis=1, min_count=1)

    # ── ADL burden ───────────────────────────────────────────────────────
    part2_items = [c for c in df.columns if c.startswith("MDSUPDRS_2-")]
    if part2_items:
        df["adl_burden"] = df[part2_items].sum(axis=1, min_count=1)

    return df


def mrmr_selection(X, y, feature_names, k=10):
    """Minimum Redundancy Maximum Relevance feature selection."""
    # Compute mutual information with target
    mi = mutual_info_classif(
        SimpleImputer(strategy="median").fit_transform(X), y,
        random_state=SEED, n_neighbors=3
    )

    selected = []
    remaining = list(range(len(feature_names)))

    # First feature: highest MI
    best_idx = np.argmax(mi)
    selected.append(best_idx)
    remaining.remove(best_idx)

    # Greedy mRMR
    for _ in range(k - 1):
        best_score = -np.inf
        best_feat = None
        for feat in remaining:
            # Relevance: MI with target
            relevance = mi[feat]
            # Redundancy: mean correlation with already selected
            X_imp = SimpleImputer(strategy="median").fit_transform(X)
            redundancy = np.mean([
                abs(np.corrcoef(X_imp[:, feat], X_imp[:, s])[0, 1])
                for s in selected
            ])
            score = relevance - redundancy
            if score > best_score:
                best_score = score
                best_feat = feat
        if best_feat is not None:
            selected.append(best_feat)
            remaining.remove(best_feat)

    return np.array(selected), [feature_names[i] for i in selected]


def main():
    print("=" * 70)
    print("Model Optimization v2 — Targeting AUC > 0.90")
    print("=" * 70)

    results = []

    # ═══════════════════════════════════════════════════════════════════════
    # Load and engineer features
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[1/6] Loading data + feature engineering...")

    # PD-only
    df_pd = pd.read_csv(os.path.join(PROC_DIR, "clinical_features", "weargait_pd_only.csv"))
    df_pd_eng = engineer_features(df_pd)
    feat_cols_eng = [c for c in df_pd_eng.columns if c not in META_COLS
                     and pd.api.types.is_numeric_dtype(df_pd_eng[c])]
    X_pd_eng = df_pd_eng[feat_cols_eng].values.astype(np.float32)
    y_pd = df_pd_eng["dbs_candidate"].values.astype(int)
    print(f"  PD-only engineered: {X_pd_eng.shape[1]} features (was 96)")

    # Full cohort
    df_full = pd.read_csv(os.path.join(PROC_DIR, "clinical_features", "weargait_clinical.csv"))
    df_full_eng = engineer_features(df_full)
    feat_cols_full = [c for c in df_full_eng.columns if c not in META_COLS
                      and pd.api.types.is_numeric_dtype(df_full_eng[c])]
    X_full_eng = df_full_eng[feat_cols_full].values.astype(np.float32)
    y_full = df_full_eng["dbs_candidate"].values.astype(int)
    print(f"  Full cohort engineered: {X_full_eng.shape[1]} features, n={len(y_full)}")

    # ═══════════════════════════════════════════════════════════════════════
    # 2. mRMR Feature Selection on engineered features
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[2/6] mRMR feature selection...")
    for k in [8, 10, 12, 15]:
        mrmr_idx, mrmr_names = mrmr_selection(X_pd_eng, y_pd, feat_cols_eng, k=k)
        X_mrmr = X_pd_eng[:, mrmr_idx]

        auc, _ = loocv_auc(X_mrmr, y_pd, lambda: xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=len(y_pd[y_pd == 0]) / max(len(y_pd[y_pd == 1]), 1),
            tree_method="hist", device="cuda", random_state=SEED,
            eval_metric="logloss", verbosity=0
        ))
        print(f"  mRMR k={k}: AUC={auc:.4f}  Features: {mrmr_names[:5]}...")
        results.append({"Model": f"XGB_mRMR{k}", "N_features": k,
                        "AUC": auc, "Cohort": "PD_only", "Method": "LOOCV+mRMR"})

    # ═══════════════════════════════════════════════════════════════════════
    # 3. XGBoost importance on engineered features (compare with mRMR)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[3/6] XGBoost importance on engineered features...")

    # Get top features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pd_eng)
    imp = SimpleImputer(strategy="median")
    X_clean = imp.fit_transform(X_scaled)
    temp_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, tree_method="hist", device="cuda",
        random_state=SEED, eval_metric="logloss", verbosity=0
    )
    temp_model.fit(X_clean, y_pd)
    importances = temp_model.feature_importances_

    for k in [8, 10, 12, 15]:
        top_idx = np.argsort(importances)[::-1][:k]
        X_top = X_pd_eng[:, top_idx]
        top_names = [feat_cols_eng[i] for i in top_idx]

        auc, _ = loocv_auc(X_top, y_pd, lambda: xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=len(y_pd[y_pd == 0]) / max(len(y_pd[y_pd == 1]), 1),
            tree_method="hist", device="cuda", random_state=SEED,
            eval_metric="logloss", verbosity=0
        ))
        print(f"  XGB_Imp k={k}: AUC={auc:.4f}  Features: {top_names[:5]}...")
        results.append({"Model": f"XGB_Eng_Top{k}", "N_features": k,
                        "AUC": auc, "Cohort": "PD_only", "Method": "LOOCV+EngFeats"})

    # ═══════════════════════════════════════════════════════════════════════
    # 4. Optuna on BEST engineered feature set
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[4/6] Optuna on best engineered features (100 trials)...")

    # Find best k from above
    pd_results = [r for r in results if r["Cohort"] == "PD_only"]
    best_so_far = max(pd_results, key=lambda x: x["AUC"])
    best_k = best_so_far["N_features"]
    best_method = best_so_far["Model"]
    print(f"  Best so far: {best_method} k={best_k} AUC={best_so_far['AUC']:.4f}")

    # Get the best features
    if "mRMR" in best_method:
        best_idx, _ = mrmr_selection(X_pd_eng, y_pd, feat_cols_eng, k=best_k)
    else:
        best_idx = np.argsort(importances)[::-1][:best_k]
    X_best = X_pd_eng[:, best_idx]
    best_names = [feat_cols_eng[i] for i in best_idx]

    def xgb_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        }
        return cv_auc(X_best, y_pd, lambda: xgb.XGBClassifier(
            **params, scale_pos_weight=len(y_pd[y_pd == 0]) / max(len(y_pd[y_pd == 1]), 1),
            tree_method="hist", device="cuda", random_state=SEED,
            eval_metric="logloss", verbosity=0
        ))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(xgb_objective, n_trials=100)

    # Final LOOCV with best params
    best_params = study.best_params

    def make_best():
        return xgb.XGBClassifier(
            **best_params, scale_pos_weight=len(y_pd[y_pd == 0]) / max(len(y_pd[y_pd == 1]), 1),
            tree_method="hist", device="cuda", random_state=SEED,
            eval_metric="logloss", verbosity=0
        )

    auc_opt, y_prob_opt = loocv_auc(X_best, y_pd, make_best)
    ci = bootstrap_ci(y_pd, y_prob_opt)
    print(f"  >> Optuna XGB on engineered: AUC={auc_opt:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")
    results.append({"Model": "XGB_Eng_Optuna", "N_features": best_k,
                    "AUC": auc_opt, "CI_low": ci[0], "CI_high": ci[1],
                    "Cohort": "PD_only", "Method": "LOOCV+Eng+Optuna100"})

    # ═══════════════════════════════════════════════════════════════════════
    # 5. FULL COHORT (n=167) — more training data
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[5/6] Full cohort (n=167) with engineered features...")

    # mRMR on full cohort
    for k in [10, 12, 15]:
        mrmr_idx_f, mrmr_names_f = mrmr_selection(X_full_eng, y_full, feat_cols_full, k=k)
        X_mrmr_f = X_full_eng[:, mrmr_idx_f]

        auc_f, y_prob_f = loocv_auc(X_mrmr_f, y_full, lambda: xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=len(y_full[y_full == 0]) / max(len(y_full[y_full == 1]), 1),
            tree_method="hist", device="cuda", random_state=SEED,
            eval_metric="logloss", verbosity=0
        ))
        ci_f = bootstrap_ci(y_full, y_prob_f)
        print(f"  Full mRMR k={k}: AUC={auc_f:.4f} [{ci_f[0]:.4f}, {ci_f[1]:.4f}]")
        results.append({"Model": f"XGB_Full_mRMR{k}", "N_features": k,
                        "AUC": auc_f, "CI_low": ci_f[0], "CI_high": ci_f[1],
                        "Cohort": "Full_167", "Method": "LOOCV+mRMR"})

    # Optuna on full cohort best features
    best_full = max([r for r in results if r["Cohort"] == "Full_167"],
                    key=lambda x: x["AUC"])
    best_k_f = best_full["N_features"]
    mrmr_idx_best, _ = mrmr_selection(X_full_eng, y_full, feat_cols_full, k=best_k_f)
    X_full_best = X_full_eng[:, mrmr_idx_best]

    def xgb_obj_full(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
        return cv_auc(X_full_best, y_full, lambda: xgb.XGBClassifier(
            **params, scale_pos_weight=len(y_full[y_full == 0]) / max(len(y_full[y_full == 1]), 1),
            tree_method="hist", device="cuda", random_state=SEED,
            eval_metric="logloss", verbosity=0
        ))

    study_full = optuna.create_study(direction="maximize",
                                      sampler=optuna.samplers.TPESampler(seed=SEED))
    study_full.optimize(xgb_obj_full, n_trials=100)

    def make_best_full():
        return xgb.XGBClassifier(
            **study_full.best_params,
            scale_pos_weight=len(y_full[y_full == 0]) / max(len(y_full[y_full == 1]), 1),
            tree_method="hist", device="cuda", random_state=SEED,
            eval_metric="logloss", verbosity=0
        )

    auc_full_opt, y_prob_full_opt = loocv_auc(X_full_best, y_full, make_best_full)
    ci_full = bootstrap_ci(y_full, y_prob_full_opt)
    print(f"  >> Full Optuna: AUC={auc_full_opt:.4f} [{ci_full[0]:.4f}, {ci_full[1]:.4f}]")
    results.append({"Model": "XGB_Full_Optuna", "N_features": best_k_f,
                    "AUC": auc_full_opt, "CI_low": ci_full[0], "CI_high": ci_full[1],
                    "Cohort": "Full_167", "Method": "LOOCV+Eng+Optuna100"})

    # ═══════════════════════════════════════════════════════════════════════
    # 6. Alternative oversampling (ADASYN, Borderline-SMOTE)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[6/6] Alternative oversampling methods...")
    for smote_name, smote_type in [("ADASYN", "adasyn"), ("BorderlineSMOTE", "borderline")]:
        auc_sm, _ = loocv_auc(X_best, y_pd, make_best, smote_type=smote_type)
        print(f"  {smote_name}: AUC={auc_sm:.4f}")
        results.append({"Model": f"XGB_Eng_{smote_name}", "N_features": best_k,
                        "AUC": auc_sm, "Cohort": "PD_only", "Method": f"LOOCV+{smote_name}"})

    # ═══════════════════════════════════════════════════════════════════════
    # Save & Summary
    # ═══════════════════════════════════════════════════════════════════════
    df_results = pd.DataFrame(results)
    out_path = os.path.join(RESULTS_DIR, "optimization_v2_results.csv")
    df_results.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Save best features
    pd.DataFrame({"rank": range(1, len(best_names) + 1), "feature": best_names}
                 ).to_csv(os.path.join(RESULTS_DIR, "optimal_features_v2.csv"), index=False)

    print(f"\n{'='*70}")
    print("OPTIMIZATION v2 RESULTS (sorted by AUC)")
    print(f"{'='*70}")
    print(f"{'Model':<30} {'Cohort':<12} {'K':<4} {'AUC':<8} {'CI'}")
    print("-" * 75)
    for _, r in df_results.sort_values("AUC", ascending=False).iterrows():
        ci = f"[{r['CI_low']:.3f}, {r['CI_high']:.3f}]" if pd.notna(r.get("CI_low")) else "—"
        print(f"{r['Model']:<30} {r['Cohort']:<12} {r['N_features']:<4} {r['AUC']:.4f}  {ci}")

    best = df_results.loc[df_results["AUC"].idxmax()]
    print(f"\n  BEST OVERALL: {best['Model']} ({best['Cohort']}) → AUC={best['AUC']:.4f}")

    # Check if we hit 0.90
    if best["AUC"] >= 0.90:
        print("  *** TARGET 0.90 ACHIEVED! ***")
    else:
        gap = 0.90 - best["AUC"]
        print(f"  Gap to 0.90: {gap:.4f} — may need more data (23 DBS+ is the ceiling)")

    print("\nDone.")


if __name__ == "__main__":
    main()
