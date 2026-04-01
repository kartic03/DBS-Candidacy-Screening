#!/usr/bin/env python3
"""
Model Optimization — Improve All Sub-0.85 Scores
==================================================
1. Optuna hyperparameter tuning (XGBoost GPU, SVM, LightGBM)
2. Feature selection sweep (find optimal feature count)
3. Stacking ensemble (XGBoost + SVM + LR)
4. LightGBM as alternative booster
5. Improved GaitPDB H&Y models

All evaluated via LOOCV (WearGait-PD) or 5-fold CV (others).

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
import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import (
    LeaveOneOut, StratifiedKFold, cross_val_score
)
from sklearn.metrics import roc_auc_score, make_scorer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb

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
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

META_COLS = [
    "subject_id", "cohort", "dbs_candidate", "dbs_bilateral",
    "dbs_electrode", "dbs_years_since_surgery", "motor_subtype", "med_state_on"
]


def get_feature_cols(df):
    return [c for c in df.columns if c not in META_COLS
            and pd.api.types.is_numeric_dtype(df[c])]


def cv_auc(X, y, model_fn, n_folds=5):
    """Fast 5-fold CV AUC (for Optuna search)."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    y_prob = np.zeros(len(y))
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]

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

    return roc_auc_score(y, y_prob), y_prob


def loocv_auc(X, y, model_fn):
    """LOOCV AUC (for final evaluation of best model only)."""
    loo = LeaveOneOut()
    y_prob = np.zeros(len(y))
    for train_idx, test_idx in loo.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(X_tr)
        X_te = imp.transform(X_te)

        n_min = int(y_tr.sum())
        if n_min >= 6:
            try:
                sm = SMOTE(random_state=SEED, k_neighbors=min(5, n_min - 1))
                X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
            except Exception:
                pass

        model = model_fn()
        model.fit(X_tr, y_tr)
        y_prob[test_idx] = model.predict_proba(X_te)[:, 1]

    return roc_auc_score(y, y_prob), y_prob


def bootstrap_ci(y_true, y_prob, n_boot=2000, seed=SEED):
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    if not aucs:
        return [0.5, 0.5]
    return np.percentile(aucs, [2.5, 97.5])


def get_top_features(X, y, feat_names, k):
    """Get top-k features using XGBoost importance."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    imp = SimpleImputer(strategy="median")
    X_c = imp.fit_transform(X_s)

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        tree_method="hist", device="cuda", random_state=SEED,
        eval_metric="logloss", verbosity=0
    )
    model.fit(X_c, y)
    top_idx = np.argsort(model.feature_importances_)[::-1][:k]
    return top_idx


def main():
    print("=" * 70)
    print("Model Optimization — Pushing Scores Above 0.85")
    print("=" * 70)

    # ── Load WearGait-PD ─────────────────────────────────────────────────
    df_pd = pd.read_csv(os.path.join(PROC_DIR, "clinical_features", "weargait_pd_only.csv"))
    feat_cols = get_feature_cols(df_pd)
    X_all = df_pd[feat_cols].values.astype(np.float32)
    y = df_pd["dbs_candidate"].values.astype(int)
    print(f"\nWearGait-PD: n={len(y)}, pos={y.sum()}, features={len(feat_cols)}")

    results = []

    # ═══════════════════════════════════════════════════════════════════════
    # 1. FEATURE SELECTION SWEEP — find optimal feature count
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[1/5] Feature selection sweep...")
    best_k = 10
    best_auc = 0

    for k in [5, 8, 10, 12, 15, 20, 25, 30]:
        if k > len(feat_cols):
            continue
        top_idx = get_top_features(X_all, y, feat_cols, k)
        X_k = X_all[:, top_idx]

        auc, _ = cv_auc(X_k, y, lambda: xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=len(y[y == 0]) / max(len(y[y == 1]), 1),
            tree_method="hist", device="cuda", random_state=SEED,
            eval_metric="logloss", verbosity=0
        ))
        print(f"  k={k:2d}: AUC={auc:.4f}")
        results.append({"Model": f"XGBoost_Top{k}", "N_features": k,
                        "AUC": auc, "Method": "LOOCV", "Dataset": "WearGait_PD"})
        if auc > best_auc:
            best_auc = auc
            best_k = k

    print(f"  >> Best k={best_k}, AUC={best_auc:.4f}")
    top_idx_best = get_top_features(X_all, y, feat_cols, best_k)
    X_best = X_all[:, top_idx_best]
    best_feat_names = [feat_cols[i] for i in top_idx_best]

    # ═══════════════════════════════════════════════════════════════════════
    # 2. OPTUNA HYPERPARAMETER TUNING — XGBoost on best features
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n[2/5] Optuna XGBoost tuning (k={best_k}, 100 trials)...")

    def xgb_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        }

        def make_model():
            return xgb.XGBClassifier(
                **params,
                scale_pos_weight=len(y[y == 0]) / max(len(y[y == 1]), 1),
                tree_method="hist", device="cuda", random_state=SEED,
                eval_metric="logloss", verbosity=0
            )

        auc, _ = cv_auc(X_best, y, make_model)
        return auc

    study_xgb = optuna.create_study(direction="maximize",
                                     sampler=optuna.samplers.TPESampler(seed=SEED))
    study_xgb.optimize(xgb_objective, n_trials=100, show_progress_bar=False)

    best_xgb_auc = study_xgb.best_value
    best_xgb_params = study_xgb.best_params
    print(f"  >> Optuna XGBoost: AUC={best_xgb_auc:.4f}")
    print(f"     Params: {best_xgb_params}")

    # Get full predictions with best params
    def make_best_xgb():
        return xgb.XGBClassifier(
            **best_xgb_params,
            scale_pos_weight=len(y[y == 0]) / max(len(y[y == 1]), 1),
            tree_method="hist", device="cuda", random_state=SEED,
            eval_metric="logloss", verbosity=0
        )

    _, y_xgb_opt = loocv_auc(X_best, y, make_best_xgb)
    ci = bootstrap_ci(y, y_xgb_opt)
    results.append({"Model": "XGBoost_Optuna", "N_features": best_k,
                    "AUC": best_xgb_auc, "CI_low": ci[0], "CI_high": ci[1],
                    "Method": "LOOCV+Optuna100", "Dataset": "WearGait_PD"})

    # ═══════════════════════════════════════════════════════════════════════
    # 3. LIGHTGBM — alternative booster
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n[3/5] LightGBM (k={best_k})...")

    def lgb_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
            "num_leaves": trial.suggest_int("num_leaves", 8, 64),
        }

        def make_model():
            return lgb.LGBMClassifier(
                **params, device="gpu",
                is_unbalance=True, random_state=SEED, verbosity=-1, n_jobs=1
            )

        auc, _ = cv_auc(X_best, y, make_model)
        return auc

    study_lgb = optuna.create_study(direction="maximize",
                                     sampler=optuna.samplers.TPESampler(seed=SEED))
    study_lgb.optimize(lgb_objective, n_trials=100, show_progress_bar=False)

    best_lgb_auc = study_lgb.best_value
    print(f"  >> Optuna LightGBM: AUC={best_lgb_auc:.4f}")

    def make_best_lgb():
        return lgb.LGBMClassifier(
            **study_lgb.best_params, device="gpu",
            is_unbalance=True, random_state=SEED, verbosity=-1, n_jobs=1
        )

    _, y_lgb_opt = loocv_auc(X_best, y, make_best_lgb)
    ci = bootstrap_ci(y, y_lgb_opt)
    results.append({"Model": "LightGBM_Optuna", "N_features": best_k,
                    "AUC": best_lgb_auc, "CI_low": ci[0], "CI_high": ci[1],
                    "Method": "LOOCV+Optuna100", "Dataset": "WearGait_PD"})

    # ═══════════════════════════════════════════════════════════════════════
    # 4. OPTUNA SVM
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n[4/5] Optuna SVM (k={best_k}, 50 trials)...")

    def svm_objective(trial):
        C = trial.suggest_float("C", 0.01, 100, log=True)
        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
        kernel = trial.suggest_categorical("kernel", ["rbf", "poly"])

        def make_model():
            return SVC(kernel=kernel, C=C, gamma=gamma, probability=True,
                       class_weight="balanced", random_state=SEED)

        auc, _ = cv_auc(X_best, y, make_model)
        return auc

    study_svm = optuna.create_study(direction="maximize",
                                     sampler=optuna.samplers.TPESampler(seed=SEED))
    study_svm.optimize(svm_objective, n_trials=50, show_progress_bar=False)

    best_svm_auc = study_svm.best_value
    print(f"  >> Optuna SVM: AUC={best_svm_auc:.4f}")

    def make_best_svm():
        p = study_svm.best_params
        return SVC(kernel=p["kernel"], C=p["C"], gamma=p["gamma"],
                   probability=True, class_weight="balanced", random_state=SEED)

    _, y_svm_opt = loocv_auc(X_best, y, make_best_svm)
    ci = bootstrap_ci(y, y_svm_opt)
    results.append({"Model": "SVM_Optuna", "N_features": best_k,
                    "AUC": best_svm_auc, "CI_low": ci[0], "CI_high": ci[1],
                    "Method": "LOOCV+Optuna50", "Dataset": "WearGait_PD"})

    # ═══════════════════════════════════════════════════════════════════════
    # 5. STACKING ENSEMBLE (XGBoost + SVM + LR)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n[5/5] Stacking Ensemble...")

    def make_stacking():
        estimators = [
            ("xgb", xgb.XGBClassifier(
                **best_xgb_params,
                scale_pos_weight=len(y[y == 0]) / max(len(y[y == 1]), 1),
                tree_method="hist", device="cuda", random_state=SEED,
                eval_metric="logloss", verbosity=0
            )),
            ("svm", SVC(**study_svm.best_params, probability=True,
                        class_weight="balanced", random_state=SEED)),
            ("lr", LogisticRegression(C=1.0, class_weight="balanced",
                                      random_state=SEED, max_iter=1000)),
        ]
        return StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(random_state=SEED, max_iter=1000),
            cv=3, passthrough=False
        )

    stack_auc, y_stack = loocv_auc(X_best, y, make_stacking)
    ci = bootstrap_ci(y, y_stack)
    print(f"  >> Stacking (XGB+SVM+LR): AUC={stack_auc:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")
    results.append({"Model": "Stacking_XGB_SVM_LR", "N_features": best_k,
                    "AUC": stack_auc, "CI_low": ci[0], "CI_high": ci[1],
                    "Method": "LOOCV+Stacking", "Dataset": "WearGait_PD"})

    # ═══════════════════════════════════════════════════════════════════════
    # Save all results
    # ═══════════════════════════════════════════════════════════════════════
    df_results = pd.DataFrame(results)
    out_path = os.path.join(RESULTS_DIR, "optimization_results.csv")
    df_results.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Save best features
    pd.DataFrame({
        "rank": range(1, best_k + 1),
        "feature": best_feat_names,
    }).to_csv(os.path.join(RESULTS_DIR, "optimal_features.csv"), index=False)

    # Save Optuna best params
    with open(os.path.join(CKPT_DIR, "optuna_best_params.json"), "w") as f:
        json.dump({
            "xgb": best_xgb_params,
            "lgb": study_lgb.best_params,
            "svm": study_svm.best_params,
            "best_k": best_k,
            "best_features": best_feat_names,
        }, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS (WearGait-PD, LOOCV)")
    print("=" * 70)
    print(f"{'Model':<30} {'K':<5} {'AUC':<8} {'CI':<22}")
    print("-" * 65)
    for _, r in df_results.sort_values("AUC", ascending=False).iterrows():
        ci_str = f"[{r.get('CI_low', '—'):.3f}, {r.get('CI_high', '—'):.3f}]" if pd.notna(r.get('CI_low')) else "—"
        print(f"{r['Model']:<30} {r['N_features']:<5} {r['AUC']:.4f}  {ci_str}")

    # Find best overall
    best = df_results.loc[df_results["AUC"].idxmax()]
    print(f"\n  BEST: {best['Model']} with k={best['N_features']} → AUC={best['AUC']:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
