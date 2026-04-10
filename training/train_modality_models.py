#!/usr/bin/env python3
"""
Train Modality-Specific Models (PADS Wearable, GaitPDB Gait, UCI Voice)
========================================================================
Trains XGBoost, SVM, and MLP on each modality's own dataset with proper labels.
GPU used for MLP via PyTorch CUDA.
5-fold CV for every model.

Output:
  results/tables/modality_model_results.csv
  results/checkpoints/modality_*.pkl / *.pt

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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef, brier_score_loss, confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings("ignore")

# ── Hardware config ──────────────────────────────────────────────────────────
N_CORES = multiprocessing.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(N_CORES)
os.environ["MKL_NUM_THREADS"] = str(N_CORES)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
print(f"Device: {device}")

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
SPLIT_DIR = os.path.join(PROJECT_ROOT, "data", "splits")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "tables")
CKPT_DIR = os.path.join(PROJECT_ROOT, "results", "checkpoints")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

N_FOLDS = 5


# ── MLP Model (GPU) ─────────────────────────────────────────────────────────
class ModalityMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 64), dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)
            ])
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def compute_metrics(y_true, y_prob, threshold=None):
    """Compute classification metrics with Youden's J threshold."""
    if threshold is None:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        j_scores = tpr - fpr
        threshold = thresholds[np.argmax(j_scores)]

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "AUC_ROC": roc_auc_score(y_true, y_prob),
        "AUC_PR": average_precision_score(y_true, y_prob),
        "Sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Brier": brier_score_loss(y_true, y_prob),
        "Threshold": threshold,
    }


def bootstrap_ci(y_true, y_prob, n_boot=2000, seed=SEED):
    rng = np.random.RandomState(seed)
    aucs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    if len(aucs) == 0:
        return [0.5, 0.5]
    return np.percentile(aucs, [2.5, 97.5])


def train_mlp_fold(X_train, y_train, X_val, y_val, input_dim, epochs=100, batch_size=64, lr=1e-3):
    """Train MLP on GPU for one fold. Returns val probabilities."""
    model = ModalityMLP(input_dim).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)],
                            dtype=torch.float32).to(device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)

    model.train()
    for epoch in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            with autocast(dtype=torch.float16):
                out = model(xb)
                loss = criterion(out, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        with autocast(dtype=torch.float16):
            logits = model(X_val_t)
        probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()

    return probs, model


def train_5fold_cv(X, y, feature_names, dataset_name, label_name, splits_path=None):
    """Train XGBoost, SVM, MLP with 5-fold CV on a dataset."""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name} | Label: {label_name}")
    print(f"N={len(y)}, Pos={int(y.sum())}, Neg={int((1-y).sum())}, Features={X.shape[1]}")
    print(f"{'='*60}")

    # Load or create splits
    if splits_path and os.path.exists(splits_path):
        with open(splits_path) as f:
            splits = json.load(f)
        cv_folds = splits["cv_folds"]
    else:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        cv_folds = []
        for train_idx, val_idx in skf.split(X, y):
            cv_folds.append({"train": train_idx.tolist(), "val": val_idx.tolist()})

    results = []

    for model_name in ["XGBoost", "SVM", "MLP"]:
        print(f"\n  Training {model_name} (5-fold CV)...")
        y_prob_all = np.zeros(len(y))
        y_assigned = np.zeros(len(y), dtype=bool)
        fold_aucs = []

        for fold_i, fold in enumerate(cv_folds):
            train_idx = np.array(fold["train"])
            val_idx = np.array(fold["val"])

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Scale within fold
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

            # Impute within fold
            imputer = SimpleImputer(strategy="median")
            X_train = imputer.fit_transform(X_train)
            X_val = imputer.transform(X_val)

            # SMOTE on training only
            n_minority = int(y_train.sum())
            if n_minority >= 3:
                try:
                    sm = SMOTE(random_state=SEED, k_neighbors=min(5, n_minority - 1))
                    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
                except Exception:
                    X_train_sm, y_train_sm = X_train, y_train
            else:
                X_train_sm, y_train_sm = X_train, y_train

            if model_name == "XGBoost":
                model = xgb.XGBClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    scale_pos_weight=len(y_train_sm[y_train_sm == 0]) / max(len(y_train_sm[y_train_sm == 1]), 1),
                    tree_method="hist", device="cuda", n_jobs=N_CORES, random_state=SEED,
                    eval_metric="logloss", verbosity=0
                )
                model.fit(X_train_sm, y_train_sm)
                y_prob_fold = model.predict_proba(X_val)[:, 1]

            elif model_name == "SVM":
                model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                            class_weight="balanced", random_state=SEED)
                model.fit(X_train_sm, y_train_sm)
                y_prob_fold = model.predict_proba(X_val)[:, 1]

            elif model_name == "MLP":
                y_prob_fold, model = train_mlp_fold(
                    X_train_sm, y_train_sm, X_val, y_val,
                    input_dim=X_train.shape[1], epochs=100, batch_size=64
                )

            y_prob_all[val_idx] = y_prob_fold
            y_assigned[val_idx] = True

            if len(np.unique(y_val)) >= 2:
                fold_auc = roc_auc_score(y_val, y_prob_fold)
                fold_aucs.append(fold_auc)
                print(f"    Fold {fold_i}: AUC={fold_auc:.4f} "
                      f"(train={len(train_idx)}, val={len(val_idx)}, val_pos={int(y_val.sum())})")

        # Overall metrics (from concatenated OOF predictions)
        mask = y_assigned
        if len(np.unique(y[mask])) >= 2:
            metrics = compute_metrics(y[mask], y_prob_all[mask])
            ci = bootstrap_ci(y[mask], y_prob_all[mask])
        else:
            metrics = {"AUC_ROC": np.nan}
            ci = [np.nan, np.nan]

        metrics["Model"] = model_name
        metrics["Dataset"] = dataset_name
        metrics["Label"] = label_name
        metrics["N"] = len(y)
        metrics["N_pos"] = int(y.sum())
        metrics["N_features"] = X.shape[1]
        metrics["Evaluation"] = f"5fold_CV"
        metrics["AUC_CI_low"] = ci[0]
        metrics["AUC_CI_high"] = ci[1]
        metrics["Mean_fold_AUC"] = np.mean(fold_aucs) if fold_aucs else np.nan
        metrics["Std_fold_AUC"] = np.std(fold_aucs) if fold_aucs else np.nan

        auc = metrics.get("AUC_ROC", np.nan)
        print(f"  >> {model_name}: OOF AUC={auc:.4f} [{ci[0]:.4f}, {ci[1]:.4f}], "
              f"Mean fold AUC={metrics['Mean_fold_AUC']:.4f} +/- {metrics['Std_fold_AUC']:.4f}")

        results.append(metrics)

    return results


def main():
    print("=" * 70)
    print("Training Modality-Specific Models (5-fold CV, GPU for MLP)")
    print("=" * 70)

    all_results = []

    # ═══════════════════════════════════════════════════════════════════════
    # 1. PADS Wearable Sensor (PD vs Healthy)
    # ═══════════════════════════════════════════════════════════════════════
    df_pads = pd.read_csv(os.path.join(PROC_DIR, "wearable_features", "pads_sensor_pd_vs_healthy.csv"))
    meta_cols = ["subject_id", "condition", "age", "gender", "height", "weight",
                 "pd_label", "n_files_processed", "handedness"]
    feat_cols = [c for c in df_pads.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df_pads[c])]
    X = df_pads[feat_cols].values.astype(np.float32)
    y = df_pads["pd_label"].values.astype(int)

    pads_results = train_5fold_cv(
        X, y, feat_cols, "PADS_Wearable", "PD_vs_Healthy",
        os.path.join(SPLIT_DIR, "pads_splits.json")
    )
    all_results.extend(pads_results)

    # ═══════════════════════════════════════════════════════════════════════
    # 2. GaitPDB (PD vs Control)
    # ═══════════════════════════════════════════════════════════════════════
    df_gait = pd.read_csv(os.path.join(PROC_DIR, "gait_features", "gaitpdb_sensor_features.csv"))
    gait_meta = ["subject_id", "Study", "Group", "Gender", "pd_label", "sex",
                 "dbs_proxy_hy25", "dbs_proxy_updrs32", "n_single_trials"]
    gait_feats = [c for c in df_gait.columns if c not in gait_meta
                  and pd.api.types.is_numeric_dtype(df_gait[c])]
    X_gait = df_gait[gait_feats].values.astype(np.float32)
    y_gait = df_gait["pd_label"].values.astype(int)

    gait_pd_results = train_5fold_cv(
        X_gait, y_gait, gait_feats, "GaitPDB_Gait", "PD_vs_Control",
        os.path.join(SPLIT_DIR, "gaitpdb_pd_splits.json")
    )
    all_results.extend(gait_pd_results)

    # ═══════════════════════════════════════════════════════════════════════
    # 3. GaitPDB H&Y >= 2.5 (DBS proxy)
    # ═══════════════════════════════════════════════════════════════════════
    df_gait_hy = pd.read_csv(os.path.join(PROC_DIR, "gait_features", "gaitpdb_with_clinical.csv"))
    df_gait_hy = df_gait_hy[df_gait_hy["dbs_proxy_hy25"].notna()].copy()
    df_gait_hy["dbs_proxy_hy25"] = df_gait_hy["dbs_proxy_hy25"].astype(int)
    X_hy = df_gait_hy[gait_feats].values.astype(np.float32)
    y_hy = df_gait_hy["dbs_proxy_hy25"].values.astype(int)

    gait_hy_results = train_5fold_cv(
        X_hy, y_hy, gait_feats, "GaitPDB_Gait", "HY_ge_2.5",
        os.path.join(SPLIT_DIR, "gaitpdb_hy_splits.json")
    )
    all_results.extend(gait_hy_results)

    # ═══════════════════════════════════════════════════════════════════════
    # 4. UCI Voice (PD vs Healthy)
    # ═══════════════════════════════════════════════════════════════════════
    df_voice = pd.read_csv(os.path.join(PROC_DIR, "voice_features", "uci_voice_features.csv"))
    voice_meta = ["name", "subject_id", "pd_status"]
    # Find label column
    if "pd_status" in df_voice.columns:
        label_col = "pd_status"
    elif "status" in df_voice.columns:
        label_col = "status"
    else:
        label_col = [c for c in df_voice.columns if df_voice[c].nunique() == 2][0]

    voice_feats = [c for c in df_voice.columns if c not in voice_meta + [label_col]
                   and pd.api.types.is_numeric_dtype(df_voice[c])]
    X_voice = df_voice[voice_feats].values.astype(np.float32)
    y_voice = df_voice[label_col].values.astype(int)

    voice_results = train_5fold_cv(
        X_voice, y_voice, voice_feats, "UCI_Voice", "PD_vs_Healthy",
        os.path.join(SPLIT_DIR, "uci_voice_splits.json")
    )
    all_results.extend(voice_results)

    # ═══════════════════════════════════════════════════════════════════════
    # Save all results
    # ═══════════════════════════════════════════════════════════════════════
    df_all = pd.DataFrame(all_results)
    out_path = os.path.join(RESULTS_DIR, "modality_model_results.csv")
    df_all.to_csv(out_path, index=False)
    print(f"\n{'='*70}")
    print(f"All results saved: {out_path}")

    # Pretty summary
    print(f"\n{'='*70}")
    print("SUMMARY: All Modality Models (5-fold CV)")
    print(f"{'='*70}")
    print(f"{'Dataset':<20} {'Model':<10} {'Label':<15} {'AUC':<8} {'CI':<20} {'Sens':<6} {'Spec':<6}")
    print("-" * 85)
    for _, r in df_all.iterrows():
        auc = r.get("AUC_ROC", np.nan)
        ci_lo = r.get("AUC_CI_low", np.nan)
        ci_hi = r.get("AUC_CI_high", np.nan)
        ci_str = f"[{ci_lo:.3f}, {ci_hi:.3f}]" if pd.notna(ci_lo) else "—"
        sens = r.get("Sensitivity", np.nan)
        spec = r.get("Specificity", np.nan)
        print(f"{r['Dataset']:<20} {r['Model']:<10} {r['Label']:<15} "
              f"{auc:.4f}  {ci_str:<20} {sens:.3f}  {spec:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
