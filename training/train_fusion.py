#!/usr/bin/env python3
"""
Train Cross-Attention Fusion on GaitPDB (Gait Sensor + Clinical Features)
==========================================================================
GaitPDB is the ONLY dataset with both gait sensor data AND clinical scores
(H&Y, UPDRS). This enables genuine 2-modality fusion.

Models:
  - Gait-only encoder (baseline)
  - Clinical-only encoder (baseline)
  - Simple concatenation (baseline)
  - Cross-attention fusion (proposed)

Target: H&Y >= 2.5 (DBS proxy, n=38 positive out of 111)
Evaluation: 5-fold CV with bootstrap CIs

Output:
  results/tables/fusion_model_results.csv
  results/checkpoints/fusion_*.pt

Author: Kartic Mishra, Gachon University
"""

import os
import sys
import json
import warnings
import multiprocessing

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef, brier_score_loss, confusion_matrix, roc_curve
from imblearn.over_sampling import SMOTE
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings("ignore")

# ── Hardware ─────────────────────────────────────────────────────────────────
N_CORES = multiprocessing.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(N_CORES)
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


# ── Model Architectures ─────────────────────────────────────────────────────
class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim=64, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query, key_value):
        q = query.unsqueeze(1)
        kv = key_value.unsqueeze(1)
        attn_out, attn_weights = self.attn(q, kv, kv)
        x = self.norm(q + attn_out)
        x = self.norm2(x + self.ff(x))
        return x.squeeze(1), attn_weights.squeeze(1)


class CrossAttentionFusion(nn.Module):
    def __init__(self, gait_dim, clinical_dim, embed_dim=64, n_heads=4, dropout=0.3):
        super().__init__()
        self.gait_encoder = ModalityEncoder(gait_dim, embed_dim, dropout)
        self.clinical_encoder = ModalityEncoder(clinical_dim, embed_dim, dropout)
        self.gait_attends_clinical = CrossAttentionBlock(embed_dim, n_heads)
        self.clinical_attends_gait = CrossAttentionBlock(embed_dim, n_heads)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 4, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x_gait, x_clinical):
        g = self.gait_encoder(x_gait)
        c = self.clinical_encoder(x_clinical)
        g_cross, attn_gc = self.gait_attends_clinical(g, c)
        c_cross, attn_cg = self.clinical_attends_gait(c, g)
        fused = torch.cat([g, g_cross, c, c_cross], dim=1)
        logits = self.classifier(fused)
        return logits, (attn_gc, attn_cg)


class SimpleConcatFusion(nn.Module):
    def __init__(self, gait_dim, clinical_dim, embed_dim=64, dropout=0.3):
        super().__init__()
        self.gait_encoder = ModalityEncoder(gait_dim, embed_dim, dropout)
        self.clinical_encoder = ModalityEncoder(clinical_dim, embed_dim, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x_gait, x_clinical):
        g = self.gait_encoder(x_gait)
        c = self.clinical_encoder(x_clinical)
        return self.classifier(torch.cat([g, c], dim=1)), None


class SingleModalityModel(nn.Module):
    def __init__(self, input_dim, embed_dim=64, dropout=0.3):
        super().__init__()
        self.encoder = ModalityEncoder(input_dim, embed_dim, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.classifier(self.encoder(x)), None


def compute_metrics(y_true, y_prob, threshold=None):
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
    if not aucs:
        return [0.5, 0.5]
    return np.percentile(aucs, [2.5, 97.5])


def train_fold(model, X_train_dict, y_train, X_val_dict, y_val,
               model_type="fusion", epochs=150, batch_size=32, lr=3e-4, patience=20):
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    weight = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scaler = GradScaler()

    if model_type in ["fusion", "concat"]:
        train_ds = TensorDataset(
            torch.tensor(X_train_dict["gait"], dtype=torch.float32),
            torch.tensor(X_train_dict["clinical"], dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
    else:
        train_ds = TensorDataset(
            torch.tensor(X_train_dict["single"], dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)

    best_val_auc = 0
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            if model_type in ["fusion", "concat"]:
                x_g, x_c, yb = [b.to(device) for b in batch]
                optimizer.zero_grad()
                with autocast(dtype=torch.float16):
                    logits, _ = model(x_g, x_c)
                    loss = criterion(logits, yb)
            else:
                x, yb = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()
                with autocast(dtype=torch.float16):
                    logits, _ = model(x)
                    loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Validation
        model.eval()
        with torch.no_grad():
            if model_type in ["fusion", "concat"]:
                x_g_v = torch.tensor(X_val_dict["gait"], dtype=torch.float32).to(device)
                x_c_v = torch.tensor(X_val_dict["clinical"], dtype=torch.float32).to(device)
                with autocast(dtype=torch.float16):
                    val_logits, _ = model(x_g_v, x_c_v)
            else:
                x_v = torch.tensor(X_val_dict["single"], dtype=torch.float32).to(device)
                with autocast(dtype=torch.float16):
                    val_logits, _ = model(x_v)

            val_probs = torch.softmax(val_logits.float(), dim=1)[:, 1].cpu().numpy()

        if len(np.unique(y_val)) >= 2:
            val_auc = roc_auc_score(y_val, val_probs)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        if model_type in ["fusion", "concat"]:
            x_g_v = torch.tensor(X_val_dict["gait"], dtype=torch.float32).to(device)
            x_c_v = torch.tensor(X_val_dict["clinical"], dtype=torch.float32).to(device)
            with autocast(dtype=torch.float16):
                val_logits, _ = model(x_g_v, x_c_v)
        else:
            x_v = torch.tensor(X_val_dict["single"], dtype=torch.float32).to(device)
            with autocast(dtype=torch.float16):
                val_logits, _ = model(x_v)
        val_probs = torch.softmax(val_logits.float(), dim=1)[:, 1].cpu().numpy()

    return val_probs, model


def main():
    print("=" * 70)
    print("Training Cross-Attention Fusion on GaitPDB")
    print("(Gait Sensor + Clinical Features — Genuine 2-Modality Fusion)")
    print("=" * 70)

    # ── Load data ────────────────────────────────────────────────────────
    print("\n[1/3] Loading data...")
    df = pd.read_csv(os.path.join(PROC_DIR, "gait_features", "gaitpdb_with_clinical.csv"))
    df = df[df["dbs_proxy_hy25"].notna()].copy()
    df["dbs_proxy_hy25"] = df["dbs_proxy_hy25"].astype(int)

    gait_meta = ["subject_id", "Study", "Group", "Gender", "pd_label", "sex",
                 "dbs_proxy_hy25", "dbs_proxy_updrs32", "n_single_trials",
                 "Age", "Height", "Weight", "HoehnYahr", "UPDRS", "UPDRSM", "TUAG"]
    clinical_cols = ["Age", "Height", "Weight", "HoehnYahr", "UPDRS", "UPDRSM", "TUAG"]
    gait_sensor_cols = [c for c in df.columns if c not in gait_meta
                        and pd.api.types.is_numeric_dtype(df[c])]

    for c in clinical_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    X_gait = df[gait_sensor_cols].values.astype(np.float32)
    X_clinical = df[clinical_cols].values.astype(np.float32)
    y = df["dbs_proxy_hy25"].values.astype(int)

    print(f"  N={len(y)}, Pos(H&Y>=2.5)={y.sum()}, Neg={(1-y).sum()}")
    print(f"  Gait sensor features: {X_gait.shape[1]}")
    print(f"  Clinical features: {X_clinical.shape[1]}")

    splits_path = os.path.join(SPLIT_DIR, "gaitpdb_hy_splits.json")
    with open(splits_path) as f:
        splits = json.load(f)
    cv_folds = splits["cv_folds"]

    # ── Train all model variants ─────────────────────────────────────────
    print("\n[2/3] Training models (5-fold CV on GPU)...")

    model_configs = {
        "CrossAttention": {"type": "fusion"},
        "SimpleConcat": {"type": "concat"},
        "GaitOnly": {"type": "single", "features": "gait"},
        "ClinicalOnly": {"type": "single", "features": "clinical"},
    }

    all_results = []

    for model_name, config in model_configs.items():
        print(f"\n  ── {model_name} ──")
        y_prob_all = np.zeros(len(y))
        fold_aucs = []

        for fold_i, fold in enumerate(cv_folds):
            train_idx = np.array(fold["train"])
            val_idx = np.array(fold["val"])

            X_g_train, X_g_val = X_gait[train_idx], X_gait[val_idx]
            X_c_train, X_c_val = X_clinical[train_idx], X_clinical[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            scaler_g = StandardScaler()
            X_g_train = scaler_g.fit_transform(X_g_train)
            X_g_val = scaler_g.transform(X_g_val)

            scaler_c = StandardScaler()
            X_c_train = scaler_c.fit_transform(X_c_train)
            X_c_val = scaler_c.transform(X_c_val)

            imp_g = SimpleImputer(strategy="median")
            X_g_train = imp_g.fit_transform(X_g_train)
            X_g_val = imp_g.transform(X_g_val)

            imp_c = SimpleImputer(strategy="median")
            X_c_train = imp_c.fit_transform(X_c_train)
            X_c_val = imp_c.transform(X_c_val)

            n_min = int(y_train.sum())
            if n_min >= 3:
                try:
                    sm = SMOTE(random_state=SEED, k_neighbors=min(5, n_min - 1))
                    X_combined = np.hstack([X_g_train, X_c_train])
                    X_combined_sm, y_train_sm = sm.fit_resample(X_combined, y_train)
                    X_g_train_sm = X_combined_sm[:, :X_g_train.shape[1]]
                    X_c_train_sm = X_combined_sm[:, X_g_train.shape[1]:]
                except Exception:
                    X_g_train_sm, X_c_train_sm, y_train_sm = X_g_train, X_c_train, y_train
            else:
                X_g_train_sm, X_c_train_sm, y_train_sm = X_g_train, X_c_train, y_train

            mtype = config["type"]
            if mtype == "fusion":
                model = CrossAttentionFusion(
                    X_g_train.shape[1], X_c_train.shape[1], embed_dim=64
                ).to(device)
                train_dict = {"gait": X_g_train_sm, "clinical": X_c_train_sm}
                val_dict = {"gait": X_g_val, "clinical": X_c_val}
            elif mtype == "concat":
                model = SimpleConcatFusion(
                    X_g_train.shape[1], X_c_train.shape[1], embed_dim=64
                ).to(device)
                train_dict = {"gait": X_g_train_sm, "clinical": X_c_train_sm}
                val_dict = {"gait": X_g_val, "clinical": X_c_val}
            else:
                feat_key = config["features"]
                if feat_key == "gait":
                    X_s_train, X_s_val = X_g_train_sm, X_g_val
                else:
                    X_s_train, X_s_val = X_c_train_sm, X_c_val
                model = SingleModalityModel(X_s_train.shape[1], embed_dim=64).to(device)
                train_dict = {"single": X_s_train}
                val_dict = {"single": X_s_val}
                mtype = "single"

            val_probs, trained_model = train_fold(
                model, train_dict, y_train_sm, val_dict, y_val,
                model_type=mtype, epochs=150, batch_size=32, lr=3e-4, patience=20
            )

            y_prob_all[val_idx] = val_probs

            if len(np.unique(y_val)) >= 2:
                fold_auc = roc_auc_score(y_val, val_probs)
                fold_aucs.append(fold_auc)
                print(f"    Fold {fold_i}: AUC={fold_auc:.4f}")

            if fold_i == len(cv_folds) - 1:
                torch.save(trained_model.state_dict(),
                           os.path.join(CKPT_DIR, f"fusion_{model_name}_fold{fold_i}.pt"))

        if len(np.unique(y)) >= 2:
            metrics = compute_metrics(y, y_prob_all)
            ci = bootstrap_ci(y, y_prob_all)
        else:
            metrics = {"AUC_ROC": np.nan}
            ci = [np.nan, np.nan]

        metrics["Model"] = model_name
        metrics["Dataset"] = "GaitPDB"
        metrics["Label"] = "HY_ge_2.5"
        metrics["N"] = len(y)
        metrics["N_pos"] = int(y.sum())
        metrics["Gait_features"] = X_gait.shape[1]
        metrics["Clinical_features"] = X_clinical.shape[1]
        metrics["AUC_CI_low"] = ci[0]
        metrics["AUC_CI_high"] = ci[1]
        metrics["Mean_fold_AUC"] = np.mean(fold_aucs) if fold_aucs else np.nan
        metrics["Std_fold_AUC"] = np.std(fold_aucs) if fold_aucs else np.nan

        all_results.append(metrics)
        print(f"  >> {model_name}: OOF AUC={metrics['AUC_ROC']:.4f} "
              f"[{ci[0]:.4f}, {ci[1]:.4f}]")

    # ── Save results ─────────────────────────────────────────────────────
    print("\n[3/3] Saving results...")
    df_results = pd.DataFrame(all_results)
    out_path = os.path.join(RESULTS_DIR, "fusion_model_results.csv")
    df_results.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")

    print(f"\n{'='*70}")
    print("FUSION ABLATION RESULTS (GaitPDB, H&Y >= 2.5)")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'AUC':<8} {'CI':<22} {'Sens':<6} {'Spec':<6} {'F1':<6}")
    print("-" * 70)
    for _, r in df_results.iterrows():
        ci_str = f"[{r['AUC_CI_low']:.3f}, {r['AUC_CI_high']:.3f}]"
        print(f"{r['Model']:<20} {r['AUC_ROC']:.4f}  {ci_str:<22} "
              f"{r['Sensitivity']:.3f}  {r['Specificity']:.3f}  {r['F1']:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
