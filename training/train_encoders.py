#!/usr/bin/env python3
"""Pretrain each modality encoder independently (gait, voice, wearable).

JBI DBS Screening Project — Encoder Pretraining
Conda env: jbi_dbs | Config: ../config.yaml
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
torch.backends.cudnn.deterministic = False

# ── Imports ─────────────────────────────────────────────────────────────────
import sys
import json
import warnings
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from copy import deepcopy

from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

# ── Project root on path ───────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.wearable_encoder import WearableResidualMLP
from models.voice_encoder import VoiceEncoder, FocalLoss
from models.gait_encoder import GaitEncoder

# ── Config ──────────────────────────────────────────────────────────────────
_CFG_PATH = PROJECT_ROOT / "config.yaml"
with open(_CFG_PATH, "r") as f:
    CFG = yaml.safe_load(f)

SEED = CFG["model"]["seed"]                      # 42
EPOCHS = CFG["model"]["epochs"]                   # 150
PATIENCE = CFG["model"]["patience"]               # 20
LR = CFG["model"]["lr"]                           # 3e-4
WD = CFG["model"]["weight_decay"]                 # 1e-2
FOCAL_GAMMA = CFG["model"]["focal_gamma"]         # 2.0
FOCAL_ALPHA = CFG["model"]["focal_alpha"]         # 0.75
N_FOLDS = CFG["training"]["n_folds"]              # 5
N_WORKERS = CFG["hardware"]["n_workers"]          # 8
PIN_MEMORY = CFG["hardware"]["pin_memory"]        # True

BATCH_SIZES = {
    "wearable": CFG["model"]["wearable_batch_size"],  # 128
    "voice": CFG["model"]["voice_batch_size"],         # 256
    "gait": CFG["model"]["gait_batch_size"],           # 256
}

# ── Paths ───────────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
PRIMARY_CSV = DATA_DIR / "processed" / "fused" / "primary_cohort.csv"
SPLITS_JSON = DATA_DIR / "splits" / "primary_splits.json"
UCI_VOICE_CSV = DATA_DIR / "processed" / "voice_features" / "uci_voice_features.csv"
UCI_UPDRS_CSV = DATA_DIR / "processed" / "voice_features" / "uci_updrs_features.csv"
CKPT_DIR = PROJECT_ROOT / "results" / "checkpoints"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# ── Reproducibility ────────────────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════
class TabularDataset(Dataset):
    """Simple tabular dataset wrapping feature matrix + labels."""

    def __init__(self, features, labels):
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels).float()
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════
def make_loader(features, labels, batch_size, shuffle=True):
    """Build a DataLoader from numpy arrays."""
    ds = TabularDataset(features, labels)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=N_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler):
    """Single training epoch with mixed-precision."""
    model.train()
    running_loss = 0.0
    n_samples = 0
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=(device.type == "cuda")):
            logits = model(X)
            # Classifier head outputs (B, 2); convert to binary logits
            if logits.dim() == 2 and logits.shape[1] == 2:
                logits = logits[:, 1] - logits[:, 0]
            loss = criterion(logits.squeeze(), y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item() * X.size(0)
        n_samples += X.size(0)

    return running_loss / n_samples


@torch.no_grad()
def evaluate(model, loader, criterion):
    """Compute val loss and AUC-ROC."""
    model.eval()
    running_loss = 0.0
    n_samples = 0
    all_probs, all_labels = [], []

    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

        with autocast(enabled=(device.type == "cuda")):
            logits = model(X)
            if logits.dim() == 2 and logits.shape[1] == 2:
                logits = logits[:, 1] - logits[:, 0]
            loss = criterion(logits.squeeze(), y)

        running_loss += loss.item() * X.size(0)
        n_samples += X.size(0)

        probs = torch.sigmoid(logits.squeeze()).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.cpu().numpy())

    avg_loss = running_loss / n_samples
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    # Guard against single-class folds
    if len(np.unique(all_labels)) < 2:
        auc = 0.5
    else:
        auc = roc_auc_score(all_labels, all_probs)

    return avg_loss, auc


def train_encoder_fold(
    encoder_name: str,
    model: torch.nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    fold: int,
    batch_size: int,
) -> list[dict]:
    """Train a single encoder on one fold. Returns list of per-epoch log dicts."""

    # ── SMOTE on training data ──────────────────────────────────────────────
    smote = SMOTE(random_state=SEED)
    try:
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    except ValueError:
        warnings.warn(f"[{encoder_name} fold {fold}] SMOTE failed — using original data")
        X_train_res, y_train_res = X_train, y_train

    # ── DataLoaders ─────────────────────────────────────────────────────────
    train_loader = make_loader(X_train_res, y_train_res, batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size, shuffle=False)

    # ── Training setup ──────────────────────────────────────────────────────
    model = model.to(device)
    criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, pct_start=0.3,
        epochs=EPOCHS, steps_per_epoch=steps_per_epoch,
    )
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # ── Training loop ───────────────────────────────────────────────────────
    best_auc = -1.0
    best_state = None
    wait = 0
    log_rows = []

    pbar = tqdm(range(1, EPOCHS + 1), desc=f"  [{encoder_name}] fold {fold}", leave=False)
    for epoch in pbar:
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler)
        val_loss, val_auc = evaluate(model, val_loader, criterion)

        log_rows.append({
            "encoder": encoder_name,
            "fold": fold,
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_auc": round(val_auc, 6),
        })

        pbar.set_postfix(trn=f"{train_loss:.4f}", val=f"{val_loss:.4f}", auc=f"{val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"    Early stop at epoch {epoch} (best AUC={best_auc:.4f})")
                break

    # ── Save checkpoint ─────────────────────────────────────────────────────
    ckpt_path = CKPT_DIR / f"{encoder_name}_fold{fold}.pt"
    torch.save(best_state, ckpt_path)
    print(f"    Saved {ckpt_path.name}  (best val-AUC = {best_auc:.4f})")

    return log_rows


# ═══════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ═══════════════════════════════════════════════════════════════════════════
def load_gait_data():
    """Load primary cohort gait features + splits.
    Returns (X, y, folds) or None if files missing.
    """
    if not PRIMARY_CSV.exists():
        warnings.warn(f"Primary cohort not found: {PRIMARY_CSV}")
        return None
    if not SPLITS_JSON.exists():
        warnings.warn(f"Splits file not found: {SPLITS_JSON}")
        return None

    df = pd.read_csv(PRIMARY_CSV)
    meta_cols = ["subject_id", "dbs_candidate", "label_type", "dataset"]
    feat_cols = [c for c in df.columns if c not in meta_cols]
    X = df[feat_cols].values.astype(np.float32)
    y = df["dbs_candidate"].values.astype(np.float32)

    with open(SPLITS_JSON, "r") as f:
        splits = json.load(f)

    return X, y, splits


def load_voice_data():
    """Load UCI voice + UPDRS datasets for voice encoder pretraining.
    Returns list of (name, X, y, n_folds_to_use) or None.
    """
    datasets = []

    if UCI_VOICE_CSV.exists():
        df = pd.read_csv(UCI_VOICE_CSV)
        meta_cols = ["subject_id", "pd_status", "label_type", "dataset"]
        feat_cols = [c for c in df.columns if c not in meta_cols]
        X = df[feat_cols].values.astype(np.float32)
        y = df["pd_status"].values.astype(np.float32)
        datasets.append(("voice_uci_voice", X, y))
    else:
        warnings.warn(f"UCI Voice features not found: {UCI_VOICE_CSV}")

    if UCI_UPDRS_CSV.exists():
        df = pd.read_csv(UCI_UPDRS_CSV)
        meta_cols = ["subject_id", "dbs_candidate", "label_type", "dataset"]
        feat_cols = [c for c in df.columns if c not in meta_cols]
        X = df[feat_cols].values.astype(np.float32)
        y = df["dbs_candidate"].values.astype(np.float32)
        datasets.append(("voice_uci_updrs", X, y))
    else:
        warnings.warn(f"UCI UPDRS features not found: {UCI_UPDRS_CSV}")

    return datasets if datasets else None


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 72)
    print("Encoder Pretraining — JBI DBS Screening Project")
    print(f"Device: {device}  |  Seed: {SEED}  |  Epochs: {EPOCHS}  |  Patience: {PATIENCE}")
    print(f"LR: {LR}  |  WD: {WD}  |  Focal(gamma={FOCAL_GAMMA}, alpha={FOCAL_ALPHA})")
    print("=" * 72)

    all_logs = []
    summary = []

    # ── 1. Gait encoder ─────────────────────────────────────────────────────
    print("\n>>> Gait Encoder (WearGait-PD primary cohort)")
    gait_data = load_gait_data()
    if gait_data is not None:
        X_gait, y_gait, splits = gait_data
        n_feats = X_gait.shape[1]
        print(f"    Data: {X_gait.shape[0]} samples, {n_feats} features")

        for k in range(N_FOLDS):
            fold_info = splits["cv_folds"][k]
            train_idx = fold_info["train"]
            val_idx = fold_info["val"]

            X_tr, y_tr = X_gait[train_idx], y_gait[train_idx]
            X_va, y_va = X_gait[val_idx], y_gait[val_idx]

            model = GaitEncoder(input_dim=n_feats, classify=True)
            fold_logs = train_encoder_fold(
                "gait", model, X_tr, y_tr, X_va, y_va,
                fold=k, batch_size=BATCH_SIZES["gait"],
            )
            all_logs.extend(fold_logs)

            best_row = max(fold_logs, key=lambda r: r["val_auc"])
            summary.append(("gait", k, best_row["val_auc"]))
    else:
        print("    SKIPPED — data not available")

    # ── 2. Voice encoder(s) ─────────────────────────────────────────────────
    print("\n>>> Voice Encoder (UCI datasets)")
    voice_datasets = load_voice_data()
    if voice_datasets is not None:
        for ds_name, X_v, y_v in voice_datasets:
            n_feats = X_v.shape[1]
            print(f"  [{ds_name}] {X_v.shape[0]} samples, {n_feats} features")

            # Generate own 5-fold CV splits for UCI datasets
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
            for k, (train_idx, val_idx) in enumerate(skf.split(X_v, y_v)):
                X_tr, y_tr = X_v[train_idx], y_v[train_idx]
                X_va, y_va = X_v[val_idx], y_v[val_idx]

                model = VoiceEncoder(input_dim=n_feats, classify=True)
                fold_logs = train_encoder_fold(
                    ds_name, model, X_tr, y_tr, X_va, y_va,
                    fold=k, batch_size=BATCH_SIZES["voice"],
                )
                all_logs.extend(fold_logs)

                best_row = max(fold_logs, key=lambda r: r["val_auc"])
                summary.append((ds_name, k, best_row["val_auc"]))
    else:
        print("    SKIPPED — data not available")

    # ── 3. Wearable encoder ─────────────────────────────────────────────────
    print("\n>>> Wearable Encoder (PADS data)")
    # Wearable pretraining requires PADS data which may not be available yet
    pads_csv = DATA_DIR / "processed" / "wearable_features" / "pads_features.csv"
    if pads_csv.exists():
        df_w = pd.read_csv(pads_csv)
        meta_cols = ["subject_id", "dbs_candidate", "label_type", "dataset",
                     "condition", "updrs_iii", "hy_stage"]
        feat_cols = [c for c in df_w.columns if c not in meta_cols]
        # Drop any remaining non-numeric columns
        feat_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df_w[c])]
        X_w = df_w[feat_cols].values.astype(np.float32)
        # Replace any NaN with 0 for training
        X_w = np.nan_to_num(X_w, nan=0.0)
        y_w = df_w["dbs_candidate"].values.astype(np.float32)
        n_feats = X_w.shape[1]
        print(f"    Data: {X_w.shape[0]} samples, {n_feats} features")

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        for k, (train_idx, val_idx) in enumerate(skf.split(X_w, y_w)):
            X_tr, y_tr = X_w[train_idx], y_w[train_idx]
            X_va, y_va = X_w[val_idx], y_w[val_idx]

            model = WearableResidualMLP(input_dim=n_feats, classify=True)
            fold_logs = train_encoder_fold(
                "wearable", model, X_tr, y_tr, X_va, y_va,
                fold=k, batch_size=BATCH_SIZES["wearable"],
            )
            all_logs.extend(fold_logs)

            best_row = max(fold_logs, key=lambda r: r["val_auc"])
            summary.append(("wearable", k, best_row["val_auc"]))
    else:
        print("    SKIPPED — PADS data not available yet")

    # ── 4. Save training log ────────────────────────────────────────────────
    if all_logs:
        log_df = pd.DataFrame(all_logs)
        log_path = TABLE_DIR / "encoder_training_log.csv"
        log_df.to_csv(log_path, index=False)
        print(f"\nTraining log saved to {log_path}")

    # ── 5. Print summary ────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("Summary — Best Val-AUC per Fold")
    print("=" * 72)
    if summary:
        for enc, fold, auc in summary:
            print(f"  {enc:<25s}  fold {fold}  AUC = {auc:.4f}")

        # Mean AUC per encoder
        print("-" * 72)
        enc_names = sorted(set(e for e, _, _ in summary))
        for enc in enc_names:
            aucs = [a for e, _, a in summary if e == enc]
            print(f"  {enc:<25s}  mean AUC = {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    else:
        print("  No encoders were trained.")
    print("=" * 72)


if __name__ == "__main__":
    main()
