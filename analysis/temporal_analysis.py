#!/usr/bin/env python3
"""
Temporal LSTM Modeling for Longitudinal PD Disease Trajectory Prediction.

Novel contribution: instead of predicting DBS candidacy from a single snapshot,
this module models the TEMPORAL TRAJECTORY of disease progression using the
UCI UPDRS telemonitoring dataset (5875 rows, 42 subjects, longitudinal).

Steps:
  1. Prepare UCI UPDRS longitudinal sequences (per-subject time series)
  2. Compute trajectory labels (UPDRS slope, progression class, DBS threshold)
  3. Train bidirectional LSTM with 5-fold GroupKFold CV
  4. Compare LSTM vs static XGBoost baseline on same data
  5. Trajectory visualisation

Outputs:
  results/tables/temporal_lstm_results.csv
  results/tables/trajectory_analysis.csv
  results/figures/temporal_trajectories.png
  results/figures/temporal_lstm_comparison.png

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
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    matthews_corrcoef,
    accuracy_score,
    precision_score,
    recall_score,
    mean_squared_error,
    r2_score,
    average_precision_score,
)
import xgboost as xgb

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Project root & model imports ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.temporal_model import TemporalLSTMEncoder, TemporalClassifier

# ── GPU setup ───────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# ── Config ──────────────────────────────────────────────────────────────────
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

SEED = CONFIG["model"]["seed"]  # 42
N_FOLDS = CONFIG["training"]["n_folds"]  # 5
DPI = CONFIG["figures"]["dpi"]

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Paths ───────────────────────────────────────────────────────────────────
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "uci_updrs" / "parkinsons_updrs.data"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Font setup (Liberation Sans) ───────────────────────────────────────────
_FONT_CANDIDATES = ["Liberation Sans", "Arial", "DejaVu Sans", "sans-serif"]
_FONT_SET = False
for _fc in _FONT_CANDIDATES:
    _matches = [f for f in fm.fontManager.ttflist if _fc.lower() in f.name.lower()]
    if _matches:
        plt.rcParams["font.family"] = _fc
        _FONT_SET = True
        break
if not _FONT_SET:
    plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = CONFIG["figures"]["font_size"]

# ── Colours (project palette) ──────────────────────────────────────────────
CLR_VOICE = CONFIG["figures"]["color_voice"]    # "#ff7f0e"
CLR_PROPOSED = CONFIG["figures"]["color_proposed"]  # "#d62728"
CLR_FUSION = CONFIG["figures"]["color_fusion"]  # "#9467bd"
CLR_GAIT = CONFIG["figures"]["color_gait"]      # "#2ca02c"
CLR_WEARABLE = CONFIG["figures"]["color_wearable"]  # "#1f77b4"

# ── UCI UPDRS voice feature columns ────────────────────────────────────────
VOICE_FEATURES = [
    "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
    "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:APQ11",
    "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "PPE",
]

DBS_THRESHOLD = 32.0  # motor_UPDRS >= 32 = proxy DBS candidate


# ============================================================================
# 1. Data Loading & Preparation
# ============================================================================
def load_uci_updrs() -> pd.DataFrame:
    """Load UCI UPDRS telemonitoring dataset."""
    print("[Step 1] Loading UCI UPDRS data...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df)} rows, {df['subject#'].nunique()} subjects")
    print(f"  motor_UPDRS range: {df['motor_UPDRS'].min():.1f} - {df['motor_UPDRS'].max():.1f}")
    print(f"  test_time range: {df['test_time'].min():.1f} - {df['test_time'].max():.1f} days")
    return df


def compute_trajectory_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-subject trajectory labels.

    For each subject, compute:
    - updrs_slope: linear regression slope of motor_UPDRS vs test_time
    - mean_updrs: mean motor UPDRS across all recordings
    - max_updrs: max motor UPDRS
    - dbs_candidate: whether motor_UPDRS >= 32 at any point
    - trajectory_class: fast / moderate / slow progressor (based on slope terciles)
    - projected_dbs_days: projected days until UPDRS reaches 32 (from linear model)
    """
    print("[Step 2] Computing trajectory labels...")
    records = []

    for subj_id, grp in df.groupby("subject#"):
        grp_sorted = grp.sort_values("test_time")
        t = grp_sorted["test_time"].values.reshape(-1, 1)
        y = grp_sorted["motor_UPDRS"].values

        # Linear regression for slope
        if len(t) >= 2 and np.std(t) > 0:
            lr = LinearRegression().fit(t, y)
            slope = lr.coef_[0]
            intercept = lr.intercept_
        else:
            slope = 0.0
            intercept = y.mean()

        mean_updrs = y.mean()
        max_updrs = y.max()
        min_updrs = y.min()
        dbs_candidate = int(max_updrs >= DBS_THRESHOLD)

        # Project when UPDRS will reach threshold
        if slope > 0 and intercept < DBS_THRESHOLD:
            projected_dbs_days = (DBS_THRESHOLD - intercept) / slope
        elif intercept >= DBS_THRESHOLD:
            projected_dbs_days = 0.0
        else:
            projected_dbs_days = np.inf  # Non-progressor or stable

        records.append({
            "subject_id": subj_id,
            "age": grp_sorted["age"].iloc[0],
            "sex": grp_sorted["sex"].iloc[0],
            "n_recordings": len(grp),
            "time_span_days": grp_sorted["test_time"].max() - grp_sorted["test_time"].min(),
            "updrs_slope": slope,
            "updrs_intercept": intercept,
            "mean_updrs": mean_updrs,
            "max_updrs": max_updrs,
            "min_updrs": min_updrs,
            "updrs_range": max_updrs - min_updrs,
            "dbs_candidate": dbs_candidate,
            "projected_dbs_days": projected_dbs_days,
        })

    traj_df = pd.DataFrame(records)

    # Trajectory class based on slope terciles
    terciles = traj_df["updrs_slope"].quantile([1 / 3, 2 / 3]).values
    traj_df["trajectory_class"] = pd.cut(
        traj_df["updrs_slope"],
        bins=[-np.inf, terciles[0], terciles[1], np.inf],
        labels=["slow", "moderate", "fast"],
    )
    traj_df["trajectory_label"] = traj_df["trajectory_class"].map(
        {"slow": 0, "moderate": 1, "fast": 2}
    )

    print(f"  DBS candidates: {traj_df['dbs_candidate'].sum()} / {len(traj_df)}")
    print(f"  Trajectory classes: {traj_df['trajectory_class'].value_counts().to_dict()}")
    print(f"  UPDRS slope range: {traj_df['updrs_slope'].min():.4f} to {traj_df['updrs_slope'].max():.4f}")

    return traj_df


# ============================================================================
# 2. Sequence Preparation
# ============================================================================
class TemporalPDDataset(Dataset):
    """PyTorch dataset for longitudinal PD voice sequences."""

    def __init__(self, sequences, labels, lengths):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)
        self.lengths = torch.LongTensor(lengths)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], self.lengths[idx]


def prepare_sequences(df: pd.DataFrame, traj_df: pd.DataFrame):
    """Build padded sequences per subject.

    Returns
    -------
    sequences : ndarray (n_subjects, max_seq_len, n_features)
    labels_dbs : ndarray (n_subjects,)  — binary DBS candidacy
    labels_traj : ndarray (n_subjects,) — trajectory class (0/1/2)
    labels_slope : ndarray (n_subjects,) — continuous UPDRS slope
    lengths : ndarray (n_subjects,)     — actual sequence lengths
    subject_ids : ndarray (n_subjects,)
    """
    print("[Step 3] Preparing padded sequences...")

    subject_ids = sorted(df["subject#"].unique())
    all_seqs = []
    all_lengths = []

    for subj_id in subject_ids:
        grp = df[df["subject#"] == subj_id].sort_values("test_time")
        feat = grp[VOICE_FEATURES].values
        all_seqs.append(feat)
        all_lengths.append(len(feat))

    max_len = max(all_lengths)
    n_features = len(VOICE_FEATURES)
    n_subjects = len(subject_ids)

    # Pad sequences with zeros
    sequences = np.zeros((n_subjects, max_len, n_features), dtype=np.float32)
    for i, seq in enumerate(all_seqs):
        sequences[i, : len(seq), :] = seq

    lengths = np.array(all_lengths)
    subject_ids_arr = np.array(subject_ids)

    # Align labels
    traj_sorted = traj_df.set_index("subject_id").loc[subject_ids_arr]
    labels_dbs = traj_sorted["dbs_candidate"].values.astype(np.float32)
    labels_traj = traj_sorted["trajectory_label"].values.astype(np.int64)
    labels_slope = traj_sorted["updrs_slope"].values.astype(np.float32)

    print(f"  Sequences: {sequences.shape}")
    print(f"  Max length: {max_len}, Min length: {min(all_lengths)}")
    print(f"  DBS labels: {int(labels_dbs.sum())} positive / {len(labels_dbs)} total")

    return sequences, labels_dbs, labels_traj, labels_slope, lengths, subject_ids_arr


# ============================================================================
# 3. LSTM Training
# ============================================================================
def train_lstm_fold(
    train_seqs,
    train_labels,
    train_lengths,
    val_seqs,
    val_labels,
    val_lengths,
    task="classification",
    epochs=100,
    lr=3e-4,
    batch_size=8,
    patience=15,
):
    """Train LSTM model for one fold.

    Returns
    -------
    dict with val predictions, metrics, and trained model state.
    """
    n_features = train_seqs.shape[2]

    # Per-feature normalisation based on training data
    scaler = StandardScaler()
    train_flat = train_seqs.reshape(-1, n_features)
    scaler.fit(train_flat[train_flat.sum(axis=1) != 0])  # Fit on non-padded rows

    # Apply scaler
    train_seqs_norm = train_seqs.copy()
    val_seqs_norm = val_seqs.copy()
    for i in range(len(train_seqs)):
        train_seqs_norm[i, : train_lengths[i], :] = scaler.transform(
            train_seqs[i, : train_lengths[i], :]
        )
    for i in range(len(val_seqs)):
        val_seqs_norm[i, : val_lengths[i], :] = scaler.transform(
            val_seqs[i, : val_lengths[i], :]
        )

    train_ds = TemporalPDDataset(train_seqs_norm, train_labels, train_lengths)
    val_ds = TemporalPDDataset(val_seqs_norm, val_labels, val_lengths)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=len(val_ds), shuffle=False,
        num_workers=0, pin_memory=True,
    )

    model = TemporalClassifier(
        input_dim=n_features, hidden_dim=64, num_layers=2,
        dropout=0.3, task=task,
    ).to(device)

    if task == "classification":
        # Weighted BCE for imbalanced data
        pos_weight = torch.tensor(
            [(len(train_labels) - train_labels.sum()) / max(train_labels.sum(), 1)]
        ).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif task == "trajectory":
        criterion = nn.CrossEntropyLoss()
    elif task == "regression":
        criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs,
        steps_per_epoch=max(len(train_loader), 1), pct_start=0.3,
    )
    scaler_amp = GradScaler()

    best_val_metric = -np.inf if task != "regression" else np.inf
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for seqs, labels, lens in train_loader:
            seqs = seqs.to(device)
            labels = labels.to(device)
            lens = lens.to(device)

            optimizer.zero_grad()
            with autocast(dtype=torch.float16):
                logits = model(seqs, lens)
                if task == "classification":
                    loss = criterion(logits.squeeze(-1), labels)
                elif task == "trajectory":
                    loss = criterion(logits, labels.long())
                elif task == "regression":
                    loss = criterion(logits.squeeze(-1), labels)

            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            scheduler.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            for seqs, labels, lens in val_loader:
                seqs = seqs.to(device)
                labels = labels.to(device)
                lens = lens.to(device)
                with autocast(dtype=torch.float16):
                    logits = model(seqs, lens)

        if task == "classification":
            probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
            y_true = labels.cpu().numpy()
            try:
                val_auc = roc_auc_score(y_true, probs)
            except ValueError:
                val_auc = 0.5
            val_metric = val_auc
            improved = val_metric > best_val_metric
        elif task == "trajectory":
            pred_probs = torch.softmax(logits, dim=1).cpu().numpy()
            y_true = labels.cpu().numpy().astype(int)
            try:
                val_auc = roc_auc_score(y_true, pred_probs, multi_class="ovr")
            except ValueError:
                val_auc = 0.5
            val_metric = val_auc
            improved = val_metric > best_val_metric
        elif task == "regression":
            preds = logits.squeeze(-1).cpu().numpy()
            y_true = labels.cpu().numpy()
            val_mse = mean_squared_error(y_true, preds)
            val_metric = val_mse
            improved = val_metric < best_val_metric

        if improved:
            best_val_metric = val_metric
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    # Load best model for final predictions
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        for seqs, labels, lens in val_loader:
            seqs = seqs.to(device)
            lens = lens.to(device)
            with autocast(dtype=torch.float16):
                logits = model(seqs, lens)

    results = {"scaler": scaler}

    if task == "classification":
        probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
        y_true = labels.cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        results.update({
            "y_true": y_true,
            "y_prob": probs,
            "y_pred": preds,
            "auc": roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.5,
            "f1": f1_score(y_true, preds, zero_division=0),
            "mcc": matthews_corrcoef(y_true, preds),
            "sensitivity": recall_score(y_true, preds, zero_division=0),
            "specificity": recall_score(1 - y_true, 1 - preds, zero_division=0),
            "auc_pr": average_precision_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.0,
        })
    elif task == "trajectory":
        pred_probs = torch.softmax(logits, dim=1).cpu().numpy()
        pred_class = pred_probs.argmax(axis=1)
        y_true = labels.cpu().numpy().astype(int)
        results.update({
            "y_true": y_true,
            "y_pred": pred_class,
            "y_prob": pred_probs,
            "auc": roc_auc_score(y_true, pred_probs, multi_class="ovr") if len(np.unique(y_true)) > 1 else 0.5,
            "accuracy": accuracy_score(y_true, pred_class),
            "f1_macro": f1_score(y_true, pred_class, average="macro", zero_division=0),
        })
    elif task == "regression":
        preds = logits.squeeze(-1).cpu().numpy()
        y_true = labels.cpu().numpy()
        results.update({
            "y_true": y_true,
            "y_pred": preds,
            "mse": mean_squared_error(y_true, preds),
            "r2": r2_score(y_true, preds),
        })

    return results


# ============================================================================
# 4. Static XGBoost Baseline (single-snapshot)
# ============================================================================
def train_static_xgboost(df, traj_df, task="classification"):
    """Train XGBoost on static (aggregated) features for comparison.

    Aggregate per-subject: mean + std of voice features -> flat feature vector.
    """
    print("  [XGBoost] Training static baseline...")
    subject_ids = sorted(df["subject#"].unique())
    records = []

    for subj_id in subject_ids:
        grp = df[df["subject#"] == subj_id][VOICE_FEATURES]
        feat_mean = grp.mean().values
        feat_std = grp.std().fillna(0).values
        records.append(np.concatenate([feat_mean, feat_std]))

    X = np.array(records, dtype=np.float32)
    feature_names = [f"{f}_mean" for f in VOICE_FEATURES] + [f"{f}_std" for f in VOICE_FEATURES]
    subject_ids_arr = np.array(subject_ids)

    traj_sorted = traj_df.set_index("subject_id").loc[subject_ids_arr]

    if task == "classification":
        y = traj_sorted["dbs_candidate"].values
    elif task == "trajectory":
        y = traj_sorted["trajectory_label"].values
    elif task == "regression":
        y = traj_sorted["updrs_slope"].values

    groups = subject_ids_arr
    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        if task in ("classification", "trajectory"):
            clf = xgb.XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                n_jobs=CONFIG["training"].get("xgb_n_jobs", 32),
                use_label_encoder=False, eval_metric="logloss", verbosity=0,
                num_class=3 if task == "trajectory" else None,
                objective="multi:softprob" if task == "trajectory" else "binary:logistic",
            )
            clf.fit(X_train, y_train)

            if task == "classification":
                probs = clf.predict_proba(X_val)[:, 1]
                preds = (probs >= 0.5).astype(int)
                auc = roc_auc_score(y_val, probs) if len(np.unique(y_val)) > 1 else 0.5
                fold_results.append({
                    "fold": fold, "auc": auc,
                    "f1": f1_score(y_val, preds, zero_division=0),
                    "mcc": matthews_corrcoef(y_val, preds),
                    "sensitivity": recall_score(y_val, preds, zero_division=0),
                    "specificity": recall_score(1 - y_val, 1 - preds, zero_division=0),
                    "auc_pr": average_precision_score(y_val, probs) if len(np.unique(y_val)) > 1 else 0.0,
                    "y_true": y_val, "y_prob": probs,
                })
            else:
                pred_probs = clf.predict_proba(X_val)
                pred_class = pred_probs.argmax(axis=1)
                auc = roc_auc_score(y_val, pred_probs, multi_class="ovr") if len(np.unique(y_val)) > 1 else 0.5
                fold_results.append({
                    "fold": fold, "auc": auc,
                    "accuracy": accuracy_score(y_val, pred_class),
                    "f1_macro": f1_score(y_val, pred_class, average="macro", zero_division=0),
                })

        elif task == "regression":
            reg = xgb.XGBRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                n_jobs=CONFIG["training"].get("xgb_n_jobs", 32), verbosity=0,
            )
            reg.fit(X_train, y_train)
            preds = reg.predict(X_val)
            fold_results.append({
                "fold": fold,
                "mse": mean_squared_error(y_val, preds),
                "r2": r2_score(y_val, preds),
            })

    return fold_results


# ============================================================================
# 5. Main Experiment
# ============================================================================
def run_temporal_experiment():
    """Run the full temporal LSTM experiment."""
    print("=" * 70)
    print("TEMPORAL LSTM ANALYSIS — Longitudinal Disease Trajectory Modeling")
    print("=" * 70)

    # ── Step 1: Load data ───────────────────────────────────────────────
    df = load_uci_updrs()
    traj_df = compute_trajectory_labels(df)

    # Save trajectory analysis
    traj_out = traj_df.copy()
    traj_out["projected_dbs_days"] = traj_out["projected_dbs_days"].replace(
        [np.inf, -np.inf], np.nan
    )
    traj_out.to_csv(TABLES_DIR / "trajectory_analysis.csv", index=False)
    print(f"  Saved: {TABLES_DIR / 'trajectory_analysis.csv'}")

    # ── Step 2: Prepare sequences ───────────────────────────────────────
    sequences, labels_dbs, labels_traj, labels_slope, lengths, subject_ids = \
        prepare_sequences(df, traj_df)

    groups = subject_ids
    gkf = GroupKFold(n_splits=N_FOLDS)

    # ── Step 3: LSTM training — 3 tasks ─────────────────────────────────
    all_results = {}
    tasks = {
        "classification": {"labels": labels_dbs, "desc": "DBS Candidacy (binary)"},
        "trajectory": {"labels": labels_traj, "desc": "Progression Class (3-class)"},
        "regression": {"labels": labels_slope, "desc": "UPDRS Slope (continuous)"},
    }

    for task_name, task_info in tasks.items():
        print(f"\n{'─' * 60}")
        print(f"[LSTM] Task: {task_info['desc']}")
        print(f"{'─' * 60}")

        labels = task_info["labels"]
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(sequences, labels, groups)):
            print(f"  Fold {fold + 1}/{N_FOLDS}...", end=" ")

            result = train_lstm_fold(
                sequences[train_idx], labels[train_idx], lengths[train_idx],
                sequences[val_idx], labels[val_idx], lengths[val_idx],
                task=task_name, epochs=100, lr=3e-4, batch_size=8, patience=15,
            )

            if task_name == "classification":
                print(f"AUC={result['auc']:.3f}  F1={result['f1']:.3f}  MCC={result['mcc']:.3f}")
                fold_metrics.append({
                    "fold": fold, "auc": result["auc"], "f1": result["f1"],
                    "mcc": result["mcc"], "sensitivity": result["sensitivity"],
                    "specificity": result["specificity"], "auc_pr": result["auc_pr"],
                })
            elif task_name == "trajectory":
                print(f"AUC={result['auc']:.3f}  Acc={result['accuracy']:.3f}  F1={result['f1_macro']:.3f}")
                fold_metrics.append({
                    "fold": fold, "auc": result["auc"],
                    "accuracy": result["accuracy"], "f1_macro": result["f1_macro"],
                })
            elif task_name == "regression":
                print(f"MSE={result['mse']:.6f}  R2={result['r2']:.3f}")
                fold_metrics.append({
                    "fold": fold, "mse": result["mse"], "r2": result["r2"],
                })

        all_results[task_name] = fold_metrics

    # ── Step 4: Static XGBoost baseline ─────────────────────────────────
    print(f"\n{'─' * 60}")
    print("[XGBoost] Static baseline comparison")
    print(f"{'─' * 60}")

    xgb_results = {}
    for task_name in tasks:
        print(f"\n  Task: {tasks[task_name]['desc']}")
        xgb_folds = train_static_xgboost(df, traj_df, task=task_name)
        xgb_results[task_name] = xgb_folds

        if task_name == "classification":
            aucs = [f["auc"] for f in xgb_folds]
            f1s = [f["f1"] for f in xgb_folds]
            print(f"    XGBoost AUC: {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}")
            print(f"    XGBoost F1:  {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}")
        elif task_name == "trajectory":
            aucs = [f["auc"] for f in xgb_folds]
            accs = [f["accuracy"] for f in xgb_folds]
            print(f"    XGBoost AUC: {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}")
            print(f"    XGBoost Acc: {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
        elif task_name == "regression":
            mses = [f["mse"] for f in xgb_folds]
            r2s = [f["r2"] for f in xgb_folds]
            print(f"    XGBoost MSE: {np.mean(mses):.6f} +/- {np.std(mses):.6f}")
            print(f"    XGBoost R2:  {np.mean(r2s):.3f} +/- {np.std(r2s):.3f}")

    # ── Step 5: Compile results table ───────────────────────────────────
    print(f"\n{'─' * 60}")
    print("Compiling results table...")
    print(f"{'─' * 60}")

    results_rows = []

    # Classification results
    lstm_cls = all_results["classification"]
    xgb_cls = xgb_results["classification"]
    for model_name, folds in [("LSTM (Temporal)", lstm_cls), ("XGBoost (Static)", xgb_cls)]:
        aucs = [f["auc"] for f in folds]
        f1s = [f["f1"] for f in folds]
        mccs = [f["mcc"] for f in folds]
        sens = [f["sensitivity"] for f in folds]
        spec = [f["specificity"] for f in folds]
        auc_prs = [f["auc_pr"] for f in folds]
        results_rows.append({
            "Task": "DBS Candidacy",
            "Model": model_name,
            "AUC-ROC": f"{np.mean(aucs):.3f} +/- {np.std(aucs):.3f}",
            "AUC-PR": f"{np.mean(auc_prs):.3f} +/- {np.std(auc_prs):.3f}",
            "F1": f"{np.mean(f1s):.3f} +/- {np.std(f1s):.3f}",
            "MCC": f"{np.mean(mccs):.3f} +/- {np.std(mccs):.3f}",
            "Sensitivity": f"{np.mean(sens):.3f} +/- {np.std(sens):.3f}",
            "Specificity": f"{np.mean(spec):.3f} +/- {np.std(spec):.3f}",
            "AUC_mean": np.mean(aucs),
        })

    # Trajectory results
    lstm_traj = all_results["trajectory"]
    xgb_traj = xgb_results["trajectory"]
    for model_name, folds in [("LSTM (Temporal)", lstm_traj), ("XGBoost (Static)", xgb_traj)]:
        aucs = [f["auc"] for f in folds]
        accs = [f["accuracy"] for f in folds]
        f1s = [f["f1_macro"] for f in folds]
        results_rows.append({
            "Task": "Progression Class",
            "Model": model_name,
            "AUC-ROC": f"{np.mean(aucs):.3f} +/- {np.std(aucs):.3f}",
            "AUC-PR": "-",
            "F1": f"{np.mean(f1s):.3f} +/- {np.std(f1s):.3f}",
            "MCC": "-",
            "Sensitivity": f"{np.mean(accs):.3f} +/- {np.std(accs):.3f}",
            "Specificity": "-",
            "AUC_mean": np.mean(aucs),
        })

    # Regression results
    lstm_reg = all_results["regression"]
    xgb_reg = xgb_results["regression"]
    for model_name, folds in [("LSTM (Temporal)", lstm_reg), ("XGBoost (Static)", xgb_reg)]:
        mses = [f["mse"] for f in folds]
        r2s = [f["r2"] for f in folds]
        results_rows.append({
            "Task": "UPDRS Slope",
            "Model": model_name,
            "AUC-ROC": "-",
            "AUC-PR": "-",
            "F1": "-",
            "MCC": "-",
            "Sensitivity": f"MSE: {np.mean(mses):.6f} +/- {np.std(mses):.6f}",
            "Specificity": f"R2: {np.mean(r2s):.3f} +/- {np.std(r2s):.3f}",
            "AUC_mean": np.mean(r2s),
        })

    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(TABLES_DIR / "temporal_lstm_results.csv", index=False)
    print(f"\n  Saved: {TABLES_DIR / 'temporal_lstm_results.csv'}")

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for _, row in results_df.iterrows():
        print(f"  {row['Task']:>20s} | {row['Model']:<20s} | AUC-ROC: {row['AUC-ROC']}")

    # ── Step 6: Trajectory visualisation ────────────────────────────────
    print(f"\n{'─' * 60}")
    print("Generating trajectory visualisation...")
    print(f"{'─' * 60}")
    plot_trajectories(df, traj_df)
    plot_lstm_comparison(all_results, xgb_results)

    print("\n" + "=" * 70)
    print("TEMPORAL ANALYSIS COMPLETE")
    print("=" * 70)

    return results_df, traj_df


# ============================================================================
# 6. Visualisation
# ============================================================================
def plot_trajectories(df: pd.DataFrame, traj_df: pd.DataFrame):
    """Plot UPDRS trajectories for example patients, coloured by DBS candidacy."""
    # Select 5 representative subjects: 2 DBS candidates, 2 non-candidates, 1 borderline
    dbs_pos = traj_df[traj_df["dbs_candidate"] == 1].nlargest(2, "updrs_slope")
    dbs_neg = traj_df[traj_df["dbs_candidate"] == 0].nsmallest(2, "updrs_slope")
    borderline = traj_df.iloc[(traj_df["mean_updrs"] - DBS_THRESHOLD).abs().argsort()[:1]]
    example_subjects = pd.concat([dbs_pos, dbs_neg, borderline]).drop_duplicates("subject_id")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── Panel A: Individual trajectories ────────────────────────────────
    ax = axes[0]
    colors_map = {1: CLR_PROPOSED, 0: CLR_WEARABLE}
    labels_map = {1: "DBS Candidate", 0: "Non-DBS"}
    plotted_labels = set()

    for _, row in example_subjects.iterrows():
        subj = int(row["subject_id"])
        grp = df[df["subject#"] == subj].sort_values("test_time")
        dbs = int(row["dbs_candidate"])
        clr = colors_map[dbs]
        lbl = labels_map[dbs] if labels_map[dbs] not in plotted_labels else None
        if lbl:
            plotted_labels.add(labels_map[dbs])

        ax.plot(grp["test_time"], grp["motor_UPDRS"], color=clr, alpha=0.7,
                linewidth=1.5, label=lbl)
        ax.scatter(grp["test_time"].iloc[0], grp["motor_UPDRS"].iloc[0],
                   color=clr, s=40, zorder=5, edgecolors="white", linewidth=0.5)

        # Add trend line
        t = grp["test_time"].values.reshape(-1, 1)
        lr = LinearRegression().fit(t, grp["motor_UPDRS"].values)
        t_range = np.linspace(t.min(), t.max(), 50).reshape(-1, 1)
        ax.plot(t_range, lr.predict(t_range), color=clr, linestyle="--",
                alpha=0.5, linewidth=1)

    ax.axhline(y=DBS_THRESHOLD, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax.text(ax.get_xlim()[1] * 0.02, DBS_THRESHOLD + 0.5,
            f"DBS Threshold (UPDRS={DBS_THRESHOLD:.0f})",
            fontsize=8, color="gray", va="bottom")
    ax.set_xlabel("Test Time (days)")
    ax.set_ylabel("Motor UPDRS")
    ax.set_title("A. Example Patient Trajectories", fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Panel B: Slope distribution by DBS status ───────────────────────
    ax = axes[1]
    dbs_slopes = traj_df[traj_df["dbs_candidate"] == 1]["updrs_slope"]
    non_dbs_slopes = traj_df[traj_df["dbs_candidate"] == 0]["updrs_slope"]

    bins = np.linspace(
        traj_df["updrs_slope"].min() - 0.001,
        traj_df["updrs_slope"].max() + 0.001,
        20,
    )
    ax.hist(non_dbs_slopes, bins=bins, alpha=0.6, color=CLR_WEARABLE,
            label=f"Non-DBS (n={len(non_dbs_slopes)})", edgecolor="white")
    ax.hist(dbs_slopes, bins=bins, alpha=0.6, color=CLR_PROPOSED,
            label=f"DBS Candidate (n={len(dbs_slopes)})", edgecolor="white")
    ax.axvline(x=0, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xlabel("UPDRS Slope (per day)")
    ax.set_ylabel("Count")
    ax.set_title("B. Disease Progression Rate by DBS Status", fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "temporal_trajectories.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'temporal_trajectories.png'}")


def plot_lstm_comparison(lstm_results: dict, xgb_results: dict):
    """Bar chart comparing LSTM vs static XGBoost across tasks."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ── Panel A: DBS Classification AUC ─────────────────────────────────
    ax = axes[0]
    lstm_aucs = [f["auc"] for f in lstm_results["classification"]]
    xgb_aucs = [f["auc"] for f in xgb_results["classification"]]

    x = np.arange(2)
    means = [np.mean(lstm_aucs), np.mean(xgb_aucs)]
    stds = [np.std(lstm_aucs), np.std(xgb_aucs)]
    bars = ax.bar(x, means, yerr=stds, capsize=5, width=0.5,
                  color=[CLR_PROPOSED, CLR_WEARABLE], edgecolor="white", linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["LSTM\n(Temporal)", "XGBoost\n(Static)"])
    ax.set_ylabel("AUC-ROC")
    ax.set_title("A. DBS Candidacy Classification", fontweight="bold")
    ax.set_ylim(0, 1.05)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.02,
                f"{m:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    # ── Panel B: Trajectory Classification AUC ──────────────────────────
    ax = axes[1]
    lstm_aucs = [f["auc"] for f in lstm_results["trajectory"]]
    xgb_aucs = [f["auc"] for f in xgb_results["trajectory"]]

    means = [np.mean(lstm_aucs), np.mean(xgb_aucs)]
    stds = [np.std(lstm_aucs), np.std(xgb_aucs)]
    bars = ax.bar(x, means, yerr=stds, capsize=5, width=0.5,
                  color=[CLR_PROPOSED, CLR_WEARABLE], edgecolor="white", linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["LSTM\n(Temporal)", "XGBoost\n(Static)"])
    ax.set_ylabel("AUC-ROC (OVR)")
    ax.set_title("B. Progression Class (3-class)", fontweight="bold")
    ax.set_ylim(0, 1.05)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.02,
                f"{m:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    # ── Panel C: UPDRS Slope Regression R^2 ─────────────────────────────
    ax = axes[2]
    lstm_r2 = [f["r2"] for f in lstm_results["regression"]]
    xgb_r2 = [f["r2"] for f in xgb_results["regression"]]

    means = [np.mean(lstm_r2), np.mean(xgb_r2)]
    stds = [np.std(lstm_r2), np.std(xgb_r2)]
    bars = ax.bar(x, means, yerr=stds, capsize=5, width=0.5,
                  color=[CLR_PROPOSED, CLR_WEARABLE], edgecolor="white", linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["LSTM\n(Temporal)", "XGBoost\n(Static)"])
    ax.set_ylabel("R$^2$")
    ax.set_title("C. UPDRS Slope Regression", fontweight="bold")
    # R2 can be negative, set reasonable limits
    y_min = min(min(means) - max(stds) - 0.1, -0.5)
    ax.set_ylim(y_min, 1.05)
    for bar, m, s in zip(bars, means, stds):
        y_pos = max(bar.get_height(), 0) + s + 0.02
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f"{m:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "temporal_lstm_comparison.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'temporal_lstm_comparison.png'}")


# ============================================================================
# Entry point
# ============================================================================
if __name__ == "__main__":
    results_df, traj_df = run_temporal_experiment()
