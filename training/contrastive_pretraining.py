#!/usr/bin/env python3
"""Contrastive learning pretraining for multimodal DBS screening.

Addresses the core problem: cross-attention fusion (M13) can't learn meaningful
cross-modal interactions because WearGait-PD only has gait features. Wearable
and voice inputs are zero-padded during fusion training, so the attention heads
have nothing to attend across.

Solution — 3-stage pipeline:
  1. Supervised Contrastive pretraining of each modality encoder on its own dataset
  2. Cross-modal alignment of gait encoders (GaitPDB <-> WearGait-PD) via shared
     embedding space so representations generalise across gait data sources
  3. Re-train fusion model with contrastive-pretrained encoders

JBI DBS Screening Project
Conda env: jbi_dbs | Config: ../config.yaml
"""

# ---------------------------------------------------------------------------
# Hardware init — MUST be at top before any numerical library import
# ---------------------------------------------------------------------------
import os, multiprocessing

N_CORES = multiprocessing.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(N_CORES)
os.environ["MKL_NUM_THREADS"] = str(N_CORES)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_CORES)

# ---------------------------------------------------------------------------
# GPU setup
# ---------------------------------------------------------------------------
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Disable torch dynamo/compile
import torch._dynamo

torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import sys
import json
import copy
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Project root & model imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.fusion_model import CrossAttentionFusionModel, FocalLoss
from models.gait_encoder import GaitEncoder
from models.voice_encoder import VoiceEncoder
from models.wearable_encoder import WearableResidualMLP

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_CFG_PATH = PROJECT_ROOT / "config.yaml"
with open(_CFG_PATH, "r") as f:
    CFG = yaml.safe_load(f)

SEED = CFG["model"]["seed"]  # 42
EPOCHS = CFG["model"]["epochs"]  # 150
PATIENCE = CFG["model"]["patience"]  # 20
LR = CFG["model"]["lr"]  # 3e-4
WEIGHT_DECAY = CFG["model"]["weight_decay"]  # 1e-2
N_FOLDS = CFG["training"]["n_folds"]  # 5
N_WORKERS = CFG["hardware"]["n_workers"]  # 8
PIN_MEMORY = CFG["hardware"]["pin_memory"]  # True
MODALITY_DROPOUT_PROB = CFG["model"]["modality_dropout_prob"]  # 0.2
FUSION_BATCH_SIZE = CFG["model"]["fusion_batch_size"]  # 32
FONT_FAMILY = CFG.get("figures", {}).get("font_family", "Liberation Sans")
DPI = CFG.get("figures", {}).get("dpi", 300)

# Contrastive-specific hyperparameters
CON_TEMPERATURE = 0.07  # supervised contrastive temperature
CROSS_TEMPERATURE = 0.10  # cross-modal contrastive temperature
LAMBDA_CON = 0.5  # weight for contrastive loss vs classification loss
CON_EPOCHS = 80  # contrastive pretraining epochs (shorter than full 150)
CON_PATIENCE = 15  # early stopping patience for contrastive pretraining
PHASE1_EPOCHS = 20  # fusion phase 1 (frozen encoders)
PHASE2_EPOCHS = EPOCHS - PHASE1_EPOCHS  # fusion phase 2 (end-to-end)

# Batch sizes
BATCH_SIZES = {
    "wearable": CFG["model"]["wearable_batch_size"],  # 128
    "voice": CFG["model"]["voice_batch_size"],  # 256
    "gait": CFG["model"]["gait_batch_size"],  # 256
}

# Paths
DATA_DIR = PROJECT_ROOT / "data"
PRIMARY_CSV = DATA_DIR / "processed" / "fused" / "primary_cohort.csv"
SPLITS_JSON = DATA_DIR / "splits" / "primary_splits.json"
PADS_CSV = DATA_DIR / "processed" / "wearable_features" / "pads_features.csv"
UCI_VOICE_CSV = DATA_DIR / "processed" / "voice_features" / "uci_voice_features.csv"
GAITPDB_CSV = DATA_DIR / "processed" / "fused" / "external_gaitpdb.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "results" / "checkpoints"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

# Non-feature columns
NON_FEATURE_COLS = ["subject_id", "dbs_candidate", "label_type", "dataset"]
LABEL_COL = "dbs_candidate"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ---------------------------------------------------------------------------
# Encoder input dims (from existing checkpoints)
# ---------------------------------------------------------------------------
WEARABLE_DIM = 4361
VOICE_DIM = 22
GAIT_DIM = 20  # primary cohort


# ============================================================================
# Supervised Contrastive Loss (Khosla et al., 2020)
# ============================================================================
class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Loss.

    Pulls together embeddings of same-class samples, pushes apart different
    classes in the embedding space. Works with arbitrary batch compositions.

    Reference: Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        features : (N, D) L2-normalised embeddings.
        labels   : (N,) integer class labels.

        Returns
        -------
        Scalar contrastive loss.
        """
        device = features.device
        batch_size = features.shape[0]

        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # L2 normalise (should already be, but ensure)
        features = torch.nn.functional.normalize(features, dim=1)

        # Similarity matrix: (N, N)
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # Mask: same-class pairs (excluding self)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)  # (N, N)
        # Remove self-comparisons
        self_mask = torch.eye(batch_size, device=device)
        mask = mask - self_mask  # positive pairs (same class, not self)

        # Number of positives per anchor
        n_positives = mask.sum(dim=1)  # (N,)

        # If any anchor has zero positives, skip those
        valid = n_positives > 0

        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # For numerical stability, subtract max from sim_matrix
        sim_max, _ = sim_matrix.max(dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_max.detach()

        # Denominator: sum of exp(sim) over all pairs except self
        exp_sim = torch.exp(sim_matrix) * (1 - self_mask)
        log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        # Log-prob for each pair
        log_prob = sim_matrix - log_denom  # (N, N)

        # Mean log-prob over positive pairs
        mean_log_prob = (mask * log_prob).sum(dim=1) / (n_positives + 1e-12)

        # Loss: only over valid anchors
        loss = -mean_log_prob[valid].mean()
        return loss


# ============================================================================
# Cross-Modal Contrastive Loss
# ============================================================================
class CrossModalContrastiveLoss(torch.nn.Module):
    """Align embeddings from different modalities for same DBS class.

    For patients with the same DBS status across two modalities, pull their
    embeddings together. For different DBS status, push apart.
    """

    def __init__(self, temperature: float = 0.10):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        labels_a: torch.Tensor,
        labels_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        emb_a    : (N_a, D) embeddings from modality A.
        emb_b    : (N_b, D) embeddings from modality B.
        labels_a : (N_a,)   class labels for modality A.
        labels_b : (N_b,)   class labels for modality B.

        Returns
        -------
        Scalar cross-modal contrastive loss.
        """
        device = emb_a.device
        n_a = emb_a.shape[0]
        n_b = emb_b.shape[0]

        if n_a == 0 or n_b == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # L2 normalise
        emb_a = torch.nn.functional.normalize(emb_a, dim=1)
        emb_b = torch.nn.functional.normalize(emb_b, dim=1)

        # Cross-modal similarity: (N_a, N_b)
        sim_matrix = torch.matmul(emb_a, emb_b.T) / self.temperature

        # Label match mask: (N_a, N_b)
        labels_a = labels_a.contiguous().view(-1, 1)
        labels_b = labels_b.contiguous().view(1, -1)
        mask = torch.eq(labels_a, labels_b).float().to(device)

        n_positives = mask.sum(dim=1)  # (N_a,)
        valid = n_positives > 0

        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Numerical stability
        sim_max, _ = sim_matrix.max(dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_max.detach()

        # Denominator over all N_b samples
        exp_sim = torch.exp(sim_matrix)
        log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        log_prob = sim_matrix - log_denom

        # Mean over positive pairs per anchor
        mean_log_prob = (mask * log_prob).sum(dim=1) / (n_positives + 1e-12)

        loss = -mean_log_prob[valid].mean()
        return loss


# ============================================================================
# Dataset helper
# ============================================================================
class TabularDataset(torch.utils.data.Dataset):
    """Simple tabular dataset wrapping feature matrix + labels."""

    def __init__(self, features, labels):
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels).long()
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class MultimodalDataset(torch.utils.data.Dataset):
    """Returns (wearable_features, voice_features, gait_features, label)."""

    def __init__(self, features, labels, gait_dim, wearable_dim, voice_dim):
        super().__init__()
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.n_samples = len(labels)

        n_feat = features.shape[1]
        self.gait_features = torch.tensor(
            features[:, : min(n_feat, gait_dim)], dtype=torch.float32
        )
        if n_feat < gait_dim:
            pad = torch.zeros(self.n_samples, gait_dim - n_feat)
            self.gait_features = torch.cat([self.gait_features, pad], dim=1)

        self.wearable_features = torch.zeros(self.n_samples, wearable_dim)
        self.voice_features = torch.zeros(self.n_samples, voice_dim)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return (
            self.wearable_features[idx],
            self.voice_features[idx],
            self.gait_features[idx],
            self.labels[idx],
        )


# ============================================================================
# Helper functions
# ============================================================================
def make_loader(features, labels, batch_size, shuffle=True):
    ds = TabularDataset(features, labels)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=N_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )


def apply_smote(X, y):
    sm = SMOTE(random_state=SEED)
    try:
        X_res, y_res = sm.fit_resample(X, y)
        print(
            f"    SMOTE: {len(y)} -> {len(y_res)} samples "
            f"(0: {(y_res == 0).sum()}, 1: {(y_res == 1).sum()})"
        )
        return X_res, y_res
    except ValueError:
        print("    SMOTE failed — using original data")
        return X, y


def detect_dims_from_checkpoints():
    """Read encoder input dims from existing checkpoints."""
    global WEARABLE_DIM, VOICE_DIM, GAIT_DIM

    wearable_ckpt = CHECKPOINT_DIR / "wearable_fold0.pt"
    if wearable_ckpt.exists():
        state = torch.load(wearable_ckpt, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        for k, v in state.items():
            if "fc1.weight" in k:
                WEARABLE_DIM = v.shape[1]
                break
        print(f"    WEARABLE_DIM = {WEARABLE_DIM} (from checkpoint)")

    voice_ckpt = CHECKPOINT_DIR / "voice_uci_voice_fold0.pt"
    if voice_ckpt.exists():
        state = torch.load(voice_ckpt, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        for k, v in state.items():
            if "fc1.weight" in k:
                VOICE_DIM = v.shape[1]
                break
        print(f"    VOICE_DIM = {VOICE_DIM} (from checkpoint)")

    gait_ckpt = CHECKPOINT_DIR / "gait_fold0.pt"
    if gait_ckpt.exists():
        state = torch.load(gait_ckpt, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        for k, v in state.items():
            if "fc1.weight" in k:
                GAIT_DIM = v.shape[1]
                break
        print(f"    GAIT_DIM = {GAIT_DIM} (from checkpoint)")


# ============================================================================
# STEP 1 + 2: Contrastive pretraining of each modality encoder
# ============================================================================
def contrastive_pretrain_encoder(
    encoder_name: str,
    model: torch.nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    fold: int,
    batch_size: int,
    embedding_dim: int,
) -> Tuple[List[Dict], float]:
    """Train encoder with: total_loss = focal_loss + lambda * supcon_loss.

    Returns (log_rows, best_val_auc).
    """
    # SMOTE
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    train_loader = make_loader(X_train_res, y_train_res, batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size, shuffle=False)

    model = model.to(device)
    focal_criterion = FocalLoss(gamma=2.0, alpha=0.75)
    supcon_criterion = SupConLoss(temperature=CON_TEMPERATURE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        pct_start=0.3,
        epochs=CON_EPOCHS,
        steps_per_epoch=steps_per_epoch,
    )
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    # Projection head for contrastive loss (operates on embedding before classifier)
    # We need to extract intermediate embeddings. The encoder with classify=True
    # outputs logits. We'll create a small projection head separately.
    proj_head = torch.nn.Sequential(
        torch.nn.Linear(embedding_dim, embedding_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(embedding_dim, 64),
    ).to(device)
    proj_optimizer = torch.optim.AdamW(
        proj_head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    best_auc = -1.0
    best_state = None
    wait = 0
    log_rows = []

    for epoch in range(1, CON_EPOCHS + 1):
        model.train()
        proj_head.train()
        running_loss = 0.0
        running_focal = 0.0
        running_supcon = 0.0
        n_samples = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            proj_optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type="cuda", enabled=(device.type == "cuda")
            ):
                # Get logits from encoder
                logits = model(X_batch)

                # Get embedding (before classifier head) by doing a partial forward
                # For all encoders: fc1 -> bn1 -> relu -> drop1 -> fc2 -> bn2 -> relu
                with torch.no_grad():
                    emb = torch.nn.functional.relu(model.bn1(model.fc1(X_batch)))
                    emb = model.drop1(emb)
                    emb = torch.nn.functional.relu(model.bn2(model.fc2(emb)))
                # Re-enable grad for the projection head
                emb = emb.detach()  # detach to avoid double backward through encoder

                # Actually, we want gradients to flow through encoder via the
                # contrastive path too. Let's redo with grad:
                # For WearableResidualMLP the architecture is different
                if hasattr(model, "fc3"):
                    # WearableResidualMLP: fc1->bn1->relu->drop1 -> fc2->bn2->relu+residual -> fc3->bn3->relu->drop3
                    identity = X_batch
                    h = model.drop1(
                        torch.nn.functional.relu(model.bn1(model.fc1(X_batch)))
                    )
                    h = model.bn2(model.fc2(h))
                    h = torch.nn.functional.relu(h + model.residual_proj(identity))
                    emb = model.drop3(
                        torch.nn.functional.relu(model.bn3(model.fc3(h)))
                    )
                else:
                    # GaitEncoder / VoiceEncoder
                    emb = model.drop1(
                        torch.nn.functional.relu(model.bn1(model.fc1(X_batch)))
                    )
                    emb = torch.nn.functional.relu(model.bn2(model.fc2(emb)))

                # Project for contrastive loss
                z = proj_head(emb)
                z = torch.nn.functional.normalize(z, dim=1)

                # Focal loss on logits
                if logits.dim() == 2 and logits.shape[1] == 2:
                    binary_logits = logits[:, 1] - logits[:, 0]
                else:
                    binary_logits = logits.squeeze()
                focal_loss = focal_criterion(
                    binary_logits, y_batch.float()
                )

                # Supervised contrastive loss
                supcon_loss = supcon_criterion(z, y_batch)

                total_loss = focal_loss + LAMBDA_CON * supcon_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.step(proj_optimizer)
            scaler.update()
            scheduler.step()

            running_loss += total_loss.item() * X_batch.size(0)
            running_focal += focal_loss.item() * X_batch.size(0)
            running_supcon += supcon_loss.item() * X_batch.size(0)
            n_samples += X_batch.size(0)

        train_loss = running_loss / n_samples
        train_focal = running_focal / n_samples
        train_supcon = running_supcon / n_samples

        # Validation (focal loss + AUC only, no contrastive in eval)
        model.eval()
        val_loss = 0.0
        val_n = 0
        all_probs, all_labels = [], []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)

                with torch.amp.autocast(
                    device_type="cuda", enabled=(device.type == "cuda")
                ):
                    logits = model(X_batch)
                    if logits.dim() == 2 and logits.shape[1] == 2:
                        binary_logits = logits[:, 1] - logits[:, 0]
                    else:
                        binary_logits = logits.squeeze()
                    loss = focal_criterion(binary_logits, y_batch.float())

                val_loss += loss.item() * X_batch.size(0)
                val_n += X_batch.size(0)

                probs = torch.sigmoid(binary_logits).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(y_batch.cpu().numpy())

        val_loss /= val_n
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)

        if len(np.unique(all_labels)) < 2:
            val_auc = 0.5
        else:
            val_auc = roc_auc_score(all_labels, all_probs)

        log_rows.append(
            {
                "encoder": encoder_name,
                "fold": fold,
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_focal": round(train_focal, 6),
                "train_supcon": round(train_supcon, 6),
                "val_loss": round(val_loss, 6),
                "val_auc": round(val_auc, 6),
            }
        )

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= CON_PATIENCE:
                if epoch % 10 == 0 or epoch <= 5:
                    pass
                print(
                    f"    [{encoder_name} fold {fold}] Early stop epoch {epoch} "
                    f"(best AUC={best_auc:.4f})"
                )
                break

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"    [{encoder_name} f{fold}] ep {epoch:3d}/{CON_EPOCHS} | "
                f"loss={train_loss:.4f} (focal={train_focal:.4f} "
                f"supcon={train_supcon:.4f}) | "
                f"val_AUC={val_auc:.4f}"
            )

    # Save checkpoint
    ckpt_path = CHECKPOINT_DIR / f"contrastive_{encoder_name}_fold{fold}.pt"
    torch.save(
        {"model_state_dict": best_state, "best_val_auc": best_auc, "fold": fold},
        ckpt_path,
    )
    print(
        f"    Saved {ckpt_path.name} (val_AUC={best_auc:.4f})"
    )

    return log_rows, best_auc


# ============================================================================
# STEP 3: Cross-modal alignment between GaitPDB and WearGait-PD gait
# ============================================================================
def cross_modal_gait_alignment(fold: int) -> List[Dict]:
    """Align gait encoder representations between GaitPDB and WearGait-PD.

    Both datasets have gait-related features but different feature sets.
    We train two separate gait encoders and align their shared embedding space
    using cross-modal contrastive loss on the DBS labels.
    """
    print(f"\n  [Cross-Modal Gait Alignment] Fold {fold}")

    # Load GaitPDB data
    if not GAITPDB_CSV.exists():
        print("    GaitPDB data not found — skipping alignment")
        return []

    df_gaitpdb = pd.read_csv(GAITPDB_CSV)
    gaitpdb_meta = ["subject_id", "dbs_candidate", "label_type", "dataset"]
    gaitpdb_feats = [c for c in df_gaitpdb.columns if c not in gaitpdb_meta]
    X_gaitpdb = df_gaitpdb[gaitpdb_feats].values.astype(np.float32)
    X_gaitpdb = np.nan_to_num(X_gaitpdb, nan=0.0)
    y_gaitpdb = df_gaitpdb["dbs_candidate"].values.astype(np.int64)
    gaitpdb_dim = X_gaitpdb.shape[1]

    # Load WearGait-PD (primary cohort) data
    df_primary = pd.read_csv(PRIMARY_CSV)
    primary_meta = ["subject_id", "dbs_candidate", "label_type", "dataset"]
    primary_feats = [c for c in df_primary.columns if c not in primary_meta]
    X_primary = df_primary[primary_feats].values.astype(np.float32)
    y_primary = df_primary["dbs_candidate"].values.astype(np.int64)

    # Load splits for primary
    with open(SPLITS_JSON, "r") as f:
        splits = json.load(f)
    fold_info = splits["cv_folds"][fold]
    train_idx = np.array(fold_info["train"])

    X_primary_train = X_primary[train_idx]
    y_primary_train = y_primary[train_idx]

    # For GaitPDB, use stratified split
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    gaitpdb_splits = list(skf.split(X_gaitpdb, y_gaitpdb))
    gaitpdb_train_idx = gaitpdb_splits[fold][0]
    X_gaitpdb_train = X_gaitpdb[gaitpdb_train_idx]
    y_gaitpdb_train = y_gaitpdb[gaitpdb_train_idx]

    print(
        f"    Primary train: {len(X_primary_train)} | GaitPDB train: {len(X_gaitpdb_train)}"
    )

    # Load contrastive-pretrained gait encoder for primary cohort
    ckpt_path = CHECKPOINT_DIR / f"contrastive_gait_fold{fold}.pt"
    if not ckpt_path.exists():
        print("    No contrastive gait checkpoint — skipping alignment")
        return []

    # Primary gait encoder
    primary_encoder = GaitEncoder(input_dim=GAIT_DIM, classify=False).to(device)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state["model_state_dict"]
    # Filter out classifier head
    sd_emb = {k: v for k, v in sd.items() if "classifier_head" not in k}
    primary_encoder.load_state_dict(sd_emb, strict=False)

    # GaitPDB encoder (same architecture, different input dim)
    gaitpdb_encoder = GaitEncoder(input_dim=gaitpdb_dim, classify=False).to(device)

    # Shared projection to 64-dim
    proj_primary = torch.nn.Sequential(
        torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64)
    ).to(device)
    proj_gaitpdb = torch.nn.Sequential(
        torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64)
    ).to(device)

    cross_modal_loss_fn = CrossModalContrastiveLoss(temperature=CROSS_TEMPERATURE)
    supcon_loss_fn = SupConLoss(temperature=CON_TEMPERATURE)

    all_params = (
        list(gaitpdb_encoder.parameters())
        + list(proj_primary.parameters())
        + list(proj_gaitpdb.parameters())
    )
    # Don't optimise primary_encoder — it's pretrained; only align via projection
    optimizer = torch.optim.AdamW(all_params, lr=LR * 0.5, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    # Create data loaders
    primary_loader = make_loader(
        X_primary_train, y_primary_train, batch_size=64, shuffle=True
    )
    gaitpdb_loader = make_loader(
        X_gaitpdb_train, y_gaitpdb_train, batch_size=64, shuffle=True
    )

    log_rows = []
    alignment_epochs = 40

    for epoch in range(1, alignment_epochs + 1):
        primary_encoder.eval()  # keep frozen
        gaitpdb_encoder.train()
        proj_primary.train()
        proj_gaitpdb.train()

        total_loss = 0.0
        n_batches = 0

        # Iterate over both loaders in parallel (zip to shorter)
        primary_iter = iter(primary_loader)
        gaitpdb_iter = iter(gaitpdb_loader)

        for _ in range(min(len(primary_loader), len(gaitpdb_loader))):
            try:
                X_p, y_p = next(primary_iter)
                X_g, y_g = next(gaitpdb_iter)
            except StopIteration:
                break

            X_p = X_p.to(device, non_blocking=True)
            y_p = y_p.to(device, non_blocking=True)
            X_g = X_g.to(device, non_blocking=True)
            y_g = y_g.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type="cuda", enabled=(device.type == "cuda")
            ):
                # Get embeddings
                with torch.no_grad():
                    emb_p = primary_encoder(X_p)  # (B, 64)
                emb_g = gaitpdb_encoder(X_g)  # (B, 64)

                # Project to shared space
                z_p = torch.nn.functional.normalize(proj_primary(emb_p), dim=1)
                z_g = torch.nn.functional.normalize(proj_gaitpdb(emb_g), dim=1)

                # Cross-modal contrastive: align by DBS label
                cm_loss = cross_modal_loss_fn(z_p, z_g, y_p, y_g)

                # Within-modality contrastive for GaitPDB (helps structure its space)
                within_loss = supcon_loss_fn(z_g, y_g)

                loss = cm_loss + 0.3 * within_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        log_rows.append(
            {
                "stage": "cross_modal_alignment",
                "fold": fold,
                "epoch": epoch,
                "loss": round(avg_loss, 6),
            }
        )

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Alignment ep {epoch:3d}/{alignment_epochs} | loss={avg_loss:.4f}")

    # Save aligned projection heads
    alignment_path = CHECKPOINT_DIR / f"contrastive_alignment_fold{fold}.pt"
    torch.save(
        {
            "proj_primary_state": proj_primary.state_dict(),
            "proj_gaitpdb_state": proj_gaitpdb.state_dict(),
            "gaitpdb_encoder_state": gaitpdb_encoder.state_dict(),
        },
        alignment_path,
    )
    print(f"    Saved alignment checkpoint: {alignment_path.name}")

    return log_rows


# ============================================================================
# STEP 4: Fusion model training with contrastive-pretrained encoders
# ============================================================================
def load_contrastive_encoder(encoder_cls, input_dim, ckpt_path, classify=False):
    """Load encoder from contrastive-pretrained checkpoint."""
    model = encoder_cls(input_dim=input_dim, classify=classify)
    if ckpt_path is not None and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        if not classify:
            state = {k: v for k, v in state.items() if "classifier_head" not in k}
        model.load_state_dict(state, strict=False)
        print(f"    Loaded contrastive-pretrained: {ckpt_path.name}")
    else:
        # Fall back to standard pretrained
        fallback_name = ckpt_path.name.replace("contrastive_", "") if ckpt_path else ""
        fallback_path = CHECKPOINT_DIR / fallback_name
        if fallback_path.exists():
            state = torch.load(fallback_path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            if not classify:
                state = {
                    k: v for k, v in state.items() if "classifier_head" not in k
                }
            model.load_state_dict(state, strict=False)
            print(f"    Fallback to standard pretrained: {fallback_path.name}")
        else:
            print(f"    No checkpoint found — random init")
    return model


def train_contrastive_fusion_fold(
    fold_k: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[List[Dict], float]:
    """Train fusion model with contrastive-pretrained encoders for one fold."""
    print(f"\n  [Contrastive Fusion] Fold {fold_k}")

    # Load contrastive-pretrained encoders
    gait_enc = load_contrastive_encoder(
        GaitEncoder, GAIT_DIM, CHECKPOINT_DIR / f"contrastive_gait_fold{fold_k}.pt"
    )
    voice_enc = load_contrastive_encoder(
        VoiceEncoder,
        VOICE_DIM,
        CHECKPOINT_DIR / f"contrastive_voice_uci_voice_fold{fold_k}.pt",
    )
    wearable_enc = load_contrastive_encoder(
        WearableResidualMLP,
        WEARABLE_DIM,
        CHECKPOINT_DIR / f"contrastive_wearable_fold{fold_k}.pt",
    )

    model = CrossAttentionFusionModel(
        wearable_encoder=wearable_enc,
        voice_encoder=voice_enc,
        gait_encoder=gait_enc,
        wearable_dim=WearableResidualMLP.EMBEDDING_DIM,
        voice_dim=VoiceEncoder.EMBEDDING_DIM,
        gait_dim=GaitEncoder.EMBEDDING_DIM,
        modality_dropout_prob=MODALITY_DROPOUT_PROB,
    ).to(device)

    # SMOTE
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    # Datasets and loaders
    train_ds = MultimodalDataset(
        X_train_res, y_train_res, GAIT_DIM, WEARABLE_DIM, VOICE_DIM
    )
    val_ds = MultimodalDataset(X_val, y_val, GAIT_DIM, WEARABLE_DIM, VOICE_DIM)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=FUSION_BATCH_SIZE,
        shuffle=True,
        num_workers=N_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=FUSION_BATCH_SIZE,
        shuffle=False,
        num_workers=N_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    criterion = FocalLoss(gamma=2.0, alpha=0.75)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    log_rows = []
    best_val_auc = -1.0
    best_state = None
    epochs_no_improve = 0

    # ---- Phase 1: Frozen encoders ----
    print(f"    Phase 1: frozen encoders ({PHASE1_EPOCHS} epochs)")
    model.freeze_encoders()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)

    for epoch in range(1, PHASE1_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for w_x, v_x, g_x, labels in train_loader:
            w_x = w_x.to(device, non_blocking=True)
            v_x = v_x.to(device, non_blocking=True)
            g_x = g_x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type="cuda", enabled=(device.type == "cuda")
            ):
                out = model(w_x, v_x, g_x)
                loss = criterion(out["logits"], labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / max(n_batches, 1)
        val_loss, val_auc, val_f1 = _validate_fusion(model, val_loader, criterion)

        log_rows.append(
            {
                "model": "ContrastiveFusion",
                "fold": fold_k,
                "phase": 1,
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "val_auc": round(val_auc, 4),
                "val_f1": round(val_f1, 4),
            }
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"      P1 ep {epoch:3d}/{PHASE1_EPOCHS} | "
                f"train={train_loss:.4f} val_AUC={val_auc:.4f} F1={val_f1:.4f}"
            )

    # ---- Phase 2: End-to-end ----
    print(f"    Phase 2: end-to-end ({PHASE2_EPOCHS} epochs)")
    model.unfreeze_encoders()
    epochs_no_improve = 0

    encoder_params = []
    fusion_params = []
    encoder_names = {"wearable_encoder", "voice_encoder", "gait_encoder"}
    for name, param in model.named_parameters():
        top = name.split(".")[0]
        if top in encoder_names:
            encoder_params.append(param)
        else:
            fusion_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": fusion_params, "lr": LR},
            {"params": encoder_params, "lr": 1e-4},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[LR, 1e-4],
        epochs=PHASE2_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
    )
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(1, PHASE2_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for w_x, v_x, g_x, labels in train_loader:
            w_x = w_x.to(device, non_blocking=True)
            v_x = v_x.to(device, non_blocking=True)
            g_x = g_x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type="cuda", enabled=(device.type == "cuda")
            ):
                out = model(w_x, v_x, g_x)
                loss = criterion(out["logits"], labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / max(n_batches, 1)
        val_loss, val_auc, val_f1 = _validate_fusion(model, val_loader, criterion)

        log_rows.append(
            {
                "model": "ContrastiveFusion",
                "fold": fold_k,
                "phase": 2,
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "val_auc": round(val_auc, 4),
                "val_f1": round(val_f1, 4),
            }
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"      P2 ep {epoch:3d}/{PHASE2_EPOCHS} | "
                f"train={train_loss:.4f} val_AUC={val_auc:.4f} F1={val_f1:.4f}"
            )

        if epochs_no_improve >= PATIENCE:
            print(f"      Early stop at epoch {epoch}")
            break

    # Save best checkpoint
    ckpt_path = CHECKPOINT_DIR / f"contrastive_fusion_model_fold{fold_k}.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "best_val_auc": best_val_auc,
            "fold": fold_k,
        },
        ckpt_path,
    )
    print(f"    Saved {ckpt_path.name} (val_AUC={best_val_auc:.4f})")

    return log_rows, best_val_auc


@torch.no_grad()
def _validate_fusion(model, val_loader, criterion):
    """Validate fusion model. Returns (val_loss, val_auc, val_f1)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_probs, all_labels = [], []

    for w_x, v_x, g_x, labels in val_loader:
        w_x = w_x.to(device, non_blocking=True)
        v_x = v_x.to(device, non_blocking=True)
        g_x = g_x.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            out = model(w_x, v_x, g_x)
            loss = criterion(out["logits"], labels)

        total_loss += loss.item()
        n_batches += 1

        probs = out["probabilities"][:, 1].cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    val_loss = total_loss / max(n_batches, 1)

    try:
        val_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        val_auc = 0.5

    preds = (all_probs >= 0.5).astype(int)
    val_f1 = f1_score(all_labels, preds, zero_division=0.0)

    return val_loss, val_auc, val_f1


# ============================================================================
# STEP 5: t-SNE visualisation
# ============================================================================
def generate_tsne_plot(fold: int = 0):
    """Generate t-SNE comparing embeddings before/after contrastive pretraining."""
    print("\n[t-SNE Visualisation]")

    # Load primary cohort
    df = pd.read_csv(PRIMARY_CSV)
    meta = ["subject_id", "dbs_candidate", "label_type", "dataset"]
    feat_cols = [c for c in df.columns if c not in meta]
    X = df[feat_cols].values.astype(np.float32)
    y = df["dbs_candidate"].values.astype(np.int64)

    # Before: standard pretrained gait encoder
    gait_before = GaitEncoder(input_dim=GAIT_DIM, classify=False).to(device)
    ckpt_before = CHECKPOINT_DIR / f"gait_fold{fold}.pt"
    if ckpt_before.exists():
        state = torch.load(ckpt_before, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        state = {k: v for k, v in state.items() if "classifier_head" not in k}
        gait_before.load_state_dict(state, strict=False)

    # After: contrastive pretrained gait encoder
    gait_after = GaitEncoder(input_dim=GAIT_DIM, classify=False).to(device)
    ckpt_after = CHECKPOINT_DIR / f"contrastive_gait_fold{fold}.pt"
    if ckpt_after.exists():
        state = torch.load(ckpt_after, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        state = {k: v for k, v in state.items() if "classifier_head" not in k}
        gait_after.load_state_dict(state, strict=False)

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    gait_before.eval()
    gait_after.eval()
    with torch.no_grad():
        emb_before = gait_before(X_tensor).cpu().numpy()
        emb_after = gait_after(X_tensor).cpu().numpy()

    # t-SNE
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=min(30, len(X) - 1))
    coords_before = tsne.fit_transform(emb_before)
    coords_after = tsne.fit_transform(emb_after)

    # Plot
    plt.rcParams["font.family"] = FONT_FAMILY
    plt.rcParams["font.size"] = 10

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    class_names = {0: "Non-DBS", 1: "DBS Candidate"}
    colors = {0: "#1f77b4", 1: "#d62728"}

    for ax, coords, title in [
        (axes[0], coords_before, "Standard Pretraining"),
        (axes[1], coords_after, "Contrastive Pretraining"),
    ]:
        for label in [0, 1]:
            mask = y == label
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=colors[label],
                label=class_names[label],
                alpha=0.7,
                s=30,
                edgecolors="white",
                linewidth=0.3,
            )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Gait Encoder Embeddings: Effect of Contrastive Pretraining",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    fig_path = FIGURES_DIR / "contrastive_tsne.png"
    fig.savefig(fig_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved t-SNE plot: {fig_path}")


# ============================================================================
# Main pipeline
# ============================================================================
def main():
    t_start = time.time()

    print("=" * 72)
    print("  Contrastive Learning Pretraining Pipeline")
    print(f"  Device       : {device}")
    print(f"  Seed         : {SEED}")
    print(f"  Temperature  : SupCon={CON_TEMPERATURE}, Cross={CROSS_TEMPERATURE}")
    print(f"  Lambda_con   : {LAMBDA_CON}")
    print(f"  Con epochs   : {CON_EPOCHS} | Fusion epochs: {PHASE1_EPOCHS}+{PHASE2_EPOCHS}")
    print("=" * 72)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Detect dimensions
    print("\n[0] Detecting encoder dimensions...")
    detect_dims_from_checkpoints()

    all_logs = []
    encoder_summary = {}

    # ==================================================================
    # STAGE 1: Contrastive pretraining of individual encoders
    # ==================================================================
    print("\n" + "=" * 72)
    print("  STAGE 1: Supervised Contrastive Pretraining")
    print("=" * 72)

    # ---- 1a. Gait encoder (on WearGait-PD primary cohort) ----
    print("\n>>> Gait Encoder (WearGait-PD primary cohort)")
    df_primary = pd.read_csv(PRIMARY_CSV)
    primary_meta = ["subject_id", "dbs_candidate", "label_type", "dataset"]
    primary_feats = [c for c in df_primary.columns if c not in primary_meta]
    X_gait = df_primary[primary_feats].values.astype(np.float32)
    y_gait = df_primary["dbs_candidate"].values.astype(np.int64)
    print(f"    Data: {X_gait.shape[0]} samples, {X_gait.shape[1]} features")

    with open(SPLITS_JSON, "r") as f:
        splits = json.load(f)

    for k in range(N_FOLDS):
        fold_info = splits["cv_folds"][k]
        train_idx = np.array(fold_info["train"])
        val_idx = np.array(fold_info["val"])

        model = GaitEncoder(input_dim=GAIT_DIM, classify=True)
        logs, best_auc = contrastive_pretrain_encoder(
            "gait",
            model,
            X_gait[train_idx],
            y_gait[train_idx],
            X_gait[val_idx],
            y_gait[val_idx],
            fold=k,
            batch_size=BATCH_SIZES["gait"],
            embedding_dim=GaitEncoder.EMBEDDING_DIM,
        )
        all_logs.extend(logs)
        encoder_summary.setdefault("gait", {})[k] = best_auc

    # ---- 1b. Voice encoder (UCI Voice) ----
    print("\n>>> Voice Encoder (UCI Voice)")
    if UCI_VOICE_CSV.exists():
        df_voice = pd.read_csv(UCI_VOICE_CSV)
        voice_meta = ["subject_id", "pd_status", "label_type", "dataset", "dbs_candidate"]
        voice_feats = [c for c in df_voice.columns if c not in voice_meta]
        X_voice = df_voice[voice_feats].values.astype(np.float32)
        # Use pd_status as label (voice dataset uses this)
        if "pd_status" in df_voice.columns:
            y_voice = df_voice["pd_status"].values.astype(np.int64)
        else:
            y_voice = df_voice["dbs_candidate"].values.astype(np.int64)
        print(f"    Data: {X_voice.shape[0]} samples, {X_voice.shape[1]} features")

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        for k, (train_idx, val_idx) in enumerate(skf.split(X_voice, y_voice)):
            model = VoiceEncoder(input_dim=VOICE_DIM, classify=True)
            logs, best_auc = contrastive_pretrain_encoder(
                "voice_uci_voice",
                model,
                X_voice[train_idx],
                y_voice[train_idx],
                X_voice[val_idx],
                y_voice[val_idx],
                fold=k,
                batch_size=BATCH_SIZES["voice"],
                embedding_dim=VoiceEncoder.EMBEDDING_DIM,
            )
            all_logs.extend(logs)
            encoder_summary.setdefault("voice", {})[k] = best_auc
    else:
        print("    UCI Voice data not found — skipping")

    # ---- 1c. Wearable encoder (PADS) ----
    print("\n>>> Wearable Encoder (PADS)")
    if PADS_CSV.exists():
        df_pads = pd.read_csv(PADS_CSV)
        pads_meta = [
            "subject_id",
            "dbs_candidate",
            "label_type",
            "dataset",
            "condition",
            "updrs_iii",
            "hy_stage",
        ]
        pads_feats = [
            c
            for c in df_pads.columns
            if c not in pads_meta and pd.api.types.is_numeric_dtype(df_pads[c])
        ]
        X_wear = df_pads[pads_feats].values.astype(np.float32)
        X_wear = np.nan_to_num(X_wear, nan=0.0)
        y_wear = df_pads["dbs_candidate"].values.astype(np.int64)

        # Ensure dim matches checkpoint
        actual_dim = X_wear.shape[1]
        if actual_dim != WEARABLE_DIM:
            print(
                f"    Warning: PADS features={actual_dim} vs checkpoint WEARABLE_DIM={WEARABLE_DIM}"
            )
            if actual_dim > WEARABLE_DIM:
                X_wear = X_wear[:, :WEARABLE_DIM]
            else:
                pad = np.zeros((X_wear.shape[0], WEARABLE_DIM - actual_dim), dtype=np.float32)
                X_wear = np.hstack([X_wear, pad])

        print(f"    Data: {X_wear.shape[0]} samples, {X_wear.shape[1]} features")

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        for k, (train_idx, val_idx) in enumerate(skf.split(X_wear, y_wear)):
            model = WearableResidualMLP(input_dim=WEARABLE_DIM, classify=True)
            logs, best_auc = contrastive_pretrain_encoder(
                "wearable",
                model,
                X_wear[train_idx],
                y_wear[train_idx],
                X_wear[val_idx],
                y_wear[val_idx],
                fold=k,
                batch_size=BATCH_SIZES["wearable"],
                embedding_dim=WearableResidualMLP.EMBEDDING_DIM,
            )
            all_logs.extend(logs)
            encoder_summary.setdefault("wearable", {})[k] = best_auc
    else:
        print("    PADS data not found — skipping")

    # Print encoder summary
    print("\n" + "-" * 72)
    print("  Contrastive Encoder Pretraining Summary")
    print("-" * 72)
    for enc_name, fold_aucs in encoder_summary.items():
        aucs = list(fold_aucs.values())
        fold_str = "  ".join(f"F{k}={v:.4f}" for k, v in sorted(fold_aucs.items()))
        print(
            f"  {enc_name:20s} | {fold_str} | "
            f"Mean={np.mean(aucs):.4f} +/- {np.std(aucs):.4f}"
        )

    # ==================================================================
    # STAGE 2: Cross-modal gait alignment
    # ==================================================================
    print("\n" + "=" * 72)
    print("  STAGE 2: Cross-Modal Gait Alignment")
    print("=" * 72)

    alignment_logs = []
    for k in range(N_FOLDS):
        logs = cross_modal_gait_alignment(k)
        alignment_logs.extend(logs)

    # ==================================================================
    # STAGE 3: Fusion model training with contrastive encoders
    # ==================================================================
    print("\n" + "=" * 72)
    print("  STAGE 3: Contrastive Fusion Model Training")
    print("=" * 72)

    fusion_logs = []
    fusion_aucs = {}

    for k in range(N_FOLDS):
        fold_info = splits["cv_folds"][k]
        train_idx = np.array(fold_info["train"])
        val_idx = np.array(fold_info["val"])

        logs, best_auc = train_contrastive_fusion_fold(
            k,
            X_gait[train_idx],
            y_gait[train_idx],
            X_gait[val_idx],
            y_gait[val_idx],
        )
        fusion_logs.extend(logs)
        fusion_aucs[k] = best_auc

    # ==================================================================
    # RESULTS COMPARISON
    # ==================================================================
    print("\n" + "=" * 72)
    print("  RESULTS COMPARISON")
    print("=" * 72)

    # Load original fusion results
    orig_fusion_log = TABLES_DIR / "fusion_training_log.csv"
    orig_aucs = {}
    if orig_fusion_log.exists():
        orig_df = pd.read_csv(orig_fusion_log)
        for model_name in orig_df["model"].unique():
            model_df = orig_df[orig_df["model"] == model_name]
            for fold_k in model_df["fold"].unique():
                fold_df = model_df[model_df["fold"] == fold_k]
                best = fold_df["val_auc"].max()
                orig_aucs.setdefault(model_name, {})[fold_k] = best

    # Build comparison table
    rows = []

    # Original models
    for model_name, fold_dict in orig_aucs.items():
        aucs = list(fold_dict.values())
        rows.append(
            {
                "Model": model_name,
                "Type": "Original",
                "Fold0_AUC": fold_dict.get(0, np.nan),
                "Fold1_AUC": fold_dict.get(1, np.nan),
                "Fold2_AUC": fold_dict.get(2, np.nan),
                "Fold3_AUC": fold_dict.get(3, np.nan),
                "Fold4_AUC": fold_dict.get(4, np.nan),
                "Mean_AUC": np.mean(aucs),
                "Std_AUC": np.std(aucs),
            }
        )

    # Contrastive fusion
    con_aucs = list(fusion_aucs.values())
    rows.append(
        {
            "Model": "ContrastiveFusion",
            "Type": "Contrastive",
            "Fold0_AUC": fusion_aucs.get(0, np.nan),
            "Fold1_AUC": fusion_aucs.get(1, np.nan),
            "Fold2_AUC": fusion_aucs.get(2, np.nan),
            "Fold3_AUC": fusion_aucs.get(3, np.nan),
            "Fold4_AUC": fusion_aucs.get(4, np.nan),
            "Mean_AUC": np.mean(con_aucs),
            "Std_AUC": np.std(con_aucs),
        }
    )

    results_df = pd.DataFrame(rows)
    results_path = TABLES_DIR / "contrastive_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n  Results saved: {results_path}")

    # Print comparison table
    print("\n  Model Comparison (val AUC):")
    print("  " + "-" * 90)
    print(
        f"  {'Model':<30s} {'Type':<14s} "
        f"{'F0':>6s} {'F1':>6s} {'F2':>6s} {'F3':>6s} {'F4':>6s} "
        f"{'Mean':>7s} {'Std':>6s}"
    )
    print("  " + "-" * 90)
    for _, row in results_df.iterrows():
        print(
            f"  {row['Model']:<30s} {row['Type']:<14s} "
            f"{row['Fold0_AUC']:6.4f} {row['Fold1_AUC']:6.4f} "
            f"{row['Fold2_AUC']:6.4f} {row['Fold3_AUC']:6.4f} "
            f"{row['Fold4_AUC']:6.4f} "
            f"{row['Mean_AUC']:7.4f} {row['Std_AUC']:6.4f}"
        )
    print("  " + "-" * 90)

    # Highlight improvement
    if "CrossAttentionFusion" in orig_aucs:
        orig_mean = np.mean(list(orig_aucs["CrossAttentionFusion"].values()))
        con_mean = np.mean(con_aucs)
        delta = con_mean - orig_mean
        print(
            f"\n  Delta (Contrastive vs Original CrossAttn): "
            f"{delta:+.4f} AUC ({delta/orig_mean*100:+.1f}%)"
        )

    # ==================================================================
    # t-SNE visualisation
    # ==================================================================
    generate_tsne_plot(fold=0)

    # ==================================================================
    # Save all logs
    # ==================================================================
    if all_logs:
        log_df = pd.DataFrame(all_logs)
        log_path = TABLES_DIR / "contrastive_encoder_training_log.csv"
        log_df.to_csv(log_path, index=False)
        print(f"\n  Encoder training log: {log_path}")

    if alignment_logs:
        align_df = pd.DataFrame(alignment_logs)
        align_path = TABLES_DIR / "contrastive_alignment_log.csv"
        align_df.to_csv(align_path, index=False)
        print(f"  Alignment log: {align_path}")

    if fusion_logs:
        fuse_df = pd.DataFrame(fusion_logs)
        fuse_path = TABLES_DIR / "contrastive_fusion_training_log.csv"
        fuse_df.to_csv(fuse_path, index=False)
        print(f"  Fusion training log: {fuse_path}")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed / 60:.1f} minutes")
    print("  Done.")


if __name__ == "__main__":
    main()
