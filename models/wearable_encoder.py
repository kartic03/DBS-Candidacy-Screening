#!/usr/bin/env python3
"""Residual MLP and TabTransformer encoders for wearable sensor features.

JBI DBS Screening Project — Wearable Modality
Conda env: jbi_dbs | Config: ../config.yaml
"""

# ── Hardware init ────────────────────────────────────────────────────────────
import os, multiprocessing
N_CORES = multiprocessing.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(N_CORES)
os.environ["MKL_NUM_THREADS"] = str(N_CORES)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_CORES)

# ── Imports ──────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from pathlib import Path

# ── GPU setup ────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# ── Config ───────────────────────────────────────────────────────────────────
_CFG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
if _CFG_PATH.exists():
    with open(_CFG_PATH, "r") as f:
        CFG = yaml.safe_load(f)
else:
    CFG = {}


# ── Focal Loss ───────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Focal loss for imbalanced binary classification.

    Parameters
    ----------
    gamma : float
        Focusing parameter (default 2.0).
    alpha : float
        Weighting factor for the positive class (default 0.75).
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - pt) ** self.gamma * bce
        return loss.mean()


# ── Weight initialisation ────────────────────────────────────────────────────
def _init_weights(module: nn.Module) -> None:
    """Kaiming normal for Linear, ones/zeros for BatchNorm."""
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


# ── Residual MLP Encoder ────────────────────────────────────────────────────
class WearableResidualMLP(nn.Module):
    """Residual MLP encoder for wearable sensor features.

    Architecture
    ------------
    Input(n) -> Linear(256) -> BN -> ReLU -> Dropout(0.3)
             -> Linear(256) -> BN -> ReLU -> (+residual: Linear(n->256))
             -> Linear(128) -> BN -> ReLU -> Dropout(0.2)
             -> Output: 128-dim embedding

    Parameters
    ----------
    input_dim : int
        Number of input features (default 500).
    classify : bool
        If True, append a 2-class classification head.
    """

    EMBEDDING_DIM = 128

    def __init__(self, input_dim: int = 500, classify: bool = False):
        super().__init__()
        self.input_dim = input_dim

        # Block 1
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.3)

        # Block 2 (with residual)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.residual_proj = nn.Linear(input_dim, 256)

        # Block 3
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.drop3 = nn.Dropout(0.2)

        # Optional classifier head
        self.classifier_head = nn.Linear(128, 2) if classify else None

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # Block 1
        out = self.drop1(F.relu(self.bn1(self.fc1(x))))

        # Block 2 + residual
        out = self.bn2(self.fc2(out))
        out = F.relu(out + self.residual_proj(identity))

        # Block 3
        out = self.drop3(F.relu(self.bn3(self.fc3(out))))

        if self.classifier_head is not None:
            out = self.classifier_head(out)

        return out

    @staticmethod
    def get_embedding_dim() -> int:
        return WearableResidualMLP.EMBEDDING_DIM


# ── TabTransformer Encoder ───────────────────────────────────────────────────
class WearableTabTransformer(nn.Module):
    """TabTransformer-based encoder for wearable features (ablation variant).

    Wraps ``tab_transformer_pytorch.FTTransformer`` and projects its output
    down to a 128-dim embedding for drop-in compatibility with the Residual MLP.

    Parameters
    ----------
    input_dim : int
        Number of *continuous* input features (default 500).
    n_heads : int
        Attention heads in transformer blocks (default 4).
    depth : int
        Number of transformer blocks (default 2).
    classify : bool
        If True, append a 2-class classification head.
    """

    EMBEDDING_DIM = 128

    def __init__(
        self,
        input_dim: int = 500,
        n_heads: int = 4,
        depth: int = 2,
        classify: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim

        from tab_transformer_pytorch import FTTransformer

        # FTTransformer supports continuous-only features (no dummy categoricals).
        self.tab = FTTransformer(
            categories=(),
            num_continuous=input_dim,
            dim=64,
            depth=depth,
            heads=n_heads,
            attn_dropout=0.1,
            ff_dropout=0.1,
            dim_out=128,
        )

        # Optional classifier head
        self.classifier_head = nn.Linear(128, 2) if classify else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FTTransformer expects (cat_input, cont_input)
        cat_empty = torch.empty(x.size(0), 0, dtype=torch.long, device=x.device)
        out = self.tab(cat_empty, x)

        if self.classifier_head is not None:
            out = self.classifier_head(out)

        return out

    @staticmethod
    def get_embedding_dim() -> int:
        return WearableTabTransformer.EMBEDDING_DIM


# ── CLI smoke-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    for name, cls in [("ResidualMLP", WearableResidualMLP), ("TabTransformer", WearableTabTransformer)]:
        try:
            model = cls(input_dim=500, classify=False).to(device)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"[{name}] Parameters: {n_params:,}")
            x = torch.randn(4, 500, device=device)
            out = model(x)
            print(f"[{name}] Input: {x.shape}  ->  Embedding: {out.shape}")
            assert out.shape == (4, 128), f"Expected (4, 128), got {out.shape}"
            print(f"[{name}] OK\n")
        except ImportError as e:
            print(f"[{name}] Skipped — {e}\n")
