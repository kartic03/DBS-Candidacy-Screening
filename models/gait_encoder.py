#!/usr/bin/env python3
"""Simple MLP encoder for gait features.

JBI DBS Screening Project — Gait Modality
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


# ── Gait MLP Encoder ────────────────────────────────────────────────────────
class GaitEncoder(nn.Module):
    """Simple MLP encoder for gait features.

    Architecture
    ------------
    Input(n) -> Linear(64) -> BN -> ReLU -> Dropout(0.3)
             -> Linear(64) -> BN -> ReLU
             -> Output: 64-dim embedding

    Parameters
    ----------
    input_dim : int
        Number of input features.
        Default 80; WearGait has 20 features.
    classify : bool
        If True, append a 2-class classification head.
    """

    EMBEDDING_DIM = 64

    def __init__(self, input_dim: int = 80, classify: bool = False):
        super().__init__()
        self.input_dim = input_dim

        # Block 1
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.drop1 = nn.Dropout(0.3)

        # Block 2
        self.fc2 = nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64)

        # Optional classifier head
        self.classifier_head = nn.Linear(64, 2) if classify else None

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop1(F.relu(self.bn1(self.fc1(x))))
        out = F.relu(self.bn2(self.fc2(out)))

        if self.classifier_head is not None:
            out = self.classifier_head(out)

        return out

    @staticmethod
    def get_embedding_dim() -> int:
        return GaitEncoder.EMBEDDING_DIM


# ── CLI smoke-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    for dim_name, dim_val in [("default", 80), ("WearGait", 20)]:
        model = GaitEncoder(input_dim=dim_val, classify=False).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[GaitEncoder  input_dim={dim_val} ({dim_name})] Parameters: {n_params:,}")
        x = torch.randn(4, dim_val, device=device)
        out = model(x)
        print(f"  Input: {x.shape}  ->  Embedding: {out.shape}")
        assert out.shape == (4, 64), f"Expected (4, 64), got {out.shape}"

    # Verify classifier head
    model_cls = GaitEncoder(input_dim=80, classify=True).to(device)
    out_cls = model_cls(torch.randn(4, 80, device=device))
    assert out_cls.shape == (4, 2), f"Classifier head: expected (4, 2), got {out_cls.shape}"
    print("\n[GaitEncoder classify=True] Logits shape:", out_cls.shape)
    print("All checks passed.")
