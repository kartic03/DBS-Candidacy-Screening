#!/usr/bin/env python3
"""LSTM-based encoder for longitudinal PD voice data.

Encodes a sequence of voice feature vectors over time per patient into a
fixed-length patient embedding that captures the disease trajectory.

JBI DBS Screening Project | Conda env: jbi_dbs | Config: ../config.yaml
"""

# ── Hardware init (MUST be at top) ──────────────────────────────────────────
import os, multiprocessing

N_CORES = multiprocessing.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(N_CORES)
os.environ["MKL_NUM_THREADS"] = str(N_CORES)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_CORES)

# ── Imports ─────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from pathlib import Path
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# ── GPU setup ───────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# ── Config ──────────────────────────────────────────────────────────────────
_CFG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
if _CFG_PATH.exists():
    with open(_CFG_PATH, "r") as f:
        CFG = yaml.safe_load(f)
else:
    CFG = {}


# ── Weight initialisation ──────────────────────────────────────────────────
def _init_weights(module: nn.Module) -> None:
    """Kaiming normal for Linear, ones/zeros for BatchNorm, orthogonal for LSTM."""
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if "weight_ih" in name:
                nn.init.kaiming_normal_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
                # Set forget gate bias to 1 for better gradient flow
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)


# ── Temporal LSTM Encoder ──────────────────────────────────────────────────
class TemporalLSTMEncoder(nn.Module):
    """Bidirectional LSTM encoder for longitudinal PD data.

    Input:  (batch, seq_len, features) — sequence of voice feature vectors
    Output: (batch, embedding_dim) — patient-level embedding capturing
            the disease trajectory over time

    Parameters
    ----------
    input_dim : int
        Number of input features per time step (16 for UCI UPDRS voice).
    hidden_dim : int
        LSTM hidden state dimension (default 64).
    num_layers : int
        Number of stacked LSTM layers (default 2).
    dropout : float
        Dropout between LSTM layers and in the projection head (default 0.3).
    embedding_dim : int
        Output embedding dimension (default 64).
    """

    EMBEDDING_DIM = 64

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        embedding_dim: int = 64,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection: normalise features before LSTM
        # Use LayerNorm to handle variable and small batch sizes during CV
        self.input_ln = nn.LayerNorm(input_dim)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        # Projection head: bidirectional -> 2x hidden_dim -> embedding_dim
        # Use LayerNorm instead of BatchNorm to handle batch_size=1 during CV
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor (batch, seq_len, features)
            Padded input sequences.
        lengths : Tensor (batch,), optional
            Actual sequence lengths (before padding). If provided,
            uses pack_padded_sequence for efficiency and correctness.

        Returns
        -------
        Tensor (batch, embedding_dim)
            Patient-level trajectory embedding.
        """
        batch_size, seq_len, feat_dim = x.shape

        # Apply layer norm across features
        x = self.input_ln(x)

        if lengths is not None:
            # Sort by length (descending) for pack_padded_sequence
            lengths = lengths.cpu().long()
            sorted_lengths, sort_idx = lengths.sort(descending=True)
            x_sorted = x[sort_idx]

            # Clamp to avoid zero-length sequences
            sorted_lengths = sorted_lengths.clamp(min=1)

            packed = pack_padded_sequence(x_sorted, sorted_lengths, batch_first=True)
            packed_out, (h_n, _) = self.lstm(packed)

            # Unsort
            _, unsort_idx = sort_idx.sort()

            # h_n: (num_layers * 2, batch, hidden_dim) for bidirectional
            # Take last layer forward + backward
            h_forward = h_n[-2][unsort_idx]  # (batch, hidden_dim)
            h_backward = h_n[-1][unsort_idx]  # (batch, hidden_dim)
        else:
            _, (h_n, _) = self.lstm(x)
            h_forward = h_n[-2]
            h_backward = h_n[-1]

        # Concatenate forward and backward final hidden states
        h_cat = torch.cat([h_forward, h_backward], dim=1)  # (batch, hidden_dim*2)

        # Project to embedding space
        embedding = self.fc(h_cat)
        return embedding

    @staticmethod
    def get_embedding_dim() -> int:
        return TemporalLSTMEncoder.EMBEDDING_DIM


# ── Temporal Classifier ────────────────────────────────────────────────────
class TemporalClassifier(nn.Module):
    """LSTM encoder + classification / regression heads.

    Supports three tasks:
    - Binary DBS candidacy classification
    - Trajectory class (fast / slow progressor)
    - UPDRS slope regression

    Parameters
    ----------
    input_dim : int
        Number of voice features per time step.
    hidden_dim : int
        LSTM hidden dimension.
    num_layers : int
        Number of LSTM layers.
    dropout : float
        Dropout rate.
    task : str
        One of 'classification', 'trajectory', 'regression'.
    """

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        task: str = "classification",
    ):
        super().__init__()
        self.task = task
        self.encoder = TemporalLSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        embedding_dim = self.encoder.EMBEDDING_DIM

        if task == "classification":
            # Binary: DBS candidate yes/no
            self.head = nn.Linear(embedding_dim, 1)
        elif task == "trajectory":
            # 3-class: slow / moderate / fast progressor
            self.head = nn.Linear(embedding_dim, 3)
        elif task == "regression":
            # Continuous: predict UPDRS slope
            self.head = nn.Linear(embedding_dim, 1)
        else:
            raise ValueError(f"Unknown task: {task}")

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        embedding = self.encoder(x, lengths)
        return self.head(embedding)


# ── CLI smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Device: {device}")

    # Test encoder
    encoder = TemporalLSTMEncoder(input_dim=16, hidden_dim=64, num_layers=2).to(device)
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"[TemporalLSTMEncoder] Parameters: {n_params:,}")

    x = torch.randn(4, 140, 16, device=device)
    lengths = torch.tensor([140, 100, 80, 50])
    out = encoder(x, lengths)
    print(f"  Input: {x.shape}  ->  Embedding: {out.shape}")
    assert out.shape == (4, 64), f"Expected (4, 64), got {out.shape}"

    # Test without lengths
    out_no_len = encoder(x)
    assert out_no_len.shape == (4, 64)
    print(f"  Without lengths: {out_no_len.shape}")

    # Test classifiers
    for task in ["classification", "trajectory", "regression"]:
        model = TemporalClassifier(input_dim=16, task=task).to(device)
        out = model(x, lengths)
        expected = {"classification": (4, 1), "trajectory": (4, 3), "regression": (4, 1)}
        assert out.shape == expected[task], f"{task}: expected {expected[task]}, got {out.shape}"
        print(f"  [{task}] Output: {out.shape}")

    print("\nAll checks passed.")
