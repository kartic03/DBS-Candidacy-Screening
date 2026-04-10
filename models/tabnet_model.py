#!/usr/bin/env python3
"""
TabNet model wrapper for the JBI DBS Screening Project.

Uses pytorch_tabnet.TabNetClassifier for tabular classification with
built-in attention mask extraction.

Conda env: jbi_dbs | Config: ../config.yaml
"""

# ── Hardware init (MUST be at top) ──────────────────────────────────────────
import os, multiprocessing

N_CORES = multiprocessing.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(N_CORES)
os.environ["MKL_NUM_THREADS"] = str(N_CORES)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_CORES)

# ── Imports ─────────────────────────────────────────────────────────────────
import numpy as np
import torch
from pathlib import Path
from pytorch_tabnet.tab_model import TabNetClassifier


# ── Config ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_tabnet(seed: int = 42, n_features: int = 20) -> TabNetClassifier:
    """Build a TabNetClassifier with tuned hyperparameters.

    Parameters
    ----------
    seed : int
        Random seed.
    n_features : int
        Number of input features.

    Returns
    -------
    TabNetClassifier (unfitted)
    """
    # Hyperparameters tuned for small tabular dataset (~150 samples, 20 features)
    clf = TabNetClassifier(
        n_d=16,              # Width of decision prediction layer
        n_a=16,              # Width of attention layer
        n_steps=3,           # Number of sequential attention steps
        gamma=1.5,           # Coefficient for feature reusage in attention
        n_independent=1,     # Number of independent GLU layers at each step
        n_shared=2,          # Number of shared GLU layers at each step
        lambda_sparse=1e-3,  # Sparsity regularization
        momentum=0.3,        # Batch momentum for BatchNorm
        mask_type="sparsemax",
        seed=seed,
        verbose=0,
        device_name="auto",
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params=dict(step_size=20, gamma=0.9),
    )
    return clf


# ── CLI smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[TabNet] Building model...")
    clf = build_tabnet(seed=42, n_features=20)
    print(f"[TabNet] Model type: {type(clf).__name__}")

    # Quick test with dummy data
    X_train = np.random.randn(100, 20).astype(np.float32)
    y_train = np.random.randint(0, 2, 100)
    X_val = np.random.randn(20, 20).astype(np.float32)
    y_val = np.random.randint(0, 2, 20)

    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=["auc"],
        max_epochs=5,
        patience=5,
        batch_size=32,
    )

    preds = clf.predict_proba(X_val)
    print(f"[TabNet] Predictions shape: {preds.shape}")

    # Get attention masks
    explain_matrix, masks = clf.explain(X_val)
    print(f"[TabNet] Explain matrix shape: {explain_matrix.shape}")
    print(f"[TabNet] Number of attention steps: {len(masks)}")
    for i, m in enumerate(masks):
        print(f"  Step {i}: mask shape = {m.shape}")

    print("\n[TabNet] Smoke test passed.")
