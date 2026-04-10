"""
Baseline models for comparison (Table 2).

Models 1-10:
  1. SVM (RBF) on wearable       2. XGBoost on wearable
  3. SVM (RBF) on voice           4. XGBoost on voice
  5. SVM (RBF) on gait            6. XGBoost on gait
  7. Early fusion -> XGBoost      8. Early fusion -> MLP (PyTorch)
  9. Late fusion (average)       10. Late fusion (Optuna-weighted)
"""

# ── Hardware init (must be at top) ──────────────────────────────────────────
import os, multiprocessing

N_CORES = multiprocessing.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(N_CORES)
os.environ["MKL_NUM_THREADS"] = str(N_CORES)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_CORES)

# ── Standard imports ────────────────────────────────────────────────────────
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import yaml
import optuna
import torch
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Config ──────────────────────────────────────────────────────────────────
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

with open(_CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

SEED = CONFIG["model"]["seed"]  # 42
XGB_N_JOBS = CONFIG["training"]["xgb_n_jobs"]  # 32
N_JOBS = CONFIG["training"]["n_jobs"]  # -1
OPTUNA_TRIALS = CONFIG["training"]["optuna_trials"]  # 100
OPTUNA_N_JOBS = CONFIG["training"]["optuna_n_jobs"]  # 4
DEVICE = CONFIG["model"]["device"]  # "cuda"
BATCH_SIZE = CONFIG["model"]["wearable_batch_size"]  # 128


# ═════════════════════════════════════════════════════════════════════════════
# Single-modality model factories
# ═════════════════════════════════════════════════════════════════════════════

def create_svm_model(random_state: int = SEED) -> Pipeline:
    """SVM (RBF kernel) with StandardScaler and balanced class weights."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            probability=True,
            random_state=random_state,
            class_weight="balanced",
        )),
    ])


def create_xgboost_model(
    n_jobs: int = XGB_N_JOBS,
    random_state: int = SEED,
) -> XGBClassifier:
    """XGBoost classifier tuned for the imbalanced dataset (130/23 ratio)."""
    return XGBClassifier(
        n_jobs=n_jobs,
        tree_method="hist",
        device="cpu",
        random_state=random_state,
        eval_metric="logloss",
        scale_pos_weight=5.65,  # 130 / 23
        use_label_encoder=False,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Early Fusion MLP (PyTorch)
# ═════════════════════════════════════════════════════════════════════════════

class EarlyFusionMLP(nn.Module):
    """
    Concat features -> Linear(n, 256) -> BN -> ReLU -> Dropout(0.3)
                    -> Linear(256, 128) -> BN -> ReLU -> Dropout(0.2)
                    -> Linear(128, 2)
    """

    def __init__(self, n_features: int, device: str = DEVICE):
        super().__init__()
        self.device_name = device
        self.net = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )
        self._n_features = n_features
        self.to(self._resolve_device())

    # ── helpers ─────────────────────────────────────────────────────────────
    def _resolve_device(self) -> torch.device:
        if self.device_name == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def n_features(self) -> int:
        return self._n_features

    # ── sklearn-like API ────────────────────────────────────────────────────
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 150,
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
        batch_size: int = BATCH_SIZE,
    ) -> "EarlyFusionMLP":
        device = self._resolve_device()
        self.train()

        X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
        y_t = torch.as_tensor(y, dtype=torch.long, device=device)

        # Class-weighted CE loss
        class_counts = np.bincount(y.astype(int), minlength=2)
        weights = torch.tensor(
            [1.0 / max(c, 1) for c in class_counts],
            dtype=torch.float32,
            device=device,
        )
        weights = weights / weights.sum()
        criterion = nn.CrossEntropyLoss(weight=weights)

        optimiser = torch.optim.AdamW(
            self.parameters(), lr=lr, weight_decay=weight_decay,
        )
        scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

        n = len(X_t)
        for _ in range(epochs):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                xb, yb = X_t[idx], y_t[idx]
                optimiser.zero_grad(set_to_none=True)
                with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                    logits = self(xb)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()

        self.eval()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        device = self._resolve_device()
        self.eval()
        with torch.no_grad():
            X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
            logits = self(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def get_params(self, deep: bool = True) -> dict:
        return {"n_features": self._n_features, "device": self.device_name}

    def set_params(self, **params) -> "EarlyFusionMLP":
        for k, v in params.items():
            setattr(self, k, v)
        return self


# ═════════════════════════════════════════════════════════════════════════════
# Late Fusion: Average
# ═════════════════════════════════════════════════════════════════════════════

class LateFusionAverage:
    """Average predicted probabilities from 3 single-modality models."""

    def __init__(self):
        self.models: Dict[str, Any] = {}  # modality_name -> fitted model

    def set_models(
        self,
        wearable_model: Any,
        voice_model: Any,
        gait_model: Any,
    ) -> "LateFusionAverage":
        self.models = {
            "wearable": wearable_model,
            "voice": voice_model,
            "gait": gait_model,
        }
        return self

    def predict_proba(
        self,
        X_wearable: np.ndarray,
        X_voice: np.ndarray,
        X_gait: np.ndarray,
    ) -> np.ndarray:
        probs = np.stack([
            self.models["wearable"].predict_proba(X_wearable),
            self.models["voice"].predict_proba(X_voice),
            self.models["gait"].predict_proba(X_gait),
        ])
        return probs.mean(axis=0)

    def predict(
        self,
        X_wearable: np.ndarray,
        X_voice: np.ndarray,
        X_gait: np.ndarray,
    ) -> np.ndarray:
        return np.argmax(
            self.predict_proba(X_wearable, X_voice, X_gait), axis=1,
        )

    def get_params(self, deep: bool = True) -> dict:
        return {"models": list(self.models.keys())}

    def set_params(self, **params) -> "LateFusionAverage":
        return self


# ═════════════════════════════════════════════════════════════════════════════
# Late Fusion: Optuna-weighted ensemble
# ═════════════════════════════════════════════════════════════════════════════

class LateFusionOptuna:
    """
    Learn optimal weights (w1, w2, w3) s.t. w1+w2+w3=1 via Optuna,
    maximising AUC-ROC on a validation set.
    """

    def __init__(
        self,
        n_trials: int = OPTUNA_TRIALS,
        n_jobs: int = OPTUNA_N_JOBS,
        seed: int = SEED,
    ):
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.seed = seed
        self.models: Dict[str, Any] = {}
        self.weights: Optional[np.ndarray] = None

    def set_models(
        self,
        wearable_model: Any,
        voice_model: Any,
        gait_model: Any,
    ) -> "LateFusionOptuna":
        self.models = {
            "wearable": wearable_model,
            "voice": voice_model,
            "gait": gait_model,
        }
        return self

    def optimise_weights(
        self,
        X_wearable_val: np.ndarray,
        X_voice_val: np.ndarray,
        X_gait_val: np.ndarray,
        y_val: np.ndarray,
    ) -> np.ndarray:
        """Run Optuna to find best modality weights on validation data."""
        prob_w = self.models["wearable"].predict_proba(X_wearable_val)[:, 1]
        prob_v = self.models["voice"].predict_proba(X_voice_val)[:, 1]
        prob_g = self.models["gait"].predict_proba(X_gait_val)[:, 1]

        def objective(trial: optuna.Trial) -> float:
            w1 = trial.suggest_float("w_wearable", 0.0, 1.0)
            w2 = trial.suggest_float("w_voice", 0.0, 1.0 - w1)
            w3 = 1.0 - w1 - w2
            blended = w1 * prob_w + w2 * prob_v + w3 * prob_g
            return roc_auc_score(y_val, blended)

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(
            objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            show_progress_bar=False,
        )

        best = study.best_params
        w1 = best["w_wearable"]
        w2 = best["w_voice"]
        w3 = 1.0 - w1 - w2
        self.weights = np.array([w1, w2, w3])
        return self.weights

    def predict_proba(
        self,
        X_wearable: np.ndarray,
        X_voice: np.ndarray,
        X_gait: np.ndarray,
    ) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError(
                "Weights not set. Call optimise_weights() first."
            )
        prob_w = self.models["wearable"].predict_proba(X_wearable)
        prob_v = self.models["voice"].predict_proba(X_voice)
        prob_g = self.models["gait"].predict_proba(X_gait)
        return (
            self.weights[0] * prob_w
            + self.weights[1] * prob_v
            + self.weights[2] * prob_g
        )

    def predict(
        self,
        X_wearable: np.ndarray,
        X_voice: np.ndarray,
        X_gait: np.ndarray,
    ) -> np.ndarray:
        return np.argmax(
            self.predict_proba(X_wearable, X_voice, X_gait), axis=1,
        )

    def get_params(self, deep: bool = True) -> dict:
        return {
            "n_trials": self.n_trials,
            "n_jobs": self.n_jobs,
            "seed": self.seed,
            "weights": self.weights,
        }

    def set_params(self, **params) -> "LateFusionOptuna":
        for k, v in params.items():
            setattr(self, k, v)
        return self


# ═════════════════════════════════════════════════════════════════════════════
# Registry
# ═════════════════════════════════════════════════════════════════════════════

def get_all_baselines(n_features_all: int = 256) -> Dict[str, Dict[str, Any]]:
    """
    Return all 10 baseline model configs for Table 2.

    Parameters
    ----------
    n_features_all : int
        Total feature dimensionality after concatenating all modalities.
        Used to initialise the EarlyFusionMLP.

    Returns
    -------
    dict  {model_name: {'model': ..., 'modality': ...}}
    """
    return {
        # ── Single-modality (1-6) ──────────────────────────────────────────
        "svm_wearable":      {"model": create_svm_model(),      "modality": "wearable"},
        "xgb_wearable":      {"model": create_xgboost_model(),  "modality": "wearable"},
        "svm_voice":         {"model": create_svm_model(),      "modality": "voice"},
        "xgb_voice":         {"model": create_xgboost_model(),  "modality": "voice"},
        "svm_gait":          {"model": create_svm_model(),      "modality": "gait"},
        "xgb_gait":          {"model": create_xgboost_model(),  "modality": "gait"},
        # ── Early fusion (7-8) ─────────────────────────────────────────────
        "early_fusion_xgb":  {"model": create_xgboost_model(),  "modality": "all"},
        "early_fusion_mlp":  {"model": EarlyFusionMLP(n_features=n_features_all), "modality": "all"},
        # ── Late fusion (9-10) ─────────────────────────────────────────────
        "late_fusion_avg":   {"model": LateFusionAverage(),      "modality": "all"},
        "late_fusion_optuna": {"model": LateFusionOptuna(),      "modality": "all"},
    }


def _count_params(model: Any) -> int:
    """Best-effort parameter count for heterogeneous model types."""
    # PyTorch
    if isinstance(model, nn.Module):
        return sum(p.numel() for p in model.parameters())
    # sklearn Pipeline
    if isinstance(model, Pipeline):
        return sum(
            _count_params(step) for _, step in model.steps
        )
    # XGBClassifier (before fitting has no trees yet)
    if isinstance(model, XGBClassifier):
        try:
            booster = model.get_booster()
            return int(booster.attr("best_iteration") or 0)
        except Exception:
            return 0  # not yet fitted
    # SVC
    if isinstance(model, SVC):
        n_sv = getattr(model, "n_support_", None)
        if n_sv is not None:
            return int(n_sv.sum())
        return 0
    # StandardScaler
    if isinstance(model, StandardScaler):
        mean = getattr(model, "mean_", None)
        return len(mean) * 2 if mean is not None else 0
    # Late fusion wrappers
    if hasattr(model, "models"):
        return sum(_count_params(m) for m in model.models.values())
    return 0


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    baselines = get_all_baselines(n_features_all=256)
    print(f"{'#':<4} {'Model':<24} {'Modality':<12} {'Params':>10}")
    print("-" * 54)
    for i, (name, cfg) in enumerate(baselines.items(), 1):
        n_params = _count_params(cfg["model"])
        print(f"{i:<4} {name:<24} {cfg['modality']:<12} {n_params:>10,}")
    print(f"\nTotal baselines: {len(baselines)}")
