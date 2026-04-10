"""
Cross-Attention Fusion Model for Multimodal DBS Candidate Classification.

Main contribution: fuses wearable, gait, and voice embeddings via
three cross-attention blocks, with modality dropout for robustness.

Also includes ablation variants (SimpleConcatFusionModel, NoVoiceFusionModel)
and FocalLoss for class-imbalanced training.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Hardware init — MUST be at top before any numerical library import
# ---------------------------------------------------------------------------
import os, multiprocessing

N_CORES = multiprocessing.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(N_CORES)
os.environ["MKL_NUM_THREADS"] = str(N_CORES)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_CORES)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Focal Loss
# ============================================================================
class FocalLoss(nn.Module):
    """Focal loss for class-imbalanced binary classification.

    Parameters
    ----------
    gamma : float
        Focusing parameter — higher values down-weight easy examples more.
    alpha : float
        Weight for the *positive* class (DBS candidates are minority).
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : (N, 2) raw logits from the classifier head.
        targets : (N,) integer class labels in {0, 1}.
        """
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # probability of the correct class

        # Per-sample alpha weight
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)

        focal_loss = alpha_t * ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ============================================================================
# Cross-Attention Fusion Model (main contribution)
# ============================================================================
class CrossAttentionFusionModel(nn.Module):
    """Multimodal fusion via cross-attention over wearable, voice, and gait
    embeddings, followed by a classification head.

    Cross-attention blocks
    ----------------------
    1. Q=Wearable, K=Gait, V=Gait   → W_gait   (wearable attends to gait)
    2. Q=Wearable, K=Voice, V=Voice  → W_voice  (wearable attends to voice)
    3. Q=Gait,     K=Voice, V=Voice  → G_voice  (gait attends to voice)

    Fusion
    ------
    Concat[W_orig, W_gait, W_voice, G_voice] → 512-dim → classification head.
    """

    def __init__(
        self,
        wearable_encoder: nn.Module,
        voice_encoder: nn.Module,
        gait_encoder: nn.Module,
        wearable_dim: int = 128,
        voice_dim: int = 64,
        gait_dim: int = 64,
        fusion_dim: int = 128,
        num_heads: int = 4,
        num_classes: int = 2,
        modality_dropout_prob: float = 0.2,
    ) -> None:
        super().__init__()

        # ----- pretrained encoders -----
        self.wearable_encoder = wearable_encoder
        self.voice_encoder = voice_encoder
        self.gait_encoder = gait_encoder

        # ----- projection layers (voice & gait → fusion_dim) -----
        self.voice_proj = nn.Linear(voice_dim, fusion_dim)
        self.gait_proj = nn.Linear(gait_dim, fusion_dim)

        # ----- cross-attention blocks -----
        # Block 1: Q=Wearable, K=Gait, V=Gait
        self.cross_attn_w_g = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=num_heads, batch_first=True
        )
        self.ln_w_g = nn.LayerNorm(fusion_dim)

        # Block 2: Q=Wearable, K=Voice, V=Voice
        self.cross_attn_w_v = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=num_heads, batch_first=True
        )
        self.ln_w_v = nn.LayerNorm(fusion_dim)

        # Block 3: Q=Gait, K=Voice, V=Voice
        self.cross_attn_g_v = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=num_heads, batch_first=True
        )
        self.ln_g_v = nn.LayerNorm(fusion_dim)

        # ----- classification head -----
        concat_dim = fusion_dim * 4  # 512
        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

        # ----- config -----
        self.modality_dropout_prob = modality_dropout_prob
        self.fusion_dim = fusion_dim

        # ----- attention weight cache (for XAI) -----
        self._attn_weights: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Encoder control
    # ------------------------------------------------------------------
    def freeze_encoders(self) -> None:
        """Freeze all three encoder sub-networks."""
        for encoder in (self.wearable_encoder, self.voice_encoder, self.gait_encoder):
            for param in encoder.parameters():
                param.requires_grad = False

    def unfreeze_encoders(self) -> None:
        """Unfreeze all three encoder sub-networks."""
        for encoder in (self.wearable_encoder, self.voice_encoder, self.gait_encoder):
            for param in encoder.parameters():
                param.requires_grad = True

    def get_attention_weights(self) -> Dict[str, torch.Tensor]:
        """Return the attention weights from the last forward pass."""
        return self._attn_weights

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        wearable_x: torch.Tensor,
        voice_x: torch.Tensor,
        gait_x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        wearable_x : input tensor for the wearable encoder.
        voice_x    : input tensor for the voice encoder.
        gait_x     : input tensor for the gait encoder.

        Returns
        -------
        dict with keys ``logits``, ``probabilities``, ``attention_weights``.
        """
        # --- encode ---
        wearable_emb = self.wearable_encoder(wearable_x)  # (B, 128)
        voice_emb = self.voice_encoder(voice_x)            # (B, 64)
        gait_emb = self.gait_encoder(gait_x)               # (B, 64)

        # --- project voice & gait to fusion_dim ---
        voice_emb = self.voice_proj(voice_emb)   # (B, 128)
        gait_emb = self.gait_proj(gait_emb)      # (B, 128)

        # --- modality dropout (training only) ---
        if self.training and self.modality_dropout_prob > 0:
            drop_idx = torch.randint(0, 3, (1,)).item()
            if torch.rand(1).item() < self.modality_dropout_prob:
                if drop_idx == 0:
                    wearable_emb = torch.zeros_like(wearable_emb)
                elif drop_idx == 1:
                    voice_emb = torch.zeros_like(voice_emb)
                else:
                    gait_emb = torch.zeros_like(gait_emb)

        # Keep original wearable embedding for the concat
        w_orig = wearable_emb  # (B, 128)

        # MultiheadAttention expects (B, S, E) with batch_first=True.
        # Our embeddings are (B, E) → unsqueeze to (B, 1, E).
        w = wearable_emb.unsqueeze(1)  # (B, 1, 128)
        v = voice_emb.unsqueeze(1)     # (B, 1, 128)
        g = gait_emb.unsqueeze(1)      # (B, 1, 128)

        # --- Block 1: Q=W, K=G, V=G ---
        w_gait, attn_w_g = self.cross_attn_w_g(w, g, g)
        w_gait = self.ln_w_g(w_gait + w)  # residual + layernorm

        # --- Block 2: Q=W, K=V, V=V ---
        w_voice, attn_w_v = self.cross_attn_w_v(w, v, v)
        w_voice = self.ln_w_v(w_voice + w)  # residual + layernorm

        # --- Block 3: Q=G, K=V, V=V ---
        g_voice, attn_g_v = self.cross_attn_g_v(g, v, v)
        g_voice = self.ln_g_v(g_voice + g)  # residual + layernorm

        # Cache attention weights for XAI visualization
        self._attn_weights = {
            "wearable_gait": attn_w_g.detach(),   # (B, 1, 1)
            "wearable_voice": attn_w_v.detach(),   # (B, 1, 1)
            "gait_voice": attn_g_v.detach(),       # (B, 1, 1)
        }

        # --- Squeeze back to (B, 128) ---
        w_gait = w_gait.squeeze(1)
        w_voice = w_voice.squeeze(1)
        g_voice = g_voice.squeeze(1)

        # --- Fusion: concat ---
        fused = torch.cat([w_orig, w_gait, w_voice, g_voice], dim=-1)  # (B, 512)

        # --- Classification ---
        logits = self.classifier(fused)                    # (B, 2)
        probabilities = F.softmax(logits, dim=-1)          # (B, 2)

        return {
            "logits": logits,
            "probabilities": probabilities,
            "attention_weights": self._attn_weights,
        }


# ============================================================================
# Ablation: Simple Concat Fusion (no cross-attention)
# ============================================================================
class SimpleConcatFusionModel(nn.Module):
    """Ablation baseline — concatenate projected embeddings without any
    cross-attention mechanism.

    Concat[W(128), V_proj(128), G_proj(128)] → 384-dim → head.
    """

    def __init__(
        self,
        wearable_encoder: nn.Module,
        voice_encoder: nn.Module,
        gait_encoder: nn.Module,
        wearable_dim: int = 128,
        voice_dim: int = 64,
        gait_dim: int = 64,
        fusion_dim: int = 128,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        self.wearable_encoder = wearable_encoder
        self.voice_encoder = voice_encoder
        self.gait_encoder = gait_encoder

        self.voice_proj = nn.Linear(voice_dim, fusion_dim)
        self.gait_proj = nn.Linear(gait_dim, fusion_dim)

        concat_dim = fusion_dim * 3  # 384
        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(
        self,
        wearable_x: torch.Tensor,
        voice_x: torch.Tensor,
        gait_x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        wearable_emb = self.wearable_encoder(wearable_x)  # (B, 128)
        voice_emb = self.voice_proj(self.voice_encoder(voice_x))  # (B, 128)
        gait_emb = self.gait_proj(self.gait_encoder(gait_x))      # (B, 128)

        fused = torch.cat([wearable_emb, voice_emb, gait_emb], dim=-1)  # (B, 384)
        logits = self.classifier(fused)
        probabilities = F.softmax(logits, dim=-1)

        return {"logits": logits, "probabilities": probabilities}


# ============================================================================
# Ablation: No-Voice Fusion (wearable + gait only)
# ============================================================================
class NoVoiceFusionModel(nn.Module):
    """Ablation — only wearable and gait, single cross-attention block.

    Block: Q=Wearable, K=Gait, V=Gait → W_gait
    Concat[W_orig(128), W_gait(128)] → 256-dim → head.
    """

    def __init__(
        self,
        wearable_encoder: nn.Module,
        gait_encoder: nn.Module,
        wearable_dim: int = 128,
        gait_dim: int = 64,
        fusion_dim: int = 128,
        num_heads: int = 4,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        self.wearable_encoder = wearable_encoder
        self.gait_encoder = gait_encoder

        self.gait_proj = nn.Linear(gait_dim, fusion_dim)

        self.cross_attn_w_g = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=num_heads, batch_first=True
        )
        self.ln_w_g = nn.LayerNorm(fusion_dim)

        concat_dim = fusion_dim * 2  # 256
        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(
        self,
        wearable_x: torch.Tensor,
        gait_x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        wearable_emb = self.wearable_encoder(wearable_x)          # (B, 128)
        gait_emb = self.gait_proj(self.gait_encoder(gait_x))      # (B, 128)

        w_orig = wearable_emb

        w = wearable_emb.unsqueeze(1)  # (B, 1, 128)
        g = gait_emb.unsqueeze(1)      # (B, 1, 128)

        w_gait, attn_w_g = self.cross_attn_w_g(w, g, g)
        w_gait = self.ln_w_g(w_gait + w)  # residual + layernorm
        w_gait = w_gait.squeeze(1)         # (B, 128)

        fused = torch.cat([w_orig, w_gait], dim=-1)  # (B, 256)
        logits = self.classifier(fused)
        probabilities = F.softmax(logits, dim=-1)

        return {
            "logits": logits,
            "probabilities": probabilities,
            "attention_weights": {"wearable_gait": attn_w_g.detach()},
        }


# ============================================================================
# Main — smoke test
# ============================================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    # ------------------------------------------------------------------
    # Try importing real encoders; fall back to simple stubs if the
    # encoder files haven't been created yet.
    # ------------------------------------------------------------------
    # Default input dims matching each encoder
    WEARABLE_INPUT_DIM = 500
    VOICE_INPUT_DIM = 88
    GAIT_INPUT_DIM = 80

    try:
        from models.wearable_encoder import WearableResidualMLP  # noqa: F401
        wearable_enc = WearableResidualMLP(input_dim=WEARABLE_INPUT_DIM, classify=False)
        print("[INFO] Loaded WearableResidualMLP from models/wearable_encoder.py")
    except (ImportError, ModuleNotFoundError):
        wearable_enc = nn.Sequential(nn.Linear(WEARABLE_INPUT_DIM, 128), nn.ReLU())
        print(f"[INFO] Using stub wearable encoder (Linear {WEARABLE_INPUT_DIM}→128)")

    try:
        from models.voice_encoder import VoiceEncoder  # noqa: F401
        voice_enc = VoiceEncoder(input_dim=VOICE_INPUT_DIM, classify=False)
        print("[INFO] Loaded VoiceEncoder from models/voice_encoder.py")
    except (ImportError, ModuleNotFoundError):
        voice_enc = nn.Sequential(nn.Linear(VOICE_INPUT_DIM, 64), nn.ReLU())
        print(f"[INFO] Using stub voice encoder (Linear {VOICE_INPUT_DIM}→64)")

    try:
        from models.gait_encoder import GaitEncoder  # noqa: F401
        gait_enc = GaitEncoder(input_dim=GAIT_INPUT_DIM, classify=False)
        print("[INFO] Loaded GaitEncoder from models/gait_encoder.py")
    except (ImportError, ModuleNotFoundError):
        gait_enc = nn.Sequential(nn.Linear(GAIT_INPUT_DIM, 64), nn.ReLU())
        print(f"[INFO] Using stub gait encoder (Linear {GAIT_INPUT_DIM}→64)")

    # ------------------------------------------------------------------
    # Build full fusion model
    # ------------------------------------------------------------------
    model = CrossAttentionFusionModel(
        wearable_encoder=wearable_enc,
        voice_encoder=voice_enc,
        gait_encoder=gait_enc,
    )
    model.eval()

    B = 8  # batch size
    wearable_x = torch.randn(B, WEARABLE_INPUT_DIM)
    voice_x = torch.randn(B, VOICE_INPUT_DIM)
    gait_x = torch.randn(B, GAIT_INPUT_DIM)

    with torch.no_grad():
        out = model(wearable_x, voice_x, gait_x)

    print("\n===== CrossAttentionFusionModel =====")
    print(f"  logits shape       : {out['logits'].shape}")
    print(f"  probabilities shape: {out['probabilities'].shape}")
    for name, aw in out["attention_weights"].items():
        print(f"  attn [{name}] shape: {aw.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  total params       : {total_params:,}")
    print(f"  trainable params   : {trainable:,}")

    # ------------------------------------------------------------------
    # Ablation: SimpleConcatFusionModel
    # ------------------------------------------------------------------
    concat_model = SimpleConcatFusionModel(
        wearable_encoder=wearable_enc,
        voice_encoder=voice_enc,
        gait_encoder=gait_enc,
    )
    concat_model.eval()
    with torch.no_grad():
        cout = concat_model(wearable_x, voice_x, gait_x)
    print("\n===== SimpleConcatFusionModel =====")
    print(f"  logits shape : {cout['logits'].shape}")
    print(f"  total params : {sum(p.numel() for p in concat_model.parameters()):,}")

    # ------------------------------------------------------------------
    # Ablation: NoVoiceFusionModel
    # ------------------------------------------------------------------
    no_voice_model = NoVoiceFusionModel(
        wearable_encoder=wearable_enc,
        gait_encoder=gait_enc,
    )
    no_voice_model.eval()
    with torch.no_grad():
        nvout = no_voice_model(wearable_x, gait_x)
    print("\n===== NoVoiceFusionModel =====")
    print(f"  logits shape : {nvout['logits'].shape}")
    print(f"  total params : {sum(p.numel() for p in no_voice_model.parameters()):,}")

    # ------------------------------------------------------------------
    # FocalLoss quick check
    # ------------------------------------------------------------------
    criterion = FocalLoss(gamma=2.0, alpha=0.75)
    dummy_logits = torch.randn(B, 2)
    dummy_targets = torch.randint(0, 2, (B,))
    loss = criterion(dummy_logits, dummy_targets)
    print(f"\n===== FocalLoss =====")
    print(f"  loss value : {loss.item():.4f}")

    print("\n[OK] All smoke tests passed.")
