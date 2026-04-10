#!/usr/bin/env python3
"""LIME-based explainability for the JBI DBS screening project.

1. LIME explanations for 3 case-study patients (high / borderline / low DBS prob).
2. Concordance analysis: Spearman rank correlation between SHAP and LIME top-10.
3. Output tables and per-patient explanation figures.

JBI DBS Screening Project | Conda env: jbi_dbs | Config: ../config.yaml
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
# Imports
# ---------------------------------------------------------------------------
import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from lime.lime_tabular import LimeTabularExplainer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Project root & model imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.baseline_models import create_xgboost_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_CFG_PATH = PROJECT_ROOT / "config.yaml"
with open(_CFG_PATH, "r") as f:
    CFG = yaml.safe_load(f)

SEED = CFG["model"]["seed"]  # 42

# Figure settings
DPI = CFG["figures"]["dpi"]  # 300
FONT_FAMILY = CFG["figures"]["font_family"]  # Arial
FONT_SIZE = CFG["figures"]["font_size"]  # 10
COLOR_WEARABLE = CFG["figures"]["color_wearable"]  # #1f77b4
COLOR_GAIT = CFG["figures"]["color_gait"]  # #2ca02c
COLOR_VOICE = CFG["figures"]["color_voice"]  # #ff7f0e

plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": FONT_SIZE,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
})

# Paths
DATA_CSV = PROJECT_ROOT / "data" / "processed" / "fused" / "primary_cohort.csv"
SPLITS_JSON = PROJECT_ROOT / "data" / "splits" / "primary_splits.json"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

# Non-feature columns
NON_FEATURE_COLS = {"subject_id", "dbs_candidate", "label_type", "dataset"}
LABEL_COL = "dbs_candidate"

# Reproducibility
np.random.seed(SEED)


# ============================================================================
# 1. Data loading & modality splitting (mirrors evaluate.py / shap_analysis.py)
# ============================================================================

def get_modality_features(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """Split features into wearable / voice / gait groups."""
    feat_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

    gait_keywords = ["gait", "posture", "toe_tapping", "leg_agility"]
    gait_cols = [c for c in feat_cols if any(kw in c.lower() for kw in gait_keywords)]

    voice_keywords = ["speech", "facial", "kinetic_tremor", "postural_tremor"]
    voice_cols = [c for c in feat_cols if any(kw in c.lower() for kw in voice_keywords)]

    wearable_cols = [c for c in feat_cols if c not in gait_cols and c not in voice_cols]

    return wearable_cols, voice_cols, gait_cols


def load_data() -> Tuple[pd.DataFrame, dict, List[str], List[str], List[str]]:
    """Load primary cohort and splits."""
    df = pd.read_csv(DATA_CSV)
    with open(SPLITS_JSON, "r") as f:
        splits = json.load(f)
    wearable_cols, voice_cols, gait_cols = get_modality_features(df)
    return df, splits, wearable_cols, voice_cols, gait_cols


def build_train_test_arrays(
    df: pd.DataFrame,
    splits: dict,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int], List[int]]:
    """Return X_train, y_train, X_test, y_test, train_idx, test_idx."""
    test_idx = splits["test_indices"]
    all_idx = set(range(len(df)))
    train_idx = sorted(all_idx - set(test_idx))

    X_train = df.iloc[train_idx][feature_cols].values.astype(np.float32)
    y_train = df.iloc[train_idx][LABEL_COL].values.astype(np.int64)
    X_test = df.iloc[test_idx][feature_cols].values.astype(np.float32)
    y_test = df.iloc[test_idx][LABEL_COL].values.astype(np.int64)

    return X_train, y_train, X_test, y_test, train_idx, test_idx


def build_modality_map(
    feature_cols: List[str],
    wearable_cols: List[str],
    voice_cols: List[str],
    gait_cols: List[str],
) -> Dict[str, str]:
    """Map each feature name to its modality label."""
    mod_map = {}
    for c in feature_cols:
        if c in wearable_cols:
            mod_map[c] = "Wearable"
        elif c in voice_cols:
            mod_map[c] = "Voice"
        elif c in gait_cols:
            mod_map[c] = "Gait"
        else:
            mod_map[c] = "Wearable"
    return mod_map


def modality_color(modality: str) -> str:
    """Return hex colour for a modality."""
    return {
        "Wearable": COLOR_WEARABLE,
        "Gait": COLOR_GAIT,
        "Voice": COLOR_VOICE,
    }.get(modality, "#999999")


# ============================================================================
# 2. Case-study patient selection (same logic as shap_analysis.py)
# ============================================================================

def select_case_studies(
    probas: np.ndarray,
    y_test: np.ndarray,
    test_idx: List[int],
    subject_ids: List[str],
) -> Dict[str, int]:
    """Select 3 patients: high prob (>0.85), borderline (0.45-0.60), low (<0.20).

    Returns dict of category -> index into test set arrays.
    """
    cases = {}

    # High probability
    high_mask = probas > 0.85
    if high_mask.any():
        candidates = np.where(high_mask)[0]
        cases["high"] = candidates[np.argmax(probas[candidates])]
    else:
        cases["high"] = np.argmax(probas)

    # Borderline
    border_mask = (probas >= 0.45) & (probas <= 0.60)
    if border_mask.any():
        candidates = np.where(border_mask)[0]
        mid = 0.525
        cases["borderline"] = candidates[np.argmin(np.abs(probas[candidates] - mid))]
    else:
        mid_val = np.median(probas)
        cases["borderline"] = np.argmin(np.abs(probas - mid_val))

    # Low probability
    low_mask = probas < 0.20
    if low_mask.any():
        candidates = np.where(low_mask)[0]
        cases["low"] = candidates[np.argmin(probas[candidates])]
    else:
        cases["low"] = np.argmin(probas)

    for cat, idx in cases.items():
        global_idx = test_idx[idx]
        sid = subject_ids[global_idx] if global_idx < len(subject_ids) else f"idx_{global_idx}"
        print(f"  Case study ({cat:>10s}): test_idx={idx}, "
              f"subject={sid}, prob={probas[idx]:.4f}, "
              f"label={y_test[idx]}")

    return cases


# ============================================================================
# 3. LIME explanations
# ============================================================================

def create_lime_explainer(
    X_train: np.ndarray,
    feature_names: List[str],
) -> LimeTabularExplainer:
    """Create a LIME tabular explainer."""
    explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=["Non-DBS", "DBS"],
        mode="classification",
        random_state=SEED,
        discretize_continuous=True,
    )
    return explainer


def explain_patient(
    explainer: LimeTabularExplainer,
    predict_fn,
    instance: np.ndarray,
    num_features: int = 20,
) -> "lime.explanation.Explanation":
    """Generate a LIME explanation for a single patient."""
    exp = explainer.explain_instance(
        instance,
        predict_fn,
        num_features=num_features,
        top_labels=1,
        num_samples=5000,
    )
    return exp


def extract_lime_rankings(
    exp,
    label: int = 1,
) -> pd.DataFrame:
    """Extract feature importance rankings from a LIME explanation.

    Returns a DataFrame with columns: rank, feature, weight.
    """
    try:
        feat_weights = exp.as_list(label=label)
    except Exception:
        # Fall back to the first available label
        feat_weights = exp.as_list()

    rows = []
    for rank, (feat_str, weight) in enumerate(feat_weights, 1):
        rows.append({
            "rank": rank,
            "feature_description": feat_str,
            "weight": weight,
        })
    return pd.DataFrame(rows)


def extract_lime_feature_name(desc: str, feature_names: List[str]) -> Optional[str]:
    """Extract the raw feature name from a LIME feature description string.

    LIME descriptions look like: 'updrs_iii_total > 0.50' or 'age <= -0.23'.
    We match the longest feature name that appears as a prefix.
    """
    # Sort by length descending so we match the longest prefix first
    for fname in sorted(feature_names, key=len, reverse=True):
        if desc.startswith(fname):
            return fname
    # Fallback: try partial match
    for fname in sorted(feature_names, key=len, reverse=True):
        if fname in desc:
            return fname
    return desc.split(" ")[0]  # best guess: first token


# ============================================================================
# 4. LIME figure for a single patient
# ============================================================================

def plot_lime_explanation(
    exp,
    category: str,
    modality_map: Dict[str, str],
    feature_names: List[str],
    save_path: Path,
    num_features: int = 15,
) -> None:
    """Save a horizontal bar chart for a LIME explanation."""
    try:
        feat_weights = exp.as_list(label=1)
    except Exception:
        feat_weights = exp.as_list()

    feat_weights = feat_weights[:num_features]

    descriptions = [fw[0] for fw in feat_weights]
    weights = [fw[1] for fw in feat_weights]

    # Determine modality colours for each bar
    colours = []
    for desc in descriptions:
        raw_name = extract_lime_feature_name(desc, feature_names)
        mod = modality_map.get(raw_name, "Wearable")
        colours.append(modality_color(mod))

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(descriptions))
    ax.barh(y_pos, weights, color=colours, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(descriptions, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("LIME Weight (contribution to DBS prediction)")
    ax.set_title(f"LIME Explanation — {category.capitalize()} DBS Probability", fontsize=12)
    ax.axvline(x=0, color="black", linewidth=0.5, linestyle="--")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_WEARABLE, label="Wearable"),
        Patch(facecolor=COLOR_GAIT, label="Gait"),
        Patch(facecolor=COLOR_VOICE, label="Voice"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8, title="Modality")

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close("all")
    print(f"[FIGURE] Saved LIME explanation ({category}): {save_path}")


# ============================================================================
# 5. SHAP-LIME concordance
# ============================================================================

def load_shap_contributions(tables_dir: Path) -> Optional[pd.DataFrame]:
    """Load SHAP modality contributions from shap_analysis.py output."""
    path = tables_dir / "shap_modality_contributions.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def compute_shap_top_features(
    tables_dir: Path,
    feature_cols: List[str],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    top_k: int = 10,
) -> Optional[List[str]]:
    """Compute SHAP top-k features by retraining XGBoost and using TreeSHAP.

    This re-computes to get per-feature rankings (not just modality-level).
    """
    try:
        import shap
    except ImportError:
        print("[WARNING] shap package not available for concordance analysis.")
        return None

    print("  Computing SHAP feature rankings via TreeExplainer...")
    xgb_model = create_xgboost_model(n_jobs=-1, random_state=SEED)
    xgb_model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_test)

    # Handle multi-output
    if shap_values.values.ndim == 3:
        sv = shap_values.values[:, :, 1]
    else:
        sv = shap_values.values

    mean_abs_shap = np.abs(sv).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:top_k]
    top_features = [feature_cols[i] for i in top_idx]

    return top_features


def compute_lime_top_features(
    explainer: LimeTabularExplainer,
    predict_fn,
    X_test: np.ndarray,
    feature_names: List[str],
    top_k: int = 10,
) -> List[str]:
    """Compute LIME global feature rankings by averaging across all test samples."""
    print("  Computing LIME global feature rankings (averaging across test set)...")
    feature_importance = {fname: 0.0 for fname in feature_names}
    n_test = len(X_test)

    for i in range(n_test):
        exp = explainer.explain_instance(
            X_test[i],
            predict_fn,
            num_features=len(feature_names),
            top_labels=1,
            num_samples=1000,
        )
        try:
            feat_weights = exp.as_list(label=1)
        except Exception:
            feat_weights = exp.as_list()

        for desc, weight in feat_weights:
            raw_name = extract_lime_feature_name(desc, feature_names)
            if raw_name in feature_importance:
                feature_importance[raw_name] += abs(weight)

    # Average
    for fname in feature_importance:
        feature_importance[fname] /= max(n_test, 1)

    # Sort by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    top_features = [f[0] for f in sorted_features[:top_k]]

    return top_features


def compute_concordance(
    shap_ranking: List[str],
    lime_ranking: List[str],
    top_k: int = 10,
) -> Dict[str, float]:
    """Compute Spearman rank correlation between SHAP and LIME top-k lists.

    Features present in both lists are ranked 1..k. Features in only one list
    are given the worst rank (k+1).
    """
    # Union of features
    all_features = list(dict.fromkeys(shap_ranking[:top_k] + lime_ranking[:top_k]))

    shap_ranks = {}
    lime_ranks = {}
    worst_rank = top_k + 1

    for feat in all_features:
        if feat in shap_ranking[:top_k]:
            shap_ranks[feat] = shap_ranking[:top_k].index(feat) + 1
        else:
            shap_ranks[feat] = worst_rank

        if feat in lime_ranking[:top_k]:
            lime_ranks[feat] = lime_ranking[:top_k].index(feat) + 1
        else:
            lime_ranks[feat] = worst_rank

    # Convert to arrays in same order
    features_ordered = sorted(all_features)
    shap_arr = np.array([shap_ranks[f] for f in features_ordered])
    lime_arr = np.array([lime_ranks[f] for f in features_ordered])

    # Spearman correlation
    if len(features_ordered) < 3:
        rho, pval = 0.0, 1.0
    else:
        rho, pval = stats.spearmanr(shap_arr, lime_arr)

    # Overlap count
    shap_set = set(shap_ranking[:top_k])
    lime_set = set(lime_ranking[:top_k])
    overlap = len(shap_set & lime_set)

    return {
        "spearman_rho": rho,
        "spearman_pvalue": pval,
        "overlap_count": overlap,
        "top_k": top_k,
    }


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 72)
    print("  LIME Analysis — JBI DBS Screening Project")
    print(f"  Seed: {SEED}")
    print("=" * 72)

    # Create output directories
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n[1] Loading data...")
    df, splits, wearable_cols, voice_cols, gait_cols = load_data()
    feature_cols = wearable_cols + voice_cols + gait_cols
    modality_map = build_modality_map(feature_cols, wearable_cols, voice_cols, gait_cols)

    print(f"    Features: {len(feature_cols)} "
          f"(Wearable={len(wearable_cols)}, Voice={len(voice_cols)}, Gait={len(gait_cols)})")

    X_train, y_train, X_test, y_test, train_idx, test_idx = build_train_test_arrays(
        df, splits, feature_cols,
    )
    print(f"    Train: {len(X_train)} | Test: {len(X_test)}")

    subject_ids = df["subject_id"].tolist()

    # ── Train XGBoost for predictions & explanations ───────────────────────
    print("\n[2] Training early-fusion XGBoost...")
    xgb_model = create_xgboost_model(n_jobs=-1, random_state=SEED)
    xgb_model.fit(X_train, y_train)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

    predict_fn = xgb_model.predict_proba

    # ── Select case study patients ─────────────────────────────────────────
    print("\n[3] Selecting case study patients...")
    cases = select_case_studies(xgb_proba, y_test, test_idx, subject_ids)

    # ── Create LIME explainer ──────────────────────────────────────────────
    print("\n[4] Creating LIME explainer...")
    lime_explainer = create_lime_explainer(X_train, feature_cols)

    # ── Explain case study patients ────────────────────────────────────────
    print("\n[5] Generating LIME explanations for case study patients...")
    all_lime_rows = []

    for category, idx in cases.items():
        print(f"\n  Explaining {category} patient (test index {idx})...")
        exp = explain_patient(lime_explainer, predict_fn, X_test[idx])

        # Extract rankings
        rankings_df = extract_lime_rankings(exp, label=1)
        rankings_df["category"] = category
        rankings_df["test_index"] = idx

        global_idx = test_idx[idx]
        sid = subject_ids[global_idx] if global_idx < len(subject_ids) else f"idx_{global_idx}"
        rankings_df["subject_id"] = sid
        rankings_df["dbs_probability"] = xgb_proba[idx]
        rankings_df["true_label"] = y_test[idx]

        # Map LIME descriptions to raw feature names and modalities
        raw_names = []
        modalities = []
        for desc in rankings_df["feature_description"]:
            raw = extract_lime_feature_name(desc, feature_cols)
            raw_names.append(raw)
            modalities.append(modality_map.get(raw, "Wearable"))
        rankings_df["feature_name"] = raw_names
        rankings_df["modality"] = modalities

        all_lime_rows.append(rankings_df)

        # Plot
        fig_path = FIGURES_DIR / f"lime_explanation_{category}.png"
        plot_lime_explanation(
            exp, category, modality_map, feature_cols, fig_path, num_features=15,
        )

    # Save all LIME explanations
    lime_df = pd.concat(all_lime_rows, ignore_index=True)
    lime_csv_path = TABLES_DIR / "lime_explanations.csv"
    lime_df.to_csv(lime_csv_path, index=False)
    print(f"\n  Saved LIME explanations: {lime_csv_path}")

    # ── SHAP-LIME concordance ──────────────────────────────────────────────
    print("\n[6] Computing SHAP-LIME concordance...")

    TOP_K = 10

    # Get SHAP top-k feature rankings
    shap_top = compute_shap_top_features(
        TABLES_DIR, feature_cols, X_train, y_train, X_test, top_k=TOP_K,
    )

    # Get LIME top-k feature rankings (global, averaged across test set)
    lime_top = compute_lime_top_features(
        lime_explainer, predict_fn, X_test, feature_cols, top_k=TOP_K,
    )

    if shap_top is not None and lime_top is not None:
        print(f"\n  SHAP top-{TOP_K}: {shap_top}")
        print(f"  LIME top-{TOP_K}: {lime_top}")

        concordance = compute_concordance(shap_top, lime_top, top_k=TOP_K)
        print(f"\n  Spearman rho = {concordance['spearman_rho']:.4f} "
              f"(p = {concordance['spearman_pvalue']:.4f})")
        print(f"  Overlap: {concordance['overlap_count']}/{TOP_K} features in common")

        # Build concordance table
        concordance_rows = []
        for rank_i in range(TOP_K):
            row = {"rank": rank_i + 1}
            row["shap_feature"] = shap_top[rank_i] if rank_i < len(shap_top) else ""
            row["lime_feature"] = lime_top[rank_i] if rank_i < len(lime_top) else ""
            concordance_rows.append(row)

        concordance_df = pd.DataFrame(concordance_rows)

        # Add summary row
        summary_row = pd.DataFrame([{
            "rank": "Summary",
            "shap_feature": f"Spearman rho={concordance['spearman_rho']:.4f}",
            "lime_feature": f"p-value={concordance['spearman_pvalue']:.4f}",
        }])
        overlap_row = pd.DataFrame([{
            "rank": "Overlap",
            "shap_feature": f"{concordance['overlap_count']}/{TOP_K} features",
            "lime_feature": "",
        }])
        concordance_df = pd.concat(
            [concordance_df, summary_row, overlap_row], ignore_index=True,
        )

        conc_path = TABLES_DIR / "shap_lime_concordance.csv"
        concordance_df.to_csv(conc_path, index=False)
        print(f"  Saved concordance table: {conc_path}")
    else:
        print("  [WARNING] Could not compute concordance (SHAP or LIME rankings unavailable).")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  LIME Analysis Complete")
    print("=" * 72)
    print(f"  Tables:  {TABLES_DIR / 'lime_explanations.csv'}")
    print(f"           {TABLES_DIR / 'shap_lime_concordance.csv'}")
    for cat in cases:
        print(f"  Figures: {FIGURES_DIR / f'lime_explanation_{cat}.png'}")
    print("\n[DONE]")


if __name__ == "__main__":
    main()
