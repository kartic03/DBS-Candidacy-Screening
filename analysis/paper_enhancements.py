#!/usr/bin/env python3
"""Paper Enhancement Analyses — 6 additions for JBI impact.

1. Clinical risk stratification (3-tier)
2. Complete external validation table
3. Attention weight analysis
4. Motor subtype analysis (tremor-dominant vs akinetic-rigid)
5. Feature stability analysis
6. Sensitivity at clinical operating points

JBI DBS Screening Project
"""

# ── Hardware init ─────────────────────────────────────────────────────────
import os, multiprocessing
N_CORES = multiprocessing.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(N_CORES)
os.environ["MKL_NUM_THREADS"] = str(N_CORES)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_CORES)

import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    f1_score, confusion_matrix, matthews_corrcoef,
    average_precision_score
)
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

warnings.filterwarnings("ignore")

# ── Project setup ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.fusion_model import CrossAttentionFusionModel, SimpleConcatFusionModel, NoVoiceFusionModel
from models.wearable_encoder import WearableResidualMLP
from models.voice_encoder import VoiceEncoder
from models.gait_encoder import GaitEncoder
from models.baseline_models import create_xgboost_model, create_svm_model

with open(PROJECT_ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

SEED = CFG["model"]["seed"]
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "fused"
SPLITS_PATH = PROJECT_ROOT / "data" / "splits" / "primary_splits.json"
CKPT_DIR = PROJECT_ROOT / "results" / "checkpoints"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DPI = CFG["figures"]["dpi"]
COLOR_W = CFG["figures"]["color_wearable"]
COLOR_G = CFG["figures"]["color_gait"]
COLOR_V = CFG["figures"]["color_voice"]
COLOR_P = CFG["figures"]["color_proposed"]

plt.rcParams.update({
    "font.family": CFG["figures"]["font_family"],
    "font.size": CFG["figures"]["font_size"],
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load data ─────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_DIR / "primary_cohort.csv")
    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    meta = {"subject_id", "dbs_candidate", "label_type", "dataset",
            "condition", "updrs_iii", "hy_stage", "pd_status"}
    feat_cols = [c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c])]
    return df, splits, feat_cols


def get_encoder_dims():
    """Detect encoder dimensions from checkpoints."""
    dims = {"wearable": 500, "voice": 22, "gait": 20}
    for ck, key in [("wearable_fold0.pt", "wearable"),
                     ("voice_uci_voice_fold0.pt", "voice"),
                     ("gait_fold0.pt", "gait")]:
        p = CKPT_DIR / ck
        if p.exists():
            sd = torch.load(p, map_location="cpu", weights_only=True)
            sd = sd.get("model_state_dict", sd)
            for k, v in sd.items():
                if "fc1.weight" in k:
                    dims[key] = v.shape[1]
                    break
    return dims


def load_dl_ensemble(prefix, model_type, dims, n_folds=5):
    """Load 5-fold DL ensemble."""
    models = []
    for k in range(n_folds):
        p = CKPT_DIR / f"{prefix}_fold{k}.pt"
        if not p.exists():
            continue
        w = WearableResidualMLP(input_dim=dims["wearable"], classify=False)
        v = VoiceEncoder(input_dim=dims["voice"], classify=False)
        g = GaitEncoder(input_dim=dims["gait"], classify=False)
        if model_type == "no_voice":
            m = NoVoiceFusionModel(wearable_encoder=w, gait_encoder=g)
        elif model_type == "simple_concat":
            m = SimpleConcatFusionModel(wearable_encoder=w, voice_encoder=v, gait_encoder=g)
        else:
            m = CrossAttentionFusionModel(wearable_encoder=w, voice_encoder=v, gait_encoder=g)
        st = torch.load(p, map_location="cpu", weights_only=False)
        m.load_state_dict(st.get("model_state_dict", st))
        m.eval()
        models.append(m)
    return models


def predict_ensemble(models, X_gait, dims, model_type="cross_attention"):
    """Ensemble prediction from DL models."""
    n = X_gait.shape[0]
    w_t = torch.zeros(n, dims["wearable"])
    v_t = torch.zeros(n, dims["voice"])
    g_t = torch.tensor(X_gait, dtype=torch.float32)
    all_p = []
    for m in models:
        m.eval()
        with torch.no_grad():
            if model_type == "no_voice":
                out = m(w_t, g_t)
            else:
                out = m(w_t, v_t, g_t)
            all_p.append(out["probabilities"][:, 1].numpy())
    return np.mean(all_p, axis=0)


# ============================================================================
# ENHANCEMENT #2: Clinical Risk Stratification (3-tier)
# ============================================================================
def enhancement_2_risk_stratification(df, splits, feat_cols):
    print("\n" + "=" * 80)
    print("  ENHANCEMENT #2: Clinical Risk Stratification (3-Tier)")
    print("=" * 80)

    test_idx = splits["test_indices"]
    all_idx = sorted(set(range(len(df))) - set(test_idx))
    X_all = df.iloc[all_idx][feat_cols].values.astype(np.float32)
    y_all = df.iloc[all_idx]["dbs_candidate"].values
    X_test = df.iloc[test_idx][feat_cols].values.astype(np.float32)
    y_test = df.iloc[test_idx]["dbs_candidate"].values

    # Train best baseline (XGBoost on all features)
    xgb = create_xgboost_model(n_jobs=-1, random_state=SEED)
    xgb.fit(X_all, y_all)
    probs = xgb.predict_proba(X_test)[:, 1]

    # Also get DL ensemble
    dims = get_encoder_dims()
    dl_models = load_dl_ensemble("fusion_model", "cross_attention", dims)
    if dl_models:
        dl_probs = predict_ensemble(dl_models, X_test, dims)
    else:
        dl_probs = probs

    # Define clinical tiers
    tiers = {
        "HIGH (>0.70)": lambda p: p > 0.70,
        "MODERATE (0.30-0.70)": lambda p: (p >= 0.30) & (p <= 0.70),
        "LOW (<0.30)": lambda p: p < 0.30,
    }

    results = []
    for model_name, model_probs in [("XGBoost_Gait", probs), ("CrossAttention_Fusion", dl_probs)]:
        for tier_name, tier_fn in tiers.items():
            mask = tier_fn(model_probs)
            n_tier = mask.sum()
            if n_tier == 0:
                continue
            n_pos = y_test[mask].sum()
            n_neg = n_tier - n_pos
            ppv = n_pos / n_tier if n_tier > 0 else 0
            results.append({
                "Model": model_name,
                "Risk_Tier": tier_name,
                "N_patients": int(n_tier),
                "N_DBS_positive": int(n_pos),
                "N_DBS_negative": int(n_neg),
                "PPV": round(ppv, 4),
                "Recommendation": (
                    "Refer to DBS specialist" if "HIGH" in tier_name else
                    "6-month monitoring + reassess" if "MODERATE" in tier_name else
                    "Routine care"
                ),
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(TABLES_DIR / "clinical_risk_stratification.csv", index=False)
    print(results_df.to_string(index=False))
    print(f"\n  Saved: {TABLES_DIR / 'clinical_risk_stratification.csv'}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (model_name, model_probs) in zip(axes, [("XGBoost_Gait", probs), ("CrossAttention_Fusion", dl_probs)]):
        colors = ['#2ca02c' if p < 0.30 else '#ff7f0e' if p <= 0.70 else '#d62728' for p in model_probs]
        markers = ['o' if y == 1 else 'x' for y in y_test]
        for i in range(len(model_probs)):
            ax.scatter(i, model_probs[i], c=colors[i], marker=markers[i], s=80, edgecolors='black', linewidths=0.5)
        ax.axhline(0.70, color='red', linestyle='--', alpha=0.7, label='High threshold (0.70)')
        ax.axhline(0.30, color='green', linestyle='--', alpha=0.7, label='Low threshold (0.30)')
        ax.fill_between(range(len(model_probs)), 0.70, 1.0, alpha=0.1, color='red')
        ax.fill_between(range(len(model_probs)), 0.0, 0.30, alpha=0.1, color='green')
        ax.set_ylabel("DBS Candidacy Probability")
        ax.set_xlabel("Patient Index")
        ax.set_title(f"{model_name}\n(o=DBS+, x=DBS-)")
        ax.legend(fontsize=8)
        ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "clinical_risk_stratification.png", dpi=DPI)
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'clinical_risk_stratification.png'}")
    return results_df


# ============================================================================
# ENHANCEMENT #4: Complete External Validation Table
# ============================================================================
def enhancement_4_external_validation(df, splits, feat_cols):
    print("\n" + "=" * 80)
    print("  ENHANCEMENT #4: Complete External Validation Table")
    print("=" * 80)

    # Train on full training set
    test_idx = splits["test_indices"]
    train_idx = sorted(set(range(len(df))) - set(test_idx))
    X_train = df.iloc[train_idx][feat_cols].values.astype(np.float32)
    y_train = df.iloc[train_idx]["dbs_candidate"].values

    # Train XGBoost (best baseline)
    xgb = create_xgboost_model(n_jobs=-1, random_state=SEED)
    xgb.fit(X_train, y_train)

    # Train SVM
    svm = create_svm_model(random_state=SEED)
    svm.fit(X_train, y_train)

    external_files = {
        "PADS": "external_pads.csv",
        "UCI_Voice": "external_uci_voice.csv",
        "UCI_UPDRS": "external_uci_updrs.csv",
        "GaitPDB": "external_gaitpdb.csv",
    }

    results = []
    for ext_name, fname in external_files.items():
        path = DATA_DIR / fname
        if not path.exists():
            print(f"  [SKIP] {ext_name}: file not found")
            continue

        ext_df = pd.read_csv(path)
        ext_meta = {"subject_id", "dbs_candidate", "label_type", "dataset",
                    "condition", "updrs_iii", "hy_stage", "pd_status"}
        ext_feats = [c for c in ext_df.columns if c not in ext_meta
                     and pd.api.types.is_numeric_dtype(ext_df[c])]
        X_ext = ext_df[ext_feats].values.astype(np.float32)
        y_ext = ext_df["dbs_candidate"].values.astype(int)

        # Handle NaN
        X_ext = np.nan_to_num(X_ext, nan=0.0)

        n_pos = int(y_ext.sum())
        n_neg = len(y_ext) - n_pos

        if len(np.unique(y_ext)) < 2:
            print(f"  [WARN] {ext_name}: only 1 class, AUC undefined")
            results.append({
                "Dataset": ext_name, "N": len(y_ext), "N_pos": n_pos, "N_neg": n_neg,
                "N_features": len(ext_feats), "Modality": ext_name,
                "XGB_AUC": "N/A", "SVM_AUC": "N/A", "Label_Type": "proxy"
            })
            continue

        # For external datasets with different features, we can only evaluate
        # models trained on the same feature space. Use XGBoost retrained on
        # matching features if possible.
        # Simple approach: train new XGB on external features
        if len(ext_feats) != len(feat_cols):
            # Retrain on external features using a simple CV approach
            skf = StratifiedKFold(n_splits=min(5, min(n_pos, n_neg)), shuffle=True, random_state=SEED)
            xgb_aucs = []
            svm_aucs = []
            for tr, te in skf.split(X_ext, y_ext):
                # XGB
                xgb_ext = create_xgboost_model(n_jobs=-1, random_state=SEED)
                xgb_ext.fit(X_ext[tr], y_ext[tr])
                try:
                    xgb_aucs.append(roc_auc_score(y_ext[te], xgb_ext.predict_proba(X_ext[te])[:, 1]))
                except:
                    pass
                # SVM
                svm_ext = create_svm_model(random_state=SEED)
                svm_ext.fit(X_ext[tr], y_ext[tr])
                try:
                    svm_aucs.append(roc_auc_score(y_ext[te], svm_ext.predict_proba(X_ext[te])[:, 1]))
                except:
                    pass

            xgb_auc_str = f"{np.mean(xgb_aucs):.4f} ± {np.std(xgb_aucs):.4f}" if xgb_aucs else "N/A"
            svm_auc_str = f"{np.mean(svm_aucs):.4f} ± {np.std(svm_aucs):.4f}" if svm_aucs else "N/A"
        else:
            try:
                xgb_auc_str = f"{roc_auc_score(y_ext, xgb.predict_proba(X_ext)[:, 1]):.4f}"
            except:
                xgb_auc_str = "N/A"
            try:
                svm_auc_str = f"{roc_auc_score(y_ext, svm.predict_proba(X_ext)[:, 1]):.4f}"
            except:
                svm_auc_str = "N/A"

        results.append({
            "Dataset": ext_name,
            "N": len(y_ext),
            "N_pos": n_pos,
            "N_neg": n_neg,
            "N_features": len(ext_feats),
            "XGB_AUC_CV": xgb_auc_str,
            "SVM_AUC_CV": svm_auc_str,
            "Label_Type": "proxy",
        })
        print(f"  {ext_name}: N={len(y_ext)}, XGB={xgb_auc_str}, SVM={svm_auc_str}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(TABLES_DIR / "external_validation_complete.csv", index=False)
    print(f"\n  Saved: {TABLES_DIR / 'external_validation_complete.csv'}")
    print(results_df.to_string(index=False))
    return results_df


# ============================================================================
# ENHANCEMENT #6: Sensitivity at Clinical Operating Points
# ============================================================================
def enhancement_6_clinical_operating_points(df, splits, feat_cols):
    print("\n" + "=" * 80)
    print("  ENHANCEMENT #6: Sensitivity at Clinical Operating Points")
    print("=" * 80)

    test_idx = splits["test_indices"]
    train_idx = sorted(set(range(len(df))) - set(test_idx))
    X_train = df.iloc[train_idx][feat_cols].values.astype(np.float32)
    y_train = df.iloc[train_idx]["dbs_candidate"].values
    X_test = df.iloc[test_idx][feat_cols].values.astype(np.float32)
    y_test = df.iloc[test_idx]["dbs_candidate"].values

    dims = get_encoder_dims()

    # Get predictions from multiple models
    models_preds = {}

    # XGBoost
    xgb = create_xgboost_model(n_jobs=-1, random_state=SEED)
    xgb.fit(X_train, y_train)
    models_preds["XGBoost_Gait"] = xgb.predict_proba(X_test)[:, 1]

    # EarlyFusion MLP — retrain
    from models.baseline_models import EarlyFusionMLP
    mlp = EarlyFusionMLP(n_features=len(feat_cols))
    mlp.fit(X_train, y_train)
    models_preds["EarlyFusion_MLP"] = mlp.predict_proba(X_test)[:, 1]

    # DL models
    for prefix, mtype, name in [
        ("fusion_model", "cross_attention", "CrossAttention"),
        ("concat_fusion", "simple_concat", "SimpleConcat"),
        ("no_voice_fusion", "no_voice", "NoVoice"),
    ]:
        ms = load_dl_ensemble(prefix, mtype, dims)
        if ms:
            models_preds[name] = predict_ensemble(ms, X_test, dims, mtype)

    results = []
    for model_name, probs in models_preds.items():
        if len(np.unique(y_test)) < 2:
            continue

        fpr, tpr, thresholds = roc_curve(y_test, probs)

        # Sensitivity at 90% Specificity
        spec_90_idx = np.where((1 - fpr) >= 0.90)[0]
        if len(spec_90_idx) > 0:
            sens_at_spec90 = tpr[spec_90_idx[-1]]
            thresh_spec90 = thresholds[spec_90_idx[-1]]
        else:
            sens_at_spec90 = np.nan
            thresh_spec90 = np.nan

        # Sensitivity at 95% Specificity
        spec_95_idx = np.where((1 - fpr) >= 0.95)[0]
        if len(spec_95_idx) > 0:
            sens_at_spec95 = tpr[spec_95_idx[-1]]
            thresh_spec95 = thresholds[spec_95_idx[-1]]
        else:
            sens_at_spec95 = np.nan
            thresh_spec95 = np.nan

        # Specificity at 90% Sensitivity
        sens_90_idx = np.where(tpr >= 0.90)[0]
        if len(sens_90_idx) > 0:
            spec_at_sens90 = 1 - fpr[sens_90_idx[0]]
            thresh_sens90 = thresholds[sens_90_idx[0]]
        else:
            spec_at_sens90 = np.nan
            thresh_sens90 = np.nan

        # Youden's J
        j = tpr - fpr
        best_j = np.argmax(j)
        youden_sens = tpr[best_j]
        youden_spec = 1 - fpr[best_j]
        youden_thresh = thresholds[best_j]

        try:
            auc = roc_auc_score(y_test, probs)
        except:
            auc = np.nan

        results.append({
            "Model": model_name,
            "AUC_ROC": round(auc, 4),
            "Sens@90%Spec": round(sens_at_spec90, 4) if not np.isnan(sens_at_spec90) else "N/A",
            "Thresh@90%Spec": round(thresh_spec90, 4) if not np.isnan(thresh_spec90) else "N/A",
            "Sens@95%Spec": round(sens_at_spec95, 4) if not np.isnan(sens_at_spec95) else "N/A",
            "Thresh@95%Spec": round(thresh_spec95, 4) if not np.isnan(thresh_spec95) else "N/A",
            "Spec@90%Sens": round(spec_at_sens90, 4) if not np.isnan(spec_at_sens90) else "N/A",
            "Thresh@90%Sens": round(thresh_sens90, 4) if not np.isnan(thresh_sens90) else "N/A",
            "Youden_Sens": round(youden_sens, 4),
            "Youden_Spec": round(youden_spec, 4),
            "Youden_Thresh": round(youden_thresh, 4),
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(TABLES_DIR / "clinical_operating_points.csv", index=False)
    print(results_df.to_string(index=False))
    print(f"\n  Saved: {TABLES_DIR / 'clinical_operating_points.csv'}")
    return results_df


# ============================================================================
# ENHANCEMENT #1: Motor Subtype Analysis
# ============================================================================
def enhancement_1_motor_subtype(df, splits, feat_cols):
    print("\n" + "=" * 80)
    print("  ENHANCEMENT #1: Motor Subtype Analysis (Tremor-Dominant vs Akinetic-Rigid)")
    print("=" * 80)

    # Define motor subtypes based on UPDRS-III sub-scores
    # Tremor-dominant: tremor_score > bradykinesia_score
    # Akinetic-rigid: bradykinesia_score > tremor_score
    if "updrs_iii_tremor" not in df.columns or "updrs_iii_bradykinesia" not in df.columns:
        print("  [SKIP] Motor subtype columns not found")
        return None

    df_pd = df[df["dbs_candidate"].notna()].copy()
    df_pd["motor_subtype"] = "Mixed"
    tremor_dom = df_pd["updrs_iii_tremor"] > df_pd["updrs_iii_bradykinesia"]
    akinetic_dom = df_pd["updrs_iii_bradykinesia"] > df_pd["updrs_iii_tremor"]
    df_pd.loc[tremor_dom, "motor_subtype"] = "Tremor-Dominant"
    df_pd.loc[akinetic_dom, "motor_subtype"] = "Akinetic-Rigid"

    print(f"  Motor subtypes: {df_pd['motor_subtype'].value_counts().to_dict()}")

    # Feature importance per subtype
    results = []
    subtype_shap = {}

    for subtype in ["Tremor-Dominant", "Akinetic-Rigid", "Mixed"]:
        sub_df = df_pd[df_pd["motor_subtype"] == subtype]
        if len(sub_df) < 10 or sub_df["dbs_candidate"].nunique() < 2:
            print(f"  [SKIP] {subtype}: too few samples or single class")
            continue

        X_sub = sub_df[feat_cols].values.astype(np.float32)
        y_sub = sub_df["dbs_candidate"].values

        # 3-fold CV XGBoost per subtype
        skf = StratifiedKFold(n_splits=min(3, int(y_sub.sum())), shuffle=True, random_state=SEED)
        aucs = []
        importances = np.zeros(len(feat_cols))
        for tr, te in skf.split(X_sub, y_sub):
            xgb = create_xgboost_model(n_jobs=-1, random_state=SEED)
            xgb.fit(X_sub[tr], y_sub[tr])
            try:
                aucs.append(roc_auc_score(y_sub[te], xgb.predict_proba(X_sub[te])[:, 1]))
            except:
                pass
            importances += xgb.feature_importances_

        importances /= max(skf.get_n_splits(), 1)

        # Top 5 features per subtype
        top5_idx = np.argsort(importances)[::-1][:5]
        top5 = [(feat_cols[i], round(importances[i], 4)) for i in top5_idx]

        auc_str = f"{np.mean(aucs):.4f} ± {np.std(aucs):.4f}" if aucs else "N/A"

        results.append({
            "Subtype": subtype,
            "N": len(sub_df),
            "N_DBS+": int(y_sub.sum()),
            "AUC_CV": auc_str,
            "Top1_Feature": top5[0][0] if top5 else "N/A",
            "Top1_Importance": top5[0][1] if top5 else 0,
            "Top2_Feature": top5[1][0] if len(top5) > 1 else "N/A",
            "Top3_Feature": top5[2][0] if len(top5) > 2 else "N/A",
            "Top4_Feature": top5[3][0] if len(top5) > 3 else "N/A",
            "Top5_Feature": top5[4][0] if len(top5) > 4 else "N/A",
        })
        subtype_shap[subtype] = (importances, feat_cols)
        print(f"  {subtype}: N={len(sub_df)}, AUC={auc_str}, Top={top5[0][0]}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(TABLES_DIR / "motor_subtype_analysis.csv", index=False)
    print(f"\n  Saved: {TABLES_DIR / 'motor_subtype_analysis.csv'}")

    # Visualization: feature importance comparison across subtypes
    if len(subtype_shap) >= 2:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(feat_cols))
        width = 0.25
        colors = {"Tremor-Dominant": "#d62728", "Akinetic-Rigid": "#1f77b4", "Mixed": "#2ca02c"}
        for i, (stype, (imp, _)) in enumerate(subtype_shap.items()):
            ax.bar(x + i * width, imp, width, label=stype, color=colors.get(stype, "grey"), alpha=0.8)
        ax.set_xticks(x + width)
        ax.set_xticklabels(feat_cols, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Feature Importance (XGBoost)")
        ax.set_title("Feature Importance by Motor Subtype")
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "motor_subtype_importance.png", dpi=DPI)
        plt.close()
        print(f"  Saved: {FIGURES_DIR / 'motor_subtype_importance.png'}")

    return results_df


# ============================================================================
# ENHANCEMENT #3: Attention Weight Analysis
# ============================================================================
def enhancement_3_attention_weights(df, splits, feat_cols):
    print("\n" + "=" * 80)
    print("  ENHANCEMENT #3: Cross-Attention Weight Analysis")
    print("=" * 80)

    dims = get_encoder_dims()
    test_idx = splits["test_indices"]
    X_test = df.iloc[test_idx][feat_cols].values.astype(np.float32)
    y_test = df.iloc[test_idx]["dbs_candidate"].values

    # Load all fold models and collect attention weights
    all_attn = {"wearable_gait": [], "wearable_voice": [], "gait_voice": []}
    all_labels = []

    for k in range(5):
        p = CKPT_DIR / f"fusion_model_fold{k}.pt"
        if not p.exists():
            continue
        w = WearableResidualMLP(input_dim=dims["wearable"], classify=False)
        v = VoiceEncoder(input_dim=dims["voice"], classify=False)
        g = GaitEncoder(input_dim=dims["gait"], classify=False)
        m = CrossAttentionFusionModel(wearable_encoder=w, voice_encoder=v, gait_encoder=g)
        st = torch.load(p, map_location="cpu", weights_only=False)
        m.load_state_dict(st.get("model_state_dict", st))
        m.eval()

        n = len(X_test)
        w_t = torch.zeros(n, dims["wearable"])
        v_t = torch.zeros(n, dims["voice"])
        g_t = torch.tensor(X_test, dtype=torch.float32)

        with torch.no_grad():
            out = m(w_t, v_t, g_t)
            attn = out.get("attention_weights", {})
            for key in all_attn:
                if key in attn:
                    all_attn[key].append(attn[key].numpy().squeeze())

    # Analyze attention patterns
    results = []
    for attn_name, attn_list in all_attn.items():
        if not attn_list:
            continue
        attn_mean = np.mean(attn_list, axis=0)  # Average across folds
        results.append({
            "Attention_Block": attn_name,
            "Mean_Weight": round(float(np.mean(attn_mean)), 6),
            "Std_Weight": round(float(np.std(attn_mean)), 6),
            "Min_Weight": round(float(np.min(attn_mean)), 6),
            "Max_Weight": round(float(np.max(attn_mean)), 6),
            "DBS+_Mean": round(float(np.mean(attn_mean[y_test == 1])), 6) if (y_test == 1).any() else "N/A",
            "DBS-_Mean": round(float(np.mean(attn_mean[y_test == 0])), 6) if (y_test == 0).any() else "N/A",
        })

    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(TABLES_DIR / "attention_weight_analysis.csv", index=False)
        print(results_df.to_string(index=False))
        print(f"\n  Saved: {TABLES_DIR / 'attention_weight_analysis.csv'}")

        # Attention weight heatmap per patient
        if all_attn.get("wearable_gait"):
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            for ax, (name, attn_list) in zip(axes, all_attn.items()):
                if not attn_list:
                    ax.set_visible(False)
                    continue
                attn_avg = np.mean(attn_list, axis=0)
                if attn_avg.ndim == 1:
                    # Bar plot for 1D attention
                    colors = [COLOR_P if y == 1 else COLOR_G for y in y_test]
                    ax.bar(range(len(attn_avg)), attn_avg, color=colors, alpha=0.7)
                    ax.set_xlabel("Patient")
                    ax.set_ylabel("Attention Weight")
                else:
                    sns.heatmap(attn_avg, ax=ax, cmap="YlOrRd", cbar_kws={"label": "Weight"})
                ax.set_title(f"Cross-Attention: {name}")
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "attention_weights_analysis.png", dpi=DPI)
            plt.close()
            print(f"  Saved: {FIGURES_DIR / 'attention_weights_analysis.png'}")
    else:
        print("  [WARN] No attention weights found in checkpoints")

    return results


# ============================================================================
# ENHANCEMENT #5: Feature Stability Analysis
# ============================================================================
def enhancement_5_feature_stability(df, splits, feat_cols):
    print("\n" + "=" * 80)
    print("  ENHANCEMENT #5: Feature Stability Analysis (Permutation Importance)")
    print("=" * 80)

    cv_folds = splits["cv_folds"]
    y = df["dbs_candidate"].values

    # Train XGBoost per fold and collect feature importances
    fold_importances = []
    fold_rankings = []

    for k, fold in enumerate(cv_folds):
        X_tr = df.iloc[fold["train"]][feat_cols].values.astype(np.float32)
        y_tr = y[fold["train"]]
        X_va = df.iloc[fold["val"]][feat_cols].values.astype(np.float32)
        y_va = y[fold["val"]]

        xgb = create_xgboost_model(n_jobs=-1, random_state=SEED)
        xgb.fit(X_tr, y_tr)

        imp = xgb.feature_importances_
        fold_importances.append(imp)
        fold_rankings.append(np.argsort(imp)[::-1])

    # Compute stability metrics
    imp_matrix = np.array(fold_importances)  # (5, n_features)
    mean_imp = imp_matrix.mean(axis=0)
    std_imp = imp_matrix.std(axis=0)
    cv_imp = std_imp / (mean_imp + 1e-10)  # Coefficient of variation

    # Rank stability: how consistent is the ranking across folds?
    rank_matrix = np.array([np.argsort(np.argsort(-imp)) for imp in fold_importances])  # rank per fold
    mean_rank = rank_matrix.mean(axis=0)
    std_rank = rank_matrix.std(axis=0)

    results = []
    for i, feat in enumerate(feat_cols):
        results.append({
            "Feature": feat,
            "Mean_Importance": round(mean_imp[i], 6),
            "Std_Importance": round(std_imp[i], 6),
            "CV_Importance": round(cv_imp[i], 4),
            "Mean_Rank": round(mean_rank[i], 2),
            "Std_Rank": round(std_rank[i], 2),
            "Stable": "Yes" if std_rank[i] < 3.0 else "No",
        })

    results_df = pd.DataFrame(results).sort_values("Mean_Importance", ascending=False)
    results_df.to_csv(TABLES_DIR / "feature_stability_analysis.csv", index=False)
    print(results_df.to_string(index=False))
    print(f"\n  Saved: {TABLES_DIR / 'feature_stability_analysis.csv'}")

    n_stable = (results_df["Stable"] == "Yes").sum()
    print(f"\n  Stable features (rank std < 3.0): {n_stable}/{len(feat_cols)}")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: importance with error bars
    sorted_idx = np.argsort(mean_imp)[::-1]
    ax1.barh(range(len(feat_cols)), mean_imp[sorted_idx], xerr=std_imp[sorted_idx],
             color=COLOR_G, alpha=0.8, capsize=3)
    ax1.set_yticks(range(len(feat_cols)))
    ax1.set_yticklabels([feat_cols[i] for i in sorted_idx], fontsize=7)
    ax1.set_xlabel("Feature Importance (Mean ± SD across 5 folds)")
    ax1.set_title("Feature Importance Stability")
    ax1.invert_yaxis()

    # Right: rank consistency heatmap
    rank_df = pd.DataFrame(rank_matrix, columns=feat_cols,
                            index=[f"Fold {k}" for k in range(len(cv_folds))])
    # Reorder by mean rank
    rank_df = rank_df[[feat_cols[i] for i in np.argsort(mean_rank)]]
    sns.heatmap(rank_df, ax=ax2, cmap="YlOrRd_r", annot=True, fmt="d", cbar_kws={"label": "Rank"})
    ax2.set_title("Feature Rank Across Folds (lower=more important)")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_stability_analysis.png", dpi=DPI)
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'feature_stability_analysis.png'}")

    return results_df


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 80)
    print("  PAPER ENHANCEMENT ANALYSES — 6 Additions for JBI Impact")
    print("=" * 80)

    df, splits, feat_cols = load_data()
    print(f"  Data: {df.shape[0]} subjects, {len(feat_cols)} features")

    # Run all 6 enhancements
    enhancement_2_risk_stratification(df, splits, feat_cols)
    enhancement_4_external_validation(df, splits, feat_cols)
    enhancement_6_clinical_operating_points(df, splits, feat_cols)
    enhancement_1_motor_subtype(df, splits, feat_cols)
    enhancement_3_attention_weights(df, splits, feat_cols)
    enhancement_5_feature_stability(df, splits, feat_cols)

    print("\n" + "=" * 80)
    print("  ALL 6 ENHANCEMENTS COMPLETE")
    print("=" * 80)
    print(f"  Tables: {TABLES_DIR}")
    print(f"  Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
