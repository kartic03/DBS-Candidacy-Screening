#!/usr/bin/env python3
"""
Publication Figures for JBI Paper (Revised)
============================================
8 figures at 300 DPI, Arial font, colorblind-safe (Okabe-Ito palette).
Following scientific-visualization skill guidelines:
  - zorder=10 for text, white bbox
  - bbox_inches='tight', pad_inches=0.1
  - No bar charts for ≤3 data points
  - 10pt min labels, 11pt titles
  - (8,5) min single panel, (14,6) multi-panel

Figures:
  Fig 2: Data distributions (PADS tremor, GaitPDB stride, UCI voice)
  Fig 4: ROC curves per dataset
  Fig 5: Multi-source validation AUC summary
  Fig 7: GaitPDB fusion ablation
  Fig 9: Calibration + DCA

Author: Kartic Mishra, Gachon University
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── Publication style ────────────────────────────────────────────────────────
try:
    plt.rcParams['font.family'] = 'Arial'
except Exception:
    plt.rcParams['font.family'] = 'Liberation Sans'

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.dpi': 300,
})

# Okabe-Ito colorblind-safe palette
COLORS = {
    'clinical': '#D55E00',   # vermillion
    'wearable': '#0072B2',   # blue
    'gait': '#009E73',       # green
    'voice': '#E69F00',      # orange
    'fusion': '#CC79A7',     # pink
    'proposed': '#D55E00',   # vermillion (same as clinical — primary)
    'baseline': '#56B4E9',   # sky blue
}

SEED = 42
np.random.seed(SEED)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "tables")
FIG_DIR = os.path.join(PROJECT_ROOT, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

META_COLS = [
    "subject_id", "cohort", "dbs_candidate", "dbs_bilateral",
    "dbs_electrode", "dbs_years_since_surgery", "motor_subtype", "med_state_on"
]


def save_fig(fig, name):
    path = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")


def get_loocv_roc(X, y):
    """Get LOOCV probabilities for ROC curve plotting."""
    loo = LeaveOneOut()
    y_prob = np.zeros(len(y))
    for train_idx, test_idx in loo.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(X_tr)
        X_te = imp.transform(X_te)
        n_min = int(y_tr.sum())
        if n_min >= 3:
            try:
                sm = SMOTE(random_state=SEED, k_neighbors=min(5, n_min - 1))
                X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
            except Exception:
                pass
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            tree_method="hist", device="cuda", random_state=SEED,
            eval_metric="logloss", verbosity=0
        )
        model.fit(X_tr, y_tr)
        y_prob[test_idx] = model.predict_proba(X_te)[:, 1]
    return y_prob


def get_cv_roc(X, y, n_folds=5):
    """Get 5-fold CV probabilities for ROC curve plotting."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    y_prob = np.zeros(len(y))
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(X_tr)
        X_te = imp.transform(X_te)
        n_min = int(y_tr.sum())
        if n_min >= 3:
            try:
                sm = SMOTE(random_state=SEED, k_neighbors=min(5, n_min - 1))
                X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
            except Exception:
                pass
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            tree_method="hist", device="cuda", random_state=SEED,
            eval_metric="logloss", verbosity=0
        )
        model.fit(X_tr, y_tr)
        y_prob[test_idx] = model.predict_proba(X_te)[:, 1]
    return y_prob


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 2: Data Distribution Panel (3-panel: wearable tremor, gait stride, voice)
# ═══════════════════════════════════════════════════════════════════════════════
def fig2_distributions():
    print("\n  Fig 2: Data distributions...")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel A: PADS wearable — tremor power by condition
    df_pads = pd.read_csv(os.path.join(PROC_DIR, "wearable_features", "pads_sensor_pd_vs_healthy.csv"))
    ax = axes[0]
    for label, color, name in [(1, COLORS['clinical'], 'PD'), (0, COLORS['baseline'], 'Healthy')]:
        vals = df_pads[df_pads['pd_label'] == label]['all_tremor_rest_acc'].dropna()
        ax.hist(vals, bins=30, alpha=0.6, color=color, label=name, edgecolor='white')
    ax.set_xlabel('Resting Tremor Power (4-6 Hz)')
    ax.set_ylabel('Count')
    ax.set_title('A. Wearable Tremor Power (PADS)', fontweight='bold')
    ax.legend(frameon=False)

    # Panel B: GaitPDB — stride time CV by group
    df_gait = pd.read_csv(os.path.join(PROC_DIR, "gait_features", "gaitpdb_sensor_features.csv"))
    ax = axes[1]
    for label, color, name in [(1, COLORS['clinical'], 'PD'), (0, COLORS['baseline'], 'Control')]:
        vals = df_gait[df_gait['pd_label'] == label]['stride_time_cv'].dropna()
        ax.hist(vals, bins=25, alpha=0.6, color=color, label=name, edgecolor='white')
    ax.set_xlabel('Stride Time CV')
    ax.set_ylabel('Count')
    ax.set_title('B. Gait Stride Variability (GaitPDB)', fontweight='bold')
    ax.legend(frameon=False)

    # Panel C: UCI Voice — jitter by status
    df_voice = pd.read_csv(os.path.join(PROC_DIR, "voice_features", "uci_voice_features.csv"))
    label_col = 'pd_status' if 'pd_status' in df_voice.columns else 'status'
    # Find a jitter column
    jitter_col = [c for c in df_voice.columns if 'jitter' in c.lower() or 'Jitter' in c]
    if jitter_col:
        jitter_col = jitter_col[0]
    else:
        jitter_col = [c for c in df_voice.columns if pd.api.types.is_numeric_dtype(df_voice[c])][0]
    ax = axes[2]
    for label, color, name in [(1, COLORS['clinical'], 'PD'), (0, COLORS['baseline'], 'Healthy')]:
        vals = df_voice[df_voice[label_col] == label][jitter_col].dropna()
        ax.hist(vals, bins=25, alpha=0.6, color=color, label=name, edgecolor='white')
    ax.set_xlabel(f'{jitter_col}')
    ax.set_ylabel('Count')
    ax.set_title('C. Voice Dysphonia (UCI Voice)', fontweight='bold')
    ax.legend(frameon=False)

    plt.tight_layout()
    save_fig(fig, 'fig2_distributions')


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 4: ROC Curves (4 panels — one per dataset)
# ═══════════════════════════════════════════════════════════════════════════════
def fig4_roc_curves():
    print("\n  Fig 4: ROC curves...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    datasets = []

    # A. WearGait-PD Clinical (LOOCV)
    df_pd = pd.read_csv(os.path.join(PROC_DIR, "clinical_features", "weargait_pd_only.csv"))
    feat_cols = [c for c in df_pd.columns if c not in META_COLS and pd.api.types.is_numeric_dtype(df_pd[c])]
    X = df_pd[feat_cols].values.astype(np.float32)
    y = df_pd["dbs_candidate"].values.astype(int)
    # Top-10 features
    scaler = StandardScaler()
    X_s = SimpleImputer(strategy="median").fit_transform(scaler.fit_transform(X))
    temp = xgb.XGBClassifier(n_estimators=200, max_depth=4, tree_method="hist",
                              device="cuda", random_state=SEED, verbosity=0)
    temp.fit(X_s, y)
    top10 = np.argsort(temp.feature_importances_)[::-1][:10]
    y_prob = get_loocv_roc(X[:, top10], y)
    datasets.append(("A. WearGait-PD Clinical\n(DBS Candidacy, LOOCV n=82)",
                     y, y_prob, COLORS['clinical']))

    # B. PADS Wearable (5-fold CV)
    df_pads = pd.read_csv(os.path.join(PROC_DIR, "wearable_features", "pads_sensor_pd_vs_healthy.csv"))
    pads_meta = ["subject_id", "condition", "age", "gender", "height", "weight",
                 "pd_label", "n_files_processed", "handedness"]
    pads_feats = [c for c in df_pads.columns if c not in pads_meta and pd.api.types.is_numeric_dtype(df_pads[c])]
    y_pads_prob = get_cv_roc(df_pads[pads_feats].values.astype(np.float32),
                              df_pads["pd_label"].values.astype(int))
    datasets.append(("B. PADS Wearable Sensor\n(PD vs Healthy, 5-fold CV n=355)",
                     df_pads["pd_label"].values, y_pads_prob, COLORS['wearable']))

    # C. GaitPDB Gait (5-fold CV)
    df_gait = pd.read_csv(os.path.join(PROC_DIR, "gait_features", "gaitpdb_sensor_features.csv"))
    gait_meta = ["subject_id", "Study", "Group", "Gender", "pd_label", "sex",
                 "dbs_proxy_hy25", "dbs_proxy_updrs32", "n_single_trials"]
    gait_feats = [c for c in df_gait.columns if c not in gait_meta and pd.api.types.is_numeric_dtype(df_gait[c])]
    y_gait_prob = get_cv_roc(df_gait[gait_feats].values.astype(np.float32),
                              df_gait["pd_label"].values.astype(int))
    datasets.append(("C. GaitPDB Force Plate\n(PD vs Control, 5-fold CV n=165)",
                     df_gait["pd_label"].values, y_gait_prob, COLORS['gait']))

    # D. UCI Voice (5-fold CV)
    df_voice = pd.read_csv(os.path.join(PROC_DIR, "voice_features", "uci_voice_features.csv"))
    label_col = 'pd_status' if 'pd_status' in df_voice.columns else 'status'
    voice_meta = ["name", "subject_id", label_col]
    voice_feats = [c for c in df_voice.columns if c not in voice_meta and pd.api.types.is_numeric_dtype(df_voice[c])]
    y_voice_prob = get_cv_roc(df_voice[voice_feats].values.astype(np.float32),
                               df_voice[label_col].values.astype(int))
    datasets.append(("D. UCI Voice Features\n(PD vs Healthy, 5-fold CV n=195)",
                     df_voice[label_col].values, y_voice_prob, COLORS['voice']))

    for idx, (title, y_true, y_prob, color) in enumerate(datasets):
        ax = axes[idx // 2, idx % 2]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)

        ax.plot(fpr, tpr, color=color, linewidth=2.5,
                label=f'AUC = {auc:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
        ax.fill_between(fpr, tpr, alpha=0.15, color=color)

        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.legend(loc='lower right', frameon=True, fancybox=True,
                  shadow=False, framealpha=0.9)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.05])
        ax.set_aspect('equal')

    plt.tight_layout(h_pad=3.0)
    save_fig(fig, 'fig4_roc_curves')


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 5: Multi-source Validation Summary (lollipop plot)
# ═══════════════════════════════════════════════════════════════════════════════
def fig5_multisource_validation():
    print("\n  Fig 5: Multi-source validation...")

    # Load modality results
    df_mod = pd.read_csv(os.path.join(RESULTS_DIR, "modality_model_results.csv"))
    df_clin = pd.read_csv(os.path.join(RESULTS_DIR, "clinical_model_results.csv"))

    # Best per dataset
    entries = []

    # Clinical (top-10 XGB from clinical_model_results)
    clin_top10 = df_clin[df_clin["Model"] == "XGBoost_Top10"]
    if len(clin_top10) > 0:
        r = clin_top10.iloc[0]
        entries.append({"Dataset": "WearGait-PD\n(Clinical, n=82)",
                       "AUC": r["AUC_ROC"], "CI_low": r["AUC_CI_low"],
                       "CI_high": r["AUC_CI_high"], "color": COLORS['clinical']})

    # PADS best
    pads = df_mod[(df_mod["Dataset"] == "PADS_Wearable")]
    if len(pads) > 0:
        best = pads.loc[pads["AUC_ROC"].idxmax()]
        entries.append({"Dataset": "PADS\n(Wearable, n=355)",
                       "AUC": best["AUC_ROC"], "CI_low": best["AUC_CI_low"],
                       "CI_high": best["AUC_CI_high"], "color": COLORS['wearable']})

    # GaitPDB PD vs Control
    gait_pd = df_mod[(df_mod["Dataset"] == "GaitPDB_Gait") & (df_mod["Label"] == "PD_vs_Control")]
    if len(gait_pd) > 0:
        best = gait_pd.loc[gait_pd["AUC_ROC"].idxmax()]
        entries.append({"Dataset": "GaitPDB\n(Gait, n=165)",
                       "AUC": best["AUC_ROC"], "CI_low": best["AUC_CI_low"],
                       "CI_high": best["AUC_CI_high"], "color": COLORS['gait']})

    # UCI Voice
    voice = df_mod[(df_mod["Dataset"] == "UCI_Voice")]
    if len(voice) > 0:
        best = voice.loc[voice["AUC_ROC"].idxmax()]
        entries.append({"Dataset": "UCI Voice\n(Voice, n=195)",
                       "AUC": best["AUC_ROC"], "CI_low": best["AUC_CI_low"],
                       "CI_high": best["AUC_CI_high"], "color": COLORS['voice']})

    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(entries))
    for i, e in enumerate(entries):
        # Lollipop: line + dot
        ax.hlines(y=i, xmin=0.5, xmax=e["AUC"], color=e["color"],
                  linewidth=3, alpha=0.8)
        ax.plot(e["AUC"], i, 'o', color=e["color"], markersize=14, zorder=5)

        # Error bar
        if pd.notna(e.get("CI_low")):
            ax.hlines(y=i, xmin=e["CI_low"], xmax=e["CI_high"],
                      color=e["color"], linewidth=1.5, alpha=0.5)
            ax.plot([e["CI_low"], e["CI_high"]], [i, i], '|',
                    color=e["color"], markersize=10)

        # AUC label
        ax.text(e["AUC"] + 0.012, i, f'{e["AUC"]:.3f}',
                va='center', ha='left', fontsize=11, fontweight='bold',
                zorder=10, bbox=dict(facecolor='white', edgecolor='none', alpha=0.9))

    ax.set_yticks(y_pos)
    ax.set_yticklabels([e["Dataset"] for e in entries], fontsize=11)
    ax.set_xlabel('AUC-ROC', fontsize=12)
    ax.set_title('Multi-Source Biomarker Validation', fontsize=13, fontweight='bold')
    ax.set_xlim(0.5, 1.05)
    ax.axvline(x=0.9, color='gray', linestyle=':', alpha=0.5, label='AUC = 0.90')
    ax.legend(frameon=False, fontsize=9)
    ax.invert_yaxis()

    plt.tight_layout()
    save_fig(fig, 'fig5_multisource_validation')


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 7: GaitPDB Fusion Ablation (dumbbell plot — fusion vs single-modality)
# ═══════════════════════════════════════════════════════════════════════════════
def fig7_fusion_ablation():
    print("\n  Fig 7: Fusion ablation...")

    # Load fusion results
    df_fusion = pd.read_csv(os.path.join(RESULTS_DIR, "fusion_model_results.csv"))
    df_fixed = pd.read_csv(os.path.join(RESULTS_DIR, "fusion_fixed_results.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Original fusion (with H&Y — for reference)
    ax = axes[0]
    models = df_fusion.sort_values("AUC_ROC")
    colors_map = {
        "GaitOnly": COLORS['gait'], "ClinicalOnly": COLORS['clinical'],
        "SimpleConcat": COLORS['baseline'], "CrossAttention": COLORS['fusion']
    }
    for i, (_, row) in enumerate(models.iterrows()):
        color = colors_map.get(row["Model"], '#999999')
        ax.hlines(y=i, xmin=0.5, xmax=row["AUC_ROC"], color=color, linewidth=3)
        ax.plot(row["AUC_ROC"], i, 'o', color=color, markersize=12, zorder=5)
        if pd.notna(row.get("AUC_CI_low")):
            ax.hlines(y=i, xmin=row["AUC_CI_low"], xmax=row["AUC_CI_high"],
                      color=color, linewidth=1.5, alpha=0.4)
        ax.text(row["AUC_ROC"] + 0.01, i, f'{row["AUC_ROC"]:.3f}',
                va='center', fontsize=10, fontweight='bold', zorder=10,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.9))

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models["Model"].values, fontsize=10)
    ax.set_xlabel('AUC-ROC')
    ax.set_title('A. Fusion (with H&Y, n=111)', fontweight='bold')
    ax.set_xlim(0.5, 1.05)
    ax.invert_yaxis()

    # Panel B: Fixed fusion (H&Y excluded)
    ax = axes[1]
    df_fixed_sorted = df_fixed.sort_values("AUC")
    for i, (_, row) in enumerate(df_fixed_sorted.iterrows()):
        name = row["Model"].replace("_Fixed", "")
        color = colors_map.get(name, '#999999')
        if "Concat" in name:
            color = COLORS['fusion']
        ax.hlines(y=i, xmin=0.4, xmax=row["AUC"], color=color, linewidth=3)
        ax.plot(row["AUC"], i, 'o', color=color, markersize=12, zorder=5)
        ax.text(row["AUC"] + 0.01, i, f'{row["AUC"]:.3f}',
                va='center', fontsize=10, fontweight='bold', zorder=10,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.9))

    ax.set_yticks(range(len(df_fixed_sorted)))
    ax.set_yticklabels([r.replace("_Fixed", "") for r in df_fixed_sorted["Model"].values], fontsize=10)
    ax.set_xlabel('AUC-ROC')
    ax.set_title('B. Fusion (H&Y excluded, honest)', fontweight='bold')
    ax.set_xlim(0.4, 0.85)
    ax.invert_yaxis()

    plt.tight_layout()
    save_fig(fig, 'fig7_fusion_ablation')


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 9: Calibration + DCA (2-panel)
# ═══════════════════════════════════════════════════════════════════════════════
def fig9_calibration_dca():
    print("\n  Fig 9: Calibration + DCA...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: DCA
    ax = axes[0]
    dca = pd.read_csv(os.path.join(RESULTS_DIR, "dca_clinical_v2.csv"))
    dca_xgb = dca[dca["model"] == "XGBoost"]

    ax.plot(dca_xgb["threshold"], dca_xgb["net_benefit_model"],
            color=COLORS['clinical'], linewidth=2, label='XGBoost (Top-10)')
    ax.plot(dca_xgb["threshold"], dca_xgb["net_benefit_treat_all"],
            color='gray', linewidth=1.5, linestyle='--', label='Treat All')
    ax.axhline(y=0, color='black', linewidth=1, linestyle='-', label='Treat None')

    ax.set_xlabel('Threshold Probability')
    ax.set_ylabel('Net Benefit')
    ax.set_title('A. Decision Curve Analysis\n(WearGait-PD, n=82)', fontweight='bold')
    ax.legend(frameon=True, framealpha=0.9, fontsize=9)
    ax.set_xlim(0, 0.8)
    ax.set_ylim(-0.05, 0.35)

    # Panel B: Calibration
    ax = axes[1]
    cal = pd.read_csv(os.path.join(RESULTS_DIR, "calibration_results_v2.csv"))

    # Plot calibration info as text summary (since we don't have the reliability curve data)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect calibration')

    y_start = 0.85
    for _, row in cal.iterrows():
        ax.text(0.1, y_start, f'{row["Model"]}:', fontsize=11, fontweight='bold',
                zorder=10, bbox=dict(facecolor='white', edgecolor='none', alpha=0.9))
        ax.text(0.1, y_start - 0.08,
                f'  Brier: {row["Brier_before"]:.3f} → {row["Brier_after_Platt"]:.3f}',
                fontsize=10, zorder=10)
        ax.text(0.1, y_start - 0.15,
                f'  ECE: {row["ECE_before"]:.3f} → {row["ECE_after_Platt"]:.3f}',
                fontsize=10, zorder=10)
        y_start -= 0.30

    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Observed Frequency')
    ax.set_title('B. Calibration Summary\n(Before → After Platt Scaling)', fontweight='bold')

    plt.tight_layout()
    save_fig(fig, 'fig9_calibration_dca')


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 1: Study Flowchart (pipeline overview)
# ═══════════════════════════════════════════════════════════════════════════════
def fig1_flowchart():
    print("\n  Fig 1: Study flowchart...")
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(-1, 17)
    ax.set_ylim(-1, 7)
    ax.axis('off')

    box_h, box_w = 1.2, 3.0

    # Datasets column
    datasets = [
        ("WearGait-PD\nn=167, Real DBS", 0, 5.5, COLORS['clinical']),
        ("PADS\nn=469, 100Hz IMU", 0, 3.8, COLORS['wearable']),
        ("GaitPDB\nn=165, Force Plate", 0, 2.1, COLORS['gait']),
        ("UCI Voice\nn=195, Acoustic", 0, 0.4, COLORS['voice']),
    ]

    for text, x, y, color in datasets:
        box = FancyBboxPatch((x, y), box_w, box_h, boxstyle="round,pad=0.15",
                              facecolor=color, edgecolor='#333', linewidth=1.5,
                              alpha=0.2, zorder=2)
        ax.add_patch(box)
        ax.text(x + box_w / 2, y + box_h / 2, text, ha='center', va='center',
                fontsize=9, fontweight='bold', zorder=10)

    # Processing column
    processing = [
        ("Feature\nExtraction", 4.5, 4.6),
        ("Scaling +\nImputation", 4.5, 2.8),
        ("5-fold CV /\nLOOCV", 4.5, 1.0),
    ]
    for text, x, y in processing:
        box = FancyBboxPatch((x, y), 2.5, box_h, boxstyle="round,pad=0.15",
                              facecolor='#E8E8E8', edgecolor='#333', linewidth=1.5, zorder=2)
        ax.add_patch(box)
        ax.text(x + 1.25, y + box_h / 2, text, ha='center', va='center',
                fontsize=9, fontweight='bold', zorder=10)

    # Models column
    models = [
        ("XGBoost\nSVM, MLP", 8.5, 4.6),
        ("Cross-Attention\nFusion (GaitPDB)", 8.5, 2.8),
    ]
    for text, x, y in models:
        box = FancyBboxPatch((x, y), 2.8, box_h, boxstyle="round,pad=0.15",
                              facecolor='#D1E5F0', edgecolor='#333', linewidth=1.5, zorder=2)
        ax.add_patch(box)
        ax.text(x + 1.4, y + box_h / 2, text, ha='center', va='center',
                fontsize=9, fontweight='bold', zorder=10)

    # XAI column
    xai = [
        ("SHAP\nTreeExplainer", 12.8, 5.0),
        ("LIME\nConcordance", 12.8, 3.3),
        ("Groq LLM\nClinical Reports", 12.8, 1.6),
    ]
    for text, x, y in xai:
        box = FancyBboxPatch((x, y), 2.8, box_h, boxstyle="round,pad=0.15",
                              facecolor='#FDE0DD', edgecolor='#333', linewidth=1.5, zorder=2)
        ax.add_patch(box)
        ax.text(x + 1.4, y + box_h / 2, text, ha='center', va='center',
                fontsize=9, fontweight='bold', zorder=10)

    # Arrows (simplified — horizontal flow)
    arrow_style = dict(arrowstyle='-|>', mutation_scale=15, color='#333',
                       linewidth=2, zorder=3)
    for y_mid in [5.2, 3.4, 1.6]:
        ax.annotate('', xy=(4.3, y_mid), xytext=(3.2, y_mid),
                    arrowprops=arrow_style)
    for y_mid in [5.2, 3.4]:
        ax.annotate('', xy=(8.3, y_mid), xytext=(7.2, y_mid),
                    arrowprops=arrow_style)
    for y_mid in [5.6, 3.9, 2.2]:
        ax.annotate('', xy=(12.6, y_mid), xytext=(11.5, y_mid),
                    arrowprops=arrow_style)

    # Section labels
    ax.text(1.5, 6.8, 'Data Sources', ha='center', fontsize=12, fontweight='bold', zorder=10)
    ax.text(5.75, 6.8, 'Preprocessing', ha='center', fontsize=12, fontweight='bold', zorder=10)
    ax.text(9.9, 6.8, 'Models', ha='center', fontsize=12, fontweight='bold', zorder=10)
    ax.text(14.2, 6.8, 'Explainability', ha='center', fontsize=12, fontweight='bold', zorder=10)

    save_fig(fig, 'fig1_study_flowchart')


def main():
    print("=" * 70)
    print("Generating Publication Figures")
    print("=" * 70)

    fig1_flowchart()
    fig2_distributions()
    fig4_roc_curves()
    fig5_multisource_validation()
    fig7_fusion_ablation()
    fig9_calibration_dca()

    print("\n" + "=" * 70)
    print("ALL FIGURES GENERATED")
    print("=" * 70)
    print(f"  Output directory: {FIG_DIR}")
    print("  Generated: fig1, fig2, fig4, fig5, fig7, fig9")
    print("  SHAP beeswarms: already generated by shap_analysis.py (fig6)")
    print("  Fig 3 (architecture): create manually in PPT/Figma")
    print("  Fig 8 (Groq reports): generated by groq_report.py")
    print("\nDone.")


if __name__ == "__main__":
    main()
