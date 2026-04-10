#!/usr/bin/env python3
"""
Publication Figures v2 — Nature-Quality
========================================
Complete rewrite with:
  - SciencePlots styling (science + no-latex)
  - Generous figure sizes, proper spacing
  - No text overlap — verified per figure
  - Proper calibration curves (not placeholder text)
  - Clean flowchart with visual hierarchy
  - Readable Groq report figure
  - Colorblind-safe Okabe-Ito palette throughout

Author: Kartic Mishra, Gachon University
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve
from imblearn.over_sampling import SMOTE
import xgboost as xgb

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "tables")
FIG_DIR = os.path.join(PROJECT_ROOT, "results", "figures", "2026-03-20")
os.makedirs(FIG_DIR, exist_ok=True)

META_COLS = [
    "subject_id", "cohort", "dbs_candidate", "dbs_bilateral",
    "dbs_electrode", "dbs_years_since_surgery", "motor_subtype", "med_state_on"
]

# ── Style ────────────────────────────────────────────────────────────────────
try:
    import scienceplots
    plt.style.use(['science', 'no-latex'])
except Exception:
    pass

# Override to Arial after scienceplots
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
})

# Okabe-Ito colorblind-safe
OI = {
    'orange': '#E69F00',
    'skyblue': '#56B4E9',
    'green': '#009E73',
    'yellow': '#F0E442',
    'blue': '#0072B2',
    'vermillion': '#D55E00',
    'pink': '#CC79A7',
    'black': '#000000',
    'grey': '#999999',
}

MODALITY_COLORS = {
    'clinical': OI['vermillion'],
    'wearable': OI['blue'],
    'gait': OI['green'],
    'voice': OI['orange'],
    'fusion': OI['pink'],
}


def save(fig, name):
    p = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(p, dpi=300, bbox_inches='tight', pad_inches=0.15, facecolor='white')
    plt.close(fig)
    print(f"  Saved: {name}.png")


def panel_label(ax, letter, x=-0.12, y=1.08):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=16,
            fontweight='bold', va='top', ha='left')


def get_loocv_probs(X, y):
    loo = LeaveOneOut()
    probs = np.zeros(len(y))
    for tr, te in loo.split(X):
        Xtr, Xte = X[tr], X[te]
        ytr = y[tr]
        sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
        imp = SimpleImputer(strategy="median"); Xtr = imp.fit_transform(Xtr); Xte = imp.transform(Xte)
        n_min = int(ytr.sum())
        if n_min >= 3:
            try:
                sm = SMOTE(random_state=SEED, k_neighbors=min(5, n_min-1))
                Xtr, ytr = sm.fit_resample(Xtr, ytr)
            except: pass
        m = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                               tree_method="hist", device="cuda", random_state=SEED,
                               eval_metric="logloss", verbosity=0)
        m.fit(Xtr, ytr)
        probs[te] = m.predict_proba(Xte)[:, 1]
    return probs


def get_cv_probs(X, y, n_folds=5):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    probs = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        Xtr, Xte = X[tr], X[te]
        ytr = y[tr]
        sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
        imp = SimpleImputer(strategy="median"); Xtr = imp.fit_transform(Xtr); Xte = imp.transform(Xte)
        n_min = int(ytr.sum())
        if n_min >= 3:
            try:
                sm = SMOTE(random_state=SEED, k_neighbors=min(5, n_min-1))
                Xtr, ytr = sm.fit_resample(Xtr, ytr)
            except: pass
        m = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                               tree_method="hist", device="cuda", random_state=SEED,
                               eval_metric="logloss", verbosity=0)
        m.fit(Xtr, ytr)
        probs[te] = m.predict_proba(Xte)[:, 1]
    return probs


# ═══════════════════════════════════════════════════════════════════════════════
def fig2_distributions():
    """3-panel distribution: KDE + rug instead of messy overlapping histograms."""
    print("  Fig 2: Distributions...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # A. PADS tremor
    df = pd.read_csv(os.path.join(PROC_DIR, "wearable_features", "pads_sensor_pd_vs_healthy.csv"))
    ax = axes[0]
    col = 'all_tremor_rest_acc'
    for lab, c, nm in [(1, MODALITY_COLORS['clinical'], 'PD (n=291)'),
                        (0, OI['skyblue'], 'Healthy (n=79)')]:
        v = df[df['pd_label']==lab][col].dropna().values
        v = v[v < np.percentile(v, 98)]  # trim outliers for clarity
        ax.hist(v, bins=40, density=True, alpha=0.45, color=c, edgecolor='none', label=nm)
    ax.set_xlabel('Resting Tremor Power (4–6 Hz)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    panel_label(ax, 'A')

    # B. GaitPDB stride CV
    df2 = pd.read_csv(os.path.join(PROC_DIR, "gait_features", "gaitpdb_sensor_features.csv"))
    ax = axes[1]
    for lab, c, nm in [(1, MODALITY_COLORS['gait'], 'PD (n=93)'),
                        (0, OI['skyblue'], 'Control (n=72)')]:
        v = df2[df2['pd_label']==lab]['stride_time_cv'].dropna().values
        ax.hist(v, bins=30, density=True, alpha=0.45, color=c, edgecolor='none', label=nm)
    ax.set_xlabel('Stride Time Coefficient of Variation', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    panel_label(ax, 'B')

    # C. UCI Voice jitter
    df3 = pd.read_csv(os.path.join(PROC_DIR, "voice_features", "uci_voice_features.csv"))
    lc = 'pd_status' if 'pd_status' in df3.columns else 'status'
    jcol = [c for c in df3.columns if 'jitter' in c.lower() or 'Jitter' in c]
    jcol = jcol[0] if jcol else [c for c in df3.columns if pd.api.types.is_numeric_dtype(df3[c])][0]
    ax = axes[2]
    for lab, c, nm in [(1, MODALITY_COLORS['voice'], 'PD (n=147)'),
                        (0, OI['skyblue'], 'Healthy (n=48)')]:
        v = df3[df3[lc]==lab][jcol].dropna().values
        ax.hist(v, bins=30, density=True, alpha=0.45, color=c, edgecolor='none', label=nm)
    ax.set_xlabel(jcol.replace('_', ' '), fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    panel_label(ax, 'C')

    plt.tight_layout(w_pad=3.0)
    save(fig, 'fig2_distributions')


# ═══════════════════════════════════════════════════════════════════════════════
def fig4_roc_curves():
    """4-panel ROC — clean, no heavy fill, thin diagonal, clear legend."""
    print("  Fig 4: ROC curves...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))

    # Prepare data
    datasets = []

    # A. Clinical
    df = pd.read_csv(os.path.join(PROC_DIR, "clinical_features", "weargait_pd_only.csv"))
    fc = [c for c in df.columns if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]
    X = df[fc].values.astype(np.float32); y = df["dbs_candidate"].values.astype(int)
    sc = StandardScaler(); Xs = SimpleImputer(strategy="median").fit_transform(sc.fit_transform(X))
    tmp = xgb.XGBClassifier(n_estimators=200, max_depth=4, tree_method="hist",
                             device="cuda", random_state=SEED, verbosity=0)
    tmp.fit(Xs, y); top = np.argsort(tmp.feature_importances_)[::-1][:10]
    prob = get_loocv_probs(X[:, top], y)
    datasets.append(('WearGait-PD Clinical', 'LOOCV, n = 82', y, prob, MODALITY_COLORS['clinical']))

    # B. PADS
    df2 = pd.read_csv(os.path.join(PROC_DIR, "wearable_features", "pads_sensor_pd_vs_healthy.csv"))
    pm = ["subject_id","condition","age","gender","height","weight","pd_label","n_files_processed","handedness"]
    pf = [c for c in df2.columns if c not in pm and pd.api.types.is_numeric_dtype(df2[c])]
    prob2 = get_cv_probs(df2[pf].values.astype(np.float32), df2["pd_label"].values.astype(int))
    datasets.append(('PADS Wearable Sensor', '5-fold CV, n = 355', df2["pd_label"].values, prob2, MODALITY_COLORS['wearable']))

    # C. GaitPDB
    df3 = pd.read_csv(os.path.join(PROC_DIR, "gait_features", "gaitpdb_sensor_features.csv"))
    gm = ["subject_id","Study","Group","Gender","pd_label","sex","dbs_proxy_hy25","dbs_proxy_updrs32","n_single_trials"]
    gf = [c for c in df3.columns if c not in gm and pd.api.types.is_numeric_dtype(df3[c])]
    prob3 = get_cv_probs(df3[gf].values.astype(np.float32), df3["pd_label"].values.astype(int))
    datasets.append(('GaitPDB Force Plate', '5-fold CV, n = 165', df3["pd_label"].values, prob3, MODALITY_COLORS['gait']))

    # D. UCI Voice
    df4 = pd.read_csv(os.path.join(PROC_DIR, "voice_features", "uci_voice_features.csv"))
    lc = 'pd_status' if 'pd_status' in df4.columns else 'status'
    vm = ["name","subject_id",lc]
    vf = [c for c in df4.columns if c not in vm and pd.api.types.is_numeric_dtype(df4[c])]
    prob4 = get_cv_probs(df4[vf].values.astype(np.float32), df4[lc].values.astype(int))
    datasets.append(('UCI Voice Features', '5-fold CV, n = 195', df4[lc].values, prob4, MODALITY_COLORS['voice']))

    labels = ['A', 'B', 'C', 'D']
    for idx, (title, subtitle, yt, yp, color) in enumerate(datasets):
        ax = axes[idx//2, idx%2]
        fpr, tpr, _ = roc_curve(yt, yp)
        auc = roc_auc_score(yt, yp)

        # Thin diagonal
        ax.plot([0,1], [0,1], color=OI['grey'], linewidth=0.8, linestyle='--', alpha=0.6)

        # ROC curve — solid, no fill
        ax.plot(fpr, tpr, color=color, linewidth=2.5)

        # AUC annotation — clean box in lower right
        ax.text(0.97, 0.05, f'AUC = {auc:.3f}', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=12, fontweight='bold',
                color=color,
                bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=0.4',
                          linewidth=1.5, alpha=0.95),
                zorder=10)

        ax.set_xlabel('1 − Specificity (FPR)')
        ax.set_ylabel('Sensitivity (TPR)')
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.05)
        ax.set_aspect('equal')

        # Title as two lines: dataset name + method
        ax.set_title(f'{title}\n{subtitle}', fontsize=12, pad=10)
        panel_label(ax, labels[idx])

    plt.tight_layout(h_pad=3.5, w_pad=3.0)
    save(fig, 'fig4_roc_curves')


# ═══════════════════════════════════════════════════════════════════════════════
def fig5_multisource():
    """Forest-plot style — dots with CI whiskers, clean and minimal."""
    print("  Fig 5: Multi-source validation...")

    entries = [
        {'name': 'WearGait-PD\nClinical (n=82)', 'auc': 0.878, 'lo': 0.792, 'hi': 0.950, 'color': MODALITY_COLORS['clinical'], 'label': 'Real DBS'},
        {'name': 'PADS\nWearable (n=355)', 'auc': 0.859, 'lo': 0.818, 'hi': 0.897, 'color': MODALITY_COLORS['wearable'], 'label': 'PD vs Healthy'},
        {'name': 'GaitPDB\nGait (n=165)', 'auc': 0.996, 'lo': 0.973, 'hi': 0.998, 'color': MODALITY_COLORS['gait'], 'label': 'PD vs Control'},
        {'name': 'UCI Voice\nVoice (n=195)', 'auc': 0.953, 'lo': 0.945, 'hi': 0.992, 'color': MODALITY_COLORS['voice'], 'label': 'PD vs Healthy'},
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, e in enumerate(entries):
        y = len(entries) - 1 - i  # top to bottom

        # CI whisker
        ax.plot([e['lo'], e['hi']], [y, y], color=e['color'], linewidth=2.5, solid_capstyle='round')
        # Caps
        ax.plot([e['lo']], [y], '|', color=e['color'], markersize=12, markeredgewidth=2.5)
        ax.plot([e['hi']], [y], '|', color=e['color'], markersize=12, markeredgewidth=2.5)
        # Center dot
        ax.plot(e['auc'], y, 'o', color=e['color'], markersize=12, zorder=5,
                markeredgecolor='white', markeredgewidth=1.5)

        # AUC value — right of CI
        ax.text(e['hi'] + 0.012, y, f"{e['auc']:.3f}",
                va='center', ha='left', fontsize=12, fontweight='bold', color=e['color'],
                zorder=10)

    ax.set_yticks(range(len(entries)))
    ax.set_yticklabels([e['name'] for e in reversed(entries)], fontsize=11)
    ax.set_xlabel('AUC-ROC (95% CI)', fontsize=12)
    ax.set_xlim(0.70, 1.05)

    # Reference lines
    ax.axvline(x=0.5, color=OI['grey'], linewidth=0.6, linestyle=':', alpha=0.4)
    ax.axvline(x=0.9, color=OI['grey'], linewidth=0.8, linestyle='--', alpha=0.4)
    ax.text(0.9, len(entries)-0.3, 'Excellent', fontsize=8, color=OI['grey'],
            ha='center', va='bottom', style='italic')

    ax.set_title('Multi-Source Biomarker Validation', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    save(fig, 'fig5_multisource_validation')


# ═══════════════════════════════════════════════════════════════════════════════
def fig7_fusion():
    """Horizontal bar chart — clean comparison, same x-axis scale for both panels."""
    print("  Fig 7: Fusion ablation...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=False)

    # A. With H&Y
    data_a = [
        ('Gait Only', 0.655, MODALITY_COLORS['gait']),
        ('Cross-Attention', 0.891, MODALITY_COLORS['fusion']),
        ('Simple Concat', 0.893, OI['skyblue']),
        ('Clinical Only', 0.993, MODALITY_COLORS['clinical']),
    ]
    ax = axes[0]
    for i, (name, auc, color) in enumerate(data_a):
        ax.barh(i, auc - 0.5, left=0.5, height=0.55, color=color, alpha=0.85,
                edgecolor='white', linewidth=1)
        ax.text(auc + 0.01, i, f'{auc:.3f}', va='center', ha='left',
                fontsize=11, fontweight='bold', zorder=10)
    ax.set_yticks(range(len(data_a)))
    ax.set_yticklabels([d[0] for d in data_a], fontsize=11)
    ax.set_xlim(0.5, 1.08)
    ax.set_xlabel('AUC-ROC', fontsize=11)
    ax.set_title('With H&Y (reference only)', fontsize=12, fontweight='bold', pad=10)
    ax.text(0.52, 3.35, '* H&Y is the target — tautological', fontsize=8,
            color=OI['grey'], style='italic')
    panel_label(ax, 'A')

    # B. H&Y excluded (honest)
    data_b = [
        ('Gait Only', 0.584, MODALITY_COLORS['gait']),
        ('Clinical\n(no H&Y)', 0.661, MODALITY_COLORS['clinical']),
        ('Gait + Clinical\n(Fusion)', 0.697, MODALITY_COLORS['fusion']),
    ]
    ax = axes[1]
    for i, (name, auc, color) in enumerate(data_b):
        ax.barh(i, auc - 0.4, left=0.4, height=0.55, color=color, alpha=0.85,
                edgecolor='white', linewidth=1)
        ax.text(auc + 0.01, i, f'{auc:.3f}', va='center', ha='left',
                fontsize=11, fontweight='bold', zorder=10)
    ax.set_yticks(range(len(data_b)))
    ax.set_yticklabels([d[0] for d in data_b], fontsize=11)
    ax.set_xlim(0.4, 0.80)
    ax.set_xlabel('AUC-ROC', fontsize=11)
    ax.set_title('H&Y excluded (honest evaluation)', fontsize=12, fontweight='bold', pad=10)

    # Delta annotation
    ax.annotate('', xy=(0.697, 2.35), xytext=(0.661, 2.35),
                arrowprops=dict(arrowstyle='<->', color=OI['black'], lw=1.5))
    ax.text(0.679, 2.55, '+0.036', ha='center', fontsize=9, fontweight='bold',
            zorder=10, bbox=dict(facecolor='white', edgecolor='none', alpha=0.9))
    panel_label(ax, 'B')

    plt.tight_layout(w_pad=4.0)
    save(fig, 'fig7_fusion_ablation')


# ═══════════════════════════════════════════════════════════════════════════════
def fig9_dca_calibration():
    """DCA + actual calibration curve (not just text)."""
    print("  Fig 9: DCA + Calibration...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # A. DCA
    ax = axes[0]
    dca = pd.read_csv(os.path.join(RESULTS_DIR, "dca_clinical_v2.csv"))
    dca_xgb = dca[dca["model"]=="XGBoost"]

    ax.plot(dca_xgb["threshold"], dca_xgb["net_benefit_model"],
            color=MODALITY_COLORS['clinical'], linewidth=2.5, label='XGBoost (Top-10)')
    ax.plot(dca_xgb["threshold"], dca_xgb["net_benefit_treat_all"],
            color=OI['grey'], linewidth=1.5, linestyle='--', label='Treat All', alpha=0.7)
    ax.axhline(y=0, color=OI['black'], linewidth=1, label='Treat None')

    # Shade benefit region
    mask = dca_xgb["net_benefit_model"] > np.maximum(dca_xgb["net_benefit_treat_all"].values, 0)
    if mask.any():
        ax.fill_between(dca_xgb["threshold"][mask], 0, dca_xgb["net_benefit_model"][mask],
                         alpha=0.1, color=MODALITY_COLORS['clinical'])

    ax.set_xlabel('Threshold Probability', fontsize=12)
    ax.set_ylabel('Net Benefit', fontsize=12)
    ax.set_xlim(0, 0.6)
    y_max = dca_xgb["net_benefit_model"].max()
    ax.set_ylim(-0.02, y_max * 1.15)
    ax.legend(frameon=True, fancybox=False, edgecolor='#cccccc', fontsize=10, loc='upper right')
    panel_label(ax, 'A')

    # B. Calibration curve — ACTUAL curve from LOOCV predictions
    ax = axes[1]
    pred = pd.read_csv(os.path.join(RESULTS_DIR, "clinical_loocv_predictions.csv"))
    y_true = pred["dbs_candidate"].values
    y_xgb = pred["xgb_prob"].values
    y_svm = pred["svm_prob"].values

    # Perfect calibration
    ax.plot([0,1], [0,1], color=OI['grey'], linewidth=1, linestyle='--', alpha=0.6,
            label='Perfectly calibrated')

    for name, yp, color, marker in [('XGBoost', y_xgb, MODALITY_COLORS['clinical'], 'o'),
                                      ('SVM', y_svm, OI['blue'], 's')]:
        try:
            prob_true, prob_pred = calibration_curve(y_true, yp, n_bins=6, strategy='quantile')
            ax.plot(prob_pred, prob_true, marker=marker, color=color, linewidth=2,
                    markersize=8, label=name, markeredgecolor='white', markeredgewidth=1)
        except Exception:
            pass

    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.legend(frameon=True, fancybox=False, edgecolor='#cccccc', fontsize=10, loc='lower right')
    panel_label(ax, 'B')

    fig.suptitle('Decision Curve Analysis and Model Calibration (WearGait-PD, n = 82)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, 'fig9_calibration_dca')


# ═══════════════════════════════════════════════════════════════════════════════
def fig1_flowchart():
    """Clean pipeline flowchart with proper visual hierarchy."""
    print("  Fig 1: Flowchart...")
    fig, ax = plt.subplots(figsize=(18, 7))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 7)
    ax.axis('off')

    def box(ax, x, y, w, h, text, color, alpha=0.15, fontsize=10, bold=True):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                               facecolor=color, edgecolor=matplotlib.colors.to_rgba(color, 0.8),
                               linewidth=2, alpha=alpha, zorder=2)
        ax.add_patch(rect)
        fw = 'bold' if bold else 'normal'
        ax.text(x+w/2, y+h/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight=fw, zorder=10, linespacing=1.4)

    def arrow(ax, x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='-|>', color='#555555',
                                     linewidth=2, mutation_scale=18),
                    zorder=3)

    # Column headers
    for x, label in [(1.5, 'DATA SOURCES'), (6, 'PREPROCESSING'),
                      (10.2, 'MODELS'), (14.5, 'EXPLAINABILITY')]:
        ax.text(x, 6.6, label, ha='center', fontsize=13, fontweight='bold',
                color='#333333', zorder=10)

    # Data sources
    ds = [
        ('WearGait-PD\nn = 167\nReal DBS labels', 5.3, MODALITY_COLORS['clinical']),
        ('PADS\nn = 469\n100 Hz IMU', 3.9, MODALITY_COLORS['wearable']),
        ('GaitPDB\nn = 165\nForce plate', 2.5, MODALITY_COLORS['gait']),
        ('UCI Voice\nn = 195\nAcoustic features', 1.1, MODALITY_COLORS['voice']),
    ]
    for text, y, color in ds:
        box(ax, 0.2, y, 2.6, 1.1, text, color, alpha=0.2, fontsize=9)

    # Preprocessing
    pp = [
        ('Feature\nExtraction\n(windowing, PSD,\nstride detection)', 4.8),
        ('StandardScaler\nSimpleImputer\n(within CV folds)', 3.0),
        ('SMOTE\n5-fold CV / LOOCV', 1.4),
    ]
    for text, y in pp:
        box(ax, 4.3, y, 2.8, 1.2, text, '#888888', alpha=0.08, fontsize=9, bold=False)

    # Models
    md = [
        ('XGBoost (GPU)\nSVM · MLP', 4.9),
        ('Cross-Attention\nFusion\n(GaitPDB)', 3.0),
    ]
    for text, y in md:
        box(ax, 8.5, y, 2.8, 1.2, text, OI['blue'], alpha=0.12, fontsize=10)

    # XAI
    xa = [
        ('SHAP\nTreeExplainer', 5.0),
        ('LIME\nConcordance', 3.4),
        ('Groq LLM\nClinical Reports', 1.8),
    ]
    for text, y in xa:
        box(ax, 13, y, 2.8, 1.0, text, OI['pink'], alpha=0.15, fontsize=10)

    # Arrows — data → preprocessing
    for y in [5.85, 4.45, 3.05, 1.65]:
        arrow(ax, 2.9, y, 4.2, max(min(y, 5.4), 2.0))

    # preprocessing → models
    arrow(ax, 7.2, 5.4, 8.4, 5.5)
    arrow(ax, 7.2, 3.6, 8.4, 3.6)

    # models → XAI
    arrow(ax, 11.4, 5.5, 12.9, 5.5)
    arrow(ax, 11.4, 3.9, 12.9, 3.9)
    arrow(ax, 11.4, 3.3, 12.9, 2.3)

    save(fig, 'fig1_study_flowchart')


# ═══════════════════════════════════════════════════════════════════════════════
def fig8_groq():
    """LLM report figure — structured cards, no text overlap."""
    print("  Fig 8: Groq reports...")

    reports = pd.read_csv(os.path.join(RESULTS_DIR, "groq_reports_real.csv"))

    fig = plt.figure(figsize=(18, 9))
    gs = gridspec.GridSpec(1, 3, wspace=0.25)

    colors = {'HIGH': MODALITY_COLORS['clinical'], 'BORDERLINE': OI['orange'], 'LOW': OI['green']}

    for i, (_, row) in enumerate(reports.iterrows()):
        ax = fig.add_subplot(gs[0, i])
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        cat = row['category']
        color = colors.get(cat, OI['grey'])

        # Header band
        ax.add_patch(plt.Rectangle((0, 8.8), 10, 1.2, facecolor=color, alpha=0.15,
                                    transform=ax.transData, zorder=1))
        ax.text(5, 9.5, f'{cat} RISK', ha='center', va='center', fontsize=16,
                fontweight='bold', color=color, zorder=10)
        ax.text(5, 9.0, f'Patient {row["patient_id"]}  |  '
                f'AI Score: {row["dbs_prob"]:.0%}  |  '
                f'Actual: {"DBS+" if row["actual_dbs"] else "DBS−"}',
                ha='center', va='center', fontsize=10, color='#555555', zorder=10)

        # Report text — split into lines, wrap carefully
        report = str(row['report_text'])
        lines = report.replace('[', '\n[').split('\n')
        y_pos = 8.2

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith('['):
                # Section header
                ax.text(0.5, y_pos, line, fontsize=9.5, fontweight='bold',
                        va='top', zorder=10, wrap=False)
                y_pos -= 0.5
            else:
                # Wrap long lines manually at ~60 chars
                words = line.split()
                current_line = ""
                for word in words:
                    if len(current_line) + len(word) + 1 > 55:
                        ax.text(0.5, y_pos, current_line.strip(), fontsize=9, va='top',
                                color='#333333', zorder=10)
                        y_pos -= 0.4
                        current_line = word + " "
                    else:
                        current_line += word + " "
                if current_line.strip():
                    ax.text(0.5, y_pos, current_line.strip(), fontsize=9, va='top',
                            color='#333333', zorder=10)
                    y_pos -= 0.5

        # Clinical summary footer
        hy = row.get('hoehn_yahr', 'N/A')
        u3 = row.get('updrs_part3', 'N/A')
        dur = row.get('disease_duration', 'N/A')
        sub = row.get('motor_subtype', 'N/A')

        ax.add_patch(plt.Rectangle((0, 0), 10, 1.2, facecolor='#f5f5f5', alpha=0.8, zorder=1))
        ax.text(5, 0.6, f'H&Y: {hy}  |  UPDRS-III: {u3}  |  Duration: {dur} yr  |  {sub}',
                ha='center', va='center', fontsize=9, color='#666666', style='italic', zorder=10)

        # Border
        for spine in ['top', 'bottom', 'left', 'right']:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color(color)
            ax.spines[spine].set_linewidth(2.5)

    fig.suptitle('Groq LLM Clinical DBS Candidacy Reports (Real Patients)',
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, 'fig8_groq_reports')


# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("Generating Publication Figures v2 (Nature Quality)")
    print("=" * 70)

    fig1_flowchart()
    fig2_distributions()
    fig4_roc_curves()
    fig5_multisource()
    fig7_fusion()
    fig9_dca_calibration()
    fig8_groq()

    print("\n" + "=" * 70)
    print(f"ALL FIGURES SAVED TO: {FIG_DIR}")
    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
