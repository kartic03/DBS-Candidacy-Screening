#!/usr/bin/env python3
"""
Publication Figures v4 — Strict Rules Applied
==============================================
ABSOLUTE RULES enforced:
  1. NO fontweight='bold' except bar VALUE labels inside bars
  2. NO annotation/delta overlap with titles (8pt clearance min)
  3. Identical legend labels across panels ("PD" / "Control")
  4. Y-axis label matches data (Density vs Count)
  5. Flowchart arrows perfectly horizontal within same row
  6. Bar value placement: single consistent rule across panels
  7. SHAP "Sum of N" row: s=6, alpha=0.35

Author: Kartic Mishra, Gachon University
"""

import os
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['font.family'] = 'Arial'

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap

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

# ── Palette (colorblind-safe) ────────────────────────────────────────────────
C = {
    'blue':    '#2166AC',
    'orange':  '#D6604D',
    'green':   '#4DAC26',
    'pink':    '#E9A3C9',
    'teal':    '#80CDC1',
    'gray':    '#969696',
    'dark':    '#252525',
}

# ── Global rcParams — NO bold anywhere ───────────────────────────────────────
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.labelweight': 'normal',      # RULE 1
    'axes.titlesize': 13,
    'axes.titleweight': 'normal',      # RULE 1
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def save(fig, name):
    p = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(p, dpi=300, bbox_inches='tight', pad_inches=0.15, facecolor='white')
    plt.close(fig)
    print(f"  Saved: {name}.png")


def panel_label(ax, letter):
    """RULE 1: panel labels are NOT bold. Moved up & left for clear separation."""
    ax.text(-0.14, 1.12, letter, transform=ax.transAxes,
            fontsize=14, fontweight='normal', va='top', ha='left')


def get_loocv_probs(X, y):
    loo = LeaveOneOut()
    probs = np.zeros(len(y))
    for tr, te in loo.split(X):
        Xtr, Xte, ytr = X[tr], X[te], y[tr]
        sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
        imp = SimpleImputer(strategy="median"); Xtr = imp.fit_transform(Xtr); Xte = imp.transform(Xte)
        n_min = int(ytr.sum())
        if n_min >= 3:
            try:
                sm = SMOTE(random_state=SEED, k_neighbors=min(5, n_min - 1))
                Xtr, ytr = sm.fit_resample(Xtr, ytr)
            except Exception:
                pass
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
        Xtr, Xte, ytr = X[tr], X[te], y[tr]
        sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
        imp = SimpleImputer(strategy="median"); Xtr = imp.fit_transform(Xtr); Xte = imp.transform(Xte)
        n_min = int(ytr.sum())
        if n_min >= 3:
            try:
                sm = SMOTE(random_state=SEED, k_neighbors=min(5, n_min - 1))
                Xtr, ytr = sm.fit_resample(Xtr, ytr)
            except Exception:
                pass
        m = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                               tree_method="hist", device="cuda", random_state=SEED,
                               eval_metric="logloss", verbosity=0)
        m.fit(Xtr, ytr)
        probs[te] = m.predict_proba(Xte)[:, 1]
    return probs


# ── Feature name cleaning ────────────────────────────────────────────────────
CLEAN = {
    'disease_duration_years': 'Disease duration (yr)',
    'total_asymmetry': 'Total motor asymmetry',
    'MDSUPDRS_2-12': 'Tremor (UPDRS II-12)',
    'subdomain_bradykinesia_UE': 'Upper-limb bradykinesia',
    'asymmetry_hand_movements': 'Hand-movement asymmetry',
    'MDSUPDRS_2-11': 'Walking/balance (II-11)',
    'MDSUPDRS_3-18': 'Rest-tremor constancy (III-18)',
    'MDSUPDRS_1-1': 'Cognitive impairment (I-1)',
    'MDSUPDRS_3-3-LLE': 'Left-leg rigidity (III-3)',
    'age': 'Age', 'bmi': 'BMI',
    'subdomain_gait_posture': 'Gait/posture subscore',
    'asymmetry_pronation_sup': 'Pronation-supination asym.',
    'MDSUPDRS_3-5-R': 'Right hand movements (III-5)',
    'MDSUPDRS_3-4-R': 'Right finger tapping (III-4)',
    'asymmetry_finger_tapping': 'Finger-tapping asymmetry',
    'hoehn_yahr': 'Hoehn & Yahr stage',
    'all_perm_entropy_gyro': 'Gyro permutation entropy',
    'action_tremor_action_ratio': 'Action-tremor power ratio',
    'all_perm_entropy_acc_var': 'Accel entropy variance',
    'all_tremor_action_ratio': 'Tremor / total power',
    'all_perm_entropy_gyro_var': 'Gyro entropy variance',
    'rest_dom_freq_acc': 'Resting dominant freq (accel)',
    'all_rom_AccZ_var': 'Vertical ROM variance',
    'rest_dom_freq_gyro': 'Resting dominant freq (gyro)',
    'action_perm_entropy_gyro': 'Action gyro entropy',
    'action_dom_freq_gyro': 'Action dominant freq (gyro)',
    'rest_tremor_rest_ratio': 'Resting-tremor ratio',
    'rest_amplitude_cv_acc': 'Resting amplitude CV (accel)',
    'rest_amplitude_cv_gyro': 'Resting amplitude CV (gyro)',
    'bilateral_asym_tremor_rest_acc': 'Bilateral tremor asymmetry',
    'UPDRSM': 'UPDRS motor score', 'UPDRS': 'UPDRS total score',
    'TUAG': 'Timed Up & Go (s)',
    'stride_time_cv': 'Stride-time CV',
    'step_asymmetry_index': 'Step-asymmetry index',
    'fog_index_total': 'FOG index',
    'force_asymmetry': 'Force asymmetry',
    'cadence_steps_per_min': 'Cadence (steps/min)',
    'peak_force_cv_L': 'Left peak-force CV',
    'peak_force_cv_R': 'Right peak-force CV',
    'stride_entropy_L': 'Left stride entropy',
    'stride_entropy_R': 'Right stride entropy',
    'loading_rate_L': 'Left loading rate',
    'loading_rate_R': 'Right loading rate',
    'HoehnYahr': 'Hoehn & Yahr stage',
    'Age': 'Age', 'Height': 'Height (cm)', 'Weight': 'Weight (kg)',
}


def cn(f):
    return CLEAN.get(f, f.replace('_', ' ').replace('all ', '').title()[:32])


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Flowchart (RULE 5: perfectly horizontal arrows)
# Wider canvas, generous margins, no cropping
# ═══════════════════════════════════════════════════════════════════════════════
def fig1():
    print("  Fig 1: Flowchart...")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(-0.3, 13.5)
    ax.set_ylim(0, 7.5)
    ax.axis('off')

    def box(x, y, w, h, text, fc, ec, fs=9):
        r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3",
                           facecolor=fc, edgecolor=ec, lw=1.8, zorder=2)
        ax.add_patch(r)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=fs, fontweight='normal', zorder=10, linespacing=1.3)

    def hdr(x, y, w, h, text):
        r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                           facecolor=C['dark'], edgecolor=C['dark'], lw=1.5, zorder=2)
        ax.add_patch(r)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=11, fontweight='normal', color='white', zorder=10)

    def harr(x1, y_mid, x2):
        """RULE 5: perfectly horizontal arrow."""
        arrow = FancyArrowPatch((x1, y_mid), (x2, y_mid),
                                arrowstyle='->', mutation_scale=14,
                                color='#555555', lw=1.8, zorder=3,
                                connectionstyle='arc3,rad=0.0',
                                shrinkA=4, shrinkB=4)
        ax.add_patch(arrow)

    def carr(x1, y1, x2, y2, rad=0.15):
        """Curved arrow for non-aligned connections."""
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                arrowstyle='->', mutation_scale=14,
                                color='#555555', lw=1.5, zorder=3,
                                connectionstyle=f'arc3,rad={rad}',
                                shrinkA=4, shrinkB=4)
        ax.add_patch(arrow)

    # ── Column layout (4 columns with generous spacing) ──
    col_x = [0.2, 3.5, 7.0, 10.5]
    col_w = [2.6, 2.8, 2.6, 2.6]

    # ── Headers ──
    hdr_y = 6.6
    for x, w, t in zip(col_x, col_w,
                        ['Data Sources', 'Preprocessing', 'Models', 'Explainability']):
        hdr(x, hdr_y, w, 0.65, t)

    # ── Row centers (4 data rows) ──
    row_cy = [5.6, 4.3, 3.0, 1.7]
    bh = 0.95  # uniform box height

    # Data source boxes (col 0)
    ds = ['WearGait-PD\nn = 167, Real DBS',
          'PADS\nn = 469, 100 Hz IMU',
          'GaitPDB\nn = 165, Force plate',
          'UCI Voice\nn = 195, Acoustic']
    for lbl, cy in zip(ds, row_cy):
        box(col_x[0], cy - bh / 2, col_w[0], bh, lbl, '#FEF0D9', '#D95F0E', fs=9)

    # Preprocessing boxes (col 1) — 3 boxes
    prep_cy = [4.95, 3.55, 1.7]
    prep_h = [1.25, 1.1, 0.95]
    prep_lbl = ['Feature extraction\n(windowing, PSD,\nstride detection)',
                'StandardScaler\nImputation\n(within CV folds)',
                'SMOTE, 5-fold CV\nLOOCV']
    for cy, h, lbl in zip(prep_cy, prep_h, prep_lbl):
        box(col_x[1], cy - h / 2, col_w[1], h, lbl, '#EDF8FB', '#2C7FB8', fs=9)

    # Model boxes (col 2) — 2 boxes
    model_cy = [4.95, 2.65]
    model_h = [1.3, 1.3]
    model_lbl = ['XGBoost (GPU)\nSVM\nMLP',
                 'Cross-attention\nfusion\n(GaitPDB)']
    for cy, h, lbl in zip(model_cy, model_h, model_lbl):
        box(col_x[2], cy - h / 2, col_w[2], h, lbl, '#E0F3DB', '#31A354', fs=10)

    # XAI boxes (col 3) — 3 boxes
    xai_cy = [5.4, 3.8, 2.2]
    xai_h = 1.0
    xai_lbl = ['SHAP\nTreeExplainer', 'LIME\nconcordance',
               'Groq LLM\nclinical reports']
    for cy, lbl in zip(xai_cy, xai_lbl):
        box(col_x[3], cy - xai_h / 2, col_w[3], xai_h, lbl, '#F1EEF6', '#756BB1', fs=10)

    # ── Arrows ──
    x_r0 = col_x[0] + col_w[0]   # right edge col 0
    x_l1 = col_x[1]               # left edge col 1
    x_r1 = col_x[1] + col_w[1]   # right edge col 1
    x_l2 = col_x[2]               # left edge col 2
    x_r2 = col_x[2] + col_w[2]   # right edge col 2
    x_l3 = col_x[3]               # left edge col 3

    # Data → Prep (rows 0,1 → prep0; row 2 → prep1; row 3 → prep2)
    harr(x_r0, row_cy[0], x_l1)                          # row0 → prep0 (horiz close)
    carr(x_r0, row_cy[1], x_l1, prep_cy[0], rad=-0.1)    # row1 → prep0
    harr(x_r0, row_cy[2], x_l1)                          # row2 → prep1 (horiz close)
    harr(x_r0, row_cy[3], x_l1)                          # row3 → prep2 (horiz)

    # Prep → Models
    harr(x_r1, prep_cy[0], x_l2)   # prep0 → model0
    harr(x_r1, prep_cy[1], x_l2)   # prep1 → model1

    # Models → XAI
    carr(x_r2, model_cy[0], x_l3, xai_cy[0], rad=0.1)    # model0 → SHAP
    carr(x_r2, model_cy[0], x_l3, xai_cy[1], rad=0.15)   # model0 → LIME
    carr(x_r2, model_cy[1], x_l3, xai_cy[2], rad=0.1)    # model1 → Groq

    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02)
    save(fig, 'fig1_study_flowchart')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Distributions (3-panel)
# RULE 3: identical legend labels ("PD" / "Control")
# RULE 4: Y-axis = "Density" only if density=True AND values look like density
# ═══════════════════════════════════════════════════════════════════════════════
def fig2():
    print("  Fig 2: Distributions...")
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.0))
    plt.subplots_adjust(wspace=0.40)

    # RULE 3: uniform group names
    GRP = [(1, C['orange'], 'PD'), (0, C['blue'], 'Control')]

    # A — PADS tremor
    df = pd.read_csv(os.path.join(PROC_DIR, "wearable_features",
                                  "pads_sensor_pd_vs_healthy.csv"))
    ax = axes[0]
    for lab, c, nm in GRP:
        v = df[df['pd_label'] == lab]['all_tremor_rest_acc'].dropna().values
        v = v[v < np.percentile(v, 97)]
        ax.hist(v, bins=35, density=True, alpha=0.55, color=c,
                edgecolor='none', label=nm)
    ax.set_xlabel('Resting tremor power\n(4-6 Hz band)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.legend(frameon=True, framealpha=0.85, edgecolor='#CCCCCC', fontsize=9)
    panel_label(ax, 'A')

    # B — GaitPDB stride CV
    df2 = pd.read_csv(os.path.join(PROC_DIR, "gait_features",
                                   "gaitpdb_sensor_features.csv"))
    ax = axes[1]
    for lab, c, nm in GRP:
        v = df2[df2['pd_label'] == lab]['stride_time_cv'].dropna().values
        ax.hist(v, bins=25, density=True, alpha=0.55, color=c,
                edgecolor='none', label=nm)
    ax.set_xlabel('Stride-time coefficient\nof variation', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.legend(frameon=True, framealpha=0.85, edgecolor='#CCCCCC', fontsize=9)
    panel_label(ax, 'B')

    # C — UCI Voice jitter
    df3 = pd.read_csv(os.path.join(PROC_DIR, "voice_features",
                                   "uci_voice_features.csv"))
    lc = 'pd_status' if 'pd_status' in df3.columns else 'status'
    jc = [c for c in df3.columns if 'jitter' in c.lower() or 'Jitter' in c]
    jc = jc[0] if jc else [c for c in df3.columns
                            if pd.api.types.is_numeric_dtype(df3[c])][0]
    ax = axes[2]
    for lab, col, nm in GRP:
        v = df3[df3[lc] == lab][jc].dropna().values
        ax.hist(v, bins=25, density=True, alpha=0.55, color=col,
                edgecolor='none', label=nm)
    ax.set_xlabel(f'{jc.replace("_", " ")}\n(voice jitter)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.legend(frameon=True, framealpha=0.85, edgecolor='#CCCCCC', fontsize=9)
    panel_label(ax, 'C')

    plt.tight_layout(pad=1.5)
    save(fig, 'fig2_distributions')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — ROC Curves (2x2)
# ═══════════════════════════════════════════════════════════════════════════════
def fig4():
    print("  Fig 4: ROC curves...")
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 7.0))
    plt.subplots_adjust(hspace=0.45, wspace=0.40)

    datasets = []

    # A — Clinical
    df = pd.read_csv(os.path.join(PROC_DIR, "clinical_features",
                                  "weargait_pd_only.csv"))
    fc = [c for c in df.columns
          if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]
    X = df[fc].values.astype(np.float32)
    y = df["dbs_candidate"].values.astype(int)
    Xs = SimpleImputer(strategy="median").fit_transform(
        StandardScaler().fit_transform(X))
    tmp = xgb.XGBClassifier(n_estimators=200, max_depth=4, tree_method="hist",
                             device="cuda", random_state=SEED, verbosity=0)
    tmp.fit(Xs, y)
    top = np.argsort(tmp.feature_importances_)[::-1][:10]
    p1 = get_loocv_probs(X[:, top], y)
    datasets.append(('WearGait-PD Clinical', 'LOOCV, n = 82',
                     y, p1, C['orange']))

    # B — PADS
    df2 = pd.read_csv(os.path.join(PROC_DIR, "wearable_features",
                                   "pads_sensor_pd_vs_healthy.csv"))
    pm = ["subject_id", "condition", "age", "gender", "height", "weight",
          "pd_label", "n_files_processed", "handedness"]
    pf = [c for c in df2.columns
          if c not in pm and pd.api.types.is_numeric_dtype(df2[c])]
    p2 = get_cv_probs(df2[pf].values.astype(np.float32),
                      df2["pd_label"].values.astype(int))
    datasets.append(('PADS Wearable', '5-fold CV, n = 370',
                     df2["pd_label"].values, p2, C['blue']))

    # C — GaitPDB
    df3 = pd.read_csv(os.path.join(PROC_DIR, "gait_features",
                                   "gaitpdb_sensor_features.csv"))
    gm = ["subject_id", "Study", "Group", "Gender", "pd_label", "sex",
          "dbs_proxy_hy25", "dbs_proxy_updrs32", "n_single_trials"]
    gf = [c for c in df3.columns
          if c not in gm and pd.api.types.is_numeric_dtype(df3[c])]
    p3 = get_cv_probs(df3[gf].values.astype(np.float32),
                      df3["pd_label"].values.astype(int))
    datasets.append(('GaitPDB Force Plate', '5-fold CV, n = 165',
                     df3["pd_label"].values, p3, C['green']))

    # D — UCI Voice
    df4 = pd.read_csv(os.path.join(PROC_DIR, "voice_features",
                                   "uci_voice_features.csv"))
    lc = 'pd_status' if 'pd_status' in df4.columns else 'status'
    vf = [c for c in df4.columns
          if c not in ["name", "subject_id", lc]
          and pd.api.types.is_numeric_dtype(df4[c])]
    p4 = get_cv_probs(df4[vf].values.astype(np.float32),
                      df4[lc].values.astype(int))
    datasets.append(('UCI Voice', '5-fold CV, n = 195',
                     df4[lc].values, p4, C['pink']))

    labels = 'ABCD'
    for idx, (title, sub, yt, yp, color) in enumerate(datasets):
        ax = axes[idx // 2, idx % 2]
        fpr, tpr, _ = roc_curve(yt, yp)
        auc = roc_auc_score(yt, yp)

        ax.plot([0, 1], [0, 1], color=C['gray'], lw=0.8, ls='--', alpha=0.5)
        ax.plot(fpr, tpr, color=color, lw=2.2)

        # AUC annotation — RULE 1: fontweight='normal'
        ax.text(0.97, 0.05, f'AUC = {auc:.3f}', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=11, fontweight='normal',
                color=color,
                bbox=dict(facecolor='white', edgecolor=color,
                          boxstyle='round,pad=0.3', lw=1.2, alpha=0.92),
                zorder=10)

        ax.set_xlabel('1 - Specificity', fontsize=10)
        ax.set_ylabel('Sensitivity', fontsize=10)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)
        ax.set_aspect('equal')
        ax.set_title(f'{title}\n{sub}', fontsize=11, pad=8)
        panel_label(ax, labels[idx])

    save(fig, 'fig4_roc_curves')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Multi-source forest plot
# ═══════════════════════════════════════════════════════════════════════════════
def fig5():
    print("  Fig 5: Multi-source validation...")
    entries = [
        {'name': 'WearGait-PD\n(Clinical, n=82)',
         'auc': 0.878, 'lo': 0.792, 'hi': 0.950, 'color': C['orange']},
        {'name': 'PADS\n(Wearable, n=370)',
         'auc': 0.860, 'lo': 0.818, 'hi': 0.897, 'color': C['blue']},
        {'name': 'GaitPDB\n(Gait, n=165)',
         'auc': 0.988, 'lo': 0.973, 'hi': 0.998, 'color': C['green']},
        {'name': 'UCI Voice\n(Voice, n=195)',
         'auc': 0.972, 'lo': 0.945, 'hi': 0.992, 'color': C['pink']},
    ]

    fig, ax = plt.subplots(figsize=(7.0, 3.5))

    for i, e in enumerate(entries):
        y = len(entries) - 1 - i
        ax.plot([e['lo'], e['hi']], [y, y], color=e['color'], lw=2.5,
                solid_capstyle='round')
        ax.plot([e['lo']], [y], '|', color=e['color'], ms=10, mew=2.5)
        ax.plot([e['hi']], [y], '|', color=e['color'], ms=10, mew=2.5)
        ax.plot(e['auc'], y, 'o', color=e['color'], ms=10, zorder=5,
                markeredgecolor='white', markeredgewidth=1.2)
        # RULE 1: no bold on annotation
        ax.text(e['hi'] + 0.008, y, f"{e['auc']:.3f}", va='center',
                ha='left', fontsize=11, fontweight='normal', color=e['color'])

    ax.set_yticks(range(len(entries)))
    ax.set_yticklabels([e['name'] for e in reversed(entries)], fontsize=10)
    ax.set_xlabel('AUC-ROC (95% CI)', fontsize=11)
    ax.set_xlim(0.72, 1.04)
    ax.axvline(0.9, color=C['gray'], lw=0.7, ls='--', alpha=0.5)
    ax.xaxis.grid(True, alpha=0.2, ls='--', lw=0.5)
    ax.set_axisbelow(True)
    ax.set_title('Multi-source biomarker validation', fontsize=13, pad=12)
    plt.tight_layout(pad=1.5)
    save(fig, 'fig5_multisource_validation')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7 — Fusion ablation (A + B horizontal bars)
# RULE 6: single consistent placement rule for bar value labels
# RULE 2: delta annotation well below title (8pt clearance)
# ═══════════════════════════════════════════════════════════════════════════════
def fig7():
    print("  Fig 7: Fusion ablation...")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.0))
    plt.subplots_adjust(wspace=0.45)

    # RULE 6: ONE placement rule for both panels
    # Rule: always place value label INSIDE the bar, right-aligned
    def bar_label_inside(ax, y_pos, auc, xleft):
        """Always inside, white text, bold (only exception allowed)."""
        ax.text(auc - 0.012, y_pos, f'{auc:.3f}', va='center', ha='right',
                fontsize=10, color='white', fontweight='bold')

    # A — with H&Y (reference)
    data_a = [
        ('Gait only',       0.655, C['green']),
        ('Cross-attention',  0.891, C['pink']),
        ('Simple concat',    0.893, C['teal']),
        ('Clinical only',    0.993, C['orange']),
    ]
    ax = axes[0]
    xleft_a = 0.45
    for i, (nm, auc, col) in enumerate(data_a):
        ax.barh(i, auc - xleft_a, left=xleft_a, height=0.55,
                color=col, alpha=0.85, edgecolor='white', lw=1)
        bar_label_inside(ax, i, auc, xleft_a)
    ax.set_yticks(range(len(data_a)))
    ax.set_yticklabels([d[0] for d in data_a], fontsize=10)
    ax.set_xlim(0.45, 1.05)
    ax.set_xlabel('AUC-ROC', fontsize=10)
    ax.set_title('With H&Y (reference)', fontsize=12, pad=8)
    ax.xaxis.grid(True, alpha=0.3, ls='--', lw=0.5)
    ax.set_axisbelow(True)
    ax.text(0.46, 3.38, '* H&Y is the target variable',
            fontsize=8, color=C['gray'], style='italic')
    panel_label(ax, 'A')

    # B — H&Y excluded (honest)
    data_b = [
        ('Gait only',              0.584, C['green']),
        ('Clinical (no H&Y)',      0.661, C['orange']),
        ('Gait + Clinical\n(fusion)', 0.697, C['blue']),
    ]
    ax = axes[1]
    xleft_b = 0.45
    for i, (nm, auc, col) in enumerate(data_b):
        ax.barh(i, auc - xleft_b, left=xleft_b, height=0.55,
                color=col, alpha=0.85, edgecolor='white', lw=1)
        bar_label_inside(ax, i, auc, xleft_b)
    ax.set_yticks(range(len(data_b)))
    ax.set_yticklabels([d[0] for d in data_b], fontsize=10)
    ax.set_xlim(0.45, 0.78)
    ax.set_xlabel('AUC-ROC', fontsize=10)
    ax.set_title('H&Y excluded (honest)', fontsize=12, pad=8)
    ax.xaxis.grid(True, alpha=0.3, ls='--', lw=0.5)
    ax.set_axisbelow(True)

    # RULE 2: delta annotation BELOW bars, not near title
    # Place between bar 1 and bar 2 (y ~ 1.5), well away from title
    ax.annotate('', xy=(0.697, 1.55), xytext=(0.661, 1.55),
                arrowprops=dict(arrowstyle='<->', color=C['dark'], lw=1.3))
    ax.text(0.679, 1.35, '+0.036', ha='center', fontsize=9,
            fontweight='normal', color=C['dark'])
    panel_label(ax, 'B')

    plt.tight_layout(pad=1.5)
    save(fig, 'fig7_fusion_ablation')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 9 — DCA + Calibration (polished, smooth curves)
# ═══════════════════════════════════════════════════════════════════════════════
def fig9():
    print("  Fig 9: DCA + Calibration...")
    from scipy.interpolate import make_interp_spline

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    plt.subplots_adjust(wspace=0.35)

    # A — DCA
    ax = axes[0]
    dca = pd.read_csv(os.path.join(RESULTS_DIR, "dca_clinical_v2.csv"))
    d = dca[dca["model"] == "XGBoost"].copy().reset_index(drop=True)

    thresh = d["threshold"].values
    nb_model = d["net_benefit_model"].values
    nb_all = d["net_benefit_treat_all"].values

    # Smooth the model curve with spline
    valid = ~np.isnan(nb_model) & ~np.isnan(thresh)
    t_v, nb_v = thresh[valid], nb_model[valid]
    if len(t_v) > 6:
        try:
            t_smooth = np.linspace(t_v.min(), t_v.max(), 200)
            spl = make_interp_spline(t_v, nb_v, k=3)
            nb_smooth = spl(t_smooth)
        except Exception:
            t_smooth, nb_smooth = t_v, nb_v
    else:
        t_smooth, nb_smooth = t_v, nb_v

    ax.plot(t_smooth, nb_smooth, color=C['orange'], lw=2.5,
            label='XGBoost (top-10)', zorder=4)
    ax.plot(thresh, nb_all, color=C['gray'], lw=1.8, ls='--',
            label='Treat all', zorder=3)
    ax.axhline(0, color=C['dark'], lw=1.2, label='Treat none', zorder=2)

    # Fill area where model beats alternatives
    mask = nb_smooth > np.maximum(
        np.interp(t_smooth, thresh[~np.isnan(nb_all)],
                  nb_all[~np.isnan(nb_all)]), 0)
    ax.fill_between(t_smooth[mask], 0, nb_smooth[mask],
                    alpha=0.15, color=C['orange'], zorder=1)

    ax.set_xlabel('Threshold probability', fontsize=11)
    ax.set_ylabel('Net benefit', fontsize=11)
    ax.set_xlim(0, 0.6)
    ymax = max(nb_smooth.max() if len(nb_smooth) > 0 else 0.3, 0.3)
    ax.set_ylim(-0.03, ymax * 1.15)
    ax.xaxis.grid(True, alpha=0.15, ls='-', lw=0.5)
    ax.yaxis.grid(True, alpha=0.15, ls='-', lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=True, framealpha=0.9, edgecolor='#DDDDDD', fontsize=9,
              loc='upper right', borderpad=0.8)
    ax.set_title('Decision curve analysis', fontsize=12, pad=10)
    panel_label(ax, 'A')

    # B — Calibration
    ax = axes[1]
    pred = pd.read_csv(os.path.join(RESULTS_DIR,
                                    "clinical_loocv_predictions.csv"))
    yt = pred["dbs_candidate"].values

    # Perfect calibration line (shaded band for reference)
    ax.fill_between([0, 1], [0, 0.9], [0.1, 1], alpha=0.06,
                    color=C['gray'], zorder=0, label='_nolegend_')
    ax.plot([0, 1], [0, 1], color=C['gray'], lw=1.5, ls='--', alpha=0.7,
            label='Perfect calibration', zorder=1)

    styles = [
        ('XGBoost', pred["xgb_prob"].values, C['orange'], 'o', 2.2),
        ('SVM', pred["svm_prob"].values, C['blue'], 's', 1.8),
    ]
    for name, yp, color, marker, lw in styles:
        try:
            pt, pp = calibration_curve(yt, yp, n_bins=6, strategy='quantile')
            # Smooth calibration curve
            if len(pp) > 3:
                try:
                    pp_s = np.linspace(pp.min(), pp.max(), 100)
                    spl = make_interp_spline(pp, pt, k=min(3, len(pp) - 1))
                    pt_s = spl(pp_s)
                    ax.plot(pp_s, pt_s, color=color, lw=lw, alpha=0.4,
                            zorder=3)
                except Exception:
                    pass
            ax.plot(pp, pt, marker=marker, color=color, lw=lw, ms=8,
                    label=name, markeredgecolor='white', markeredgewidth=1.0,
                    markevery=1, zorder=4)
        except Exception:
            pass

    ax.set_xlabel('Mean predicted probability', fontsize=11)
    ax.set_ylabel('Fraction of positives', fontsize=11)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.05)
    ax.set_aspect('equal')
    ax.xaxis.grid(True, alpha=0.15, ls='-', lw=0.5)
    ax.yaxis.grid(True, alpha=0.15, ls='-', lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=True, framealpha=0.9, edgecolor='#DDDDDD', fontsize=9,
              loc='lower right', borderpad=0.8)
    ax.set_title('Calibration plot', fontsize=12, pad=10)
    panel_label(ax, 'B')

    fig.suptitle('Decision curve analysis and calibration (WearGait-PD, n = 82)',
                 fontsize=13, y=1.01)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, 'fig9_calibration_dca')


# ═══════════════════════════════════════════════════════════════════════════════
# SHAP BEESWARMS
# RULE 7: "Sum of N other features" row gets s=6, alpha=0.35
# ═══════════════════════════════════════════════════════════════════════════════
def shap_beeswarm(X, y, feat_names, title, subtitle, filename, max_display=15):
    print(f"  {filename}...")
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    imp = SimpleImputer(strategy="median")
    Xc = imp.fit_transform(Xs)
    m = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                           tree_method="hist", device="cuda", random_state=SEED,
                           eval_metric="logloss", verbosity=0)
    m.fit(Xc, y)
    explainer = shap.TreeExplainer(m)
    sv = explainer.shap_values(Xc)

    clean_names = [cn(f) for f in feat_names]
    explanation = shap.Explanation(values=sv,
                                  base_values=explainer.expected_value,
                                  data=Xc,
                                  feature_names=clean_names)

    fig_h = max_display * 0.45 + 2.0
    fig, ax = plt.subplots(figsize=(8.5, fig_h))

    plt.sca(ax)
    shap.plots.beeswarm(explanation, max_display=max_display, show=False,
                        color_bar=True, plot_size=None, s=10, alpha=0.75)

    # RULE 7: de-emphasize "Sum of N other features" (bottom row)
    # Find the PathCollection for the bottom row and reduce size/alpha
    collections = ax.collections
    if collections:
        # The last scatter collection is the "Sum of N" row
        last_coll = collections[-1]
        last_coll.set_sizes(np.full(len(last_coll.get_offsets()), 6))
        last_coll.set_alpha(0.35)

    # Clean up — RULE 1: no bold on title/xlabel
    ax.set_title(f'{title}\n{subtitle}', fontsize=13, fontweight='normal',
                 pad=12)
    ax.set_xlabel('SHAP value (impact on model output)', fontsize=11)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='x', labelsize=10)

    # Horizontal gridlines behind data
    for ytick in ax.get_yticks():
        ax.axhline(y=ytick, color='gray', alpha=0.15, lw=0.5, zorder=0)

    # Zero line
    ax.axvline(x=0, color='#555555', lw=1.0, ls='-', zorder=1)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.subplots_adjust(left=0.38)
    p = os.path.join(FIG_DIR, f"{filename}.png")
    fig.savefig(p, dpi=300, bbox_inches='tight', pad_inches=0.15,
                facecolor='white')
    plt.close(fig)
    print(f"    Saved: {filename}.png")


def all_shap():
    # Clinical
    df = pd.read_csv(os.path.join(PROC_DIR, "clinical_features",
                                  "weargait_pd_only.csv"))
    fc = [c for c in df.columns
          if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]
    shap_beeswarm(df[fc].values.astype(np.float32),
                  df["dbs_candidate"].values.astype(int),
                  fc, 'Clinical features',
                  'DBS candidacy (real labels, n = 82)',
                  'shap_clinical_beeswarm')

    # Wearable
    df2 = pd.read_csv(os.path.join(PROC_DIR, "wearable_features",
                                   "pads_sensor_pd_vs_healthy.csv"))
    pm = ["subject_id", "condition", "age", "gender", "height", "weight",
          "pd_label", "n_files_processed", "handedness"]
    pf = [c for c in df2.columns
          if c not in pm and pd.api.types.is_numeric_dtype(df2[c])]
    shap_beeswarm(df2[pf].values.astype(np.float32),
                  df2["pd_label"].values.astype(int),
                  pf, 'Wearable sensor features',
                  'PD vs Control (PADS, n = 370)',
                  'shap_wearable_beeswarm')

    # Gait
    df3 = pd.read_csv(os.path.join(PROC_DIR, "gait_features",
                                   "gaitpdb_sensor_features.csv"))
    gm = ["subject_id", "Study", "Group", "Gender", "pd_label", "sex",
          "dbs_proxy_hy25", "dbs_proxy_updrs32", "n_single_trials"]
    gf = [c for c in df3.columns
          if c not in gm and pd.api.types.is_numeric_dtype(df3[c])]
    shap_beeswarm(df3[gf].values.astype(np.float32),
                  df3["pd_label"].values.astype(int),
                  gf, 'Gait force-plate features',
                  'PD vs Control (GaitPDB, n = 165)',
                  'shap_gait_beeswarm')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 8 — Groq LLM Reports (HORIZONTAL 1×3 layout, matching example)
# ═══════════════════════════════════════════════════════════════════════════════
def fig8():
    print("  Fig 8: Groq reports...")
    reports = pd.read_csv(os.path.join(RESULTS_DIR, "groq_reports_real.csv"))

    # Color scheme: header bg + card bg (light tint)
    card_style = {
        'HIGH':       {'hdr': '#D6604D', 'bg': '#FDF0EE', 'border': '#D6604D'},
        'BORDERLINE': {'hdr': '#E6A817', 'bg': '#FFF8E7', 'border': '#E6A817'},
        'LOW':        {'hdr': '#4DAC26', 'bg': '#EFF8EC', 'border': '#4DAC26'},
    }

    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(1, 3, wspace=0.08, left=0.03, right=0.97,
                           top=0.90, bottom=0.02)

    for i, (_, row) in enumerate(reports.iterrows()):
        ax = fig.add_subplot(gs[0, i])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        cat = row['category']
        style = card_style.get(cat, card_style['LOW'])

        # ── Card background ──
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=style['bg'],
                                   alpha=0.6, transform=ax.transAxes,
                                   zorder=0))

        # ── Colored header band (top 12%) ──
        ax.add_patch(plt.Rectangle((0, 0.88), 1, 0.12,
                                   facecolor=style['hdr'], alpha=0.25,
                                   transform=ax.transAxes, zorder=1))

        # Header text
        ax.text(0.5, 0.945, f'{cat} ({row["dbs_prob"]:.0%})',
                ha='center', va='center', fontsize=14, fontweight='normal',
                color=style['hdr'], transform=ax.transAxes, zorder=10)

        # Patient subtitle
        actual = "DBS+" if row["actual_dbs"] else "DBS\u2212"
        ax.text(0.5, 0.895,
                f'Patient: {row["patient_id"]}',
                ha='center', va='center', fontsize=9, color='#666',
                transform=ax.transAxes, zorder=10)

        # ── Parse report into structured sections ──
        report = str(row['report_text'])
        # Extract section content
        sections = {}
        current_key = None
        for part in report.replace('[', '\n[').split('\n'):
            part = part.strip()
            if not part:
                continue
            if part.startswith('[') and ']' in part:
                bracket_end = part.index(']')
                current_key = part[1:bracket_end]
                content = part[bracket_end + 1:].strip()
                sections[current_key] = content
            elif current_key:
                sections[current_key] = sections.get(current_key, '') + ' ' + part

        # ── Render sections as monospace-like structured text ──
        y_pos = 0.82
        line_h = 0.038
        max_chars = 42  # chars per line for wrapping

        section_order = ['DBS_Score', 'Motor_Profile', 'Key_Drivers',
                         'Recommendation', 'Caution_Flags']

        for key in section_order:
            if key not in sections:
                continue
            val = sections[key].strip()

            # Section header
            if y_pos < 0.08:
                break
            ax.text(0.05, y_pos, f'{key}:', fontsize=9,
                    fontweight='normal', va='top', color=style['hdr'],
                    transform=ax.transAxes, zorder=10,
                    fontfamily='monospace')
            y_pos -= line_h

            # Wrap content text
            words = val.split()
            line = "  "
            for word in words:
                if len(line) + len(word) + 1 > max_chars:
                    if y_pos < 0.08:
                        break
                    ax.text(0.05, y_pos, line.rstrip(), fontsize=8.5,
                            va='top', color='#333',
                            transform=ax.transAxes, zorder=10,
                            fontfamily='monospace')
                    y_pos -= line_h
                    line = "  " + word + " "
                else:
                    line += word + " "
            if line.strip() and y_pos >= 0.08:
                ax.text(0.05, y_pos, line.rstrip(), fontsize=8.5,
                        va='top', color='#333',
                        transform=ax.transAxes, zorder=10,
                        fontfamily='monospace')
                y_pos -= line_h * 1.4  # extra space between sections

        # ── Footer bar ──
        ax.add_patch(plt.Rectangle((0, 0), 1, 0.06, facecolor='#F0F0F0',
                                   alpha=0.9, transform=ax.transAxes,
                                   zorder=1))
        hy = row.get('hoehn_yahr', 'N/A')
        u3 = row.get('updrs_part3', 'N/A')
        dur = row.get('disease_duration', 'N/A')
        ax.text(0.5, 0.03,
                f'H&Y: {hy}  |  UPDRS-III: {u3}  |  Duration: {dur} yr',
                ha='center', va='center', fontsize=8, color='#666',
                style='italic', transform=ax.transAxes, zorder=10)

        # ── Card border ──
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(style['border'])
            sp.set_linewidth(2.5)

    fig.suptitle('Figure 8: LLM-generated DBS candidacy reports '
                 '(Llama 3.3 70B via Groq)',
                 fontsize=13, fontweight='normal', y=0.96)
    save(fig, 'fig8_groq_reports')


# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("Generating Publication Figures v4 (All Rules Enforced)")
    print("=" * 70)
    fig1()
    fig2()
    fig4()
    fig5()
    fig7()
    fig9()
    all_shap()
    fig8()
    print("\n" + "=" * 70)
    print(f"ALL FIGURES -> {FIG_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
