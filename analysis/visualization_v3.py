#!/usr/bin/env python3
"""
Publication Figures v3 — Strict Journal-Quality Spec
=====================================================
Following exact typography, color, sizing, and layout rules for
Nature/JAMA/JBI-grade figures.

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
from matplotlib.patches import FancyBboxPatch
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

# ── Palette (colorblind-safe, journal standard) ──────────────────────────────
C = {
    'blue':    '#2166AC',
    'orange':  '#D6604D',
    'green':   '#4DAC26',
    'pink':    '#E9A3C9',
    'teal':    '#80CDC1',
    'gray':    '#969696',
    'dark':    '#252525',
}

# ── Global rcParams ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.labelweight': 'bold',
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
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
    ax.text(-0.08, 1.05, letter, transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='left')


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
                sm = SMOTE(random_state=SEED, k_neighbors=min(5, n_min-1))
                Xtr, ytr = sm.fit_resample(Xtr, ytr)
            except: pass
        m = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                               tree_method="hist", device="cuda", random_state=SEED,
                               eval_metric="logloss", verbosity=0)
        m.fit(Xtr, ytr); probs[te] = m.predict_proba(Xte)[:, 1]
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
                sm = SMOTE(random_state=SEED, k_neighbors=min(5, n_min-1))
                Xtr, ytr = sm.fit_resample(Xtr, ytr)
            except: pass
        m = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                               tree_method="hist", device="cuda", random_state=SEED,
                               eval_metric="logloss", verbosity=0)
        m.fit(Xtr, ytr); probs[te] = m.predict_proba(Xte)[:, 1]
    return probs


# ── Feature name cleaning ────────────────────────────────────────────────────
CLEAN = {
    'disease_duration_years':'Disease duration (yr)',
    'total_asymmetry':'Total motor asymmetry',
    'MDSUPDRS_2-12':'Tremor (UPDRS II-12)',
    'subdomain_bradykinesia_UE':'Upper-limb bradykinesia',
    'asymmetry_hand_movements':'Hand-movement asymmetry',
    'MDSUPDRS_2-11':'Walking/balance (II-11)',
    'MDSUPDRS_3-18':'Rest-tremor constancy (III-18)',
    'MDSUPDRS_1-1':'Cognitive impairment (I-1)',
    'MDSUPDRS_3-3-LLE':'Left-leg rigidity (III-3)',
    'age':'Age','bmi':'BMI',
    'subdomain_gait_posture':'Gait/posture subscore',
    'asymmetry_pronation_sup':'Pronation–supination asym.',
    'MDSUPDRS_3-5-R':'Right hand movements (III-5)',
    'MDSUPDRS_3-4-R':'Right finger tapping (III-4)',
    'asymmetry_finger_tapping':'Finger-tapping asymmetry',
    'hoehn_yahr':'Hoehn & Yahr stage',
    'all_perm_entropy_gyro':'Gyro permutation entropy',
    'action_tremor_action_ratio':'Action-tremor power ratio',
    'all_perm_entropy_acc_var':'Accel entropy variance',
    'all_tremor_action_ratio':'Tremor / total power',
    'all_perm_entropy_gyro_var':'Gyro entropy variance',
    'rest_dom_freq_acc':'Resting dominant freq (accel)',
    'all_rom_AccZ_var':'Vertical ROM variance',
    'rest_dom_freq_gyro':'Resting dominant freq (gyro)',
    'action_perm_entropy_gyro':'Action gyro entropy',
    'action_dom_freq_gyro':'Action dominant freq (gyro)',
    'rest_tremor_rest_ratio':'Resting-tremor ratio',
    'rest_amplitude_cv_acc':'Resting amplitude CV (accel)',
    'rest_amplitude_cv_gyro':'Resting amplitude CV (gyro)',
    'bilateral_asym_tremor_rest_acc':'Bilateral tremor asymmetry',
    'UPDRSM':'UPDRS motor score','UPDRS':'UPDRS total score',
    'TUAG':'Timed Up & Go (s)',
    'stride_time_cv':'Stride-time CV',
    'step_asymmetry_index':'Step-asymmetry index',
    'fog_index_total':'FOG index',
    'force_asymmetry':'Force asymmetry',
    'cadence_steps_per_min':'Cadence (steps min⁻¹)',
    'peak_force_cv_L':'Left peak-force CV',
    'peak_force_cv_R':'Right peak-force CV',
    'stride_entropy_L':'Left stride entropy',
    'stride_entropy_R':'Right stride entropy',
    'loading_rate_L':'Left loading rate',
    'loading_rate_R':'Right loading rate',
    'HoehnYahr':'Hoehn & Yahr stage',
    'Age':'Age','Height':'Height (cm)','Weight':'Weight (kg)',
}

def cn(f):
    return CLEAN.get(f, f.replace('_',' ').replace('all ','').title()[:32])


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Distributions (3-panel)
# ═══════════════════════════════════════════════════════════════════════════════
def fig2():
    print("  Fig 2: Distributions...")
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.0))
    plt.subplots_adjust(wspace=0.40)

    # A — PADS tremor
    df = pd.read_csv(os.path.join(PROC_DIR,"wearable_features","pads_sensor_pd_vs_healthy.csv"))
    ax = axes[0]
    for lab,c,nm in [(1,C['orange'],'PD'),(0,C['blue'],'Healthy')]:
        v = df[df['pd_label']==lab]['all_tremor_rest_acc'].dropna().values
        v = v[v < np.percentile(v, 97)]
        ax.hist(v, bins=35, density=True, alpha=0.55, color=c, edgecolor='none', label=nm)
    ax.set_xlabel('Resting tremor power\n(4–6 Hz band)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Density', fontsize=10, fontweight='bold')
    ax.legend(frameon=True, framealpha=0.85, edgecolor='#CCCCCC', fontsize=9)
    panel_label(ax, 'A')

    # B — GaitPDB stride CV
    df2 = pd.read_csv(os.path.join(PROC_DIR,"gait_features","gaitpdb_sensor_features.csv"))
    ax = axes[1]
    for lab,c,nm in [(1,C['orange'],'PD'),(0,C['blue'],'Control')]:
        v = df2[df2['pd_label']==lab]['stride_time_cv'].dropna().values
        ax.hist(v, bins=25, density=True, alpha=0.55, color=c, edgecolor='none', label=nm)
    ax.set_xlabel('Stride-time coefficient\nof variation', fontsize=10, fontweight='bold')
    ax.set_ylabel('Density', fontsize=10, fontweight='bold')
    ax.legend(frameon=True, framealpha=0.85, edgecolor='#CCCCCC', fontsize=9)
    panel_label(ax, 'B')

    # C — UCI Voice jitter
    df3 = pd.read_csv(os.path.join(PROC_DIR,"voice_features","uci_voice_features.csv"))
    lc = 'pd_status' if 'pd_status' in df3.columns else 'status'
    jc = [c for c in df3.columns if 'jitter' in c.lower() or 'Jitter' in c]
    jc = jc[0] if jc else [c for c in df3.columns if pd.api.types.is_numeric_dtype(df3[c])][0]
    ax = axes[2]
    for lab,col,nm in [(1,C['orange'],'PD'),(0,C['blue'],'Healthy')]:
        v = df3[df3[lc]==lab][jc].dropna().values
        ax.hist(v, bins=25, density=True, alpha=0.55, color=col, edgecolor='none', label=nm)
    ax.set_xlabel(f'{jc.replace("_"," ")}\n(voice jitter)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Density', fontsize=10, fontweight='bold')
    ax.legend(frameon=True, framealpha=0.85, edgecolor='#CCCCCC', fontsize=9)
    panel_label(ax, 'C')

    plt.tight_layout(pad=1.5)
    save(fig, 'fig2_distributions')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — ROC Curves (2×2)
# ═══════════════════════════════════════════════════════════════════════════════
def fig4():
    print("  Fig 4: ROC curves...")
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 7.0))
    plt.subplots_adjust(hspace=0.45, wspace=0.40)

    datasets = []

    # A — Clinical
    df = pd.read_csv(os.path.join(PROC_DIR,"clinical_features","weargait_pd_only.csv"))
    fc = [c for c in df.columns if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]
    X = df[fc].values.astype(np.float32); y = df["dbs_candidate"].values.astype(int)
    Xs = SimpleImputer(strategy="median").fit_transform(StandardScaler().fit_transform(X))
    tmp = xgb.XGBClassifier(n_estimators=200,max_depth=4,tree_method="hist",
                             device="cuda",random_state=SEED,verbosity=0)
    tmp.fit(Xs, y); top = np.argsort(tmp.feature_importances_)[::-1][:10]
    p1 = get_loocv_probs(X[:,top], y)
    datasets.append(('WearGait-PD Clinical','LOOCV, n = 82',y,p1,C['orange']))

    # B — PADS
    df2 = pd.read_csv(os.path.join(PROC_DIR,"wearable_features","pads_sensor_pd_vs_healthy.csv"))
    pm = ["subject_id","condition","age","gender","height","weight","pd_label","n_files_processed","handedness"]
    pf = [c for c in df2.columns if c not in pm and pd.api.types.is_numeric_dtype(df2[c])]
    p2 = get_cv_probs(df2[pf].values.astype(np.float32), df2["pd_label"].values.astype(int))
    datasets.append(('PADS Wearable','5-fold CV, n = 355',df2["pd_label"].values,p2,C['blue']))

    # C — GaitPDB
    df3 = pd.read_csv(os.path.join(PROC_DIR,"gait_features","gaitpdb_sensor_features.csv"))
    gm = ["subject_id","Study","Group","Gender","pd_label","sex","dbs_proxy_hy25","dbs_proxy_updrs32","n_single_trials"]
    gf = [c for c in df3.columns if c not in gm and pd.api.types.is_numeric_dtype(df3[c])]
    p3 = get_cv_probs(df3[gf].values.astype(np.float32), df3["pd_label"].values.astype(int))
    datasets.append(('GaitPDB Force Plate','5-fold CV, n = 165',df3["pd_label"].values,p3,C['green']))

    # D — UCI Voice
    df4 = pd.read_csv(os.path.join(PROC_DIR,"voice_features","uci_voice_features.csv"))
    lc = 'pd_status' if 'pd_status' in df4.columns else 'status'
    vf = [c for c in df4.columns if c not in ["name","subject_id",lc] and pd.api.types.is_numeric_dtype(df4[c])]
    p4 = get_cv_probs(df4[vf].values.astype(np.float32), df4[lc].values.astype(int))
    datasets.append(('UCI Voice','5-fold CV, n = 195',df4[lc].values,p4,C['pink']))

    labels = 'ABCD'
    for idx,(title,sub,yt,yp,color) in enumerate(datasets):
        ax = axes[idx//2, idx%2]
        fpr,tpr,_ = roc_curve(yt,yp); auc = roc_auc_score(yt,yp)

        ax.plot([0,1],[0,1], color=C['gray'], lw=0.8, ls='--', alpha=0.5)
        ax.plot(fpr, tpr, color=color, lw=2.2)

        ax.text(0.97, 0.05, f'AUC = {auc:.3f}', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=11, fontweight='bold', color=color,
                bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=0.3',
                          lw=1.2, alpha=0.92), zorder=10)

        ax.set_xlabel('1 − Specificity', fontsize=10, fontweight='bold')
        ax.set_ylabel('Sensitivity', fontsize=10, fontweight='bold')
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.05)
        ax.set_aspect('equal')
        ax.set_title(f'{title}\n{sub}', fontsize=11, fontweight='bold', pad=8)
        panel_label(ax, labels[idx])

    save(fig, 'fig4_roc_curves')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Multi-source forest plot
# ═══════════════════════════════════════════════════════════════════════════════
def fig5():
    print("  Fig 5: Multi-source validation...")
    entries = [
        {'name':'WearGait-PD\n(Clinical, n=82)', 'auc':0.878,'lo':0.792,'hi':0.950,'color':C['orange']},
        {'name':'PADS\n(Wearable, n=355)',       'auc':0.859,'lo':0.818,'hi':0.897,'color':C['blue']},
        {'name':'GaitPDB\n(Gait, n=165)',        'auc':0.996,'lo':0.973,'hi':0.998,'color':C['green']},
        {'name':'UCI Voice\n(Voice, n=195)',     'auc':0.953,'lo':0.945,'hi':0.992,'color':C['pink']},
    ]

    fig, ax = plt.subplots(figsize=(7.0, 3.5))

    for i,e in enumerate(entries):
        y = len(entries)-1-i
        ax.plot([e['lo'],e['hi']], [y,y], color=e['color'], lw=2.5, solid_capstyle='round')
        ax.plot([e['lo']],[y], '|', color=e['color'], ms=10, mew=2.5)
        ax.plot([e['hi']],[y], '|', color=e['color'], ms=10, mew=2.5)
        ax.plot(e['auc'], y, 'o', color=e['color'], ms=10, zorder=5,
                markeredgecolor='white', markeredgewidth=1.2)
        ax.text(e['hi']+0.008, y, f"{e['auc']:.3f}", va='center', ha='left',
                fontsize=11, fontweight='bold', color=e['color'])

    ax.set_yticks(range(len(entries)))
    ax.set_yticklabels([e['name'] for e in reversed(entries)], fontsize=10)
    ax.set_xlabel('AUC-ROC (95 % CI)', fontsize=11, fontweight='bold')
    ax.set_xlim(0.72, 1.04)
    ax.axvline(0.9, color=C['gray'], lw=0.7, ls='--', alpha=0.5)
    ax.xaxis.grid(True, alpha=0.2, ls='--', lw=0.5); ax.set_axisbelow(True)
    ax.set_title('Multi-Source Biomarker Validation', fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout(pad=1.5)
    save(fig, 'fig5_multisource_validation')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7 — Fusion ablation (A+B horizontal bars)
# ═══════════════════════════════════════════════════════════════════════════════
def fig7():
    print("  Fig 7: Fusion ablation...")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.0))
    plt.subplots_adjust(wspace=0.45)

    # A — with H&Y (reference)
    data_a = [
        ('Gait only',         0.655, C['green']),
        ('Cross-attention',   0.891, C['pink']),
        ('Simple concat',     0.893, C['teal']),
        ('Clinical only',     0.993, C['orange']),
    ]
    ax = axes[0]
    for i,(nm,auc,col) in enumerate(data_a):
        ax.barh(i, auc-0.45, left=0.45, height=0.55, color=col, alpha=0.85, edgecolor='white', lw=1)
        if auc > 0.75:
            ax.text(auc-0.015, i, f'{auc:.3f}', va='center', ha='right',
                    fontsize=10, color='white', fontweight='bold')
        else:
            ax.text(auc+0.01, i, f'{auc:.3f}', va='center', ha='left',
                    fontsize=10, color=C['dark'], fontweight='bold')
    ax.set_yticks(range(len(data_a)))
    ax.set_yticklabels([d[0] for d in data_a], fontsize=10)
    ax.set_xlim(0.45, 1.05)
    ax.set_xlabel('AUC-ROC', fontsize=10, fontweight='bold')
    ax.set_title('With H&Y (reference)', fontsize=12, fontweight='bold', pad=8)
    ax.xaxis.grid(True, alpha=0.3, ls='--', lw=0.5); ax.set_axisbelow(True)
    ax.text(0.46, 3.38, '* H&Y is the target variable', fontsize=8, color=C['gray'], style='italic')
    panel_label(ax, 'A')

    # B — H&Y excluded (honest)
    data_b = [
        ('Gait only',               0.584, C['green']),
        ('Clinical (no H&Y)',       0.661, C['orange']),
        ('Gait + Clinical\n(fusion)',0.697, C['blue']),
    ]
    ax = axes[1]
    for i,(nm,auc,col) in enumerate(data_b):
        ax.barh(i, auc-0.45, left=0.45, height=0.55, color=col, alpha=0.85, edgecolor='white', lw=1)
        if auc > 0.62:
            ax.text(auc-0.01, i, f'{auc:.3f}', va='center', ha='right',
                    fontsize=10, color='white', fontweight='bold')
        else:
            ax.text(auc+0.01, i, f'{auc:.3f}', va='center', ha='left',
                    fontsize=10, color=C['dark'], fontweight='bold')
    ax.set_yticks(range(len(data_b)))
    ax.set_yticklabels([d[0] for d in data_b], fontsize=10)
    ax.set_xlim(0.45, 0.78)
    ax.set_xlabel('AUC-ROC', fontsize=10, fontweight='bold')
    ax.set_title('H&Y excluded (honest)', fontsize=12, fontweight='bold', pad=8)
    ax.xaxis.grid(True, alpha=0.3, ls='--', lw=0.5); ax.set_axisbelow(True)

    # Delta annotation
    ax.annotate('', xy=(0.697, 2.4), xytext=(0.661, 2.4),
                arrowprops=dict(arrowstyle='<->', color=C['dark'], lw=1.3))
    ax.text(0.679, 2.6, 'Δ +0.036', ha='center', fontsize=9, fontweight='bold', color=C['dark'])
    panel_label(ax, 'B')

    plt.tight_layout(pad=1.5)
    save(fig, 'fig7_fusion_ablation')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 9 — DCA + Calibration
# ═══════════════════════════════════════════════════════════════════════════════
def fig9():
    print("  Fig 9: DCA + Calibration...")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.8))
    plt.subplots_adjust(wspace=0.40)

    # A — DCA
    ax = axes[0]
    dca = pd.read_csv(os.path.join(RESULTS_DIR,"dca_clinical_v2.csv"))
    d = dca[dca["model"]=="XGBoost"]

    ax.plot(d["threshold"], d["net_benefit_model"], color=C['orange'], lw=2.0, label='XGBoost (top-10)')
    ax.plot(d["threshold"], d["net_benefit_treat_all"], color=C['gray'], lw=1.5, ls='--', label='Treat all')
    ax.axhline(0, color=C['dark'], lw=1.0, label='Treat none')

    mask = d["net_benefit_model"].values > np.maximum(d["net_benefit_treat_all"].values, 0)
    ax.fill_between(d["threshold"][mask], 0, d["net_benefit_model"][mask],
                     alpha=0.12, color=C['orange'], zorder=1)

    ax.set_xlabel('Threshold probability', fontsize=10, fontweight='bold')
    ax.set_ylabel('Net benefit', fontsize=10, fontweight='bold')
    ax.set_xlim(0, 0.6); ax.set_ylim(-0.02, d["net_benefit_model"].max()*1.12)
    ax.legend(frameon=True, framealpha=0.85, edgecolor='#CCCCCC', fontsize=9,
              loc='upper right', bbox_to_anchor=(0.98, 0.98))
    panel_label(ax, 'A')

    # B — Calibration
    ax = axes[1]
    pred = pd.read_csv(os.path.join(RESULTS_DIR,"clinical_loocv_predictions.csv"))
    yt = pred["dbs_candidate"].values
    ax.plot([0,1],[0,1], color=C['gray'], lw=1.2, ls='--', alpha=0.6, label='Perfect', zorder=1)

    for name,yp,color,marker in [('XGBoost',pred["xgb_prob"].values,C['orange'],'o'),
                                   ('SVM',pred["svm_prob"].values,C['blue'],'s')]:
        try:
            pt,pp = calibration_curve(yt, yp, n_bins=6, strategy='quantile')
            ax.plot(pp, pt, marker=marker, color=color, lw=1.8, ms=7,
                    label=name, markeredgecolor='white', markeredgewidth=0.8, markevery=1)
        except: pass

    ax.set_xlabel('Mean predicted probability', fontsize=10, fontweight='bold')
    ax.set_ylabel('Fraction of positives', fontsize=10, fontweight='bold')
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.05)
    ax.set_aspect('equal')
    ax.xaxis.grid(True, alpha=0.2, ls='--', lw=0.5)
    ax.yaxis.grid(True, alpha=0.2, ls='--', lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=True, framealpha=0.85, edgecolor='#CCCCCC', fontsize=9, loc='lower right')
    panel_label(ax, 'B')

    fig.suptitle('Decision Curve Analysis and Calibration (WearGait-PD, n = 82)',
                 fontsize=13, fontweight='bold', y=1.03)
    plt.tight_layout(pad=1.5)
    save(fig, 'fig9_calibration_dca')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Flowchart
# ═══════════════════════════════════════════════════════════════════════════════
def fig1():
    print("  Fig 1: Flowchart...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7)
    ax.axis('off')

    def box(x,y,w,h,text,fc,ec,fs=9,bold=True):
        r = FancyBboxPatch((x,y),w,h, boxstyle="round,pad=0.3",
                            facecolor=fc, edgecolor=ec, lw=1.8, zorder=2)
        ax.add_patch(r)
        fw = 'bold' if bold else 'normal'
        ax.text(x+w/2, y+h/2, text, ha='center', va='center',
                fontsize=fs, fontweight=fw, zorder=10, linespacing=1.3)

    def hdr(x,y,w,h,text):
        r = FancyBboxPatch((x,y),w,h, boxstyle="round,pad=0.2",
                            facecolor=C['dark'], edgecolor=C['dark'], lw=1.5, zorder=2)
        ax.add_patch(r)
        ax.text(x+w/2, y+h/2, text, ha='center', va='center',
                fontsize=10, fontweight='bold', color='white', zorder=10)

    def arr(x1,y1,x2,y2):
        ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                    arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5,
                                     connectionstyle='arc3,rad=0.0'),
                    zorder=3)

    # Headers
    col_x = [0.3, 2.8, 5.5, 7.8]
    col_w = [2.0, 2.2, 2.0, 2.0]
    for x,w,t in zip(col_x, col_w,
                      ['Data Sources','Preprocessing','Models','Explainability']):
        hdr(x, 6.1, w, 0.6, t)

    # Data sources (warm)
    ds = [('WearGait-PD\nn = 167, Real DBS',5.0),
          ('PADS\nn = 469, 100 Hz IMU',3.8),
          ('GaitPDB\nn = 165, Force plate',2.6),
          ('UCI Voice\nn = 195, Acoustic',1.4)]
    for t,y in ds:
        box(0.3, y, 2.0, 0.9, t, '#FEF0D9', '#D95F0E', fs=8)

    # Preprocessing (cool blue)
    pp = [('Feature extraction\n(windowing, PSD,\nstride detection)',4.6),
          ('StandardScaler\nImputation\n(within CV folds)',3.0),
          ('SMOTE, 5-fold CV\nLOOCV',1.7)]
    for t,y in pp:
        box(2.8, y, 2.2, 1.1, t, '#EDF8FB', '#2C7FB8', fs=8, bold=False)

    # Models (green)
    box(5.5, 4.3, 2.0, 1.2, 'XGBoost (GPU)\nSVM\nMLP', '#E0F3DB', '#31A354', fs=9)
    box(5.5, 2.6, 2.0, 1.2, 'Cross-attention\nfusion\n(GaitPDB)', '#E0F3DB', '#31A354', fs=9)

    # XAI (purple)
    xa = [('SHAP\nTreeExplainer',4.8),('LIME\nconcordance',3.4),('Groq LLM\nclinical reports',2.0)]
    for t,y in xa:
        box(7.8, y, 2.0, 0.9, t, '#F1EEF6', '#756BB1', fs=9)

    # Arrows: data→preproc
    for dy in [5.45, 4.25, 3.05, 1.85]:
        arr(2.35, dy, 2.75, max(min(dy, 5.15), 2.25))
    # preproc→models
    arr(5.05, 5.15, 5.45, 4.9)
    arr(5.05, 3.55, 5.45, 3.2)
    # models→XAI
    arr(7.55, 4.9, 7.75, 5.25)
    arr(7.55, 3.8, 7.75, 3.85)
    arr(7.55, 3.0, 7.75, 2.45)

    plt.tight_layout(pad=1.5)
    save(fig, 'fig1_study_flowchart')


# ═══════════════════════════════════════════════════════════════════════════════
# SHAP BEESWARMS — strict spec
# ═══════════════════════════════════════════════════════════════════════════════
def shap_beeswarm(X, y, feat_names, title, subtitle, filename, max_display=15):
    print(f"  {filename}...")
    sc = StandardScaler(); Xs = sc.fit_transform(X)
    imp = SimpleImputer(strategy="median"); Xc = imp.fit_transform(Xs)
    m = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                           tree_method="hist", device="cuda", random_state=SEED,
                           eval_metric="logloss", verbosity=0)
    m.fit(Xc, y)
    explainer = shap.TreeExplainer(m)
    sv = explainer.shap_values(Xc)

    clean_names = [cn(f) for f in feat_names]
    explanation = shap.Explanation(values=sv, base_values=explainer.expected_value,
                                   data=Xc, feature_names=clean_names)

    fig_h = max_display * 0.45 + 2.0
    fig, ax = plt.subplots(figsize=(8.5, fig_h))

    plt.sca(ax)
    shap.plots.beeswarm(explanation, max_display=max_display, show=False,
                         color_bar=True, plot_size=None, s=10, alpha=0.75)

    # Clean up
    ax.set_title(f'{title}\n{subtitle}', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('SHAP value (impact on model output)', fontsize=11, fontweight='bold')
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
    fig.savefig(p, dpi=300, bbox_inches='tight', pad_inches=0.15, facecolor='white')
    plt.close(fig)
    print(f"    Saved: {filename}.png")


def all_shap():
    # Clinical
    df = pd.read_csv(os.path.join(PROC_DIR,"clinical_features","weargait_pd_only.csv"))
    fc = [c for c in df.columns if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]
    shap_beeswarm(df[fc].values.astype(np.float32), df["dbs_candidate"].values.astype(int),
                  fc, 'Clinical Features', 'DBS Candidacy (real labels, n = 82)',
                  'shap_clinical_beeswarm')

    # Wearable
    df2 = pd.read_csv(os.path.join(PROC_DIR,"wearable_features","pads_sensor_pd_vs_healthy.csv"))
    pm = ["subject_id","condition","age","gender","height","weight","pd_label","n_files_processed","handedness"]
    pf = [c for c in df2.columns if c not in pm and pd.api.types.is_numeric_dtype(df2[c])]
    shap_beeswarm(df2[pf].values.astype(np.float32), df2["pd_label"].values.astype(int),
                  pf, 'Wearable Sensor Features', 'PD vs Healthy (PADS, n = 355)',
                  'shap_wearable_beeswarm')

    # Gait
    df3 = pd.read_csv(os.path.join(PROC_DIR,"gait_features","gaitpdb_sensor_features.csv"))
    gm = ["subject_id","Study","Group","Gender","pd_label","sex","dbs_proxy_hy25","dbs_proxy_updrs32","n_single_trials"]
    gf = [c for c in df3.columns if c not in gm and pd.api.types.is_numeric_dtype(df3[c])]
    shap_beeswarm(df3[gf].values.astype(np.float32), df3["pd_label"].values.astype(int),
                  gf, 'Gait Force-Plate Features', 'PD vs Control (GaitPDB, n = 165)',
                  'shap_gait_beeswarm')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 8 — Groq LLM Reports
# ═══════════════════════════════════════════════════════════════════════════════
def fig8():
    print("  Fig 8: Groq reports...")
    reports = pd.read_csv(os.path.join(RESULTS_DIR,"groq_reports_real.csv"))

    colors_map = {'HIGH':C['orange'],'BORDERLINE':'#CC8800','LOW':C['green']}
    fig = plt.figure(figsize=(7.5, 10))
    gs = gridspec.GridSpec(3, 1, hspace=0.15, top=0.94, bottom=0.02, left=0.03, right=0.97)

    for i,(_, row) in enumerate(reports.iterrows()):
        ax = fig.add_subplot(gs[i, 0])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

        cat = row['category']
        color = colors_map.get(cat, C['gray'])

        # Header
        ax.add_patch(plt.Rectangle((0, 0.85), 1, 0.15, facecolor=color, alpha=0.12,
                                    transform=ax.transAxes, zorder=1))
        ax.text(0.5, 0.94, f'{cat} RISK — {row["dbs_prob"]:.0%}',
                ha='center', va='center', fontsize=14, fontweight='bold',
                color=color, transform=ax.transAxes, zorder=10)
        actual = "DBS+" if row["actual_dbs"] else "DBS−"
        ax.text(0.5, 0.875, f'Patient {row["patient_id"]}   |   Actual: {actual}',
                ha='center', va='center', fontsize=9, color='#555',
                transform=ax.transAxes, zorder=10)

        # Report text
        report = str(row['report_text'])
        sections = report.replace('[', '\n[').split('\n')
        y_pos = 0.80
        max_chars = 90

        for section in sections:
            section = section.strip()
            if not section: continue
            if section.startswith('['):
                y_pos -= 0.01
                if y_pos < 0.08: break
                ax.text(0.03, y_pos, section, fontsize=9, fontweight='bold',
                        va='top', color=color, transform=ax.transAxes, zorder=10)
                y_pos -= 0.055
            else:
                words = section.split()
                line = ""
                for word in words:
                    if len(line)+len(word)+1 > max_chars:
                        if y_pos < 0.08: break
                        ax.text(0.03, y_pos, line.strip(), fontsize=9,
                                va='top', color='#333', transform=ax.transAxes, zorder=10)
                        y_pos -= 0.045
                        line = word+" "
                    else:
                        line += word+" "
                if line.strip() and y_pos >= 0.08:
                    ax.text(0.03, y_pos, line.strip(), fontsize=9,
                            va='top', color='#333', transform=ax.transAxes, zorder=10)
                    y_pos -= 0.055

        # Footer
        ax.add_patch(plt.Rectangle((0, 0), 1, 0.07, facecolor='#F5F5F5', alpha=0.9,
                                    transform=ax.transAxes, zorder=1))
        hy=row.get('hoehn_yahr','N/A'); u3=row.get('updrs_part3','N/A')
        dur=row.get('disease_duration','N/A'); sub=row.get('motor_subtype','N/A')
        ax.text(0.5, 0.035,
                f'H&Y: {hy}   |   UPDRS-III: {u3}   |   Duration: {dur} yr   |   {sub}',
                ha='center', va='center', fontsize=8, color='#666', style='italic',
                transform=ax.transAxes, zorder=10)

        # Border
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(color); sp.set_linewidth(2)

    fig.suptitle('Groq LLM Clinical DBS Candidacy Reports — Real Patients',
                 fontsize=14, fontweight='bold', y=0.97)
    save(fig, 'fig8_groq_reports')


# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("="*70)
    print("Generating Publication Figures v3 (Strict Journal Spec)")
    print("="*70)
    fig1()
    fig2()
    fig4()
    fig5()
    fig7()
    fig9()
    all_shap()
    fig8()
    print("\n" + "="*70)
    print(f"ALL FIGURES → {FIG_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()
