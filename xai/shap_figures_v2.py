#!/usr/bin/env python3
"""
Clean SHAP Figures + Groq Report Figure (v2)
==============================================
Fix issues:
  - SHAP: Clean feature names, show only top 15, larger figure, no clutter
  - Groq: Proper text wrapping, no overlap, structured layout

Author: Kartic Mishra, Gachon University
"""

import os
import warnings
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "tables")
FIG_DIR = os.path.join(PROJECT_ROOT, "results", "figures", "2026-03-20")

try:
    import scienceplots
    plt.style.use(['science', 'no-latex'])
except: pass

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
})

OI = {
    'orange': '#E69F00', 'skyblue': '#56B4E9', 'green': '#009E73',
    'blue': '#0072B2', 'vermillion': '#D55E00', 'pink': '#CC79A7',
}

META_COLS = [
    "subject_id", "cohort", "dbs_candidate", "dbs_bilateral",
    "dbs_electrode", "dbs_years_since_surgery", "motor_subtype", "med_state_on"
]

# ── Clean feature name mapping ───────────────────────────────────────────────
CLEAN_NAMES = {
    'disease_duration_years': 'Disease Duration (years)',
    'total_asymmetry': 'Total Motor Asymmetry',
    'MDSUPDRS_2-12': 'UPDRS II-12: Tremor',
    'MDSUPDRS_2-13': 'UPDRS II-13: Freezing',
    'MDSUPDRS_2-11': 'UPDRS II-11: Walking/Balance',
    'MDSUPDRS_2-3': 'UPDRS II-3: Eating',
    'MDSUPDRS_2-7': 'UPDRS II-7: Handwriting',
    'subdomain_bradykinesia_UE': 'Upper Limb Bradykinesia',
    'asymmetry_hand_movements': 'Hand Movement Asymmetry',
    'MDSUPDRS_3-18': 'UPDRS III-18: Rest Tremor Constancy',
    'MDSUPDRS_3-9': 'UPDRS III-9: Arising from Chair',
    'MDSUPDRS_3-11': 'UPDRS III-11: Freezing of Gait',
    'MDSUPDRS_3-3-LLE': 'UPDRS III-3: Left Leg Rigidity',
    'MDSUPDRS_3-5-R': 'UPDRS III-5: Right Hand Movements',
    'MDSUPDRS_3-4-R': 'UPDRS III-4: Right Finger Tapping',
    'MDSUPDRS_3-4-L': 'UPDRS III-4: Left Finger Tapping',
    'MDSUPDRS_3-8-R': 'UPDRS III-8: Right Leg Agility',
    'MDSUPDRS_3-3-LUE': 'UPDRS III-3: Left Arm Rigidity',
    'MDSUPDRS_1-1': 'UPDRS I-1: Cognitive Impairment',
    'asymmetry_finger_tapping': 'Finger Tapping Asymmetry',
    'asymmetry_toe_tapping': 'Toe Tapping Asymmetry',
    'hoehn_yahr': 'Hoehn & Yahr Stage',
    'age': 'Age',
    'bmi': 'BMI',
    'sex': 'Sex',
    'updrs_part3_total': 'UPDRS Part III Total',
    'updrs_part2_total': 'UPDRS Part II Total',
    'subdomain_bradykinesia_LE': 'Lower Limb Bradykinesia',
    # Wearable features
    'all_perm_entropy_gyro': 'Gyroscope Permutation Entropy',
    'action_tremor_action_ratio': 'Action Tremor Power Ratio',
    'all_perm_entropy_acc_var': 'Accelerometer Entropy Variance',
    'all_tremor_action_ratio': 'Tremor/Total Power Ratio',
    'all_perm_entropy_gyro_var': 'Gyro Entropy Variance',
    'rest_dom_freq_acc': 'Resting Dominant Freq (Acc)',
    'all_rom_AccZ_var': 'Vertical ROM Variance',
    'rest_dom_freq_gyro': 'Resting Dominant Freq (Gyro)',
    'action_perm_entropy_gyro': 'Action Gyro Entropy',
    'action_dom_freq_gyro': 'Action Dominant Freq (Gyro)',
    'rest_tremor_rest_ratio': 'Resting Tremor Ratio',
    'rest_amplitude_cv_acc': 'Resting Amplitude CV (Acc)',
    'rest_amplitude_cv_gyro': 'Resting Amplitude CV (Gyro)',
    'bilateral_asym_tremor_rest_acc': 'Bilateral Tremor Asymmetry',
    'rest_energy_decay': 'Resting Energy Decay',
    'action_amplitude_cv_gyro': 'Action Amplitude CV (Gyro)',
    'rest_tremor_action_ratio': 'Rest/Action Tremor Ratio',
    # Gait features
    'UPDRSM': 'UPDRS Motor Score',
    'UPDRS': 'UPDRS Total Score',
    'TUAG': 'Timed Up & Go (sec)',
    'stride_time_cv': 'Stride Time CV',
    'step_asymmetry_index': 'Step Asymmetry Index',
    'fog_index_total': 'FOG Index',
    'force_asymmetry': 'Force Asymmetry',
    'cadence_steps_per_min': 'Cadence (steps/min)',
    'peak_force_cv_L': 'Left Peak Force CV',
    'peak_force_cv_R': 'Right Peak Force CV',
    'stride_entropy_L': 'Left Stride Entropy',
    'stride_entropy_R': 'Right Stride Entropy',
    'loading_rate_L': 'Left Loading Rate',
    'loading_rate_R': 'Right Loading Rate',
    'Age': 'Age',
    'Height': 'Height',
    'Weight': 'Weight',
}


def clean_name(feat):
    return CLEAN_NAMES.get(feat, feat.replace('_', ' ').title()[:35])


def make_shap_fig(X, y, feature_names, title, subtitle, color, filename, max_display=15):
    """Train XGBoost, compute SHAP, create clean beeswarm."""
    print(f"  {filename}...")

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    imp = SimpleImputer(strategy="median")
    X_c = imp.fit_transform(X_s)

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        tree_method="hist", device="cuda", random_state=SEED,
        eval_metric="logloss", verbosity=0
    )
    model.fit(X_c, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_c)

    # Clean feature names
    clean_names = [clean_name(f) for f in feature_names]

    explanation = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=X_c,
        feature_names=clean_names
    )

    # Create figure with generous size
    fig, ax = plt.subplots(figsize=(10, max_display * 0.55 + 2))

    plt.sca(ax)
    shap.plots.beeswarm(explanation, max_display=max_display, show=False,
                         color_bar=True, plot_size=None)

    # Clean up the auto-generated plot
    ax.set_title(f'{title}\n{subtitle}', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('SHAP Value (impact on prediction)', fontsize=11)

    # Make y-tick labels readable
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='x', labelsize=10)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, f"{filename}.png")
    fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.2, facecolor='white')
    plt.close(fig)
    print(f"    Saved: {filename}.png")


def fig8_groq_v2():
    """Clean Groq report cards — proper text wrapping, no overlap."""
    print("  fig8_groq_reports (v2)...")

    reports = pd.read_csv(os.path.join(RESULTS_DIR, "groq_reports_real.csv"))

    colors = {'HIGH': OI['vermillion'], 'BORDERLINE': OI['orange'], 'LOW': OI['green']}

    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(1, 3, wspace=0.15, left=0.02, right=0.98, top=0.90, bottom=0.02)

    for i, (_, row) in enumerate(reports.iterrows()):
        ax = fig.add_subplot(gs[0, i])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        cat = row['category']
        color = colors.get(cat, '#999999')

        # ── Header ───────────────────────────────────────────────────────
        ax.add_patch(plt.Rectangle((0, 0.88), 1, 0.12,
                                    facecolor=color, alpha=0.15,
                                    transform=ax.transAxes, zorder=1))
        ax.text(0.5, 0.95, f'{cat} RISK — {row["dbs_prob"]:.0%}',
                ha='center', va='center', fontsize=15, fontweight='bold',
                color=color, zorder=10, transform=ax.transAxes)
        actual = "DBS+" if row["actual_dbs"] else "DBS−"
        ax.text(0.5, 0.895, f'Patient {row["patient_id"]}  |  Actual: {actual}',
                ha='center', va='center', fontsize=10, color='#555',
                zorder=10, transform=ax.transAxes)

        # ── Report body ──────────────────────────────────────────────────
        report = str(row['report_text'])
        # Parse sections
        sections = report.replace('[', '\n[').split('\n')

        y_pos = 0.84
        line_height = 0.035
        max_chars = 48  # chars per line

        for section in sections:
            section = section.strip()
            if not section:
                continue

            if section.startswith('['):
                # Section header — bold, with space above
                y_pos -= 0.01
                if y_pos < 0.12:
                    break
                ax.text(0.04, y_pos, section, fontsize=9.5, fontweight='bold',
                        va='top', zorder=10, transform=ax.transAxes,
                        color=color)
                y_pos -= line_height * 1.2
            else:
                # Body text — wrap manually
                words = section.split()
                line = ""
                for word in words:
                    if len(line) + len(word) + 1 > max_chars:
                        if y_pos < 0.12:
                            break
                        ax.text(0.04, y_pos, line.strip(), fontsize=9,
                                va='top', color='#333', zorder=10,
                                transform=ax.transAxes)
                        y_pos -= line_height
                        line = word + " "
                    else:
                        line += word + " "
                if line.strip() and y_pos >= 0.12:
                    ax.text(0.04, y_pos, line.strip(), fontsize=9,
                            va='top', color='#333', zorder=10,
                            transform=ax.transAxes)
                    y_pos -= line_height * 1.3

        # ── Footer ───────────────────────────────────────────────────────
        ax.add_patch(plt.Rectangle((0, 0), 1, 0.09,
                                    facecolor='#f5f5f5', alpha=0.9,
                                    transform=ax.transAxes, zorder=1))
        hy = row.get('hoehn_yahr', 'N/A')
        u3 = row.get('updrs_part3', 'N/A')
        dur = row.get('disease_duration', 'N/A')
        sub = row.get('motor_subtype', 'N/A')
        ax.text(0.5, 0.045,
                f'H&Y: {hy}   |   UPDRS-III: {u3}   |   Duration: {dur} yr   |   {sub}',
                ha='center', va='center', fontsize=8.5, color='#666',
                style='italic', zorder=10, transform=ax.transAxes)

        # ── Card border ──────────────────────────────────────────────────
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(color)
            spine.set_linewidth(2.5)

    fig.suptitle('Groq LLM Clinical DBS Candidacy Reports — Real Patients',
                 fontsize=16, fontweight='bold', y=0.97)
    path = os.path.join(FIG_DIR, "fig8_groq_reports.png")
    fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.15, facecolor='white')
    plt.close(fig)
    print(f"    Saved: fig8_groq_reports.png")


def main():
    print("=" * 70)
    print("SHAP + Groq Figures v2 (Clean)")
    print("=" * 70)

    # ── SHAP: Clinical ────────────────────────────────────────────────────
    df = pd.read_csv(os.path.join(PROC_DIR, "clinical_features", "weargait_pd_only.csv"))
    fc = [c for c in df.columns if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]
    make_shap_fig(df[fc].values.astype(np.float32), df["dbs_candidate"].values.astype(int),
                  fc, "Clinical Features", "DBS Candidacy (Real Labels, n = 82)",
                  OI['vermillion'], "shap_clinical_beeswarm", max_display=15)

    # ── SHAP: Wearable ───────────────────────────────────────────────────
    df2 = pd.read_csv(os.path.join(PROC_DIR, "wearable_features", "pads_sensor_pd_vs_healthy.csv"))
    pm = ["subject_id","condition","age","gender","height","weight","pd_label","n_files_processed","handedness"]
    pf = [c for c in df2.columns if c not in pm and pd.api.types.is_numeric_dtype(df2[c])]
    make_shap_fig(df2[pf].values.astype(np.float32), df2["pd_label"].values.astype(int),
                  pf, "Wearable Sensor Features", "PD vs Healthy (PADS, n = 355)",
                  OI['blue'], "shap_wearable_beeswarm", max_display=15)

    # ── SHAP: Gait ───────────────────────────────────────────────────────
    df3 = pd.read_csv(os.path.join(PROC_DIR, "gait_features", "gaitpdb_sensor_features.csv"))
    gm = ["subject_id","Study","Group","Gender","pd_label","sex","dbs_proxy_hy25","dbs_proxy_updrs32","n_single_trials"]
    gf = [c for c in df3.columns if c not in gm and pd.api.types.is_numeric_dtype(df3[c])]
    make_shap_fig(df3[gf].values.astype(np.float32), df3["pd_label"].values.astype(int),
                  gf, "Gait Force Plate Features", "PD vs Control (GaitPDB, n = 165)",
                  OI['green'], "shap_gait_beeswarm", max_display=15)

    # ── Groq Reports ─────────────────────────────────────────────────────
    fig8_groq_v2()

    print("\nDone.")


if __name__ == "__main__":
    main()
