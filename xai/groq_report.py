#!/usr/bin/env python3
"""
Groq LLM Clinical DBS Reports (Revised) — REAL Patients
=========================================================
Generates clinician-readable DBS candidacy reports for 3 REAL patients
from the WearGait-PD test set using Llama 3.3 70B via Groq API.

Selects: 1 high-risk DBS+, 1 borderline, 1 low-risk DBS-
Uses actual UPDRS scores and SHAP-derived feature importance.

Output:
  results/tables/groq_reports_real.csv
  results/figures/fig8_groq_reports.png

Author: Kartic Mishra, Gachon University
"""

import os
import time
import warnings
import yaml

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "tables")
FIG_DIR = os.path.join(PROJECT_ROOT, "results", "figures")

# Load config
with open(os.path.join(PROJECT_ROOT, "config.yaml")) as f:
    config = yaml.safe_load(f)

GROQ_API_KEY = config.get("groq", {}).get("api_key", "")
GROQ_MODEL = config.get("groq", {}).get("model", "llama-3.3-70b-versatile")
FALLBACK_MODEL = config.get("groq", {}).get("fallback_model", "qwen-qwq-32b")


def groq_generate(system_prompt, user_prompt, max_tokens=250, temperature=0.3):
    """Call Groq API with retry logic."""
    try:
        from groq import Groq
    except ImportError:
        print("  WARNING: groq package not installed. Generating placeholder reports.")
        return None

    if not GROQ_API_KEY or GROQ_API_KEY == "YOUR_GROQ_API_KEY":
        print("  WARNING: Groq API key not set. Generating placeholder reports.")
        return None

    client = Groq(api_key=GROQ_API_KEY)

    for attempt in range(3):
        try:
            model = GROQ_MODEL if attempt < 2 else FALLBACK_MODEL
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** attempt)

    return None


SYSTEM_PROMPT = """You are a board-certified movement disorder neurologist specializing in DBS patient selection. Generate a concise evidence-based DBS candidacy assessment.

Format your response with these sections:
[DBS_Score] The AI-predicted probability
[Motor_Profile] Brief motor characterization
[Key_Drivers] Top SHAP-identified features driving the prediction
[Recommendation] Clinical recommendation
[Caution_Flags] Any concerns or limitations

Length: 100-150 words. Tone: clinical, direct, evidence-based."""


def main():
    print("=" * 70)
    print("Groq LLM Clinical DBS Reports (Real Patients)")
    print("=" * 70)

    # Load LOOCV predictions
    pred_path = os.path.join(RESULTS_DIR, "clinical_loocv_predictions.csv")
    df_pred = pd.read_csv(pred_path)

    # Load full clinical data for patient details
    df_pd = pd.read_csv(os.path.join(
        PROJECT_ROOT, "data", "processed", "clinical_features", "weargait_pd_only.csv"
    ))

    # Load SHAP importance
    shap_path = os.path.join(RESULTS_DIR, "shap_clinical_importance.csv")
    df_shap = pd.read_csv(shap_path)
    top_features = df_shap.head(10)["feature"].tolist()

    # Select 3 real patients
    # 1. High risk: DBS+ with highest probability
    dbs_pos = df_pred[df_pred["dbs_candidate"] == 1].sort_values("xgb_prob", ascending=False)
    high_patient = dbs_pos.iloc[0] if len(dbs_pos) > 0 else None

    # 2. Borderline: closest to 0.5 threshold
    df_pred["dist_to_05"] = abs(df_pred["xgb_prob"] - 0.5)
    borderline_patient = df_pred.sort_values("dist_to_05").iloc[0]

    # 3. Low risk: DBS- with lowest probability
    dbs_neg = df_pred[df_pred["dbs_candidate"] == 0].sort_values("xgb_prob", ascending=True)
    low_patient = dbs_neg.iloc[0] if len(dbs_neg) > 0 else None

    cases = []
    for category, patient_row in [("HIGH", high_patient), ("BORDERLINE", borderline_patient),
                                    ("LOW", low_patient)]:
        if patient_row is None:
            continue

        sid = patient_row["subject_id"]
        prob = patient_row["xgb_prob"]
        actual = int(patient_row["dbs_candidate"])

        # Get clinical details
        patient_data = df_pd[df_pd["subject_id"] == sid]
        if len(patient_data) == 0:
            continue
        p = patient_data.iloc[0]

        hy = p.get("hoehn_yahr", "N/A")
        updrs3 = p.get("updrs_part3_total", "N/A")
        duration = p.get("disease_duration_years", "N/A")
        age = p.get("age", "N/A")
        subtype = p.get("motor_subtype", "N/A")

        # Build prompt
        user_prompt = f"""Patient: {sid} | DBS AI Score: {prob:.1%} ({category} RISK)
Actual DBS Status: {"DBS Recipient" if actual == 1 else "Non-DBS"}
Age: {age} | Disease Duration: {duration} years | H&Y: {hy}
MDS-UPDRS Part III Total: {updrs3} | Motor Subtype: {subtype}
Top SHAP Drivers: {', '.join(top_features[:5])}
Full Feature Set: 65 MDS-UPDRS items + bilateral asymmetry indices + demographics"""

        print(f"\n  Generating report for {sid} ({category}, prob={prob:.3f})...")
        report = groq_generate(SYSTEM_PROMPT, user_prompt)

        if report is None:
            report = (f"[DBS_Score] {prob:.1%} ({category} risk)\n"
                     f"[Motor_Profile] {'Advanced' if prob > 0.5 else 'Mild'} PD, "
                     f"H&Y {hy}, UPDRS-III {updrs3}\n"
                     f"[Key_Drivers] {', '.join(top_features[:3])}\n"
                     f"[Recommendation] {'Consider DBS evaluation' if prob > 0.5 else 'Routine follow-up'}\n"
                     f"[Caution_Flags] AI-generated assessment requires clinical confirmation")

        cases.append({
            "patient_id": sid,
            "dbs_prob": prob,
            "actual_dbs": actual,
            "category": category,
            "report_text": report,
            "hoehn_yahr": hy,
            "updrs_part3": updrs3,
            "disease_duration": duration,
            "motor_subtype": subtype,
        })
        print(f"  Report generated ({len(report)} chars)")

    # Save reports
    df_reports = pd.DataFrame(cases)
    out_path = os.path.join(RESULTS_DIR, "groq_reports_real.csv")
    df_reports.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")

    # Generate Fig 8: Report visualization
    print("\n  Generating Fig 8...")
    try:
        plt.rcParams['font.family'] = 'Arial'
    except Exception:
        plt.rcParams['font.family'] = 'Liberation Sans'

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    colors = {"HIGH": "#D55E00", "BORDERLINE": "#E69F00", "LOW": "#009E73"}

    for i, case in enumerate(cases):
        ax = axes[i]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Header
        color = colors.get(case["category"], "#999999")
        ax.add_patch(plt.Rectangle((0, 0.88), 1, 0.12, facecolor=color, alpha=0.2,
                                    transform=ax.transAxes, zorder=1))
        ax.text(0.5, 0.94, f'{case["category"]} RISK — {case["dbs_prob"]:.0%}',
                ha='center', va='center', fontsize=14, fontweight='bold',
                color=color, zorder=10, transform=ax.transAxes)
        ax.text(0.5, 0.89, f'Patient {case["patient_id"]} | '
                f'Actual: {"DBS+" if case["actual_dbs"] else "DBS-"}',
                ha='center', va='center', fontsize=10, color='#555',
                zorder=10, transform=ax.transAxes)

        # Report text
        report_lines = case["report_text"].split('\n')
        y_pos = 0.82
        for line in report_lines:
            if line.strip():
                fontweight = 'bold' if line.strip().startswith('[') else 'normal'
                fontsize = 9 if fontweight == 'normal' else 10
                ax.text(0.05, y_pos, line.strip(), fontsize=fontsize,
                        fontweight=fontweight, va='top', wrap=True,
                        zorder=10, transform=ax.transAxes)
                y_pos -= 0.06

        # Clinical summary box
        ax.text(0.05, 0.12, f'H&Y: {case["hoehn_yahr"]}  |  '
                f'UPDRS-III: {case["updrs_part3"]}  |  '
                f'Duration: {case["disease_duration"]}yr  |  '
                f'Subtype: {case["motor_subtype"]}',
                fontsize=8, va='top', color='#666', style='italic',
                zorder=10, transform=ax.transAxes,
                bbox=dict(facecolor='#f5f5f5', edgecolor='#ddd', boxstyle='round,pad=0.5'))

        # Border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(color)
            spine.set_linewidth(2)

    fig.suptitle('Groq LLM Clinical DBS Candidacy Reports (Real Patients)',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = os.path.join(FIG_DIR, "fig8_groq_reports.png")
    fig.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor='white')
    plt.close(fig)
    print(f"  Saved: {fig_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
