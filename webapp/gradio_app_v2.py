#!/usr/bin/env python3
"""
DBS Candidacy Screening Tool v2 - Polished Gradio Web Application
==================================================================
An Explainable Multi-Source AI Framework for Deep Brain Stimulation
Candidacy Screening in Parkinson's Disease

Features:
  - Tab 1: Interactive DBS screening with real-time SHAP explanations
  - Tab 2: Paper results dashboard (ROC, SHAP beeswarms, model comparison)
  - Tab 3: Case study gallery (3 real patients with LLM reports)
  - Tab 4: About / methodology

Author: Kartic Mishra, Gachon University | JBI 2026
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import shap
import gradio as gr
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
FIG_DIR = PROJECT_ROOT / "results" / "figures" / "2026-03-20"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
print(f"[JBI App] APP_DIR: {APP_DIR}")
print(f"[JBI App] PROJECT_ROOT: {PROJECT_ROOT}")
print(f"[JBI App] FIG_DIR exists: {FIG_DIR.exists()}")

# Load top-10 model (same as paper's lead result: AUC=0.87 LOOCV)
model = joblib.load(APP_DIR / "xgb_top10_model.joblib")
features = joblib.load(APP_DIR / "xgb_top10_features.joblib")
model_scaler = joblib.load(APP_DIR / "xgb_top10_scaler.joblib")
model_imputer = joblib.load(APP_DIR / "xgb_top10_imputer.joblib")
explainer = shap.TreeExplainer(model)
print(f"[JBI App] Loaded top-10 model: {features}")

# Load Groq API key
GROQ_KEY = ""
try:
    import yaml
    cfg_path = PROJECT_ROOT / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        GROQ_KEY = cfg.get("groq", {}).get("api_key", "")
        if GROQ_KEY == "YOUR_GROQ_API_KEY":
            GROQ_KEY = ""
except Exception:
    pass

# Load results data
def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

df_results = safe_read_csv(TABLE_DIR / "2026-03-20" / "supplementary" / "clinical_model_results.csv")
df_modality = safe_read_csv(TABLE_DIR / "2026-03-20" / "supplementary" / "modality_model_results.csv")
df_groq = safe_read_csv(TABLE_DIR / "2026-03-20" / "supplementary" / "groq_reports_real.csv")
df_shap_clin = safe_read_csv(TABLE_DIR / "2026-03-20" / "supplementary" / "shap_clinical_importance.csv")
df_mann_whitney = safe_read_csv(TABLE_DIR / "2026-03-20" / "supplementary" / "mann_whitney_v2.csv")

# Feature display names (top-10 from XGBoost importance selection)
FEATURE_LABELS = {
    "MDSUPDRS_3-4-L": "Left finger tapping (III-4-L)",
    "MDSUPDRS_3-11": "Freezing of gait (III-11)",
    "disease_duration_years": "Disease duration (years)",
    "MDSUPDRS_3-6-R": "Right pronation-supination (III-6-R)",
    "asymmetry_toe_tapping": "Toe tapping asymmetry",
    "asymmetry_finger_tapping": "Finger tapping asymmetry",
    "MDSUPDRS_3-4-R": "Right finger tapping (III-4-R)",
    "subdomain_bradykinesia_UE": "Upper-limb bradykinesia",
    "MDSUPDRS_1-1": "Cognitive impairment (I-1)",
    "total_asymmetry": "Total motor asymmetry",
}

# ══════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
/* Global */
.gradio-container { max-width: 1400px !important; }

/* Plotly chart containers - white card with rounded corners */
.plot-container, .js-plotly-plot { border-radius: 10px !important; overflow: hidden; }

/* ═══════════════ LIGHT MODE (default) ═══════════════ */

/* Header */
.app-header {
    background: #00332b;
    color: #e0f2f1;
    padding: 2rem 2.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    border-left: 5px solid #4db6ac;
}
.app-header h1 {
    font-size: 1.9rem; margin: 0 0 0.5rem 0; font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.3px;
}
.app-header p {
    font-size: 0.95rem; margin: 0.2rem 0;
    color: #b2dfdb;
    line-height: 1.5;
}

/* Risk cards */
.risk-high {
    background: linear-gradient(135deg, #ffebee, #ffcdd2);
    border-left: 5px solid #d32f2f;
    padding: 1.2rem 1.5rem; border-radius: 8px;
    color: #333;
}
.risk-high h2 { color: #c62828; }
.risk-moderate {
    background: linear-gradient(135deg, #fff3e0, #ffe0b2);
    border-left: 5px solid #f57c00;
    padding: 1.2rem 1.5rem; border-radius: 8px;
    color: #333;
}
.risk-moderate h2 { color: #e65100; }
.risk-low {
    background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
    border-left: 5px solid #2e7d32;
    padding: 1.2rem 1.5rem; border-radius: 8px;
    color: #333;
}
.risk-low h2 { color: #1b5e20; }

/* Stat cards */
.stat-card {
    background: linear-gradient(135deg, #e0f2f1, #ffffff);
    border: 1px solid #b2dfdb;
    border-radius: 10px;
    padding: 1.2rem;
    text-align: center;
    transition: box-shadow 200ms ease;
}
.stat-card:hover { box-shadow: 0 4px 16px rgba(0,77,64,0.12); }
.stat-card h3 { font-size: 2rem; color: #004d40; margin: 0; }
.stat-card p { font-size: 0.85rem; color: #666; margin: 0.3rem 0 0 0; }

/* Case cards */
.case-card {
    border: 2px solid #e0e0e0;
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    background: #ffffff;
    color: #333;
}
.case-high { border-color: #d32f2f; }
.case-moderate { border-color: #f57c00; }
.case-low { border-color: #2e7d32; }
.case-card pre { color: #333; }
.report-box {
    background: #f5f5f5;
    padding: 1rem;
    border-radius: 6px;
    margin-top: 0.5rem;
}

/* Footer */
.app-footer {
    text-align: center;
    color: #888;
    font-size: 0.82rem;
    margin-top: 2rem;
    padding: 1.2rem;
    border-top: 1px solid #e0e0e0;
}

/* ═══════════════ DARK MODE ═══════════════ */

/* Header - slightly deeper in dark mode */
.dark .app-header {
    background: #001a15 !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5) !important;
    border-left-color: #80cbc4 !important;
}

/* Risk cards */
.dark .risk-high {
    background: linear-gradient(135deg, #3e1215, #4a1a1d) !important;
    color: #f5c6cb !important;
}
.dark .risk-high h2 { color: #ef9a9a !important; }
.dark .risk-high p { color: #e0b0b0 !important; }
.dark .risk-moderate {
    background: linear-gradient(135deg, #3e2c10, #4a3415) !important;
    color: #ffe0b2 !important;
}
.dark .risk-moderate h2 { color: #ffcc80 !important; }
.dark .risk-moderate p { color: #e0c8a0 !important; }
.dark .risk-low {
    background: linear-gradient(135deg, #102e18, #1a3e22) !important;
    color: #c8e6c9 !important;
}
.dark .risk-low h2 { color: #a5d6a7 !important; }
.dark .risk-low p { color: #b0d0b0 !important; }

/* Stat cards */
.dark .stat-card {
    background: linear-gradient(135deg, #1a2e28, #1e3830) !important;
    border-color: #2e4e44 !important;
}
.dark .stat-card h3 { color: #80cbc4 !important; }
.dark .stat-card p { color: #9ca3af !important; }

/* Case cards */
.dark .case-card {
    background: #1e1e2e !important;
    border-color: #3a3a50 !important;
    color: #d0d0d0 !important;
}
.dark .case-card p { color: #b0b0b0 !important; }
.dark .case-card .report-box {
    background: #111118 !important;
}
.dark .case-card .report-box pre {
    color: #d0d0d0 !important;
}
.dark .case-high { border-color: #ef5350 !important; }
.dark .case-moderate { border-color: #ffa726 !important; }
.dark .case-low { border-color: #66bb6a !important; }

/* Footer */
.dark .app-footer {
    color: #888 !important;
    border-top-color: #333 !important;
}

/* Tables */
.dark .dataframe th {
    background: #1e2e28 !important;
    color: #e0e0e0 !important;
}
.dark .dataframe td {
    color: #ccc !important;
}
"""


# ══════════════════════════════════════════════════════════════════════
# PLOTLY WHITE CARD CONSTANTS (readable in both light and dark mode)
# ══════════════════════════════════════════════════════════════════════

PLOT_BG = "#ffffff"
PLOT_FONT = dict(family="Inter, Arial, sans-serif", size=12, color="#333333")
PLOT_GRID = "#eeeeee"


def apply_white_card(fig):
    """Apply opaque white background + explicit text colors to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=PLOT_FONT,
        xaxis=dict(gridcolor=PLOT_GRID),
        yaxis=dict(gridcolor=PLOT_GRID),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════
# PREDICTION ENGINE
# ══════════════════════════════════════════════════════════════════════

def predict_dbs(finger_L, fog, disease_duration, pronation_R, asym_toe,
                asym_finger, finger_R, brady_UE, cognitive, total_asym):

    vals = {
        "MDSUPDRS_3-4-L": float(finger_L),
        "MDSUPDRS_3-11": float(fog),
        "disease_duration_years": float(disease_duration),
        "MDSUPDRS_3-6-R": float(pronation_R),
        "asymmetry_toe_tapping": float(asym_toe),
        "asymmetry_finger_tapping": float(asym_finger),
        "MDSUPDRS_3-4-R": float(finger_R),
        "subdomain_bradykinesia_UE": float(brady_UE),
        "MDSUPDRS_1-1": float(cognitive),
        "total_asymmetry": float(total_asym),
    }

    X_raw = np.array([[vals[f] for f in features]], dtype=np.float32)
    X = model_imputer.transform(X_raw)
    X = model_scaler.transform(X)
    prob = float(model.predict_proba(X)[0, 1])
    pct = prob * 100

    # Risk tier
    if prob > 0.70:
        tier, css, icon, rec = "HIGH RISK", "risk-high", "&#x1F534;", "Recommend referral to DBS specialist for comprehensive evaluation."
    elif prob > 0.30:
        tier, css, icon, rec = "MODERATE RISK", "risk-moderate", "&#x1F7E1;", "Consider monitoring with repeat assessment in 6 months. Discuss DBS possibility."
    else:
        tier, css, icon, rec = "LOW RISK", "risk-low", "&#x1F7E2;", "Continue current treatment. Reassess if symptoms significantly progress."

    # Risk card HTML
    risk_html = f"""<div class="{css}">
<h2 style="margin:0 0 0.5rem 0;">{icon} {tier} - {pct:.1f}% DBS Probability</h2>
<p style="margin:0; font-size:0.95rem;">{rec}</p>
</div>"""

    # SHAP values
    shap_vals = explainer(X)
    sv = shap_vals.values[0, :, 1] if shap_vals.values.ndim == 3 else shap_vals.values[0]

    # SHAP waterfall (Plotly)
    abs_sv = np.abs(sv)
    top_idx = np.argsort(abs_sv)[::-1][:12]

    names = [FEATURE_LABELS.get(features[i], features[i]) for i in top_idx]
    values = [sv[i] for i in top_idx]
    colors = ["#c62828" if v > 0 else "#00695c" for v in values]

    fig_shap = go.Figure(go.Bar(
        y=names[::-1], x=values[::-1],
        orientation="h",
        marker=dict(color=colors[::-1], line=dict(width=0.5, color="#333")),
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>"
    ))
    fig_shap.update_layout(
        title=dict(text="Feature contributions to this prediction", font=dict(size=14, color="#333")),
        xaxis_title="SHAP value (impact on DBS probability)",
        yaxis_title="",
        height=420, margin=dict(l=200, r=30, t=50, b=50),
        shapes=[dict(type="line", x0=0, x1=0, y0=-0.5, y1=len(top_idx)-0.5,
                     line=dict(color="#999", width=1, dash="dash"))],
    )
    apply_white_card(fig_shap)
    fig_shap.add_annotation(x=max(values)*0.6, y=len(top_idx)-1,
                            text="Red = increases risk", showarrow=False,
                            font=dict(color="#c62828", size=10))
    fig_shap.add_annotation(x=min(values)*0.6, y=len(top_idx)-2,
                            text="Teal = decreases risk", showarrow=False,
                            font=dict(color="#00695c", size=10))

    # Top drivers markdown
    drivers = []
    for rank, idx in enumerate(top_idx[:5], 1):
        fname = FEATURE_LABELS.get(features[idx], features[idx])
        direction = "increases" if sv[idx] > 0 else "decreases"
        icon_d = "&#x1F53A;" if sv[idx] > 0 else "&#x1F53B;"
        drivers.append(f"{rank}. {icon_d} **{fname}** = {vals[features[idx]]:.2f} - {direction} risk (SHAP: {sv[idx]:+.4f})")

    drivers_md = "\n\n".join(drivers)

    # Clinical summary
    profile = []
    if brady_UE > 8: profile.append("bradykinetic")
    if fog > 1: profile.append("gait-impaired")
    if abs(asym_finger) > 0.3 or abs(asym_toe) > 0.3: profile.append("asymmetric")
    if not profile: profile = ["mild motor symptoms"]

    summary_md = f"""### Clinical summary

| Parameter | Value |
|-----------|-------|
| PD duration | {int(disease_duration)} years |
| Upper-limb bradykinesia | {brady_UE:.0f} |
| Finger tapping (L/R) | {finger_L:.0f} / {finger_R:.0f} |
| Freezing of gait | {fog:.0f} |
| Motor asymmetry | {total_asym:.2f} |
| Motor profile | {', '.join(profile).capitalize()} |
| DBS probability | **{pct:.1f}%** ({tier}) |

---
*Top-10 feature model (LOOCV AUC = 0.87). Research prototype, not for clinical use.*"""

    return risk_html, fig_shap, drivers_md, summary_md


# ══════════════════════════════════════════════════════════════════════
# GROQ LLM REPORT
# ══════════════════════════════════════════════════════════════════════

def generate_groq_report(finger_L, fog, disease_duration, pronation_R, asym_toe,
                         asym_finger, finger_R, brady_UE, cognitive, total_asym):
    if not GROQ_KEY:
        return "Groq API key not configured. Set groq.api_key in config.yaml to enable LLM reports."

    try:
        from groq import Groq
        client = Groq(api_key=GROQ_KEY)

        vals = {
            "MDSUPDRS_3-4-L": float(finger_L), "MDSUPDRS_3-11": float(fog),
            "disease_duration_years": float(disease_duration),
            "MDSUPDRS_3-6-R": float(pronation_R),
            "asymmetry_toe_tapping": float(asym_toe),
            "asymmetry_finger_tapping": float(asym_finger),
            "MDSUPDRS_3-4-R": float(finger_R),
            "subdomain_bradykinesia_UE": float(brady_UE),
            "MDSUPDRS_1-1": float(cognitive),
            "total_asymmetry": float(total_asym),
        }
        X_raw = np.array([[vals[f] for f in features]], dtype=np.float32)
        X = model_scaler.transform(model_imputer.transform(X_raw))
        prob = float(model.predict_proba(X)[0, 1]) * 100
        tier = "HIGH" if prob > 70 else "MODERATE" if prob > 30 else "LOW"

        profile = []
        if brady_UE > 8: profile.append("bradykinetic")
        if fog > 1: profile.append("gait-impaired")
        if not profile: profile = ["mild"]

        system_prompt = (
            "You are a board-certified movement disorder neurologist specializing in DBS "
            "patient selection. Generate a concise evidence-based DBS candidacy report. "
            "Format: [DBS_Score][Motor_Profile][Key_Drivers][Recommendation][Caution_Flags]. "
            "Length: 120-160 words. Tone: clinical, direct."
        )
        user_prompt = (
            f"DBS Prob: {prob:.1f}% ({tier}) | "
            f"Duration: {int(disease_duration)}yr | Bradykinesia UE: {brady_UE:.0f} | "
            f"Finger tap L/R: {finger_L:.0f}/{finger_R:.0f} | FOG: {fog:.0f} | "
            f"Motor asymmetry: {total_asym:.2f} | Cognitive: {cognitive:.0f} | "
            f"Motor: {', '.join(profile)}"
        )

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=300, temperature=0.3
        )
        report = response.choices[0].message.content
        return f"### Llama 3.3 70B Clinical Report (via Groq)\n\n{report}\n\n---\n*Generated by Llama 3.3 70B. For research purposes only.*"

    except Exception as e:
        return f"Error generating report: {str(e)}"


# ══════════════════════════════════════════════════════════════════════
# RESULTS DASHBOARD
# ══════════════════════════════════════════════════════════════════════

def build_model_comparison_plot():
    """AUC comparison across all datasets."""
    data = [
        ("XGBoost Top-10\n(WearGait-PD)", 0.87, 0.79, 0.95, "#004d40"),
        ("SVM Top-10\n(WearGait-PD)", 0.85, 0.75, 0.93, "#00897b"),
        ("XGBoost\n(PADS Wearable)", 0.86, 0.82, 0.90, "#0277bd"),
        ("MLP\n(PADS Wearable)", 0.86, 0.81, 0.90, "#4fc3f7"),
        ("XGBoost\n(GaitPDB Gait)", 0.99, 0.97, 1.00, "#2e7d32"),
        ("MLP\n(UCI Voice)", 0.97, 0.95, 0.99, "#7b1fa2"),
    ]

    names = [d[0] for d in data]
    aucs = [d[1] for d in data]
    lows = [d[1] - d[2] for d in data]
    highs = [d[3] - d[1] for d in data]
    colors = [d[4] for d in data]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=aucs,
        error_y=dict(type="data", symmetric=False, array=highs, arrayminus=lows,
                     color="#333", thickness=1.5),
        marker=dict(color=colors, line=dict(width=1, color="#333")),
        hovertemplate="<b>%{x}</b><br>AUC: %{y:.2f}<extra></extra>",
    ))
    # Add text annotations above error bars
    for i, (name, auc, hi) in enumerate(zip(names, aucs, highs)):
        fig.add_annotation(x=name, y=auc + hi + 0.02, text=f"{auc:.2f}",
                           showarrow=False, font=dict(size=11, color="#333"),
                           yanchor="bottom")
    fig.add_shape(type="line", x0=-0.5, x1=5.5, y0=0.5, y1=0.5,
                  line=dict(color="#999", dash="dot", width=1))
    fig.add_annotation(x=5.5, y=0.5, text="Chance (0.5)", showarrow=False,
                       font=dict(size=9, color="#999"), xanchor="right")
    fig.update_layout(
        title=dict(text="Model performance across datasets (AUC-ROC with 95% CI)", font=dict(color="#333")),
        yaxis=dict(title="AUC-ROC", range=[0.4, 1.10]),
        xaxis_title="",
        height=450,
        margin=dict(t=60, b=100),
        showlegend=False,
    )
    apply_white_card(fig)
    return fig


def build_shap_importance_plot():
    """Top clinical features by SHAP importance."""
    if df_shap_clin.empty:
        return go.Figure().add_annotation(text="SHAP data not available", showarrow=False)

    df = df_shap_clin.head(15).copy()
    if "feature" not in df.columns or "mean_abs_shap" not in df.columns:
        cols = df.columns.tolist()
        if len(cols) >= 2:
            df.columns = ["feature", "mean_abs_shap"] + cols[2:]

    names = [FEATURE_LABELS.get(f, f.replace("_", " ").title()[:30]) for f in df["feature"]]
    vals = df["mean_abs_shap"].values

    fig = go.Figure(go.Bar(
        y=names[::-1], x=vals[::-1],
        orientation="h",
        marker=dict(
            color=vals[::-1],
            colorscale=[[0, "#b2dfdb"], [0.5, "#00897b"], [1, "#004d40"]],
            line=dict(width=0.5, color="#333"),
        ),
        hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Top 15 clinical features by SHAP importance (WearGait-PD)", font=dict(color="#333")),
        xaxis_title="Mean |SHAP value|",
        height=500, margin=dict(l=250, r=30, t=50, b=50),
    )
    apply_white_card(fig)
    return fig


def build_dataset_pie():
    """Dataset sizes donut chart."""
    labels = ["WearGait-PD (82)", "PADS (370)", "GaitPDB (165)", "UCI Voice (195)"]
    values = [82, 370, 165, 195]
    colors_pie = ["#004d40", "#0277bd", "#2e7d32", "#7b1fa2"]

    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.4,
                           marker=dict(colors=colors_pie),
                           textinfo="label+value", textfont=dict(size=11),
                           hovertemplate="<b>%{label}</b><br>n = %{value}<extra></extra>"))
    fig.update_layout(title=dict(text="Dataset sizes (n = 812 total)", font=dict(size=14, color="#333")),
                      height=350, showlegend=False,
                      margin=dict(t=50, b=20, l=20, r=20),
                      paper_bgcolor=PLOT_BG, font=PLOT_FONT)
    return fig


def build_auc_by_modality():
    """AUC by modality bar chart."""
    modalities = ["Clinical", "Wearable", "Gait", "Voice"]
    aucs = [0.87, 0.86, 0.99, 0.97]
    colors_bar = ["#004d40", "#0277bd", "#2e7d32", "#7b1fa2"]

    fig = go.Figure(go.Bar(x=modalities, y=aucs,
                           marker=dict(color=colors_bar, line=dict(width=1, color="#333")),
                           text=[f"{a:.2f}" for a in aucs],
                           textposition="outside",
                           textfont=dict(color="#333333", size=12),
                           width=0.55))
    fig.update_layout(title=dict(text="Best AUC by modality", font=dict(size=14, color="#333")),
                      yaxis=dict(title="AUC-ROC", range=[0.7, 1.08]),
                      height=350, showlegend=False,
                      margin=dict(t=50, b=40, l=50, r=30),
                      paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG, font=PLOT_FONT,
                      xaxis=dict(gridcolor=PLOT_GRID), yaxis_gridcolor=PLOT_GRID)
    return fig


# ══════════════════════════════════════════════════════════════════════
# CASE STUDY TAB
# ══════════════════════════════════════════════════════════════════════

def get_case_studies():
    if df_groq.empty:
        return "Case study data not available."

    cases = []
    colors = {"HIGH": "#d32f2f", "BORDERLINE": "#f57c00", "LOW": "#2e7d32"}
    css_classes = {"HIGH": "case-high", "BORDERLINE": "case-moderate", "LOW": "case-low"}

    for _, row in df_groq.iterrows():
        cat = row.get("category", "UNKNOWN")
        pid = row.get("patient_id", "N/A")
        prob = row.get("dbs_prob", 0) * 100
        actual = "DBS+" if row.get("actual_dbs", 0) == 1 else "DBS-"
        hy = row.get("hoehn_yahr", "N/A")
        updrs = row.get("updrs_part3", "N/A")
        dur = row.get("disease_duration", "N/A")
        subtype = row.get("motor_subtype", "N/A")
        report = row.get("report_text", "No report available.")
        color = colors.get(cat, "#999")
        css_cls = css_classes.get(cat, "")

        cases.append(f"""<div class="case-card {css_cls}">
<h3 style="color:{color}; margin:0 0 0.5rem 0;">Patient {pid} - {cat} RISK ({prob:.1f}%)</h3>
<p><b>Actual outcome:</b> {actual} | <b>H&Y:</b> {hy} | <b>UPDRS-III:</b> {updrs} | <b>Duration:</b> {dur} yr | <b>Subtype:</b> {subtype}</p>
<div class="report-box">
<pre style="white-space:pre-wrap; font-size:0.85rem; font-family:inherit; margin:0;">{report}</pre>
</div>
</div>""")

    return "\n".join(cases)


# ══════════════════════════════════════════════════════════════════════
# ABOUT TAB
# ══════════════════════════════════════════════════════════════════════

ABOUT_MD = """
## About this tool

This interactive demonstration accompanies the manuscript:

> **"Explainable multi-source AI framework for deep brain stimulation candidacy screening in Parkinson's disease"**
>
> Kartic, Gachon University

### How it works

1. **Input** clinical parameters (MDS-UPDRS items, disease duration, motor asymmetry indices)
2. **XGBoost classifier** predicts DBS candidacy probability using the top-10 selected clinical features
3. **SHAP TreeExplainer** computes per-feature contributions to explain the prediction
4. **Risk stratification** classifies patients as HIGH (>70%), MODERATE (30-70%), or LOW (<30%)
5. **LLM report** (optional) generates a structured clinical narrative via Groq Llama 3.3 70B

### Key results

| Metric | Value |
|--------|-------|
| Primary AUC (LOOCV, n=82) | 0.87 (95% CI: 0.79-0.95) |
| Permutation test | p = 0.003 |
| Wearable validation (PADS, n=370) | AUC = 0.86 |
| Gait validation (GaitPDB, n=165) | AUC = 0.99 |
| Voice validation (UCI, n=195) | AUC = 0.97 |
| Total subjects across 4 datasets | 812 |

### Datasets

- **WearGait-PD** (FDA CDRH): 82 PD patients, 23 DBS+, real DBS labels
- **PADS** (PhysioNet): 370 subjects, 100 Hz smartwatch IMU, 11 motor tasks
- **GaitPDB** (PhysioNet): 165 subjects, 100 Hz bilateral force plates
- **UCI Parkinson's** (UCI MLR): 195 recordings, 22 acoustic voice features

### Limitations

- Small DBS-positive cohort (n = 23) limits generalizability
- Model trained on WearGait-PD clinical scores only (not raw sensor data)
- No prospective clinical validation
- Research prototype: **not intended for clinical use**

### Contact

Kartic | Gachon University, Republic of Korea
"""


# ══════════════════════════════════════════════════════════════════════
# GRADIO APP
# ══════════════════════════════════════════════════════════════════════

with gr.Blocks(
    title="DBS Candidacy Screening Tool",
    css=CUSTOM_CSS,
    theme=gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="emerald",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
) as demo:

    # ── Header ────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="app-header">
        <p style="opacity:0.6; font-size:0.75rem; margin:0 0 0.4rem 0; letter-spacing:1px; text-transform:uppercase;">
        Pre-Implant Screening &middot; Clinical + Wearable + Gait + Voice</p>
        <h1>DBS Candidacy Screening Tool</h1>
        <p>An Explainable Multi-Source AI Framework for Deep Brain Stimulation
        Candidacy Screening in Parkinson's Disease</p>
        <p style="opacity:0.7; font-size:0.85rem; margin-top:0.3rem;">
        Kartic | Gachon University</p>
    </div>
    """)

    with gr.Tabs():

        # ══════════════════════════════════════════════════════════════
        # TAB 1: SCREENING TOOL
        # ══════════════════════════════════════════════════════════════
        with gr.Tab("Screening tool", id="screening"):
            with gr.Row(equal_height=False):

                # LEFT: Patient inputs
                with gr.Column(scale=2, min_width=350):
                    gr.Markdown("### Patient parameters (top-10 features)")

                    with gr.Group():
                        gr.Markdown("**Disease history**")
                        dur_in = gr.Slider(0, 30, value=6, step=0.5, label="Disease duration (years)")

                    with gr.Group():
                        gr.Markdown("**MDS-UPDRS motor items**")
                        with gr.Row():
                            finger_L_in = gr.Slider(0, 4, value=1, step=1, label="Finger tapping - Left (III-4-L)")
                            finger_R_in = gr.Slider(0, 4, value=1, step=1, label="Finger tapping - Right (III-4-R)")
                        with gr.Row():
                            pronation_R_in = gr.Slider(0, 4, value=1, step=1, label="Pronation-supination - Right (III-6-R)")
                            fog_in = gr.Slider(0, 4, value=0, step=1, label="Freezing of gait (III-11)")
                        brady_UE_in = gr.Slider(0, 20, value=6, step=1, label="Upper-limb bradykinesia sub-score")
                        cognitive_in = gr.Slider(0, 4, value=0, step=1, label="Cognitive impairment (I-1)")

                    with gr.Group():
                        gr.Markdown("**Motor asymmetry indices** (-1 to 1; 0 = symmetric)")
                        with gr.Row():
                            asym_finger_in = gr.Slider(-1, 1, value=0.0, step=0.05, label="Finger tapping asymmetry")
                            asym_toe_in = gr.Slider(-1, 1, value=0.0, step=0.05, label="Toe tapping asymmetry")
                        total_asym_in = gr.Slider(0, 0.7, value=0.3, step=0.02, label="Total motor asymmetry")

                # RIGHT: Results
                with gr.Column(scale=3, min_width=500):
                    risk_output = gr.HTML(label="Risk assessment")
                    shap_output = gr.Plot(label="SHAP feature contributions")
                    gr.Markdown("### Top 5 risk drivers")
                    drivers_output = gr.Markdown()
                    summary_output = gr.Markdown()

                    with gr.Accordion("Generate LLM clinical report (Groq Llama 3.3 70B)", open=False):
                        llm_btn = gr.Button("Generate report", variant="secondary", size="sm")
                        llm_output = gr.Markdown()

            # Wire inputs (same order as predict_dbs arguments)
            all_inputs = [
                finger_L_in, fog_in, dur_in, pronation_R_in, asym_toe_in,
                asym_finger_in, finger_R_in, brady_UE_in, cognitive_in, total_asym_in,
            ]
            all_outputs = [risk_output, shap_output, drivers_output, summary_output]

            for inp in all_inputs:
                inp.change(fn=predict_dbs, inputs=all_inputs, outputs=all_outputs)
            demo.load(fn=predict_dbs, inputs=all_inputs, outputs=all_outputs)

            llm_btn.click(fn=generate_groq_report, inputs=all_inputs, outputs=llm_output)

        # ══════════════════════════════════════════════════════════════
        # TAB 2: RESULTS DASHBOARD
        # ══════════════════════════════════════════════════════════════
        with gr.Tab("Results dashboard", id="results"):
            gr.Markdown("### Paper results at a glance")

            # Key metrics row
            gr.HTML("""
            <div style="display:grid; grid-template-columns: repeat(5, 1fr); gap:1rem; margin:1rem 0;">
                <div class="stat-card"><h3>0.87</h3><p>Primary AUC<br>(LOOCV)</p></div>
                <div class="stat-card"><h3>0.86</h3><p>Wearable AUC<br>(PADS)</p></div>
                <div class="stat-card"><h3>0.99</h3><p>Gait AUC<br>(GaitPDB)</p></div>
                <div class="stat-card"><h3>0.97</h3><p>Voice AUC<br>(UCI)</p></div>
                <div class="stat-card"><h3>812</h3><p>Total subjects<br>(4 datasets)</p></div>
            </div>
            """)

            with gr.Row():
                with gr.Column():
                    gr.Plot(value=build_dataset_pie, label="Dataset sizes")
                with gr.Column():
                    gr.Plot(value=build_auc_by_modality, label="AUC by modality")

            with gr.Row():
                with gr.Column():
                    gr.Plot(value=build_model_comparison_plot, label="Model comparison")
                with gr.Column():
                    gr.Plot(value=build_shap_importance_plot, label="SHAP feature importance")

            # Manuscript figures gallery
            gr.Markdown("### Manuscript figures")
            with gr.Row():
                for fname, label in [
                    ("fig4_roc_curves.png", "ROC curves"),
                    ("fig9_calibration_dca.png", "Calibration + DCA"),
                    ("fig5_multisource_validation.png", "Multi-source validation"),
                ]:
                    fpath = FIG_DIR / fname
                    if fpath.exists():
                        gr.Image(value=str(fpath), label=label, show_label=True)

            with gr.Row():
                for fname, label in [
                    ("shap_clinical_beeswarm.png", "SHAP - Clinical"),
                    ("shap_wearable_beeswarm.png", "SHAP - Wearable"),
                    ("shap_gait_beeswarm.png", "SHAP - Gait"),
                ]:
                    fpath = FIG_DIR / fname
                    if fpath.exists():
                        gr.Image(value=str(fpath), label=label, show_label=True)

        # ══════════════════════════════════════════════════════════════
        # TAB 3: CASE STUDIES
        # ══════════════════════════════════════════════════════════════
        with gr.Tab("Case studies", id="cases"):
            gr.Markdown("### Real patient case studies from WearGait-PD")
            gr.Markdown("Three representative patients with LLM-generated clinical screening reports "
                        "(Groq Llama 3.3 70B). These are real patients with actual DBS outcomes.")
            gr.HTML(value=get_case_studies)

            # Show Groq reports figure if available
            groq_fig = FIG_DIR / "fig8_groq_reports.png"
            if groq_fig.exists():
                gr.Image(value=str(groq_fig), label="LLM report cards (Fig. 5 in manuscript)",
                         show_label=True)

        # ══════════════════════════════════════════════════════════════
        # TAB 4: ABOUT
        # ══════════════════════════════════════════════════════════════
        with gr.Tab("About", id="about"):
            gr.Markdown(ABOUT_MD)

            # Study flowchart
            flow_fig = FIG_DIR / "fig1_study_flowchart.png"
            if flow_fig.exists():
                gr.Image(value=str(flow_fig), label="Study design (Fig. 1)",
                         show_label=True)

    # ── Footer ────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="app-footer">
        <b>DBS Candidacy Screening Tool</b> v2.0 | Pre-Implant Screening | Research Prototype<br>
        Clinical + Wearable + Gait + Voice Biomarkers<br>
        Kartic | Gachon University<br>
        <em>For research purposes only. Not intended for clinical decision-making without proper validation.</em>
    </div>
    """)


# ══════════════════════════════════════════════════════════════════════
# LAUNCH
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
