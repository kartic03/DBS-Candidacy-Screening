#!/usr/bin/env python3
"""
Statistical Tests (Revised) for JBI Paper
===========================================
1. Mann-Whitney U + BH-FDR for DBS+ vs DBS- feature comparisons
2. Motor subtype analysis (tremor-dominant vs akinetic-rigid)
3. Post-hoc power analysis for DeLong tests
4. Permutation test summary

Output:
  results/tables/mann_whitney_v2.csv
  results/tables/motor_subtype_v2.csv
  results/tables/power_analysis.csv

Author: Kartic Mishra, Gachon University
"""

import os
import warnings
import multiprocessing

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection

warnings.filterwarnings("ignore")

N_CORES = multiprocessing.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(N_CORES)
SEED = 42

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "tables")

META_COLS = [
    "subject_id", "cohort", "dbs_candidate", "dbs_bilateral",
    "dbs_electrode", "dbs_years_since_surgery", "motor_subtype", "med_state_on"
]


def main():
    print("=" * 70)
    print("Statistical Tests (Revised)")
    print("=" * 70)

    # ── Load data ────────────────────────────────────────────────────────
    df = pd.read_csv(os.path.join(PROC_DIR, "clinical_features", "weargait_pd_only.csv"))
    feat_cols = [c for c in df.columns if c not in META_COLS
                 and pd.api.types.is_numeric_dtype(df[c])]
    y = df["dbs_candidate"].values

    # ═══════════════════════════════════════════════════════════════════════
    # 1. Mann-Whitney U + BH-FDR (DBS+ vs DBS-)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[1/3] Mann-Whitney U tests (DBS+ vs DBS-)...")
    dbs_pos = df[df["dbs_candidate"] == 1]
    dbs_neg = df[df["dbs_candidate"] == 0]

    mw_results = []
    for col in feat_cols:
        pos_vals = dbs_pos[col].dropna()
        neg_vals = dbs_neg[col].dropna()
        if len(pos_vals) < 3 or len(neg_vals) < 3:
            continue
        u_stat, p_val = stats.mannwhitneyu(pos_vals, neg_vals, alternative='two-sided')
        effect_size = abs(pos_vals.median() - neg_vals.median()) / (
            pd.concat([pos_vals, neg_vals]).std() + 1e-12
        )
        mw_results.append({
            "feature": col,
            "DBS+_median": pos_vals.median(),
            "DBS-_median": neg_vals.median(),
            "U_statistic": u_stat,
            "p_value": p_val,
            "effect_size_d": effect_size,
            "DBS+_n": len(pos_vals),
            "DBS-_n": len(neg_vals),
        })

    df_mw = pd.DataFrame(mw_results).sort_values("p_value")

    # BH-FDR correction
    reject, pvals_corrected = fdrcorrection(df_mw["p_value"].values, alpha=0.05)
    df_mw["p_fdr"] = pvals_corrected
    df_mw["significant_fdr05"] = reject

    n_sig = df_mw["significant_fdr05"].sum()
    print(f"  Tested: {len(df_mw)} features")
    print(f"  Significant (FDR<0.05): {n_sig}")

    # Top significant features
    sig_feats = df_mw[df_mw["significant_fdr05"]].head(15)
    print(f"\n  Top significant features:")
    for _, row in sig_feats.iterrows():
        print(f"    {row['feature']:<40} p_fdr={row['p_fdr']:.4f}  "
              f"d={row['effect_size_d']:.3f}  "
              f"DBS+={row['DBS+_median']:.2f} vs DBS-={row['DBS-_median']:.2f}")

    df_mw.to_csv(os.path.join(RESULTS_DIR, "mann_whitney_v2.csv"), index=False)
    print(f"  Saved: mann_whitney_v2.csv")

    # ═══════════════════════════════════════════════════════════════════════
    # 2. Motor Subtype Analysis
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[2/3] Motor subtype analysis...")
    subtypes = df["motor_subtype"].value_counts()
    print(f"  Subtype distribution: {subtypes.to_dict()}")

    subtype_results = []
    for subtype in df["motor_subtype"].unique():
        sub = df[df["motor_subtype"] == subtype]
        n_total = len(sub)
        n_dbs = int(sub["dbs_candidate"].sum())
        if n_total >= 5:
            dbs_rate = n_dbs / n_total
            subtype_results.append({
                "subtype": subtype,
                "n": n_total,
                "n_dbs_positive": n_dbs,
                "dbs_rate": dbs_rate,
                "mean_updrs_part3": sub["updrs_part3_total"].mean(),
                "mean_hy": sub["hoehn_yahr"].mean(),
                "mean_duration": sub["disease_duration_years"].mean(),
                "mean_total_asymmetry": sub["total_asymmetry"].mean(),
            })
            print(f"  {subtype}: n={n_total}, DBS+={n_dbs} ({dbs_rate:.1%}), "
                  f"UPDRS-III={sub['updrs_part3_total'].mean():.1f}")

    # Chi-square test: DBS rate by subtype
    ar = df[df["motor_subtype"] == "Akinetic-Rigid"]
    td = df[df["motor_subtype"] == "Tremor-Dominant"]
    if len(ar) >= 5 and len(td) >= 5:
        contingency = np.array([
            [int(ar["dbs_candidate"].sum()), int((1 - ar["dbs_candidate"]).sum())],
            [int(td["dbs_candidate"].sum()), int((1 - td["dbs_candidate"]).sum())]
        ])
        chi2, p_chi, _, _ = stats.chi2_contingency(contingency)
        print(f"\n  Chi-square (AR vs TD DBS rate): chi2={chi2:.3f}, p={p_chi:.4f}")
        for r in subtype_results:
            r["chi2_p_value"] = p_chi

    pd.DataFrame(subtype_results).to_csv(
        os.path.join(RESULTS_DIR, "motor_subtype_v2.csv"), index=False
    )
    print(f"  Saved: motor_subtype_v2.csv")

    # ═══════════════════════════════════════════════════════════════════════
    # 3. Post-hoc Power Analysis
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[3/3] Post-hoc power analysis...")

    # For DeLong test: what N would we need to detect AUC difference of 0.05?
    # Using normal approximation for AUC comparison
    auc1 = 0.878  # Best model
    auc2 = 0.846  # Second best
    delta = auc1 - auc2
    alpha = 0.05
    z_alpha = stats.norm.ppf(1 - alpha / 2)

    # Approximate variance of AUC (Hanley & McNeil)
    n_pos = 23
    n_neg = 59

    def auc_variance(auc, n1, n2):
        q1 = auc / (2 - auc)
        q2 = 2 * auc ** 2 / (1 + auc)
        return (auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) +
                (n2 - 1) * (q2 - auc ** 2)) / (n1 * n2)

    var1 = auc_variance(auc1, n_pos, n_neg)
    var2 = auc_variance(auc2, n_pos, n_neg)

    # Current power
    z_current = delta / np.sqrt(var1 + var2)
    power_current = 1 - stats.norm.cdf(z_alpha - z_current)

    # Required N for 80% power
    z_beta = stats.norm.ppf(0.8)
    required_ratio = ((z_alpha + z_beta) / (delta + 1e-12)) ** 2
    # Rough: multiply current N by ratio
    n_needed_pos = max(int(n_pos * required_ratio * (var1 + var2) / delta ** 2), n_pos)

    power_results = [{
        "comparison": "XGB_Top10 vs SVM_Top10",
        "AUC1": auc1, "AUC2": auc2, "delta": delta,
        "current_n_pos": n_pos, "current_n_neg": n_neg,
        "current_power": power_current,
        "needed_n_pos_80pct_power": min(n_needed_pos, 10000),
        "note": "Small delta + small N = underpowered"
    }]

    print(f"  Current power to detect AUC {auc1:.3f} vs {auc2:.3f}: {power_current:.3f}")
    print(f"  Estimated N(DBS+) needed for 80% power: ~{min(n_needed_pos, 10000)}")
    print(f"  Conclusion: Study underpowered for between-model DeLong comparisons")

    pd.DataFrame(power_results).to_csv(
        os.path.join(RESULTS_DIR, "power_analysis.csv"), index=False
    )
    print(f"  Saved: power_analysis.csv")

    # ═══════════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STATISTICAL TESTS COMPLETE")
    print("=" * 70)
    print(f"""
  Mann-Whitney U: {n_sig}/{len(df_mw)} features significant (FDR<0.05)
  Motor subtypes: AR={len(ar)} (DBS+ {ar['dbs_candidate'].sum()}), TD={len(td)} (DBS+ {td['dbs_candidate'].sum()})
  Power: {power_current:.3f} (underpowered for between-model DeLong)
  All results saved to {RESULTS_DIR}/
    """)
    print("Done.")


if __name__ == "__main__":
    main()
