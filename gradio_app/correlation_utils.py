"""
Correlation analysis utilities.
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau


def compute_all_correlations(
    emid_values: list[float],
    emid_ub_values: list[float],
    emi_values: list[float],
    rp_values: list[float],
) -> dict:
    """
    Compute:
      - Pearson correlation between EMID and EMID_UB
      - Spearman correlation between EMI and RP score
      - Kendall Tau between EMI and RP score

    Returns a dict with all results.
    """
    results = {}

    # --- EMID vs EMID_UB (Pearson) ---
    if len(emid_values) >= 2 and len(emid_ub_values) >= 2:
        pr, p_pval = pearsonr(emid_values, emid_ub_values)
        results["EMID_vs_EMID_UB"] = {
            "Pearson Correlation": round(pr, 6),
            "p-value": p_pval,
            "num_pairs": len(emid_values),
        }
    else:
        results["EMID_vs_EMID_UB"] = {"error": "Not enough data points (need >= 2)"}

    # --- EMI vs RP (Spearman + Kendall) ---
    if len(emi_values) >= 2 and len(rp_values) >= 2:
        sp_corr, sp_pval = spearmanr(emi_values, rp_values)
        kt_corr, kt_pval = kendalltau(emi_values, rp_values)
        results["EMI_vs_RP"] = {
            "Spearman Correlation": round(sp_corr, 6),
            "Spearman p-value": sp_pval,
            "Kendall Tau": round(kt_corr, 6),
            "Kendall p-value": kt_pval,
            "num_pairs": len(emi_values),
        }
    else:
        results["EMI_vs_RP"] = {"error": "Not enough data points (need >= 2)"}

    return results
