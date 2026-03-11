import os
import json
from scipy.stats import spearmanr, kendalltau

RP_SCORES_DIR = "rp_scores"
RESULTS_DIR = "results_old"
OUTPUT_PATH = "correlation_results_captions_llama_0.json"

# File paths
RP_SCORES_PATH = os.path.join(RP_SCORES_DIR, "rp_scores_all_splits_llama.json")
RESULTS_NATURAL_PATH = os.path.join(RESULTS_DIR, "pairs-NATURAL-ALL-llava-v1.5-13b-CLUB_global.json")
RESULTS_SYNTHETIC_PATH = os.path.join(RESULTS_DIR, "pairs-SYNTHETIC-ALL-llava-v1.5-13b-CLUB_global.json")

# Load RP scores (structured as {"natural": {...}, "synthetic": {...}})
rp_data = json.load(open(RP_SCORES_PATH))

# Load EMI scores from results folder
emi_data = {
    "natural": json.load(open(RESULTS_NATURAL_PATH)),
    "synthetic": json.load(open(RESULTS_SYNTHETIC_PATH)),
}


def compute_correlations(rp_section, emi_section):
    """Compute Spearman and Kendall Tau correlations between RP and EMI scores."""
    emi_dict = emi_section["EMI"]

    # Use only datasets present in both RP and EMI
    common_keys = sorted(set(rp_section.keys()) & set(emi_dict.keys()))

    emi_scores = []
    rp_scores = []
    for key in common_keys:
        emi_scores.append(emi_dict[key])
        rp_scores.append(rp_section[key]["rp_score"])

    spearman_corr, spearman_pval = spearmanr(emi_scores, rp_scores)
    kendall_corr, kendall_pval = kendalltau(emi_scores, rp_scores)

    return {
        "num_datasets": len(common_keys),
        "datasets": common_keys,
        "Spearman Correlation": {
            "correlation": spearman_corr,
            "p-value": spearman_pval,
        },
        "Kendall Tau Correlation": {
            "correlation": kendall_corr,
            "p-value": kendall_pval,
        },
    }


# Calculate correlations separately for natural and synthetic shifts
results = {}
for shift_type in ["natural", "synthetic"]:
    results[shift_type] = compute_correlations(
        rp_data[shift_type], emi_data[shift_type]
    )

# Save results to a JSON file
with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))