"""
Calculate AUC-ROC and AUC-PR scores for EMID detection (ID vs OOD).
Labels: 0 = D1 (ID), 1 = D2-D5 (OOD)
"""

import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, auc, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt


def load_emid_scores(results_dir: Path = Path("results/concept_drift_detection")) -> dict[str, list[float]]:
	"""Load EMID scores from JSON files."""
	emid_scores = {}
	
	# Load EMID scores file
	scores_file = results_dir / "emid_subset_pair_scores.json"
	
	if not scores_file.exists():
		raise FileNotFoundError(f"Scores file not found: {scores_file}")
	
	with open(scores_file) as f:
		data = json.load(f)
	
	# Extract EMID scores for each dataset
	# D1 is ID, D2-D5 are OOD
	emid_scores["D1_pairs_emid"] = data.get("D1_pairs_emid", [])
	emid_scores["D2_migrated"] = data.get("D2_migrated", [])
	emid_scores["D3_migrated"] = data.get("D3_migrated", [])
	emid_scores["D4_migrated"] = data.get("D4_migrated", [])
	emid_scores["D5_migrated"] = data.get("D5_migrated", [])
	
	return emid_scores


def calculate_auc_scores(emid_scores: dict[str, list[float]]) -> tuple[float, float]:
	"""Calculate AUC-ROC and AUC-PR scores."""
	
	# Prepare labels and scores
	y_true = []
	y_scores = []
	
	# D1 (ID) = label 0
	d1_scores = emid_scores.get("D1_pairs_emid", [])
	y_true.extend([0] * len(d1_scores))
	y_scores.extend(d1_scores)
	
	# D2-D5 (OOD) = label 1
	for key in ["D2_migrated", "D3_migrated", "D4_migrated", "D5_migrated"]:
		ood_scores = emid_scores.get(key, [])
		y_true.extend([1] * len(ood_scores))
		y_scores.extend(ood_scores)
	
	y_true = np.array(y_true)
	y_scores = np.array(y_scores)
	
	# Calculate AUC-ROC
	auc_roc = roc_auc_score(y_true, -y_scores)
	
	# Calculate ROC curve
	fpr, tpr, _ = roc_curve(y_true, -y_scores)
	
	# Calculate AUC-PR
	precision, recall, _ = precision_recall_curve(y_true, -y_scores)
	auc_pr = auc(recall, precision)
	
	return auc_roc, auc_pr, y_true, y_scores, fpr, tpr, precision, recall


def plot_auc_curves(
	auc_roc: float,
	auc_pr: float,
	fpr: np.ndarray,
	tpr: np.ndarray,
	precision: np.ndarray,
	recall: np.ndarray,
	output_dir: Path = Path("results"),
) -> None:
	"""Plot ROC and PR curves."""
	
	fig, axes = plt.subplots(1, 2, figsize=(15, 5))
	
	# ROC Curve
	axes[0].plot(fpr, tpr, color="darkorange", lw=2.5, label=f"ROC Curve (AUC = {auc_roc:.4f})")
	axes[0].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")
	axes[0].set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
	axes[0].set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
	axes[0].set_title("ROC Curve (D1 vs OOD)", fontsize=13, fontweight="bold")
	axes[0].legend(fontsize=11, loc="lower right")
	axes[0].grid(True, alpha=0.3, linestyle="--")
	axes[0].set_xlim([0.0, 1.0])
	axes[0].set_ylim([0.0, 1.05])
	
	# PR Curve
	axes[1].plot(recall, precision, color="darkgreen", lw=2.5, label=f"PR Curve (AUC = {auc_pr:.4f})")
	axes[1].axhline(y=0.5, color="navy", lw=2, linestyle="--", label="Random (~0.5 for balanced)")
	axes[1].set_xlabel("Recall", fontsize=12, fontweight="bold")
	axes[1].set_ylabel("Precision", fontsize=12, fontweight="bold")
	axes[1].set_title("Precision-Recall Curve (D1 vs OOD)", fontsize=13, fontweight="bold")
	axes[1].legend(fontsize=11, loc="best")
	axes[1].grid(True, alpha=0.3, linestyle="--")
	axes[1].set_xlim([0.0, 1.0])
	axes[1].set_ylim([0.0, 1.05])
	
	plt.tight_layout()
	output_path = output_dir / "auc_roc_pr_curves.png"
	plt.savefig(output_path, dpi=300, bbox_inches="tight")
	print(f"✓ Saved AUC curves plot: {output_path}")
	plt.close()


def main():
	"""Main function."""
	results_dir = Path("results/concept_drift_detection")
	
	print("=" * 70)
	print("AUC-ROC and AUC-PR Score Calculation")
	print("=" * 70)
	
	# Load EMID scores
	print("\n[1] Loading EMID scores...")
	emid_scores = load_emid_scores(results_dir)
	
	for key, scores in emid_scores.items():
		print(f"  {key}: {len(scores)} scores (mean={np.mean(scores):.6f})")
	
	# Calculate AUC scores
	print("\n[2] Calculating AUC scores...")
	auc_roc, auc_pr, y_true, y_scores, fpr, tpr, precision, recall = calculate_auc_scores(emid_scores)
	
	print(f"\n  AUC-ROC Score: {auc_roc:.6f}")
	print(f"  AUC-PR Score:  {auc_pr:.6f}")
	
	# Summary statistics
	print(f"\n[3] Summary Statistics:")
	print(f"  Total samples: {len(y_true)}")
	print(f"  ID (D1) samples: {np.sum(y_true == 0)}")
	print(f"  OOD (D2-D5) samples: {np.sum(y_true == 1)}")
	print(f"  Min EMID score: {np.min(y_scores):.6f}")
	print(f"  Max EMID score: {np.max(y_scores):.6f}")
	print(f"  Mean EMID score (ID): {np.mean(y_scores[y_true == 0]):.6f}")
	print(f"  Mean EMID score (OOD): {np.mean(y_scores[y_true == 1]):.6f}")
	
	# Plot AUC curves
	print("\n[4] Plotting AUC curves...")
	plot_auc_curves(auc_roc, auc_pr, fpr, tpr, precision, recall, results_dir)
	
	print("\n" + "=" * 70)
	print("✓ AUC calculation and plotting complete!")
	print("=" * 70)


if __name__ == "__main__":
	main()
