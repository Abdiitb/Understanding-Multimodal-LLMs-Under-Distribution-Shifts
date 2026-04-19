"""
Plot EMID distributions (histograms/PDF with KDE) across all d_k datasets in one graph.
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def load_emid_scores(emid_scores_path: Path) -> dict[str, list[float]]:
	"""Load EMID scores from JSON file."""
	with emid_scores_path.open("r", encoding="utf-8") as f:
		emid_data = json.load(f)
	
	# Extract scores, excluding the description
	scores = {}
	for key, value in emid_data.items():
		if key != "description" and isinstance(value, list):
			scores[key] = value
	
	return scores


def plot_emid_histograms(
	emid_scores: dict[str, list[float]],
	output_path: Path,
) -> None:
	"""Plot overlaid histograms and KDE curves of EMID distributions for all datasets."""
	
	# Rename for consistency
	plot_scores = {}
	for key, value in emid_scores.items():
		if key == "D1_pairs_emid":
			plot_scores["D1 (ID)"] = value
		else:
			plot_scores[key.replace("_migrated", "")] = value
	
	# Sort by key for consistent ordering (D1 first)
	sorted_keys = sorted(plot_scores.keys(), key=lambda x: (x != "D1 (ID)", x))
	
	# Create figure
	fig, ax = plt.subplots(figsize=(12, 7))
	
	# Define colors
	colors = ["green", "steelblue", "coral", "purple", "brown"]
	
	# Plot KDE curves only
	for idx, label in enumerate(sorted_keys):
		scores = np.array(plot_scores[label])
		color = colors[idx % len(colors)]
		
		# Plot KDE curve
		try:
			kde = stats.gaussian_kde(scores)
			x_range = np.linspace(scores.min() - 0.02, scores.max() + 0.02, 200)
			kde_values = kde(x_range)
			
			kde_linewidth = 3.0 if label == "D1 (ID)" else 2.3
			ax.plot(x_range, kde_values, color=color, linewidth=kde_linewidth, label=label)
		except Exception as e:
			print(f"  Warning: Failed to compute KDE for {label}: {e}")
	
	# Plot mean lines for each dataset
	for idx, label in enumerate(sorted_keys):
		scores = plot_scores[label]
		mean_val = np.mean(scores)
		color = colors[idx % len(colors)]
		linestyle = "-" if label == "D1 (ID)" else "--"
		linewidth_mean = 2.5 if label == "D1 (ID)" else 2.0
		
		ax.axvline(
			mean_val,
			color=color,
			linestyle=linestyle,
			linewidth=linewidth_mean,
			alpha=0.8,
		)
	
	# Labels and title
	ax.set_xlabel("EMID Score", fontsize=13, fontweight="bold")
	ax.set_ylabel("Probability Density", fontsize=13, fontweight="bold")
	ax.set_title("EMID Distribution Across All Datasets (KDE Curves + Means)", fontsize=14, fontweight="bold")
	ax.legend(fontsize=11, loc="upper right")
	ax.grid(True, alpha=0.3, linestyle="--")
	
	# Save figure
	plt.tight_layout()
	plt.savefig(output_path, dpi=300, bbox_inches="tight")
	print(f"✓ Saved EMID histogram with KDE: {output_path}")
	
	# Print statistics
	print("\nStatistics (vertical lines show mean, curves show KDE density):")
	for label in sorted_keys:
		scores = plot_scores[label]
		print(f"  {label:15s}: mean={np.mean(scores):9.6f}, std={np.std(scores):9.6f}, min={np.min(scores):9.6f}, max={np.max(scores):9.6f}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Plot EMID histogram across all d_k datasets"
	)
	parser.add_argument(
		"--emid-scores-path",
		type=str,
		default="results/concept_drift_detection/emid_subset_pair_scores.json",
		help="Path to EMID scores JSON file",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default="results/concept_drift_detection",
		help="Directory to save plot",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	
	emid_scores_path = Path(args.emid_scores_path)
	output_dir = Path(args.output_dir)
	
	if not emid_scores_path.exists():
		raise FileNotFoundError(f"EMID scores file not found: {emid_scores_path}")
	
	# Load scores
	print(f"Loading EMID scores from {emid_scores_path}...")
	emid_scores = load_emid_scores(emid_scores_path)
	
	print(f"Found EMID scores for: {', '.join(emid_scores.keys())}\n")
	
	# Create plot
	output_path = output_dir / "emid_histogram.png"
	plot_emid_histograms(emid_scores, output_path)


if __name__ == "__main__":
	main()
