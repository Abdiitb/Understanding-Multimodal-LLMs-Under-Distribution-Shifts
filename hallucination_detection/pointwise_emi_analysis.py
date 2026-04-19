from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

try:
    from hallucination_detection.bootstrap_utils import bootstrap_confidence_interval
except ModuleNotFoundError:
    from bootstrap_utils import bootstrap_confidence_interval


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)]

    if isinstance(obj, dict):
        if "records" in obj and isinstance(obj["records"], list):
            return [r for r in obj["records"] if isinstance(r, dict)]

    raise ValueError(f"Unsupported JSON structure in {path}. Expected list[dict] or dict with 'records'.")


def _extract_labels_and_scores(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    y_true: list[int] = []
    pemi_scores: list[float] = []

    for idx, row in enumerate(rows):
        if "hallucination_label" not in row or "pointwise_emi" not in row:
            continue

        label_raw = row["hallucination_label"]
        score_raw = row["pointwise_emi"]

        try:
            label = int(label_raw)
            score = float(score_raw)
        except Exception as exc:
            raise ValueError(f"Invalid label/score at row index {idx}: {exc}") from exc

        if label not in {0, 1}:
            raise ValueError(f"hallucination_label must be 0 or 1 at row index {idx}, got {label}")

        y_true.append(label)
        pemi_scores.append(score)

    if not y_true:
        raise ValueError("No valid rows found with hallucination_label and pointwise_emi")

    return np.asarray(y_true, dtype=np.int64), np.asarray(pemi_scores, dtype=np.float64)


def _extract_labels_scores_and_categories(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[str]]:
	"""Extract labels, scores, and categories (if present) from rows."""
	y_true: list[int] = []
	pemi_scores: list[float] = []
	categories: list[str] = []

	for idx, row in enumerate(rows):
		if "hallucination_label" not in row or "pointwise_emi" not in row:
			continue

		label_raw = row["hallucination_label"]
		score_raw = row["pointwise_emi"]
		category = row.get("category", "unknown")

		try:
			label = int(label_raw)
			score = float(score_raw)
		except Exception as exc:
			raise ValueError(f"Invalid label/score at row index {idx}: {exc}") from exc

		if label not in {0, 1}:
			raise ValueError(f"hallucination_label must be 0 or 1 at row index {idx}, got {label}")

		y_true.append(label)
		pemi_scores.append(score)
		categories.append(str(category))

	if not y_true:
		raise ValueError("No valid rows found with hallucination_label and pointwise_emi")

	return np.asarray(y_true, dtype=np.int64), np.asarray(pemi_scores, dtype=np.float64), categories


def predict_from_threshold(pemi_scores: np.ndarray, threshold: float) -> np.ndarray:
    """
    Rule:
      if pemi > threshold -> predict no hallucination (0)
      else -> predict hallucination (1)
    """
    return np.where(pemi_scores > threshold, 0, 1).astype(np.int64)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def compute_pemi_ranges(y_true: np.ndarray, pemi_scores: np.ndarray) -> dict[str, Any]:
    def _class_range(label: int) -> dict[str, Any]:
        values = pemi_scores[y_true == label]
        if values.size == 0:
            return {
                "count": 0,
                "min": None,
                "max": None,
                "range": None,
            }
        min_val = float(np.min(values))
        max_val = float(np.max(values))
        return {
            "count": int(values.size),
            "min": min_val,
            "max": max_val,
            "range": [min_val, max_val],
        }

    return {
        "hallucination_label_1": _class_range(1),
        "hallucination_label_0": _class_range(0),
    }


def compute_per_category_metrics(
	y_true: np.ndarray,
	pemi_scores: np.ndarray,
	categories: list[str],
	bootstrap_samples: int = 1000,
	test_repeats: int = 200,
	seed: int = 42,
) -> dict[str, Any]:
	"""Compute metrics per category (e.g., adversarial, popular, random for POPE)."""
	unique_categories = sorted(set(categories))
	category_metrics = {}
	
	for cat in unique_categories:
		mask = np.array([c == cat for c in categories])
		cat_y_true = y_true[mask]
		cat_pemi_scores = pemi_scores[mask]
		
		if cat_y_true.size == 0:
			category_metrics[cat] = {"error": "No samples in category"}
			continue
		
		cat_metrics: dict[str, Any] = {
			"num_samples": int(cat_y_true.size),
			"num_hallucination_label_0": int(np.sum(cat_y_true == 0)),
			"num_hallucination_label_1": int(np.sum(cat_y_true == 1)),
			"pemi_ranges": compute_pemi_ranges(cat_y_true, cat_pemi_scores),
		}
		
		# Compute ROC AUC
		if np.unique(cat_y_true).size >= 2:
			cat_metrics["roc_auc"] = float(roc_auc_score(cat_y_true, -cat_pemi_scores))
			cat_metrics["pr_auc"] = float(average_precision_score(cat_y_true, -cat_pemi_scores))
			
			# Bootstrap CI for ROC-AUC
			auc_ci = bootstrap_auc_confidence_interval(
				y_true=cat_y_true,
				pemi_scores=cat_pemi_scores,
				num_bootstrap=bootstrap_samples,
				seed=seed + hash(cat) % 1000,
			)
			cat_metrics["roc_auc_ci"] = {
				"lower": auc_ci["ci_lower"],
				"upper": auc_ci["ci_upper"],
			}
			
			# Robustness tests
			cat_metrics["robustness_tests"] = {
				"label_shuffle_test": label_shuffle_test(
					y_true=cat_y_true,
					pemi_scores=cat_pemi_scores,
					repeats=test_repeats,
					seed=seed + hash(cat) % 1000,
				),
				"input_output_swap_test": alignment_swap_test(
					y_true=cat_y_true,
					pemi_scores=cat_pemi_scores,
					repeats=test_repeats,
					seed=seed + hash(cat) % 1000 + 1,
				),
			}
		else:
			cat_metrics["roc_auc"] = None
			cat_metrics["pr_auc"] = None
			cat_metrics["warning"] = "Insufficient unique labels for ROC/PR computation"
		
		category_metrics[cat] = cat_metrics
	
	return category_metrics


def plot_and_save_roc(y_true: np.ndarray, pemi_scores: np.ndarray, output_path: Path) -> float:
    """
    Positive class is hallucination (1).
    Since higher pemi implies no hallucination, use -pemi as positive score.
    """
    hallucination_score = -pemi_scores
    fpr, tpr, _ = roc_curve(y_true, hallucination_score)
    roc_auc = float(auc(fpr, tpr))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve: Hallucination Detection using Pointwise EMI")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return roc_auc


def plot_and_save_pr_curve(y_true: np.ndarray, pemi_scores: np.ndarray, output_path: Path) -> float:
    """
    Positive class is hallucination (1).
    Since higher pemi implies no hallucination, use -pemi as positive score.
    """
    hallucination_score = -pemi_scores
    precision, recall, _ = precision_recall_curve(y_true, hallucination_score)
    pr_auc = float(auc(recall, precision))
    
    # Also compute average precision score as reference
    avg_precision = float(average_precision_score(y_true, hallucination_score))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"PR curve (AUC = {pr_auc:.4f})", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve: Hallucination Detection using Pointwise EMI")
    plt.legend(loc="best")
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return pr_auc


def bootstrap_auc_confidence_interval(
	y_true: np.ndarray,
	pemi_scores: np.ndarray,
	num_bootstrap: int = 1000,
	seed: int = 42,
	lower_percentile: float = 2.5,
	upper_percentile: float = 97.5,
) -> dict[str, Any]:
    """
    Bootstrap CI for ROC-AUC.

    Positive class is hallucination (1), so we use -pemi_scores as decision score.
    """
    hallucination_score = -pemi_scores

    def _auc_stat(y_batch: np.ndarray, score_batch: np.ndarray) -> float:
        if np.unique(y_batch).size < 2:
            return float("nan")
        return float(roc_auc_score(y_batch, score_batch))

    ci_lower, ci_upper, auc_arr = bootstrap_confidence_interval(
        x=y_true,
        y=hallucination_score,
        statistic_fn=_auc_stat,
        num_bootstrap=num_bootstrap,
        seed=seed,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )

    return {
        "num_bootstrap": int(num_bootstrap),
        "num_valid_bootstrap": int(auc_arr.size),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def _auc_from_pemi(y_true: np.ndarray, pemi_scores: np.ndarray) -> float:
    return float(roc_auc_score(y_true, -pemi_scores))


def label_shuffle_test(y_true: np.ndarray, pemi_scores: np.ndarray, repeats: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    aucs: list[float] = []
    for _ in range(repeats):
        y_perm = rng.permutation(y_true)
        if np.unique(y_perm).size < 2:
            continue
        aucs.append(_auc_from_pemi(y_perm, pemi_scores))

    if not aucs:
        raise ValueError("Label shuffle test produced no valid AUC samples")

    arr = np.asarray(aucs, dtype=np.float64)
    return {
        "mean_auc": float(np.mean(arr)),
        "std_auc": float(np.std(arr)),
        "min_auc": float(np.min(arr)),
        "max_auc": float(np.max(arr)),
        "num_repeats": int(repeats),
        "num_valid": int(arr.size),
    }


def alignment_swap_test(y_true: np.ndarray, pemi_scores: np.ndarray, repeats: int, seed: int) -> dict[str, Any]:
    """
    Proxy for input-output swap check using pEMI-only data:
    shuffles pEMI assignments across samples (breaks alignment while preserving score distribution).
    """
    rng = np.random.default_rng(seed)
    aucs: list[float] = []
    for _ in range(repeats):
        pemi_perm = rng.permutation(pemi_scores)
        aucs.append(_auc_from_pemi(y_true, pemi_perm))

    arr = np.asarray(aucs, dtype=np.float64)
    return {
        "mean_auc": float(np.mean(arr)),
        "std_auc": float(np.std(arr)),
        "min_auc": float(np.min(arr)),
        "max_auc": float(np.max(arr)),
        "num_repeats": int(repeats),
    }


def random_score_test(y_true: np.ndarray, repeats: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    aucs: list[float] = []
    for _ in range(repeats):
        random_scores = rng.normal(size=n)
        aucs.append(float(roc_auc_score(y_true, random_scores)))

    arr = np.asarray(aucs, dtype=np.float64)
    return {
        "mean_auc": float(np.mean(arr)),
        "std_auc": float(np.std(arr)),
        "min_auc": float(np.min(arr)),
        "max_auc": float(np.max(arr)),
        "num_repeats": int(repeats),
    }


def plot_pemi_distributions(y_true: np.ndarray, pemi_scores: np.ndarray, output_path: Path) -> dict[str, Any]:
    values_h = pemi_scores[y_true == 1]
    values_nh = pemi_scores[y_true == 0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    if values_h.size > 0:
        plt.hist(values_h, bins=30, alpha=0.6, label="Hallucination (label=1)", density=True)
    if values_nh.size > 0:
        plt.hist(values_nh, bins=30, alpha=0.6, label="No hallucination (label=0)", density=True)
    plt.xlabel("Pointwise EMI")
    plt.ylabel("Density")
    plt.title("Pointwise EMI Distributions by Hallucination Label")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return {
        "hallucination_mean": float(np.mean(values_h)) if values_h.size > 0 else None,
        "hallucination_median": float(np.median(values_h)) if values_h.size > 0 else None,
        "no_hallucination_mean": float(np.mean(values_nh)) if values_nh.size > 0 else None,
        "no_hallucination_median": float(np.median(values_nh)) if values_nh.size > 0 else None,
        "mean_gap_label1_minus_label0": (
            float(np.mean(values_h) - np.mean(values_nh)) if values_h.size > 0 and values_nh.size > 0 else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pointwise EMI analysis with ROC-AUC and optional threshold metrics")
    parser.add_argument("--input-json", type=str, required=True, help="Path to JSON with hallucination_label and pointwise_emi")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional decision threshold for pointwise EMI. If omitted, only ROC/AUC + pEMI ranges are computed.",
    )
    parser.add_argument(
        "--output-metrics-json",
        type=str,
        default="results/emi_vs_hallucination/pemi_threshold_metrics.json",
        help="Path to save threshold metrics JSON",
    )
    parser.add_argument(
        "--output-roc-png",
        type=str,
        default="results/emi_vs_hallucination/pemi_roc_auc.png",
        help="Path to save ROC plot PNG",
    )
    parser.add_argument(
        "--output-pr-png",
        type=str,
        default="results/emi_vs_hallucination/pemi_pr_auc.png",
        help="Path to save PR curve plot PNG",
    )
    parser.add_argument(
        "--output-dist-png",
        type=str,
        default="results/emi_vs_hallucination/pemi_distribution.png",
        help="Path to save pEMI distribution plot PNG",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Number of bootstrap samples for AUC confidence interval",
    )
    parser.add_argument(
        "--test-repeats",
        type=int,
        default=200,
        help="Number of repeats for shuffle/random robustness tests",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for bootstrap")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_json)
    rows = _load_rows(input_path)
    
    # Try to extract with categories (for POPE dataset)
    try:
        y_true, pemi_scores, categories = _extract_labels_scores_and_categories(rows)
        has_categories = True
        print(f"✓ Extracted categories from data. Found {len(set(categories))} unique categories: {sorted(set(categories))}")
    except:
        y_true, pemi_scores = _extract_labels_and_scores(rows)
        categories = None
        has_categories = False
        print("✓ No category information found in data")

    metrics: dict[str, Any] = {}
    if args.threshold is not None:
        y_pred = predict_from_threshold(pemi_scores, threshold=args.threshold)
        metrics.update(compute_metrics(y_true, y_pred))
        metrics["threshold"] = float(args.threshold)

    metrics["pemi_value_ranges"] = compute_pemi_ranges(y_true, pemi_scores)

    roc_auc = plot_and_save_roc(y_true, pemi_scores, output_path=Path(args.output_roc_png))
    auc_ci = bootstrap_auc_confidence_interval(
        y_true=y_true,
        pemi_scores=pemi_scores,
        num_bootstrap=args.bootstrap_samples,
        seed=args.seed,
    )
    metrics["num_samples"] = int(y_true.shape[0])
    metrics["roc_auc"] = roc_auc
    metrics["roc_auc_ci"] = {
        "lower": auc_ci["ci_lower"],
        "upper": auc_ci["ci_upper"],
        "num_bootstrap": auc_ci["num_bootstrap"],
        "num_valid_bootstrap": auc_ci["num_valid_bootstrap"],
    }

    # Compute PR curve
    pr_auc = plot_and_save_pr_curve(y_true, pemi_scores, output_path=Path(args.output_pr_png))
    metrics["pr_auc"] = pr_auc

    # Additional tests requested
    metrics["robustness_tests"] = {
        "label_shuffle_test": label_shuffle_test(
            y_true=y_true,
            pemi_scores=pemi_scores,
            repeats=args.test_repeats,
            seed=args.seed,
        ),
        "input_output_swap_test": alignment_swap_test(
            y_true=y_true,
            pemi_scores=pemi_scores,
            repeats=args.test_repeats,
            seed=args.seed + 1,
        ),
        "random_score_test": random_score_test(
            y_true=y_true,
            repeats=args.test_repeats,
            seed=args.seed + 2,
        ),
    }
    metrics["distribution_separation"] = plot_pemi_distributions(
        y_true=y_true,
        pemi_scores=pemi_scores,
        output_path=Path(args.output_dist_png),
    )

    # Per-category metrics (if categories present)
    if has_categories and categories:
        print("\nComputing per-category metrics...")
        metrics["per_category_metrics"] = compute_per_category_metrics(
            y_true=y_true,
            pemi_scores=pemi_scores,
            categories=categories,
            bootstrap_samples=args.bootstrap_samples,
            test_repeats=args.test_repeats,
            seed=args.seed,
        )
        for cat, cat_metrics in metrics["per_category_metrics"].items():
            print(f"  {cat}: {cat_metrics.get('num_samples', 0)} samples, " +
                  f"ROC AUC={cat_metrics.get('roc_auc', 'N/A')}")

    output_metrics_path = Path(args.output_metrics_json)
    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with output_metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved metrics:", output_metrics_path)
    print("Saved ROC plot:", args.output_roc_png)
    print("Saved PR curve plot:", args.output_pr_png)
    print("Saved distribution plot:", args.output_dist_png)
    print("ROC AUC:", roc_auc)
    print("PR AUC:", pr_auc)


if __name__ == "__main__":
    main()
