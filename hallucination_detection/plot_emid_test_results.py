from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


def _load_results(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict) or "results" not in payload or not isinstance(payload["results"], list):
        raise ValueError("Expected JSON with top-level key 'results' as a list")

    rows = [r for r in payload["results"] if isinstance(r, dict)]
    if not rows:
        raise ValueError("No valid rows in 'results'")
    return rows


def _bootstrap_ci_from_subset_pairs(
    subset_emid: list[float], subset_hr: list[float], num_bootstrap: int, seed: int
) -> tuple[float, float]:
    emid = np.asarray(subset_emid, dtype=np.float64)
    hr = np.asarray(subset_hr, dtype=np.float64)
    if emid.shape[0] != hr.shape[0] or emid.shape[0] < 2:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    n = emid.shape[0]
    samples: list[float] = []

    for _ in range(num_bootstrap):
        idx = rng.integers(0, n, size=n)
        emid_b = emid[idx]
        hr_b = hr[idx]
        if np.allclose(emid_b, emid_b[0]) or np.allclose(hr_b, hr_b[0]):
            continue
        rho_b, _ = spearmanr(emid_b, hr_b)
        if not np.isnan(rho_b):
            samples.append(float(rho_b))

    if not samples:
        return float("nan"), float("nan")

    arr = np.asarray(samples, dtype=np.float64)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def _extract_plot_arrays(
    rows: list[dict[str, Any]], num_bootstrap: int, seed: int
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    rows_sorted = sorted(rows, key=lambda r: int(r.get("K", 0)))

    k_vals: list[int] = []
    rho_avg_vals: list[float] = []
    rho_class_vals: list[float] = []

    ci_lowers_avg: list[float] = []
    ci_uppers_avg: list[float] = []
    has_any_ci_avg = False

    ci_lowers_class: list[float] = []
    ci_uppers_class: list[float] = []
    has_any_ci_class = False

    for row_idx, row in enumerate(rows_sorted):
        if "K" not in row:
            continue

        k = int(row["K"])
        if "rho_avg_pemi" not in row or "rho_emi_class" not in row:
            continue
        rho_avg = float(row["rho_avg_pemi"])
        rho_class = float(row["rho_emi_class"])

        k_vals.append(k)
        rho_avg_vals.append(rho_avg)
        rho_class_vals.append(rho_class)

        ci = row.get("ci")
        if isinstance(ci, list) and len(ci) == 2:
            has_any_ci_avg = True
            ci_lowers_avg.append(float(ci[0]))
            ci_uppers_avg.append(float(ci[1]))
        else:
            subset_emid = row.get("subset_emid_avg_pemi")
            subset_hr = row.get("subset_hallucination_ratio")
            if isinstance(subset_emid, list) and isinstance(subset_hr, list):
                ci_low, ci_up = _bootstrap_ci_from_subset_pairs(
                    subset_emid=subset_emid,
                    subset_hr=subset_hr,
                    num_bootstrap=num_bootstrap,
                    seed=seed + row_idx,
                )
                if not np.isnan(ci_low) and not np.isnan(ci_up):
                    has_any_ci_avg = True
                    ci_lowers_avg.append(ci_low)
                    ci_uppers_avg.append(ci_up)
                else:
                    ci_lowers_avg.append(float(rho_avg))
                    ci_uppers_avg.append(float(rho_avg))
            else:
                ci_lowers_avg.append(float(rho_avg))
                ci_uppers_avg.append(float(rho_avg))

        subset_emid_class = row.get("subset_emid_emi_class")
        subset_hr = row.get("subset_hallucination_ratio")
        if isinstance(subset_emid_class, list) and isinstance(subset_hr, list):
            ci_low_c, ci_up_c = _bootstrap_ci_from_subset_pairs(
                subset_emid=subset_emid_class,
                subset_hr=subset_hr,
                num_bootstrap=num_bootstrap,
                seed=seed + 10000 + row_idx,
            )
            if not np.isnan(ci_low_c) and not np.isnan(ci_up_c):
                has_any_ci_class = True
                ci_lowers_class.append(ci_low_c)
                ci_uppers_class.append(ci_up_c)
            else:
                ci_lowers_class.append(float(rho_class))
                ci_uppers_class.append(float(rho_class))
        else:
            ci_lowers_class.append(float(rho_class))
            ci_uppers_class.append(float(rho_class))

    if not k_vals:
        raise ValueError("No plottable rows found with K and rho")

    ci_low_avg_arr: np.ndarray | None = None
    ci_up_avg_arr: np.ndarray | None = None
    if has_any_ci_avg:
        ci_low_avg_arr = np.asarray(ci_lowers_avg, dtype=np.float64)
        ci_up_avg_arr = np.asarray(ci_uppers_avg, dtype=np.float64)

    ci_low_class_arr: np.ndarray | None = None
    ci_up_class_arr: np.ndarray | None = None
    if has_any_ci_class:
        ci_low_class_arr = np.asarray(ci_lowers_class, dtype=np.float64)
        ci_up_class_arr = np.asarray(ci_uppers_class, dtype=np.float64)

    return (
        np.asarray(k_vals, dtype=np.int64),
        np.asarray(rho_avg_vals, dtype=np.float64),
        np.asarray(rho_class_vals, dtype=np.float64),
        ci_low_avg_arr,
        ci_up_avg_arr,
        ci_low_class_arr,
        ci_up_class_arr,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot rho vs K with CI error bars from EMID test results JSON")
    parser.add_argument("--input-json", type=str, required=True, help="Path to emid_vs_hallucination_rate output JSON")
    parser.add_argument(
        "--output-png",
        type=str,
        default="results/emi_vs_hallucination/emid_rho_vs_k.png",
        help="Path to save rho-vs-K plot",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Bootstrap samples to estimate CI when 'ci' field is missing",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for CI bootstrap")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _load_results(Path(args.input_json))
    k_vals, rho_avg_vals, rho_class_vals, ci_lowers_avg, ci_uppers_avg, ci_lowers_class, ci_uppers_class = _extract_plot_arrays(
        rows,
        num_bootstrap=args.bootstrap_samples,
        seed=args.seed,
    )

    output_path = Path(args.output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    if ci_lowers_avg is not None and ci_uppers_avg is not None:
        ci_lowers_avg = np.minimum(ci_lowers_avg, rho_avg_vals)
        ci_uppers_avg = np.maximum(ci_uppers_avg, rho_avg_vals)
        yerr_lower_avg = np.maximum(0.0, rho_avg_vals - ci_lowers_avg)
        yerr_upper_avg = np.maximum(0.0, ci_uppers_avg - rho_avg_vals)
        yerr_avg = np.vstack([yerr_lower_avg, yerr_upper_avg])
        plt.errorbar(k_vals, rho_avg_vals, yerr=yerr_avg, fmt="o-", capsize=4, linewidth=2, label="EMID via avg pEMI")
    else:
        plt.plot(k_vals, rho_avg_vals, "o-", linewidth=2, label="EMID via avg pEMI")

    if ci_lowers_class is not None and ci_uppers_class is not None:
        ci_lowers_class = np.minimum(ci_lowers_class, rho_class_vals)
        ci_uppers_class = np.maximum(ci_uppers_class, rho_class_vals)
        yerr_lower_class = np.maximum(0.0, rho_class_vals - ci_lowers_class)
        yerr_upper_class = np.maximum(0.0, ci_uppers_class - rho_class_vals)
        yerr_class = np.vstack([yerr_lower_class, yerr_upper_class])
        plt.errorbar(k_vals, rho_class_vals, yerr=yerr_class, fmt="s--", capsize=4, linewidth=2, label="EMID via EMI class")
    else:
        plt.plot(k_vals, rho_class_vals, "s--", linewidth=2, label="EMID via EMI class")

    plt.title("EMID Test: rho vs K (two EMI methods)")
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("K (number of subsets)")
    plt.ylabel("Spearman rho (EMID vs Hallucination Rate)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print("Saved plot:", output_path)


if __name__ == "__main__":
    main()
