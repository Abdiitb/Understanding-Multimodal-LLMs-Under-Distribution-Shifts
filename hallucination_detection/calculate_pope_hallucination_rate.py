"""
Compute POPE category-wise hallucination rate from pope_model_responses.json.

Expected input format:
{
  "model_id": "...",
  "categories": {
    "adversarial": [
      {
        "reference_answer": "yes"|"no",
        "model_answer": "yes"|"no",
        ...
      },
      ...
    ],
    "popular": [...],
    "random": [...]
  }
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _normalize_yes_no(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"yes", "y", "1", "true"}:
        return "yes"
    if text in {"no", "n", "0", "false"}:
        return "no"
    return text


def _get_categories(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    categories = payload.get("categories")
    if not isinstance(categories, dict):
        raise ValueError("Expected top-level key 'categories' with a dict of category -> records")

    parsed: dict[str, list[dict[str, Any]]] = {}
    for category, rows in categories.items():
        if not isinstance(rows, list):
            continue
        parsed[category] = [r for r in rows if isinstance(r, dict)]
    return parsed


def compute_category_hallucination_rate(categories: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, float | int]]:
    metrics: dict[str, dict[str, float | int]] = {}

    for category, rows in categories.items():
        total = 0
        yes_yes = 0
        yes_no = 0
        no_yes = 0
        no_no = 0

        for row in rows:
            reference_answer = _normalize_yes_no(row.get("reference_answer", ""))
            model_answer = _normalize_yes_no(row.get("model_answer", ""))

            if reference_answer not in {"yes", "no"} or model_answer not in {"yes", "no"}:
                continue

            total += 1
            if reference_answer == "yes" and model_answer == "yes":
                yes_yes += 1
            elif reference_answer == "yes" and model_answer == "no":
                yes_no += 1
            elif reference_answer == "no" and model_answer == "yes":
                no_yes += 1
            elif reference_answer == "no" and model_answer == "no":
                no_no += 1

        no_ground_truth_total = no_no + no_yes
        hallucination_rate_pct = (100.0 * no_yes / no_ground_truth_total) if no_ground_truth_total > 0 else 0.0
        metrics[category] = {
            "num_evaluated": total,
            "num_false_positives": no_yes,
            "num_no_ground_truth": no_ground_truth_total,
            "hallucination_rate_percent": round(hallucination_rate_pct, 2),
            "confusion": {
                "yes_to_yes": yes_yes,
                "yes_to_no": yes_no,
                "no_to_yes": no_yes,
                "no_to_no": no_no,
            },
        }

    # Overall across all categories
    overall_total = sum(int(m["num_evaluated"]) for m in metrics.values())
    overall_false_positives = sum(int(m["num_false_positives"]) for m in metrics.values())
    overall_no_ground_truth = sum(int(m["num_no_ground_truth"]) for m in metrics.values())
    overall_hallucination_rate = (100.0 * overall_false_positives / overall_no_ground_truth) if overall_no_ground_truth > 0 else 0.0
    overall_yes_yes = sum(int(m["confusion"]["yes_to_yes"]) for m in metrics.values())
    overall_yes_no = sum(int(m["confusion"]["yes_to_no"]) for m in metrics.values())
    overall_no_yes = sum(int(m["confusion"]["no_to_yes"]) for m in metrics.values())
    overall_no_no = sum(int(m["confusion"]["no_to_no"]) for m in metrics.values())
    metrics["overall"] = {
        "num_evaluated": overall_total,
        "num_false_positives": overall_false_positives,
        "num_no_ground_truth": overall_no_ground_truth,
        "hallucination_rate_percent": round(overall_hallucination_rate, 2),
        "confusion": {
            "yes_to_yes": overall_yes_yes,
            "yes_to_no": overall_yes_no,
            "no_to_yes": overall_no_yes,
            "no_to_no": overall_no_no,
        },
    }

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute POPE category-wise hallucination rate")
    parser.add_argument(
        "--input-json",
        type=str,
        default="results/pope/pope_model_responses.json",
        help="Path to POPE model responses JSON",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional path to save computed metrics JSON",
    )
    args = parser.parse_args()

    input_path = Path(args.input_json)
    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    categories = _get_categories(payload)
    metrics = compute_category_hallucination_rate(categories)

    model_id = payload.get("model_id", "unknown_model")
    print(f"Model: {model_id}")
    print("Category-wise hallucination rate (% = false_positives / no-ground-truth samples):")
    for category, values in metrics.items():
        conf = values["confusion"]
        print(
            f"  - {category}: {values['hallucination_rate_percent']}% "
            f"({values['num_false_positives']}/{values['num_no_ground_truth']})"
        )
        print(
            "      confusion: "
            f"yes->yes={conf['yes_to_yes']}, yes->no={conf['yes_to_no']}, "
            f"no->yes={conf['no_to_yes']}, no->no={conf['no_to_no']}"
        )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump({"model_id": model_id, "hallucination_rate": metrics}, f, indent=2)
        print(f"Saved metrics to: {output_path}")


if __name__ == "__main__":
    main()
