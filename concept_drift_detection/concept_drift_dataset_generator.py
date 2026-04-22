from __future__ import annotations

import argparse
import base64
import json
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Image, load_dataset

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable=None, total=None, **kwargs):
        if iterable is None:
            class _NoOpTqdm:
                def __enter__(self):
                    return self
                def __exit__(self, exc_type, exc_val, exc_tb):
                    return False
                def update(self, n=1):
                    pass
            return _NoOpTqdm()
        return iterable


@dataclass
class FilteredSample:
    x: dict[str, Any]
    answers_counter: Counter[str]


def _normalize_answer(answer: str) -> str:
    return answer.strip().lower()


def _extract_answers(sample: dict[str, Any]) -> list[str]:
    answers_raw = sample.get("answers")
    if answers_raw is None:
        return []

    normalized: list[str] = []

    if isinstance(answers_raw, list):
        for item in answers_raw:
            if isinstance(item, dict):
                text = item.get("answer")
                if isinstance(text, str):
                    norm = _normalize_answer(text)
                    if norm:
                        normalized.append(norm)
            elif isinstance(item, str):
                norm = _normalize_answer(item)
                if norm:
                    normalized.append(norm)
        return normalized

    if isinstance(answers_raw, dict):
        ans_field = answers_raw.get("answer")
        if isinstance(ans_field, list):
            for item in ans_field:
                if isinstance(item, str):
                    norm = _normalize_answer(item)
                    if norm:
                        normalized.append(norm)
        return normalized

    return normalized


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    return str(value)


def _serialize_image_value(image_value: Any, include_image_bytes: bool = False) -> Any:
    if isinstance(image_value, dict):
        path_val = image_value.get("path")
        bytes_val = image_value.get("bytes")
        if "path" in image_value or "bytes" in image_value:
            num_bytes = None
            payload: dict[str, Any] = {
                "path": path_val,
                "num_bytes": num_bytes,
            }
            if isinstance(bytes_val, (bytes, bytearray)):
                num_bytes = len(bytes_val)
                payload["num_bytes"] = num_bytes
                if include_image_bytes:
                    payload["bytes_base64"] = base64.b64encode(bytes(bytes_val)).decode("utf-8")
            return payload

    if hasattr(image_value, "size") and hasattr(image_value, "mode"):
        size_val = getattr(image_value, "size", None)
        if isinstance(size_val, tuple):
            size_val = list(size_val)
        return {
            "type": type(image_value).__name__,
            "mode": getattr(image_value, "mode", None),
            "size": size_val,
            "format": getattr(image_value, "format", None),
        }

    return _to_jsonable(image_value)


def _extract_x(sample: dict[str, Any], include_image_bytes: bool = False) -> dict[str, Any]:
    wanted_fields = ["answers", "image_id", "question_id", "question", "image"]
    x: dict[str, Any] = {}
    for key in wanted_fields:
        if key in sample:
            if key == "image":
                x[key] = _serialize_image_value(sample[key], include_image_bytes=include_image_bytes)
            else:
                x[key] = _to_jsonable(sample[key])
    return x


# =======================
# 🔥 NEW: CORRUPTION LOGIC
# =======================

def construct_dataset_clean(filtered: list[FilteredSample]) -> list[dict[str, Any]]:
    """D1: clean dataset (true mapping)"""
    d = []
    for item in filtered:
        ranked = sorted(item.answers_counter.items(), key=lambda t: (-t[1], t[0]))
        d.append({"x": item.x, "y": ranked[0][0]})
    return d


def construct_dataset_with_corruption(
    filtered: list[FilteredSample],
    corruption_ratio: float,
) -> list[dict[str, Any]]:
    """
    Create dataset with partial corruption of labels
    """
    # collect all answers globally
    all_answers = []
    for item in filtered:
        all_answers.extend(list(item.answers_counter.keys()))

    d = []
    for item in filtered:
        ranked = sorted(item.answers_counter.items(), key=lambda t: (-t[1], t[0]))
        true_answer = ranked[0][0]

        if random.random() < corruption_ratio:
            y = random.choice(all_answers)
        else:
            y = true_answer

        d.append({"x": item.x, "y": y})

    return d


# =======================

def build_filtered_samples(
    split: str,
    n_target: int,
    streaming: bool,
    include_image_bytes: bool,
) -> list[FilteredSample]:
    # Limit threading to prevent GIL issues with streaming datasets
    if streaming:
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        os.environ.setdefault('MKL_NUM_THREADS', '1')
        os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
        os.environ.setdefault('HF_DATASETS_DISABLE_MULTIPROCESSING', '1')
    
    dataset = load_dataset("lmms-lab/VQAv2", split=split, streaming=streaming)
    dataset = dataset.cast_column("image", Image(decode=False))

    filtered: list[FilteredSample] = []
    try:
        with tqdm(total=n_target, desc="Collecting samples") as pbar:
            for sample in dataset:
                answers = _extract_answers(sample)
                if not answers:
                    continue

                answers_counter = Counter(answers)
                filtered.append(
                    FilteredSample(
                        x=_extract_x(sample, include_image_bytes=include_image_bytes),
                        answers_counter=answers_counter,
                    )
                )

                pbar.update(1)
                if len(filtered) >= n_target:
                    break
    finally:
        # Ensure dataset iterators are cleaned up
        if hasattr(dataset, 'cleanup_cache'):
            try:
                dataset.cleanup_cache()
            except Exception:
                pass

    return filtered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-target", type=int, default=1000)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument(
        "--include-image-bytes",
        action="store_true",
        help="Include image bytes as base64 in output JSON (larger and more memory intensive)",
    )
    parser.add_argument("--output-dir", type=str, default="results/concept_drift_detection/datasets")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _terminate_process(exit_code: int, streaming: bool) -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    if streaming:
        os._exit(exit_code)
    raise SystemExit(exit_code)


def main() -> None:
    args: argparse.Namespace | None = None
    try:
        args = parse_args()
        random.seed(args.seed)

        filtered = build_filtered_samples(
            split=args.split,
            n_target=args.n_target,
            streaming=args.streaming,
            include_image_bytes=args.include_image_bytes,
        )

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\nCreating datasets with increasing drift...")
        dataset_specs = [
            ("D1.json", 0.0),
            ("D2.json", 0.2),
            ("D3.json", 0.4),
            ("D4.json", 0.6),
            ("D5.json", 0.8),
        ]

        for name, corruption_ratio in dataset_specs:
            if corruption_ratio == 0.0:
                data = construct_dataset_clean(filtered)
            else:
                data = construct_dataset_with_corruption(filtered, corruption_ratio)
            path = output_dir / name
            with path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"Saved {name}: {len(data)} samples")
            del data

        print("\n✅ Done! Concept drift datasets created.")
        _terminate_process(exit_code=0, streaming=args.streaming)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr, flush=True)
        _terminate_process(exit_code=1, streaming=bool(args and args.streaming))


if __name__ == "__main__":
    main()