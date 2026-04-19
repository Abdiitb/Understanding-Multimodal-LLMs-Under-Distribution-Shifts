from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import XLMRobertaModel, XLMRobertaTokenizer

import sys

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from main import EMI


def _normalize_yes_no(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"yes", "y", "1", "true"}:
        return "yes"
    if text in {"no", "n", "0", "false"}:
        return "no"
    return text


def _first_present(row: dict[str, Any], keys: list[str]) -> object | None:
    for key in keys:
        if key in row:
            return row[key]
    return None


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    item = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path} line {line_idx}: {exc}") from exc
                if isinstance(item, dict):
                    rows.append(item)
        if not rows:
            raise ValueError(f"No valid rows found in JSONL file: {path}")
        return rows

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)]

    if isinstance(obj, dict):
        if "records" in obj and isinstance(obj["records"], list):
            return [r for r in obj["records"] if isinstance(r, dict)]
        if "categories" in obj and isinstance(obj["categories"], dict):
            rows: list[dict[str, Any]] = []
            for category, items in obj["categories"].items():
                if not isinstance(items, list):
                    continue
                for row in items:
                    if isinstance(row, dict):
                        merged = dict(row)
                        merged.setdefault("category", category)
                        rows.append(merged)
            return rows

    raise ValueError(f"Unsupported JSON structure in {path}")


def _extract_arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    pemi_vals: list[float] = []
    reference_answers: list[str] = []
    model_answers: list[str] = []
    questions: list[str] = []

    for idx, row in enumerate(rows):
        if "pointwise_emi" not in row:
            continue
        question_val = _first_present(row, ["question", "prompt"])
        ref_val = _first_present(row, ["reference_answer"])
        model_val = _first_present(row, ["model_answer", "text"])

        if ref_val is None or model_val is None or question_val is None:
            continue

        try:
            pemi = float(row["pointwise_emi"])
        except Exception as exc:
            raise ValueError(f"Invalid pointwise_emi at row {idx}: {exc}") from exc

        pemi_vals.append(pemi)
        reference_answers.append(_normalize_yes_no(ref_val))
        model_answers.append(_normalize_yes_no(model_val))
        questions.append(str(question_val))

    if not pemi_vals:
        raise ValueError("No valid rows with pointwise_emi/reference_answer/model_answer/question")

    return (
        np.asarray(pemi_vals, dtype=np.float64),
        np.asarray(reference_answers, dtype=object),
        np.asarray(model_answers, dtype=object),
        questions,
    )


def _extract_qa_arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    reference_answers: list[str] = []
    model_answers: list[str] = []
    questions: list[str] = []

    for row in rows:
        question_val = _first_present(row, ["question", "prompt"])
        ref_val = _first_present(row, ["reference_answer"])
        model_val = _first_present(row, ["model_answer", "text"])

        if ref_val is None or model_val is None or question_val is None:
            continue

        reference_answers.append(_normalize_yes_no(ref_val))
        model_answers.append(_normalize_yes_no(model_val))
        questions.append(str(question_val))

    if not questions:
        raise ValueError("No valid rows with question/prompt + reference_answer + model_answer/text")

    return (
        np.asarray(reference_answers, dtype=object),
        np.asarray(model_answers, dtype=object),
        questions,
    )


def _load_reference_rows(path: Path) -> list[dict[str, Any]]:
    return _load_rows(path)


def _auto_find_reference_file(id_path: Path) -> Path | None:
    combined_dir = repo_root / "combined_dataset"
    if not combined_dir.exists() or not combined_dir.is_dir():
        return None

    stem = id_path.stem
    candidates = list(combined_dir.glob("*.jsonl"))
    if not candidates:
        return None

    best: Path | None = None
    best_len = -1
    for p in candidates:
        s = p.stem
        if stem.startswith(s) and len(s) > best_len:
            best = p
            best_len = len(s)
    return best


def _enrich_rows_with_reference(rows: list[dict[str, Any]], reference_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_qid: dict[int, dict[str, Any]] = {}
    by_question: dict[str, dict[str, Any]] = {}

    for r in reference_rows:
        qid_obj = r.get("question_id")
        if isinstance(qid_obj, int):
            by_qid[qid_obj] = r
        q_text = _first_present(r, ["question", "prompt"])
        if q_text is not None:
            by_question[str(q_text).strip()] = r

    out: list[dict[str, Any]] = []
    for row in rows:
        merged = dict(row)
        if "reference_answer" not in merged:
            matched = None
            qid_obj = merged.get("question_id")
            if isinstance(qid_obj, int):
                matched = by_qid.get(qid_obj)
            if matched is None:
                q_text = _first_present(merged, ["question", "prompt"])
                if q_text is not None:
                    matched = by_question.get(str(q_text).strip())
            if matched is not None and "reference_answer" in matched:
                merged["reference_answer"] = matched["reference_answer"]
            if "question" not in merged:
                q_text_ref = _first_present(matched or {}, ["question", "prompt"])
                if q_text_ref is not None:
                    merged["question"] = q_text_ref
        out.append(merged)

    return out


def _encode_texts(texts: list[str], model_name: str, batch_size: int = 64) -> torch.Tensor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
    model = XLMRobertaModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embeddings: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            tok = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            tok = {k: v.to(device) for k, v in tok.items()}

            mask = tok["attention_mask"].unsqueeze(-1)
            h = model(**tok, output_hidden_states=True).hidden_states[0] * mask
            h = h[:, 1:, :].sum(dim=1) / mask[:, 1:, :].sum(dim=1)
            all_embeddings.append(h.detach().cpu())

    return torch.cat(all_embeddings, dim=0).float()


def _compute_emi_from_class(emi_estimator: EMI, x: torch.Tensor, y_model: torch.Tensor, y_ref: torch.Tensor) -> float:
    with torch.inference_mode():
        x_d = x.to(emi_estimator.device)
        y_model_d = y_model.to(emi_estimator.device)
        y_ref_d = y_ref.to(emi_estimator.device)
        model_mi = float(emi_estimator.mi_est(x_d, y_model_d).item())
        ref_mi = float(emi_estimator.mi_est(x_d, y_ref_d).item())
    return float(model_mi - ref_mi)


def _to_pairs(x: torch.Tensor, y: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(x[i], y[i]) for i in range(x.shape[0])]


def _infer_mi_est_dim_from_ckpt(ckpt_path: Path) -> int | None:
    try:
        state = torch.load(ckpt_path, map_location="cpu")
    except Exception:
        return None

    if not isinstance(state, dict):
        return None

    w = state.get("p_mu.0.weight")
    if isinstance(w, torch.Tensor) and w.ndim == 2 and w.shape[0] > 0:
        return int(w.shape[0] * 2)

    b = state.get("p_mu.0.bias")
    if isinstance(b, torch.Tensor) and b.ndim == 1 and b.shape[0] > 0:
        return int(b.shape[0] * 2)

    return None


def _split_sorted_indices_by_pemi(pemi: np.ndarray, k: int) -> list[np.ndarray]:
    idx_sorted = np.argsort(pemi)
    subsets = np.array_split(idx_sorted, k)
    return [s for s in subsets if s.size > 0]


def _subset_hallucination_ratio(ref_subset: np.ndarray, pred_subset: np.ndarray) -> float:
    no_yes = np.sum((ref_subset == "no") & (pred_subset == "yes"))
    no_no = np.sum((ref_subset == "no") & (pred_subset == "no"))
    denom = no_yes + no_no
    if denom == 0:
        return 0.0
    return float(no_yes / denom)


def _hallucination_mask(ref: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return (ref == "no") & (pred == "yes")


def _apply_balanced_sampling(
    pemi: np.ndarray,
    ref: np.ndarray,
    pred: np.ndarray,
    questions: list[str],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, int]]:
    if pemi.shape[0] != ref.shape[0] or pemi.shape[0] != pred.shape[0] or pemi.shape[0] != len(questions):
        raise ValueError("Balanced sampling input length mismatch")

    mask_pos = _hallucination_mask(ref, pred)
    pos_idx = np.where(mask_pos)[0]
    neg_idx = np.where(~mask_pos)[0]

    n_pos = int(pos_idx.shape[0])
    n_neg = int(neg_idx.shape[0])

    if n_pos == 0:
        raise ValueError("Cannot balance classes: no hallucinated samples found in OOD data")
    if n_neg == 0:
        raise ValueError("Cannot balance classes: no non-hallucinated samples found in OOD data")

    rng = np.random.default_rng(seed)
    keep_neg_idx = rng.choice(neg_idx, size=n_pos, replace=False) if n_neg > n_pos else neg_idx
    keep_idx = np.concatenate([pos_idx, keep_neg_idx])
    rng.shuffle(keep_idx)

    q_arr = np.asarray(questions, dtype=object)
    pemi_b = pemi[keep_idx]
    ref_b = ref[keep_idx]
    pred_b = pred[keep_idx]
    q_b = q_arr[keep_idx].tolist()

    meta = {
        "num_ood_original": int(pemi.shape[0]),
        "num_hallucinated_original": n_pos,
        "num_non_hallucinated_original": n_neg,
        "num_ood_balanced": int(pemi_b.shape[0]),
        "num_hallucinated_balanced": int(np.sum(_hallucination_mask(ref_b, pred_b))),
        "num_non_hallucinated_balanced": int(np.sum(~_hallucination_mask(ref_b, pred_b))),
    }
    return pemi_b, ref_b, pred_b, q_b, meta


def evaluate_k(
    pemi_ood: np.ndarray,
    ref: np.ndarray,
    pred: np.ndarray,
    k: int,
    emi_id_class: float,
    x_ood: torch.Tensor,
    y_model_ood: torch.Tensor,
    y_ref_ood: torch.Tensor,
    emi_estimator: EMI,
) -> dict[str, Any]:
    subsets = _split_sorted_indices_by_pemi(pemi_ood, k)

    subset_emi_ood_avg_pemi: list[float] = []
    subset_emid_avg_pemi: list[float] = []
    subset_emi_ood_emi_class: list[float] = []
    subset_emid_emi_class: list[float] = []
    subset_hr: list[float] = []

    for s in subsets:
        emi_k_avg_pemi = float(np.mean(pemi_ood[s]))
        emid_k_avg_pemi = float(emi_id_class - emi_k_avg_pemi)

        s_t = torch.as_tensor(s, dtype=torch.long)
        emi_k_class = _compute_emi_from_class(
            emi_estimator=emi_estimator,
            x=x_ood.index_select(0, s_t),
            y_model=y_model_ood.index_select(0, s_t),
            y_ref=y_ref_ood.index_select(0, s_t),
        )
        emid_k_class = float(emi_id_class - emi_k_class)

        hr_k = _subset_hallucination_ratio(ref[s], pred[s])
        subset_emi_ood_avg_pemi.append(emi_k_avg_pemi)
        subset_emid_avg_pemi.append(emid_k_avg_pemi)
        subset_emi_ood_emi_class.append(emi_k_class)
        subset_emid_emi_class.append(emid_k_class)
        subset_hr.append(hr_k)

    emid_avg_arr = np.asarray(subset_emid_avg_pemi, dtype=np.float64)
    emid_class_arr = np.asarray(subset_emid_emi_class, dtype=np.float64)
    hr_arr = np.asarray(subset_hr, dtype=np.float64)

    rho_avg, p_value_avg = spearmanr(emid_avg_arr, hr_arr)
    rho_class, p_value_class = spearmanr(emid_class_arr, hr_arr)
    if np.isnan(rho_avg):
        rho_avg = 0.0
    if np.isnan(p_value_avg):
        p_value_avg = 1.0
    if np.isnan(rho_class):
        rho_class = 0.0
    if np.isnan(p_value_class):
        p_value_class = 1.0

    return {
        "K": int(k),
        "rho_avg_pemi": float(rho_avg),
        "p_value_avg_pemi": float(p_value_avg),
        "rho_emi_class": float(rho_class),
        "p_value_emi_class": float(p_value_class),
        "num_subsets": int(len(subsets)),
        "subset_emid_avg_pemi": subset_emid_avg_pemi,
        "subset_emi_ood_avg_pemi": subset_emi_ood_avg_pemi,
        "subset_emid_emi_class": subset_emid_emi_class,
        "subset_emi_ood_emi_class": subset_emi_ood_emi_class,
        "subset_hallucination_ratio": subset_hr,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correlation between subset EMID and subset hallucination ratio across K bins using two EMI methods"
    )
    parser.add_argument("--input-json", type=str, required=True, help="OOD JSON/JSONL with pointwise_emi, reference_answer, model_answer")
    parser.add_argument("--id-json", type=str, required=True, help="ID JSON/JSONL (llava-coco-eng) with question/prompt + reference_answer + model_answer/text")
    parser.add_argument("--id-reference-json", type=str, default="", help="Optional ID reference JSON/JSONL (auto-detected from combined_dataset if omitted)")
    parser.add_argument("--club-ckpt-path", type=str, required=True, help="Path to trained CLUB checkpoint for EMI class method")
    parser.add_argument("--text-embedder", type=str, default="xlm-roberta-base")
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument("--feature-dim", type=int, default=768)
    parser.add_argument("--mi-est-dim", type=int, default=256)
    parser.add_argument("--k-values", type=str, default="10,15,20,25", help="Comma-separated K values")
    parser.add_argument(
        "--balanced-classes",
        action="store_true",
        help="Balance OOD classes by downsampling non-hallucinated samples to match hallucinated sample count",
    )
    parser.add_argument("--balance-seed", type=int, default=42, help="Random seed for balanced class sampling")
    parser.add_argument(
        "--output-json",
        type=str,
        default="results/emi_vs_hallucination/emid_vs_hallucination_rate.json",
        help="Path to save results JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    inferred_mi_est_dim = _infer_mi_est_dim_from_ckpt(Path(args.club_ckpt_path))
    if inferred_mi_est_dim is not None and inferred_mi_est_dim != args.mi_est_dim:
        print(
            f"[Info] Overriding --mi-est-dim from {args.mi_est_dim} to {inferred_mi_est_dim} "
            f"to match checkpoint: {args.club_ckpt_path}"
        )
        args.mi_est_dim = inferred_mi_est_dim

    rows_ood = _load_rows(Path(args.input_json))
    pemi_ood, ref, pred, q_ood = _extract_arrays(rows_ood)
    balance_meta = {
        "num_ood_original": int(len(pemi_ood)),
        "num_hallucinated_original": int(np.sum(_hallucination_mask(ref, pred))),
        "num_non_hallucinated_original": int(np.sum(~_hallucination_mask(ref, pred))),
        "num_ood_balanced": int(len(pemi_ood)),
        "num_hallucinated_balanced": int(np.sum(_hallucination_mask(ref, pred))),
        "num_non_hallucinated_balanced": int(np.sum(~_hallucination_mask(ref, pred))),
    }
    if args.balanced_classes:
        pemi_ood, ref, pred, q_ood, balance_meta = _apply_balanced_sampling(
            pemi=pemi_ood,
            ref=ref,
            pred=pred,
            questions=q_ood,
            seed=args.balance_seed,
        )

    rows_id = _load_rows(Path(args.id_json))
    if args.id_reference_json:
        ref_rows = _load_reference_rows(Path(args.id_reference_json))
        rows_id = _enrich_rows_with_reference(rows_id, ref_rows)
    else:
        auto_ref = _auto_find_reference_file(Path(args.id_json))
        if auto_ref is not None:
            ref_rows = _load_reference_rows(auto_ref)
            rows_id = _enrich_rows_with_reference(rows_id, ref_rows)

    ref_id, pred_id, q_id = _extract_qa_arrays(rows_id)

    x_ood = _encode_texts(q_ood, model_name=args.text_embedder, batch_size=args.embed_batch_size)
    y_ref_ood = _encode_texts([str(v) for v in ref], model_name=args.text_embedder, batch_size=args.embed_batch_size)
    y_model_ood = _encode_texts([str(v) for v in pred], model_name=args.text_embedder, batch_size=args.embed_batch_size)

    x_id = _encode_texts(q_id, model_name=args.text_embedder, batch_size=args.embed_batch_size)
    y_ref_id = _encode_texts([str(v) for v in ref_id], model_name=args.text_embedder, batch_size=args.embed_batch_size)
    y_model_id = _encode_texts([str(v) for v in pred_id], model_name=args.text_embedder, batch_size=args.embed_batch_size)

    emi_estimator = EMI(
        feature_dim=args.feature_dim,
        mi_est_dim=args.mi_est_dim,
        mi_ckpt_path=args.club_ckpt_path,
        v_embedder_name=None,
        t_embedder_name=None,
    )
    emi_estimator.eval()

    emi_id_class = _compute_emi_from_class(
        emi_estimator=emi_estimator,
        x=x_id,
        y_model=y_model_id,
        y_ref=y_ref_id,
    )

    k_values = [int(v.strip()) for v in args.k_values.split(",") if v.strip()]
    if any(k <= 1 for k in k_values):
        raise ValueError("All K values must be > 1")
    if any(k > len(pemi_ood) for k in k_values):
        raise ValueError(f"Each K must be <= number of valid OOD samples ({len(pemi_ood)})")

    results = [
        evaluate_k(
            pemi_ood=pemi_ood,
            ref=ref,
            pred=pred,
            k=k,
            emi_id_class=emi_id_class,
            x_ood=x_ood,
            y_model_ood=y_model_ood,
            y_ref_ood=y_ref_ood,
            emi_estimator=emi_estimator,
        )
        for k in k_values
    ]

    payload = {
        "num_samples_ood": int(len(pemi_ood)),
        "num_samples_id": int(len(q_id)),
        "balanced_classes": bool(args.balanced_classes),
        "balance_seed": int(args.balance_seed),
        "class_balance": balance_meta,
        "emi_id_emi_class": float(emi_id_class),
        "k_values": k_values,
        "hallucination_ratio_definition": "no->yes / (no->yes + no->no)",
        "emid_definitions": {
            "emid_avg_pemi": "EMID_avg_pemi = EMI_ID_emi_class - EMI_OOD_avg_pemi",
            "emid_emi_class": "EMID_emi_class = EMI_ID_emi_class - EMI_OOD_emi_class",
        },
        "methods": {
            "id_method": "EMI_ID computed only with EMI class in main.py (CLUB estimator)",
            "ood_method_avg_pemi": "EMI_OOD from arithmetic mean of pointwise_emi",
            "ood_method_emi_class": "EMI_OOD from EMI class in main.py (CLUB estimator)",
        },
        "results": results,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Saved results:", output_path)


if __name__ == "__main__":
    main()
