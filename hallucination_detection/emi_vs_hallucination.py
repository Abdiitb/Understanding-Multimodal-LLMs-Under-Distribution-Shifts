from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import XLMRobertaModel, XLMRobertaTokenizer

import sys

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
	sys.path.insert(0, str(repo_root))

from gradio_app.estimator import CLUB
from hallucination_detection.bootstrap_utils import bootstrap_confidence_interval
from hallucination_detection.pointwise_emi import PointwiseEMI


@dataclass
class CorrelationResult:
	rho: float
	p_value: float
	ci_lower: float
	ci_upper: float


def _normalize_yes_no(value: object) -> str:
	text = str(value).strip().lower()
	if text in {"yes", "y", "1", "true"}:
		return "yes"
	if text in {"no", "n", "0", "false"}:
		return "no"
	return text


def _hallucination_label(reference_answer: object, model_answer: object) -> int:
	ref = _normalize_yes_no(reference_answer)
	pred = _normalize_yes_no(model_answer)
	return 1 if (ref == "no" and pred == "yes") else 0


def _load_response_rows(path: Path) -> list[dict]:
	with path.open("r", encoding="utf-8") as f:
		obj = json.load(f)

	if isinstance(obj, list):
		return [r for r in obj if isinstance(r, dict)]

	if isinstance(obj, dict):
		if "records" in obj and isinstance(obj["records"], list):
			return [r for r in obj["records"] if isinstance(r, dict)]

		if "categories" in obj and isinstance(obj["categories"], dict):
			rows: list[dict] = []
			for category, items in obj["categories"].items():
				if not isinstance(items, list):
					continue
				for row in items:
					if isinstance(row, dict):
						merged = dict(row)
						merged.setdefault("category", category)
						rows.append(merged)
			return rows

	raise ValueError(f"Unsupported response JSON structure: {path}")


def _load_tensor(path: Path, name: str) -> torch.Tensor:
	suffix = path.suffix.lower()
	if suffix in {".pt", ".pth"}:
		obj = torch.load(path, map_location="cpu")
		if isinstance(obj, torch.Tensor):
			return obj
		raise ValueError(f"{name} file must contain a tensor when using .pt/.pth: {path}")

	if suffix == ".npy":
		arr = np.load(path)
		return torch.from_numpy(arr)

	raise ValueError(f"Unsupported file format for {name}: {path} (use .pt/.pth/.npy)")


def _to_pairs(x: torch.Tensor, y: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
	return [(x[i], y[i]) for i in range(x.shape[0])]


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

	return torch.cat(all_embeddings, dim=0)


def _extract_embeddings_from_responses_json(
	response_rows: list[dict],
	text_embedder_name: str,
	batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	questions = [str(r.get("question", "")) for r in response_rows]
	ref_answers = [str(r.get("reference_answer", "")) for r in response_rows]
	model_answers = [str(r.get("model_answer", "")) for r in response_rows]

	if any(q.strip() == "" for q in questions):
		raise ValueError("responses-json contains empty question field(s)")

	x = _encode_texts(questions, model_name=text_embedder_name, batch_size=batch_size).float()
	y_true = _encode_texts(ref_answers, model_name=text_embedder_name, batch_size=batch_size).float()
	y_model = _encode_texts(model_answers, model_name=text_embedder_name, batch_size=batch_size).float()
	return x, y_true, y_model


def compute_emi_hallucination_correlation(
	emi_calculator: PointwiseEMI,
	pairs_true: list[tuple[torch.Tensor, torch.Tensor]],
	pairs_model: list[tuple[torch.Tensor, torch.Tensor]],
	h: torch.Tensor,
	bootstrap_samples: int,
	one_sided_if_negative: bool,
	seed: int,
) -> CorrelationResult:
	# Step 1: pointwise EMI
	result = emi_calculator.compute_from_pairs(pairs_true, pairs_model)
	e = result.pointwise_emi

	# Step 2: to numpy
	e_np = e.detach().cpu().numpy()
	h_np = h.detach().cpu().numpy()

	if e_np.ndim != 1 or h_np.ndim != 1:
		raise ValueError("Both pointwise EMI and hallucination labels must be 1D")
	if e_np.shape[0] != h_np.shape[0]:
		raise ValueError(f"Length mismatch: len(e)={e_np.shape[0]} vs len(h)={h_np.shape[0]}")

	# Step 3: Spearman rho + p-value
	rho, p_value = spearmanr(e_np, h_np)
	if np.isnan(rho) or np.isnan(p_value):
		raise ValueError("Spearman correlation returned NaN; check variance in inputs")

	if one_sided_if_negative and rho < 0:
		p_value = p_value / 2.0

	# Step 4: bootstrap CI (reusable utility)
	def _rho_stat(a: np.ndarray, b: np.ndarray) -> float:
		rho_b, _ = spearmanr(a, b)
		return float(rho_b)

	ci_lower, ci_upper, _ = bootstrap_confidence_interval(
		x=e_np,
		y=h_np,
		statistic_fn=_rho_stat,
		num_bootstrap=bootstrap_samples,
		seed=seed,
		lower_percentile=2.5,
		upper_percentile=97.5,
	)

	return CorrelationResult(
		rho=float(rho),
		p_value=float(p_value),
		ci_lower=ci_lower,
		ci_upper=ci_upper,
	)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Test correlation between pointwise EMI and hallucination labels")
	parser.add_argument(
		"--responses-json",
		type=str,
		required=True,
		help="JSON/JSON dict-list of responses with keys reference_answer/model_answer to derive hallucination labels",
	)
	parser.add_argument("--club-ckpt-path", type=str, required=True, help="Path to trained CLUB checkpoint (.pt)")
	parser.add_argument("--feature-dim", type=int, default=768)
	parser.add_argument("--club-hidden-dim", type=int, default=256)
	parser.add_argument("--text-embedder", type=str, default="xlm-roberta-base")
	parser.add_argument("--embed-batch-size", type=int, default=64)
	parser.add_argument("--num-negative-samples", type=int, default=50)
	parser.add_argument("--bootstrap-samples", type=int, default=1000)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--one-sided-if-negative", action="store_true")
	parser.add_argument("--output-json", type=str, default="")
	parser.add_argument(
		"--output-records-json",
		type=str,
		default="",
		help="Optional path to save per-sample records with hallucination_label and pointwise_emi",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	response_rows = _load_response_rows(Path(args.responses_json))
	x, y_true, y_model = _extract_embeddings_from_responses_json(
		response_rows=response_rows,
		text_embedder_name=args.text_embedder,
		batch_size=args.embed_batch_size,
	)

	h = torch.tensor(
		[_hallucination_label(r.get("reference_answer", ""), r.get("model_answer", "")) for r in response_rows],
		dtype=torch.float32,
	)

	if x.ndim != 2 or y_true.ndim != 2 or y_model.ndim != 2:
		raise ValueError("x, y_true, and y_model must be 2D tensors [N, D]")
	if x.shape != y_true.shape or x.shape != y_model.shape:
		raise ValueError(f"Shape mismatch: x={tuple(x.shape)}, y_true={tuple(y_true.shape)}, y_model={tuple(y_model.shape)}")
	if h.shape[0] != x.shape[0]:
		raise ValueError(f"Length mismatch: h has {h.shape[0]} entries but x has {x.shape[0]} samples")
	if response_rows and len(response_rows) != x.shape[0]:
		raise ValueError(f"Length mismatch: responses-json has {len(response_rows)} rows but x has {x.shape[0]} samples")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	club = CLUB(args.feature_dim, args.feature_dim, args.club_hidden_dim).to(device)
	club.load_state_dict(torch.load(args.club_ckpt_path, map_location=device))
	club.eval()

	emi_calculator = PointwiseEMI(
		club_estimator=club,
		num_negative_samples=args.num_negative_samples,
		device=device,
		seed=args.seed,
	)

	pairs_true = _to_pairs(x, y_true)
	pairs_model = _to_pairs(x, y_model)
	pointwise_result = emi_calculator.compute_from_pairs(pairs_true, pairs_model)

	corr = compute_emi_hallucination_correlation(
		emi_calculator=emi_calculator,
		pairs_true=pairs_true,
		pairs_model=pairs_model,
		h=h,
		bootstrap_samples=args.bootstrap_samples,
		one_sided_if_negative=args.one_sided_if_negative,
		seed=args.seed,
	)

	result_payload = {
		"rho": corr.rho,
		"p_value": corr.p_value,
		"ci": [corr.ci_lower, corr.ci_upper],
		"bootstrap_samples": args.bootstrap_samples,
		"num_negative_samples": args.num_negative_samples,
		"num_points": int(x.shape[0]),
	}

	print("Spearman rho:", corr.rho)
	print("p-value:", corr.p_value)
	print("95% CI:", [corr.ci_lower, corr.ci_upper])

	if args.output_json:
		output_path = Path(args.output_json)
		output_path.parent.mkdir(parents=True, exist_ok=True)
		with output_path.open("w", encoding="utf-8") as f:
			json.dump(result_payload, f, indent=2)
		print("Saved result JSON:", output_path)

	if args.output_records_json:
		record_scores = pointwise_result.pointwise_emi.detach().cpu().tolist()
		h_vals = h.detach().cpu().tolist()

		if response_rows:
			enriched_rows: list[dict] = []
			for idx, row in enumerate(response_rows):
				updated = dict(row)
				updated["hallucination_label"] = int(h_vals[idx])
				updated["pointwise_emi"] = float(record_scores[idx])
				enriched_rows.append(updated)
		else:
			enriched_rows = [
				{
					"index": idx,
					"hallucination_label": int(h_vals[idx]),
					"pointwise_emi": float(record_scores[idx]),
				}
				for idx in range(len(record_scores))
			]

		rows_output_path = Path(args.output_records_json)
		rows_output_path.parent.mkdir(parents=True, exist_ok=True)
		with rows_output_path.open("w", encoding="utf-8") as f:
			json.dump(enriched_rows, f, indent=2)
		print("Saved per-sample records JSON:", rows_output_path)


if __name__ == "__main__":
	main()

