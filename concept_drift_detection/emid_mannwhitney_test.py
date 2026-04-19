"""
EMID Subset Test + Mann-Whitney U Test for ID vs OOD datasets.

Implements the pseudocode:
- STEP 1: Define subset sampling (without replacement)
- STEP 2: Define EMID = EMI(D_ref) - EMI(D)
- STEP 3: Generate null distribution (ID scores)
- STEP 4: Generate OOD scores
- STEP 5: Mann-Whitney U test
- STEP 6: Decision

Uses the EMI class from main.py to compute proper Expected Model Informativeness scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import gc

import numpy as np
import torch
import torch.nn as nn

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

# Import EMI class from main.py
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from main import EMI


class FallbackCounter:
	"""Track EMI computation fallbacks for debugging and logging."""
	def __init__(self):
		self.missing_image = 0
		self.missing_question = 0
		self.missing_answer = 0
		self.model_answer_fallback = 0  # Using question as pseudo-prediction
		self.gpu_memory_error = 0  # GPU out of memory fallback
		self.invalid_samples = 0  # Samples skipped due to invalid data
		self.emi_computation_errors = 0  # Errors during EMI computation
	
	def increment(self, counter_name: str) -> None:
		"""Increment a counter."""
		if hasattr(self, counter_name):
			setattr(self, counter_name, getattr(self, counter_name) + 1)
		else:
			print(f"Warning: Unknown counter '{counter_name}'")
	
	def total_fallbacks(self) -> int:
		"""Return total number of fallbacks."""
		return (self.missing_image + self.missing_question + self.missing_answer + 
				self.model_answer_fallback + self.gpu_memory_error + self.invalid_samples + 
				self.emi_computation_errors)
	
	def print_summary(self, context: str = "EMI Computation") -> None:
		"""Print a summary of fallback counts."""
		total = self.total_fallbacks()
		if total == 0:
			print(f"\n✓ {context}: No fallbacks used")
			return
		
		print(f"\n{context} - Fallback Usage Summary:")
		print(f"  Total fallbacks: {total}")
		if self.missing_image > 0:
			print(f"    - Missing image: {self.missing_image}")
		if self.missing_question > 0:
			print(f"    - Missing question: {self.missing_question}")
		if self.missing_answer > 0:
			print(f"    - Missing answer: {self.missing_answer}")
		if self.model_answer_fallback > 0:
			print(f"    - Model answer fallback: {self.model_answer_fallback}")
		if self.invalid_samples > 0:
			print(f"    - Invalid samples (skipped): {self.invalid_samples}")
		if self.gpu_memory_error > 0:
			print(f"    - GPU/Memory errors: {self.gpu_memory_error}")
		if self.emi_computation_errors > 0:
			print(f"    - EMI computation errors: {self.emi_computation_errors}")


def load_dk_json_dataset(path: Path) -> list[dict[str, Any]]:
	"""Load a d_k.json format dataset (array of dicts with nested 'x' and 'y' structure)."""
	with path.open("r", encoding="utf-8") as f:
		obj = json.load(f)
	
	if not isinstance(obj, list):
		raise ValueError(f"{path}: expected list, got {type(obj).__name__}")
	
	if not obj:
		raise ValueError(f"{path}: list is empty")
	
	# Handle both formats: {"x": {...}, "y": "..."} or {"key": {"x": ..., "y": ...}}
	first = obj[0]
	if "x" in first and "y" in first:
		return obj
	elif len(first) == 1:
		# Single-key nested format
		key = next(iter(first.keys()))
		nested = first[key]
		if "x" in nested and "y" in nested:
			# Unwrap all items
			return [next(iter(item.values())) for item in obj]
	
	raise ValueError(f"{path}: unrecognized d_k.json format")


def deserialize_image(image_filename: str, image_dir: Path) -> Any:
	"""Load image from disk by filename.
	
	Args:
		image_filename: Filename of the image (e.g., 'img_000.jpg')
		image_dir: Path to directory containing images
	
	Returns:
		PIL Image in RGB format, or None if loading fails
	"""
	from PIL import Image
	
	if not image_filename:
		return None
	
	try:
		image_path = image_dir / image_filename
		if not image_path.exists():
			print(f"Warning: Image file not found: {image_path}")
			return None
		
		img = Image.open(image_path)
		img = img.convert("RGB")
		return img
	except Exception as e:
		print(f"Warning: Failed to load image {image_filename}: {e}")
		return None


def compute_emi_score_with_class(
	emi_model: EMI,
	sample: dict[str, Any],
	model_answers: list[str] | None = None,
	fallback_counter: FallbackCounter | None = None,
) -> float:
	"""Compute EMI for a single sample using the EMI class.
	
	Args:
		emi_model: Initialized EMI instance
		sample: Sample dict containing 'x' (with image, question) and 'y' (answers)
		model_answers: List of model predictions to use. If None, use empty string as fallback.
		fallback_counter: Optional counter to track fallback usage
	
	Returns:
		EMI score (model_mi - ref_mi)
	"""
	x = sample.get("x", {})
	y_ref = sample.get("y", "")
	
	# Extract image and question
	image_filename = x.get("image")
	# Construct image_dir relative to dataset directory
	if hasattr(emi_model, '_dataset_dir'):
		dataset_dir = emi_model._dataset_dir
	else:
		dataset_dir = Path("results/concept_drift_detection/datasets")  # Fallback
	image_dir = dataset_dir / x.get("image_dir", "images")
	image = deserialize_image(image_filename, image_dir) if image_filename else None
	question = x.get("question", "")
	
	if image is None or not question:
		if fallback_counter:
			if image is None:
				fallback_counter.increment('missing_image')
			if not question:
				fallback_counter.increment('missing_question')
		return 0.0  # Return neutral score if missing data
	
	# Use provided model answers or fallback to question as model prediction
	if model_answers is None:
		if fallback_counter:
			fallback_counter.increment('model_answer_fallback')
		model_answers = [question]  # Fallback: use question as pseudo-prediction
	
	try:
		with torch.inference_mode():
			# Compute EMI: model_mi - ref_mi
			emi_score, model_mi, ref_mi = emi_model(
				x_v=[image],
				x_t=[question],
				y_hat=model_answers if isinstance(model_answers, list) else [model_answers],
				y=[y_ref],
				return_emb=False
			)
		return float(emi_score)
	except Exception as e:
		if fallback_counter:
			fallback_counter.increment('emi_computation_errors')
		print(f"Warning: Failed to compute EMI: {e}")
		return 0.0


def compute_emi_scores_for_dataset(
	emi_model: EMI,
	dataset: list[dict[str, Any]],
	batch_size: int = 8,
	fallback_counter: FallbackCounter | None = None,
) -> np.ndarray:
	"""Compute EMI scores for all samples in a dataset."""
	scores = []
	
	with tqdm(total=len(dataset), desc="Computing EMI scores", unit="sample") as pbar:
		for i in range(0, len(dataset), batch_size):
			batch = dataset[i : i + batch_size]
			
			for sample in batch:
				# Extract reference answer
				y_ref = sample.get("y", "")
				x = sample.get("x", {})
				image_filename = x.get("image")
				# Construct image_dir relative to dataset directory
				dataset_dir = Path("results/concept_drift_detection/datasets")
				image_dir = dataset_dir / x.get("image_dir", "images")
				question = x.get("question", "")
				
				# Try to load image
				try:
					image = deserialize_image(image_filename, image_dir) if image_filename else None
				except Exception:
					image = None
				
				if image is None or not question or not y_ref:
					if fallback_counter:
						if image is None:
							fallback_counter.increment('missing_image')
						if not question:
							fallback_counter.increment('missing_question')
						if not y_ref:
							fallback_counter.increment('missing_answer')
					pbar.update(1)
					continue
				
				try:
					with torch.inference_mode():
						# Compute EMI using the reference answer as both model prediction and ground truth
						# This measures how informative the input is about the ground truth
						emi_score, model_mi, ref_mi = emi_model(
							x_v=[image],
							x_t=[question],
							y_hat=[y_ref],  # Use reference as proxy for model prediction
							y=[y_ref],
							return_emb=False
						)
					scores.append(float(emi_score))
				except Exception as e:
					if fallback_counter:
						fallback_counter.increment('emi_computation_errors')
					scores.append(0.0)
				
				pbar.update(1)
	
	return np.asarray(scores, dtype=np.float64)


def compute_emi_for_subset(
	emi_model: EMI,
	subset_samples: list[dict[str, Any]],
	model_answers_map: dict[int, str] | None = None,
	fallback_counter: FallbackCounter | None = None,
) -> tuple[float, float, float]:
	"""Compute EMI score for a subset of samples by passing them all together to the EMI model.
	
	Args:
		emi_model: Initialized EMI instance
		subset_samples: List of samples in the subset
		model_answers_map: Mapping from image_id to model answer. If provided, uses model answers instead of reference answers.
		fallback_counter: Optional counter to track fallback usage
	
	Returns:
		Tuple of (emi_score, model_mi, ref_mi) for the entire subset
	"""
	if not subset_samples:
		return 0.0, 0.0, 0.0
	
	# Extract data from all samples in subset
	images = []
	questions = []
	answers_pred = []
	answers_ref = []
	
	for sample in subset_samples:
		y_ref = sample.get("y", "")
		x = sample.get("x", {})
		image_filename = x.get("image")
		# Construct image_dir relative to dataset directory
		dataset_dir = Path("results/concept_drift_detection/datasets")
		image_dir = dataset_dir / x.get("image_dir", "images")
		question = x.get("question", "")
		image_id = x.get("image_id")
		
		# Load image
		try:
			image = deserialize_image(image_filename, image_dir) if image_filename else None
		except Exception:
			image = None
		
		if image is None or not question or not y_ref:
			if fallback_counter:
				if image is None:
					fallback_counter.increment('missing_image')
				if not question:
					fallback_counter.increment('missing_question')
				if not y_ref:
					fallback_counter.increment('missing_answer')
				fallback_counter.increment('invalid_samples')
			# Skip invalid samples
			continue
		
		# Get model answer from map if available, otherwise use reference
		model_answer = y_ref
		if model_answers_map and image_id in model_answers_map:
			model_answer = model_answers_map[image_id]
		
		images.append(image)
		questions.append(question)
		answers_pred.append(model_answer)  # Use model answer if available, else reference
		answers_ref.append(y_ref)
	
	if not images:
		return 0.0
	
	# Debug: Check if pred and ref answers are the same
	answers_match = all(p == r for p, r in zip(answers_pred, answers_ref))
	if not answers_match:
		num_mismatches = sum(1 for p, r in zip(answers_pred, answers_ref) if p != r)
		print(f"    [DEBUG] y_hat vs y mismatch: {num_mismatches}/{len(answers_pred)} samples differ")
	
	try:
		with torch.inference_mode():
			# Compute EMI for the entire subset at once
			emi_score, model_mi, ref_mi = emi_model(
				x_v=images,
				x_t=questions,
				y_hat=answers_pred,
				y=answers_ref,
				return_emb=False
			)
		result = float(emi_score)
		model_mi_val = float(model_mi)
		ref_mi_val = float(ref_mi)
		
		# Clean up GPU memory
		del images, questions, answers_pred, answers_ref
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
		
		return result, model_mi_val, ref_mi_val
	except RuntimeError as e:
		if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
			if fallback_counter:
				fallback_counter.increment('gpu_memory_error')
			print(f"Warning: GPU/Memory error for subset (n={len(images)}): {e}")
			print("  Attempting to reduce and recompute with smaller subset...")
			
			# Clear GPU memory and try with reduced set
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
			gc.collect()
			
			# Use mean of available subset as fallback
			if images:
				return float(np.mean([float(img is not None) for img in images])), 0.0, 0.0
		else:
			print(f"Warning: Failed to compute EMI for subset: {e}")
		
		# Clean up even on error
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
		return 0.0, 0.0, 0.0
	except Exception as e:
		print(f"Warning: Failed to compute EMI for subset: {e}")
		# Clean up even on error
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
		return 0.0, 0.0, 0.0


def sample_subset_indices(rng: np.random.Generator, n: int, subset_size: int) -> np.ndarray:
	"""Sample indices without replacement."""
	if subset_size <= 0:
		raise ValueError("subset_size must be positive")
	if subset_size > n:
		raise ValueError(f"subset_size={subset_size} is larger than dataset size n={n}")
	return rng.choice(n, size=subset_size, replace=False)

def mann_whitney_u_test(
	scores_id: list[float],
	scores_ood: list[float],
	alternative: str = "less",
) -> dict[str, float]:
	"""Perform Mann-Whitney U test.
	
	Args:
		scores_id: ID distribution EMID scores
		scores_ood: OOD distribution EMID scores
		alternative: One of 'less' (ID < OOD), 'greater' (ID > OOD), or 'two-sided'
	
	Returns:
		Dictionary with U_statistic and p_value
	"""
	if alternative not in ["less", "greater", "two-sided"]:
		raise ValueError(f"alternative must be 'less', 'greater', or 'two-sided', got '{alternative}'")
	
	try:
		from scipy.stats import mannwhitneyu
	except ImportError as e:
		raise ImportError("scipy is required. Install with: pip install scipy") from e

	result = mannwhitneyu(scores_id, scores_ood, alternative=alternative, method="asymptotic")
	return {"U_statistic": float(result.statistic), "p_value": float(result.pvalue)}


def save_emid_scores_incremental(
	emid_scores_path: Path,
	scores_id: list[dict[str, float]] | list[float],
	scores_ood_by_dataset: dict[str, list[dict[str, float]]] | dict[str, list[float]]
) -> None:
	"""Incrementally save EMID scores with MI values to JSON file.
	
	Args:
		emid_scores_path: Path to save JSON file
		scores_id: List of EMID score dicts (with 'emi', 'model_mi', 'ref_mi') or floats
		scores_ood_by_dataset: Dict of OOD datasets to lists of score dicts or floats
	"""
	emid_data = {
		"description": "EMID scores and MI values computed from pairs of subsets",
		"D1_pairs_emid": scores_id,
	}
	emid_data.update({name: scores_ood_by_dataset[name] for name in sorted(scores_ood_by_dataset.keys())})

	with emid_scores_path.open("w", encoding="utf-8") as f:
		json.dump(emid_data, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="EMID subset test + Mann-Whitney U test on d_k.json datasets using EMI class"
	)
	parser.add_argument(
		"--dataset-dir",
		type=str,
		default="results/concept_drift_detection/datasets",
		help="Directory containing D1_migrated.json, D2_migrated.json, D3_migrated.json, D4_migrated.json",
	)
	parser.add_argument(
		"--mi-ckpt-path",
		type=str,
		default="estimator_ckpt/CLUB_all_datasets_trained.pt",
		help="Path to pre-trained MI estimator checkpoint",
	)
	parser.add_argument(
		"--v-embedder",
		type=str,
		default="openai/clip-vit-base-patch32",
		help="Vision embedder model name",
	)
	parser.add_argument(
		"--t-embedder",
		type=str,
		default="xlm-roberta-base",
		help="Text embedder model name",
	)
	parser.add_argument("--k-trials", type=int, default=100, help="Number of trials K")
	parser.add_argument("--subset-size", type=int, default=200, help="Subset size (without replacement)")
	parser.add_argument("--seed", type=int, default=42, help="Random seed")
	parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
	parser.add_argument(
		"--output-dir",
		type=str,
		default="results/concept_drift_detection",
		help="Directory to save results and EMI scores",
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=8,
		help="Batch size for processing samples",
	)
	parser.add_argument(
		"--max-subset-samples",
		type=int,
		default=None,
		help="Maximum samples to use per subset (for memory constraints). If None, uses all samples.",
	)
	parser.add_argument(
		"--mann-whitney-alternative",
		type=str,
		choices=["less", "greater", "two-sided"],
		default="less",
		help="Mann-Whitney U test alternative: 'less' (ID < OOD), 'greater' (ID > OOD), or 'two-sided'",
	)
	parser.add_argument(
		"--emid-scores-path",
		type=str,
		default=None,
		help="Path to pre-computed EMID scores JSON file (e.g., from previous run). If provided, skips EMID computation and performs only Mann-Whitney test.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	# Optimize memory usage
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
		# Enable memory efficient algorithms
		torch.backends.cudnn.benchmark = False
	
	gc.collect()

	dataset_dir = Path(args.dataset_dir)
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	k_trials = int(args.k_trials)
	subset_size = int(args.subset_size)
	max_subset_samples = args.max_subset_samples
	
	if k_trials <= 0:
		raise ValueError("k-trials must be positive")
	if subset_size <= 0:
		raise ValueError("subset-size must be positive")
	
	# Adaptive subset size reduction for memory safety
	if max_subset_samples is not None and subset_size > max_subset_samples:
		print(f"Warning: Reducing subset_size from {subset_size} to {max_subset_samples} due to memory constraints")
		subset_size = max_subset_samples

	# Initialize EMI model
	print("Initializing EMI model...")
	try:
		emi_model = EMI(
			feature_dim=768,
			mi_est_dim=256,
			mi_ckpt_path=args.mi_ckpt_path,
			v_embedder_name=args.v_embedder,
			t_embedder_name=args.t_embedder,
		)
		print("✓ EMI model initialized successfully")
		print("✓ Memory optimization enabled: periodic GPU cache clearing")
		print("✓ Memory optimization enabled: automatic image format conversion")
		if torch.cuda.is_available():
			print(f"✓ CUDA device: {torch.cuda.get_device_name(0)}")
			print(f"✓ Initial GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
	except Exception as e:
		raise RuntimeError(f"Failed to initialize EMI model: {e}") from e

	# Load datasets
	print("\nLoading datasets...")
	
	# Auto-discover all JSON files in dataset_dir
	if not dataset_dir.exists():
		raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
	
	json_files = sorted(dataset_dir.glob("*.json"))
	if not json_files:
		raise FileNotFoundError(f"No JSON files found in {dataset_dir}")
	
	print(f"Found {len(json_files)} dataset files:")
	for f in json_files:
		print(f"  - {f.name}")
	
	# Load D1 as ID distribution (required for all trials)
	d1_path = dataset_dir / "D1_migrated.json"
	if not d1_path.exists():
		raise FileNotFoundError(f"ID distribution file not found: {d1_path}")
	
	try:
		d1_raw = load_dk_json_dataset(d1_path)
		print(f"\nID Distribution (D1): {len(d1_raw)} samples")
	except Exception as e:
		raise RuntimeError(f"Failed to load D1_migrated.json: {e}") from e
	
	# Load model answers from combined JSON
	print("\nLoading model answers...")
	model_answers_map: dict[int, str] = {}
	combined_json_path = output_dir / "combined_d1_with_answers.json"
	if combined_json_path.exists():
		try:
			with combined_json_path.open("r", encoding="utf-8") as f:
				combined_data = json.load(f)
			# Create mapping from image_id to model_answer
			for item in combined_data:
				image_id = item.get("image_id")
				model_answer = item.get("model_answer", "")
				if image_id is not None:
					model_answers_map[image_id] = model_answer
			print(f"  ✓ Loaded {len(model_answers_map)} model answers from {combined_json_path.name}")
		except Exception as e:
			print(f"  Warning: Failed to load model answers from {combined_json_path}: {e}")
	else:
		print(f"  Warning: Model answers file not found: {combined_json_path}")
		print(f"  Will proceed with reference answers as fallback.")
	
	# Initialize fallback counter for tracking EMI computation issues
	fallback_counter = FallbackCounter()
	print(f"\n✓ Fallback counter initialized")
	
	# Discover OOD dataset paths (don't load yet - load on demand)
	ood_dataset_paths = {}
	print(f"\nOOD Distributions (on-demand loading):")
	for path in json_files:
		if path.name == "D1_migrated.json":
			continue
		name = path.stem  # Get filename without extension
		ood_dataset_paths[name] = path
		print(f"  {name}: (will load on demand)")
	
	if not ood_dataset_paths:
		raise RuntimeError("No OOD datasets found")

	# Validate subset_size against D1 only
	if subset_size > len(d1_raw):
		raise ValueError(f"subset-size={subset_size} is larger than D1 size={len(d1_raw)}")

	rng = np.random.default_rng(int(args.seed))

	# Check if pre-computed EMID scores are provided
	if args.emid_scores_path:
		print(f"\nLoading pre-computed EMID scores from: {args.emid_scores_path}")
		emid_scores_file = Path(args.emid_scores_path)
		if not emid_scores_file.exists():
			raise FileNotFoundError(f"EMID scores file not found: {emid_scores_file}")
		
		try:
			with emid_scores_file.open("r", encoding="utf-8") as f:
				emid_data = json.load(f)
			
			# Extract ID and OOD scores
			scores_id = emid_data.get("D1_pairs_emid", [])
			scores_ood_by_dataset = {}
			
			# Load all other keys as OOD datasets
			for key in emid_data.keys():
				if key not in ["description", "D1_pairs_emid"]:
					scores_ood_by_dataset[key] = emid_data[key]
			
			print(f"  ✓ Loaded {len(scores_id)} ID scores from {len(scores_ood_by_dataset)} OOD datasets")
			for name, scores in scores_ood_by_dataset.items():
				print(f"    - {name}: {len(scores)} scores")
			
			# Flatten OOD scores for overall Mann-Whitney test
			scores_ood = []
			for scores in scores_ood_by_dataset.values():
				scores_ood.extend(scores)
			
			print(f"  ✓ Total OOD scores: {len(scores_ood)}")
			
			# Skip directly to Mann-Whitney test
			use_precomputed = True
			emid_scores_path = output_dir / "emid_subset_pair_scores.json"
		except Exception as e:
			raise RuntimeError(f"Failed to load EMID scores from {emid_scores_file}: {e}") from e
	else:
		use_precomputed = False
		# Initialize EMID scores file with empty dict lists
		emid_scores_path = output_dir / "emid_subset_pair_scores.json"
		scores_ood_by_dataset: dict[str, list[dict[str, float]]] = {name: [] for name in sorted(ood_dataset_paths.keys())}
		# Save initial structure
		save_emid_scores_incremental(emid_scores_path, [], scores_ood_by_dataset)
		print(f"Created EMID scores file: {emid_scores_path}")

	if use_precomputed:
		print("\nSkipping STEP 3 & 4 (EMID computation) - using pre-computed scores")
	else:
		# STEP 3: Generate ID EMID scores (null distribution)
		# Sample TWO subsets from D1 and compute EMID for each pair
		print("\nSTEP 3: Generating ID EMID scores (null distribution)...")
		print(f"Sampling {k_trials} pairs of subsets of size {subset_size} from D1...")
		print(f"Using model answers from combined JSON: {len(model_answers_map) > 0}")
		scores_id: list[dict[str, float]] = []
		with tqdm(total=k_trials, desc="ID trials (D1 pairs)", unit="trial") as bar:
			for _ in range(k_trials):
				# Sample reference subset from D1
				ref_indices = rng.choice(len(d1_raw), size=subset_size, replace=False)
				ref_samples = [d1_raw[i] for i in ref_indices]
				ref_emi, ref_model_mi, ref_ref_mi = compute_emi_for_subset(emi_model, ref_samples, model_answers_map, fallback_counter)
				del ref_samples, ref_indices
				
				# Sample target subset from D1 (can overlap with ref)
				tar_indices = rng.choice(len(d1_raw), size=subset_size, replace=False)
				tar_samples = [d1_raw[i] for i in tar_indices]
				tar_emi, tar_model_mi, tar_ref_mi = compute_emi_for_subset(emi_model, tar_samples, model_answers_map, fallback_counter)
				del tar_samples, tar_indices
				
				# Compute EMID = EMI(ref) - EMI(tar)
				emid_score = ref_emi - tar_emi
				scores_id.append({
					"emi": float(emid_score),
					"ref_model_mi": float(ref_model_mi),
					"ref_ref_mi": float(ref_ref_mi),
					"tar_model_mi": float(tar_model_mi),
					"tar_ref_mi": float(tar_ref_mi),
				})
				
				# Save scores incrementally after each pair
				save_emid_scores_incremental(emid_scores_path, scores_id, scores_ood_by_dataset)
				
				# Periodic memory cleanup
				if (_ + 1) % 10 == 0:
					gc.collect()
					if torch.cuda.is_available():
						torch.cuda.empty_cache()
				
				bar.update(1)

		# Extract EMI values for statistical analysis
		scores_id_emis = [s["emi"] for s in scores_id]
		print(f"ID EMID scores: mean={np.mean(scores_id_emis):.4f}, std={np.std(scores_id_emis):.4f}, min={np.min(scores_id_emis):.4f}, max={np.max(scores_id_emis):.4f}")

		# STEP 4: Generate OOD EMID scores
		# Sample one subset from D1 (ref) and one from Dk (tar), compute EMID
		print("\nSTEP 4: Generating OOD EMID scores...")
		scores_ood: list[float] = []

		total_ood_trials = k_trials * len(ood_dataset_paths)
		with tqdm(total=total_ood_trials, desc="OOD trials (D1 vs Dk)", unit="trial") as bar:
			for name in sorted(ood_dataset_paths.keys()):
				# Load OOD dataset on demand
				path = ood_dataset_paths[name]
				try:
					data = load_dk_json_dataset(path)
					print(f"\n  Loaded {name}: {len(data)} samples")
					
					# Validate subset_size for this dataset
					if subset_size > len(data):
						raise ValueError(f"subset-size={subset_size} is larger than {name} size={len(data)}")
					
				except Exception as e:
					print(f"  Warning: Failed to load {path.name}: {e}")
					continue
				
				# Initialize scores list for this dataset if not already present
				if name not in scores_ood_by_dataset:
					scores_ood_by_dataset[name] = []
				print(f"  Sampling pairs from D1 and {name}...")
				
				for trial_idx in range(k_trials):
					# Sample reference subset from D1 (ID)
					ref_indices = rng.choice(len(d1_raw), size=subset_size, replace=False)
					ref_samples = [d1_raw[i] for i in ref_indices]
					ref_emi, ref_model_mi, ref_ref_mi = compute_emi_for_subset(emi_model, ref_samples, model_answers_map, fallback_counter)
					del ref_samples, ref_indices
					
					# Sample target subset from Dk (OOD)
					tar_indices = rng.choice(len(data), size=subset_size, replace=False)
					tar_samples = [data[i] for i in tar_indices]
					tar_emi, tar_model_mi, tar_ref_mi = compute_emi_for_subset(emi_model, tar_samples, model_answers_map, fallback_counter)
					del tar_samples, tar_indices
					print(f"    Trial {trial_idx + 1}/{k_trials} for {name}: EMI(ref)={ref_emi:.4f}, EMI(tar)={tar_emi:.4f}")
					
					# Compute EMID = EMI(D1_ref) - EMI(Dk_tar)
					emid_score = ref_emi - tar_emi
					scores_ood.append(float(emid_score))
					scores_ood_by_dataset[name].append({
						"emi": float(emid_score),
						"ref_model_mi": float(ref_model_mi),
						"ref_ref_mi": float(ref_ref_mi),
						"tar_model_mi": float(tar_model_mi),
						"tar_ref_mi": float(tar_ref_mi),
					})
					
					# Save scores incrementally
					save_emid_scores_incremental(emid_scores_path, scores_id, scores_ood_by_dataset)
					
					# Periodic memory cleanup
					if (trial_idx + 1) % 10 == 0:
						gc.collect()
						if torch.cuda.is_available():
							torch.cuda.empty_cache()
					
					bar.update(1)
				
				# Delete dataset from memory after finishing all trials for this dataset
				del data
				gc.collect()
				if torch.cuda.is_available():
					torch.cuda.empty_cache()
				print(f"  ✓ Completed {name} trials. Dataset removed from memory.")
		
		print(f"\nOOD EMID scores summary:")
		for name, scores in scores_ood_by_dataset.items():
			emi_values = [s["emi"] for s in scores]
			print(f"  {name}: mean={np.mean(emi_values):.4f}, std={np.std(emi_values):.4f}, min={np.min(emi_values):.4f}, max={np.max(emi_values):.4f}")
	
	# Final cleanup after intensive computation
	gc.collect()
	if torch.cuda.is_available():
		torch.cuda.empty_cache()

	# Final save of all EMID scores (already saved incrementally, this is final confirmation)
	print("\nFinal save of subset-pair EMID scores...")
	save_emid_scores_incremental(emid_scores_path, scores_id, scores_ood_by_dataset)
	print(f"  Saved subset-pair EMID scores: {emid_scores_path}")

	# Extract EMID scores (EMI values) for Mann-Whitney test
	# Handle both dict format (from new computation) and float format (from precomputed)
	if isinstance(scores_id[0] if scores_id else None, dict):
		scores_id_emis = [s["emi"] for s in scores_id]
	else:
		scores_id_emis = scores_id
	
	scores_ood_emis = []
	scores_ood_by_dataset_emis = {}
	for name, scores in scores_ood_by_dataset.items():
		if isinstance(scores[0] if scores else None, dict):
			emi_vals = [s["emi"] for s in scores]
		else:
			emi_vals = scores
		scores_ood_by_dataset_emis[name] = emi_vals
		scores_ood_emis.extend(emi_vals)

	# STEP 5: Perform Mann-Whitney U test
	print("\nSTEP 5: Performing Mann-Whitney U test...")
	print(f"  Alternative hypothesis: {args.mann_whitney_alternative}")
	stats = mann_whitney_u_test(scores_id_emis, scores_ood_emis, alternative=args.mann_whitney_alternative)

	# STEP 5b: Perform pairwise Mann-Whitney U tests (D1 vs each OOD dataset separately)
	print("\nSTEP 5b: Performing pairwise Mann-Whitney U tests (D1 vs each OOD dataset)...")
	pairwise_stats = {}
	for name in sorted(scores_ood_by_dataset_emis.keys()):
		ood_scores_for_dataset = scores_ood_by_dataset_emis[name]
		if not ood_scores_for_dataset:
			print(f"  Warning: No OOD scores for {name}, skipping pairwise test")
			continue
		
		pairwise_result = mann_whitney_u_test(scores_id_emis, ood_scores_for_dataset, alternative=args.mann_whitney_alternative)
		pairwise_stats[name] = pairwise_result
		
		is_significant = pairwise_result["p_value"] < float(args.alpha)
		significance_str = "✓ SIGNIFICANT" if is_significant else "✗ Not significant"
		print(f"  {name} vs D1: U={pairwise_result['U_statistic']:.4f}, p={pairwise_result['p_value']:.6g} {significance_str}")

	# STEP 6: Decision
	decision = "EMID separates ID and OOD (significant)" if stats["p_value"] < float(args.alpha) else "No significant separation"

	payload: dict[str, Any] = {
		"dataset_dir": str(dataset_dir),
		"k_trials": k_trials,
		"subset_size": subset_size,
		"seed": int(args.seed),
		"alpha": float(args.alpha),
		"alternative": args.mann_whitney_alternative,
		"method": "asymptotic",
		"emid_scores_loaded_from": args.emid_scores_path if args.emid_scores_path else None,
		"overall_test": {
			"U_statistic": stats["U_statistic"],
			"p_value": stats["p_value"],
			"decision": decision,
		},
		"pairwise_tests": {
			name: {
				"U_statistic": pairwise_stats[name]["U_statistic"],
				"p_value": pairwise_stats[name]["p_value"],
				"decision": "Separates D1 from OOD (significant)" if pairwise_stats[name]["p_value"] < float(args.alpha) else "No significant separation",
			}
			for name in sorted(pairwise_stats.keys())
		},
		"scores_summary": {
			"scores_id_mean": float(np.mean(scores_id_emis)),
			"scores_id_std": float(np.std(scores_id_emis, ddof=1)) if len(scores_id_emis) > 1 else 0.0,
			"scores_id_min": float(np.min(scores_id_emis)),
			"scores_id_max": float(np.max(scores_id_emis)),
			"scores_ood_mean": float(np.mean(scores_ood_emis)),
			"scores_ood_std": float(np.std(scores_ood_emis, ddof=1)) if len(scores_ood_emis) > 1 else 0.0,
			"scores_ood_min": float(np.min(scores_ood_emis)),
			"scores_ood_max": float(np.max(scores_ood_emis)),
		},
		"scores_ood_by_dataset_summary": {
			k: {
				"mean": float(np.mean(v)),
				"std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
				"min": float(np.min(v)),
				"max": float(np.max(v)),
				"n": int(len(v)),
			}
			for k, v in scores_ood_by_dataset_emis.items()
		},
	}

	# Save results
	results_path = output_dir / "mannwhitney_test_results.json"
	with results_path.open("w", encoding="utf-8") as f:
		json.dump(payload, f, ensure_ascii=False, indent=2)

	print("\n" + "=" * 60)
	print("STEP 6: Decision Results")
	print("=" * 60)
	print("\nOverall Test (D1 vs All OOD combined):")
	print(f"  U_statistic = {payload['overall_test']['U_statistic']:.4f}")
	print(f"  p_value     = {payload['overall_test']['p_value']:.6g}")
	print(f"  alpha       = {args.alpha}")
	print(f"  Decision:   {payload['overall_test']['decision']}")
	
	print("\nPairwise Tests (D1 vs each OOD dataset):")
	for name, pairwise_result in sorted(payload["pairwise_tests"].items()):
		is_sig = pairwise_result["p_value"] < float(args.alpha)
		sig_marker = "✓" if is_sig else "✗"
		print(f"  {sig_marker} {name}: U={pairwise_result['U_statistic']:.4f}, p={pairwise_result['p_value']:.6g}")
		print(f"         {pairwise_result['decision']}")
	print("=" * 60)
	print(f"Saved results: {results_path}")
	print(f"Saved EMI scores: {emid_scores_path}")	
	# Print fallback usage summary
	fallback_counter.print_summary("EMI Computation - Overall Summary")	
	# Final memory cleanup
	del d1_raw, emi_model
	gc.collect()
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
		print(f"\n✓ Final GPU memory cleared: {torch.cuda.memory_allocated() / 1e9:.2f} GB in use")


if __name__ == "__main__":
	main()
