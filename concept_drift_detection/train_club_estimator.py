"""
Train CLUB (Contrastive Learning Upper Bound) MI estimator on all d_k.json datasets.

This script:
1. Discovers all d_k.json files in a directory (e.g., D1_migrated.json, D2_migrated.json, etc.)
2. Loads and combines all datasets
3. Extracts embeddings using CLIP vision encoder and XLM-RoBERTa text encoder
4. Trains a CLUB estimator to approximate MI(X, Y) where X is multimodal features and Y is labels
5. Saves the trained checkpoint
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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

# Import CLUB class from main.py
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from main import CLUB


def load_dk_json_dataset(path: Path) -> list[dict[str, Any]]:
    """Load a d_k.json format dataset."""
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
        key = next(iter(first.keys()))
        nested = first[key]
        if "x" in nested and "y" in nested:
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


def extract_embeddings(
    dataset: list[dict[str, Any]],
    base_dir: Path | str = ".",
    v_embedder_name: str = "openai/clip-vit-base-patch32",
    t_embedder_name: str = "xlm-roberta-base",
    batch_size: int = 4,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract vision and text embeddings from dataset.
    
    Args:
        dataset: List of data samples
        base_dir: Base directory for resolving relative image paths
        v_embedder_name: Vision model name
        t_embedder_name: Text model name
        batch_size: Batch size for embedding extraction
        device: Device to use
    
    Returns:
        x_embeddings: Tensor of shape [N, feature_dim] - multimodal features
        y_embeddings: Tensor of shape [N, feature_dim] - answer embeddings
    """
    from transformers import CLIPModel, CLIPProcessor, XLMRobertaModel, XLMRobertaTokenizer
    
    base_dir = Path(base_dir)
    
    print(f"\nLoading vision and text encoders...")
    v_model = CLIPModel.from_pretrained(v_embedder_name).to(device)
    v_processor = CLIPProcessor.from_pretrained(v_embedder_name)
    t_model = XLMRobertaModel.from_pretrained(t_embedder_name).to(device)
    t_processor = XLMRobertaTokenizer.from_pretrained(t_embedder_name)
    
    x_embeddings = []
    y_embeddings = []
    
    valid_count = 0
    skipped_count = 0
    
    print(f"\nExtracting embeddings from {len(dataset)} samples...")
    with tqdm(total=len(dataset), desc="Extracting embeddings", unit="sample") as pbar:
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            
            # Collect valid samples
            valid_samples = []
            valid_indices = []
            for idx, sample in enumerate(batch):
                y_ref = sample.get("y", "")
                x = sample.get("x", {})
                image_filename = x.get("image")
                # Construct image_dir relative to base directory
                image_dir = base_dir / x.get("image_dir", "images")
                question = x.get("question", "")
                
                try:
                    image = deserialize_image(image_filename, image_dir) if image_filename else None
                except Exception:
                    image = None
                
                if image is not None and question and y_ref:
                    valid_samples.append((image, question, y_ref))
                    valid_indices.append(idx)
                else:
                    skipped_count += 1
                
                pbar.update(1)
            
            if not valid_samples:
                continue
            
            # Extract vision embeddings
            images = [s[0] for s in valid_samples]
            questions = [s[1] for s in valid_samples]
            answers = [s[2] for s in valid_samples]
            
            try:
                v_inputs = v_processor(images=images, return_tensors="pt", padding=True)
                v_inputs = {k: v.to(device) for k, v in v_inputs.items()}
                
                with torch.inference_mode():
                    v_outputs = v_model.vision_model(
                        pixel_values=v_inputs['pixel_values'],
                        output_hidden_states=True
                    )
                    # Extract visual features (mean pooling over patch tokens)
                    z_v = v_outputs.hidden_states[-1][:, 1:, :].mean(dim=1).float()  # [batch, 768]
                
                # Extract text embeddings for questions
                t_inputs = t_processor(questions, return_tensors="pt", padding=True, truncation=True, max_length=512)
                t_inputs = {k: v.to(device) for k, v in t_inputs.items()}
                attn_mask = t_inputs['attention_mask'].unsqueeze(-1)  # [batch, seq_len, 1]
                
                with torch.inference_mode():
                    t_outputs = t_model(**t_inputs, output_hidden_states=True)
                    z_t = (t_outputs.hidden_states[0] * attn_mask).sum(dim=1) / attn_mask.sum(dim=1)  # [batch, 768]
                
                # Combine vision and text features
                z_x = (z_v + z_t) / 2  # [batch, 768] - multimodal features
                
                # Extract answer embeddings
                y_inputs = t_processor(answers, return_tensors="pt", padding=True, truncation=True, max_length=512)
                y_inputs = {k: v.to(device) for k, v in y_inputs.items()}
                y_attn_mask = y_inputs['attention_mask'].unsqueeze(-1)
                
                with torch.inference_mode():
                    y_outputs = t_model(**y_inputs, output_hidden_states=True)
                    z_y = (y_outputs.hidden_states[0] * y_attn_mask).sum(dim=1) / y_attn_mask.sum(dim=1)  # [batch, 768]
                
                x_embeddings.append(z_x.cpu())
                y_embeddings.append(z_y.cpu())
                valid_count += len(valid_samples)
            
            except Exception as e:
                print(f"Warning: Failed to extract embeddings for batch: {e}")
    
    print(f"\nExtraction complete: {valid_count} valid samples, {skipped_count} skipped")
    
    x_embeddings = torch.cat(x_embeddings, dim=0)  # [N, 768]
    y_embeddings = torch.cat(y_embeddings, dim=0)  # [N, 768]
    
    return x_embeddings, y_embeddings


def train_club(
    x_embeddings: torch.Tensor,
    y_embeddings: torch.Tensor,
    feature_dim: int = 768,
    hidden_dim: int = 256,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    device: str = "cuda",
) -> nn.Module:
    """
    Train CLUB estimator on embeddings.
    
    Args:
        x_embeddings: [N, feature_dim] - multimodal features
        y_embeddings: [N, feature_dim] - answer embeddings
        feature_dim: Embedding dimension
        hidden_dim: Hidden layer dimension of CLUB network
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        device: Device to train on
    
    Returns:
        Trained CLUB model
    """
    print(f"\nInitializing CLUB estimator...")
    print(f"  Feature dimension: {feature_dim}")
    print(f"  Hidden dimension: {hidden_dim}")
    club = CLUB(feature_dim, feature_dim, hidden_dim).to(device)
    
    optimizer = optim.Adam(club.parameters(), lr=learning_rate)
    
    # Create dataset loader
    dataset = torch.utils.data.TensorDataset(x_embeddings, y_embeddings)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"\nStarting training...")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Total batches per epoch: {len(loader)}")
    
    club.train()
    for epoch in range(epochs):
        total_loss = 0.0
        with tqdm(total=len(loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                
                # CLUB loss = -loglikelihood
                loss = club.learning_loss(x_batch, y_batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.update(1)
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1:3d}: Average Loss = {avg_loss:.6f}")
    
    club.eval()
    return club


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CLUB MI estimator on all d_k.json datasets"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="results/concept_drift_detection/datasets",
        help="Path to directory containing d_k.json files (e.g., D1_migrated.json, D2_migrated.json, ...)",
    )
    parser.add_argument(
        "--dataset-pattern",
        type=str,
        default="*_migrated.json",
        help="Glob pattern to match dataset files (default: *_migrated.json for D[1-5]_migrated.json)",
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
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=768,
        help="Feature (embedding) dimension",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden layer dimension of CLUB network",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-path",
        type=str,
        default="estimator_ckpt/CLUB_all_datasets_trained.pt",
        help="Path to save trained checkpoint",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Discover all dataset files matching the pattern
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    dataset_files = sorted(dataset_dir.glob(args.dataset_pattern))
    if not dataset_files:
        raise FileNotFoundError(
            f"No dataset files found matching pattern '{args.dataset_pattern}' in {dataset_dir}"
        )
    
    print(f"Found {len(dataset_files)} dataset files:")
    for f in dataset_files:
        print(f"  - {f.name}")
    
    # Load and combine all datasets
    combined_dataset = []
    for dataset_path in dataset_files:
        print(f"\nLoading dataset from {dataset_path.name}...")
        try:
            dataset = load_dk_json_dataset(dataset_path)
            combined_dataset.extend(dataset)
            print(f"  Loaded {len(dataset)} samples")
        except Exception as e:
            print(f"  Warning: Failed to load {dataset_path.name}: {e}")
    
    if not combined_dataset:
        raise ValueError("No valid samples loaded from any dataset file")
    
    print(f"\nTotal combined dataset: {len(combined_dataset)} samples from {len(dataset_files)} files")
    
    # Extract embeddings
    x_embeddings, y_embeddings = extract_embeddings(
        combined_dataset,
        base_dir=dataset_dir,
        v_embedder_name=args.v_embedder,
        t_embedder_name=args.t_embedder,
        device=device,
    )
    
    print(f"\nEmbedding shapes:")
    print(f"  X (multimodal features): {x_embeddings.shape}")
    print(f"  Y (answer embeddings): {y_embeddings.shape}")
    
    # Train CLUB estimator
    club_model = train_club(
        x_embeddings,
        y_embeddings,
        feature_dim=int(args.feature_dim),
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        device=device,
    )
    
    # Save checkpoint
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(club_model.state_dict(), output_path)
    print(f"\nSaved trained CLUB checkpoint to: {output_path}")
    print(f"Model trained on {len(dataset_files)} datasets with {len(combined_dataset)} total samples")


if __name__ == "__main__":
    main()
