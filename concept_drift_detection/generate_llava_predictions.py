"""
Generate model predictions for D1 dataset using LLaVA 7B model.

Uses LLaVA 7B (from HuggingFace) to get inference predictions on each sample.
"""

import json
import torch
from pathlib import Path
from PIL import Image

try:
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from tqdm.auto import tqdm
except ImportError as e:
    raise ImportError(f"Required packages not found. Install with: pip install transformers pillow") from e


def deserialize_image(image_filename: str, image_dir: Path) -> Image.Image:
    """Load image from disk by filename.
    
    Args:
        image_filename: Filename of the image (e.g., 'img_000.jpg')
        image_dir: Path to directory containing images
    
    Returns:
        PIL Image in RGB format, or None if loading fails
    """
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


def generate_llava_predictions(dataset_path: Path, model_id: str = "llava-hf/llava-1.5-7b-hf"):
    """
    Generate predictions using LLaVA 7B model.
    
    Args:
        dataset_path: Path to migrated D_k.json file
        model_id: HuggingFace model ID for LLaVA
    
    Returns:
        Dict with predictions and metadata
    """
    # Load model and processor
    print(f"Loading {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    
    # Load dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} samples from {dataset_path}")
    
    predictions = []
    failed_count = 0
    
    with tqdm(total=len(dataset), desc="Generating LLaVA predictions", unit="sample") as pbar:
        for idx, sample in enumerate(dataset):
            try:
                # Extract image and question
                x = sample.get('x', {})
                image_filename = x.get('image')
                # Construct image_dir relative to dataset directory
                dataset_dir = dataset_path.parent
                image_dir = dataset_dir / x.get('image_dir', 'images')
                question = x.get('question', '')
                
                if not image_filename or not question:
                    predictions.append('')
                    failed_count += 1
                    pbar.update(1)
                    continue
                
                # Load image from disk
                image = deserialize_image(image_filename, image_dir)
                if image is None:
                    predictions.append('')
                    failed_count += 1
                    pbar.update(1)
                    continue
                
                # Prepare prompt - instruct model to give short answers (1-3 words)
                prompt = f"<image>\nQuestion: {question}\nAnswer with only one to three words:"
                
                # Process inputs
                inputs = processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt"
                ).to(device)
                
                # Generate prediction - shorter max_new_tokens for brief answers
                with torch.inference_mode():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=15,
                        temperature=0.5,
                        use_cache=True
                    )
                
                # Decode output
                prediction = processor.decode(output_ids[0], skip_special_tokens=True)
                
                # Extract just the answer part (after prompt)
                if "Answer with only one to three words:" in prediction:
                    answer = prediction.split("Answer with only one to three words:")[-1].strip()
                else:
                    answer = prediction.strip()
                
                # Post-process: enforce 1-3 words limit by taking first N words
                words = answer.split()
                if len(words) > 3:
                    answer = " ".join(words[:3])
                
                predictions.append(answer)
                
                # Periodic memory cleanup
                if (idx + 1) % 100 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"\nWarning: Failed to process sample {idx}: {e}")
                predictions.append('')
                failed_count += 1
            
            pbar.update(1)
    
    print(f"\nGeneration complete. Failed: {failed_count}/{len(dataset)}")
    
    return {
        'description': 'LLaVA 7B model predictions for D1 dataset',
        'model_name': 'llava-1.5-7b-hf',
        'num_samples': len(predictions),
        'num_successful': len(predictions) - failed_count,
        'predictions': predictions
    }


def main():
    dataset_dir = Path('results/concept_drift_detection/datasets')
    output_dir = Path('results/concept_drift_detection')
    
    # Generate predictions for D1
    d1_path = dataset_dir / 'D1.json'
    
    if not d1_path.exists():
        print(f"Error: {d1_path} not found")
        return
    
    results = generate_llava_predictions(d1_path)
    
    # Save predictions
    output_path = output_dir / 'D1_llava_predictions.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved LLaVA predictions to {output_path}")
    print(f"  Total samples: {results['num_samples']}")
    print(f"  Successful predictions: {results['num_successful']}")
    
    # Show some examples
    with open(d1_path, 'r') as f:
        d1 = json.load(f)
    
    print(f"\nExample predictions (first 5 samples):")
    for i in range(min(5, len(d1))):
        ground_truth = d1[i].get('y', '')
        question = d1[i].get('x', {}).get('question', '')
        pred = results['predictions'][i]
        print(f"  Q{i+1}: {question}")
        print(f"       GT: {ground_truth}")
        print(f"       Pred: {pred}")
        print()


if __name__ == '__main__':
    main()
