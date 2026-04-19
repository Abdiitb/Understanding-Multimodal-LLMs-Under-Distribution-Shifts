#!/usr/bin/env python3
"""
Script to create a combined JSON with questions, ground truth answers, and model answers.

This script combines:
- D1.json: Dataset with questions and ground truth answers
- D1_llava_predictions.json: Model predictions in the same order as D1.json

Output: combined_d1_with_answers.json
Structure of each item:
{
  "question_id": <index>,
  "question_data": <original x field>,
  "ground_truth_answer": <y field>,
  "model_answer": <corresponding prediction>
}
"""

import json
import sys
from pathlib import Path

def create_combined_dataset(
    d1_path="results/concept_drift_detection/datasets/D1_migrated.json",
    predictions_path="results/concept_drift_detection/D1_llava_predictions.json",
    output_path="results/concept_drift_detection/combined_d1_with_answers.json",
):
    """
    Combine D1 dataset with model predictions.
    
    Args:
        d1_path: Path to D1_migrated.json
        predictions_path: Path to D1_llava_predictions.json
        output_path: Path to save the combined JSON
    """
    
    print("=" * 80)
    print("Creating Combined Dataset")
    print("=" * 80)
    
    # Load predictions first
    print(f"\n1. Loading predictions from {predictions_path}...")
    with open(predictions_path, 'r') as f:
        predictions_data = json.load(f)
    
    predictions = predictions_data.get('predictions', [])
    print(f"   ✓ Loaded {len(predictions)} predictions")
    print(f"   Model: {predictions_data.get('model_name', 'Unknown')}")
    
    # Load D1 data and combine with predictions
    print(f"\n2. Loading dataset from {d1_path}...")
    with open(d1_path, 'r') as f:
        d1_data = json.load(f)
    
    print(f"   ✓ Loaded {len(d1_data)} samples")
    
    # Validate matching lengths
    if len(d1_data) != len(predictions):
        print(f"\n❌ ERROR: Mismatch in sizes!")
        print(f"   D1 samples: {len(d1_data)}")
        print(f"   Predictions: {len(predictions)}")
        sys.exit(1)
    
    # Create combined dataset
    print(f"\n3. Combining dataset with predictions...")
    combined_data = []
    
    for idx, (d1_item, prediction) in enumerate(zip(d1_data, predictions)):
        x_data = d1_item.get('x', {})
        combined_item = {
            "question_id": x_data.get('question_id'),
            "image_id": x_data.get('image_id'),
            "question": x_data.get('question'),
            "gt_answer": d1_item.get('y'),
            "model_answer": prediction.lower()
        }
        combined_data.append(combined_item)
        
        # Print progress every 1000 items
        if (idx + 1) % 1000 == 0:
            print(f"   Processed: {idx + 1}/{len(d1_data)}")
    
    print(f"   ✓ Combined {len(combined_data)} samples")
    
    # Save combined dataset
    print(f"\n4. Saving combined dataset to {output_path}...")
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(combined_data, f, indent=2)
    
    print(f"   ✓ Saved successfully!")
    
    # Print summary
    print(f"\n5. Summary:")
    print(f"   Output file: {output_path}")
    print(f"   Total items: {len(combined_data)}")
    print(f"   File size: {Path(output_path).stat().st_size / (1024 * 1024):.2f} MB")
    
    # Print sample
    print(f"\n6. Sample combined item (first one):")
    print(json.dumps(combined_data[0], indent=2)[:500] + "...")
    
    print("\n" + "=" * 80)
    print("✓ Combined dataset created successfully!")
    print("=" * 80)


if __name__ == "__main__":
    create_combined_dataset()
