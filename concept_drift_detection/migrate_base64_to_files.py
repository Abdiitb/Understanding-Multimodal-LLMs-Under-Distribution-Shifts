#!/usr/bin/env python3
"""
Migration script: Convert base64-encoded images in JSON files to file-based storage.

This script:
1. Finds all D*.json files in a directory
2. Extracts images only from D1.json and saves them to disk
3. For other datasets, reuses the extracted images from D1
4. Updates JSON files to reference images by filename
5. Creates new D*_migrated.json files

Usage:
    python migrate_base64_to_files.py --dataset-dir path/to/datasets \
                                      --images-dir images
    
    Or process all D*.json files in default location:
    python migrate_base64_to_files.py
"""

import argparse
import base64
import json
import shutil
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm


def migrate_dataset(
    input_json: Path,
    images_dir_name: str = "images",
    extract_images: bool = True,
    shared_images_dir: Path = None,
) -> tuple:
    """
    Migrate a JSON dataset from base64 to file-based storage.
    
    Images are saved with their image_id from the JSON (if available),
    otherwise a sequential counter is used. A new migrated JSON file is created.
    Original JSON file is NOT deleted by this function.
    
    Args:
        input_json: Path to input JSON file with base64 images
        images_dir_name: Name of subdirectory for images
        extract_images: Whether to extract and save images (True for D1, False for others)
        shared_images_dir: Path to shared images directory (used when extract_images=False)
        
    Returns:
        Tuple of (output_json_path, images_dir_path)
    """
    # Create images directory next to the JSON file
    json_dir = input_json.parent
    
    if extract_images:
        # D1: create its own images directory and extract images
        images_dir = json_dir / images_dir_name
        images_dir.mkdir(exist_ok=True)
    else:
        # Other datasets: use shared images directory from D1
        if shared_images_dir is None:
            raise ValueError("shared_images_dir must be provided when extract_images=False")
        images_dir = shared_images_dir
    
    print(f"\n{'=' * 80}")
    print(f"Migrating base64 images to file-based storage")
    print(f"{'=' * 80}")
    print(f"Input JSON:      {input_json}")
    print(f"Images dir:      {images_dir}")
    
    # Load input JSON
    print(f"\nLoading JSON file...")
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data).__name__}")
    
    print(f"Loaded {len(data)} items")
    
    # Process each item
    migrated_data = []
    image_count = 0
    skipped_count = 0
    
    with tqdm(total=len(data), desc="Migrating items", unit="item") as pbar:
        for idx, item in enumerate(data):
            try:
                migrated_item = item.copy()
                
                # Get image_id from top level or nested 'x' structure
                image_id = item.get('image_id')
                
                # Check if item has nested 'x' structure (common in d_k.json format)
                if 'x' in item and isinstance(item['x'], dict):
                    x = item['x'].copy()
                    if image_id is None:
                        image_id = x.get('image_id')
                    
                    image_data = x.get('image')
                    
                    if isinstance(image_data, dict) and 'bytes_base64' in image_data:
                        # Extract base64 and save to file only if extract_images is True
                        if extract_images:
                            bytes_base64 = image_data['bytes_base64']
                            try:
                                image_bytes = base64.b64decode(bytes_base64)
                                img = Image.open(BytesIO(image_bytes))
                                img = img.convert('RGB')
                                
                                # Save with image_id as filename, or fallback to counter
                                if image_id is not None:
                                    image_filename = f"{image_id}.jpg"
                                else:
                                    image_filename = f"img_{image_count:06d}.jpg"
                                
                                image_path = images_dir / image_filename
                                img.save(image_path, 'JPEG', quality=95)
                                
                                # Update JSON to reference the file
                                x['image'] = image_filename
                                x['image_dir'] = str(images_dir_name)
                                migrated_item['x'] = x
                                
                                image_count += 1
                            except Exception as e:
                                print(f"\n  Warning: Failed to extract image from item {idx}: {e}")
                                skipped_count += 1
                        else:
                            # For non-D1 datasets, assume image files already exist
                            # Just update reference without extracting
                            if image_id is not None:
                                image_filename = f"{image_id}.jpg"
                            else:
                                image_filename = f"img_{image_count:06d}.jpg"
                            
                            x['image'] = image_filename
                            x['image_dir'] = str(images_dir_name)
                            migrated_item['x'] = x
                            image_count += 1
                    elif isinstance(image_data, str):
                        # Already a filename reference
                        x['image_dir'] = str(images_dir_name)
                        migrated_item['x'] = x
                    
                    migrated_data.append(migrated_item)
                else:
                    # No image data, keep as-is
                    migrated_data.append(migrated_item)
                
            except Exception as e:
                print(f"\n  Warning: Failed to process item {idx}: {e}")
                migrated_data.append(item)
                skipped_count += 1
            
            pbar.update(1)
    
    # Save migrated JSON with _migrated suffix
    output_json = json_dir / (input_json.stem + "_migrated.json")
    print(f"\nSaving migrated JSON to {output_json.name}...")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(migrated_data, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print(f"\n{'=' * 80}")
    print(f"Migration Complete!")
    print(f"{'=' * 80}")
    print(f"Items processed:     {len(migrated_data)}")
    print(f"Images extracted:    {image_count}")
    print(f"Items skipped:       {skipped_count}")
    print(f"Output JSON size:    {output_json.stat().st_size / 1024:.2f} KB")
    if extract_images and image_count > 0:
        total_image_size = sum(f.stat().st_size for f in images_dir.glob('*.jpg'))
        print(f"Total image size:    {total_image_size / (1024 * 1024):.2f} MB")
    print(f"\nOutput files:")
    print(f"  - JSON: {output_json}")
    if extract_images:
        print(f"  - Images: {images_dir}")
    
    return output_json, images_dir


def main():
    parser = argparse.ArgumentParser(
        description="Migrate base64-encoded images in all d_k.json files to file-based storage"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("concept_drift_detection/datasets"),
        help="Directory containing d_k.json files (default: concept_drift_detection/datasets)",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="images",
        help="Subdirectory name for images (default: images)",
    )
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    # Find all D*.json files
    json_files = sorted(dataset_dir.glob("D*.json"))
    
    if not json_files:
        print(f"No D*.json files found in {dataset_dir}")
        return
    
    # Find D1.json
    d1_file = next((f for f in json_files if f.stem == "D1"), None)
    if not d1_file:
        print(f"Warning: D1.json not found. Images will not be extracted.")
        shared_images_dir = None
    
    print(f"\nFound {len(json_files)} D*.json files to process:")
    for f in json_files:
        print(f"  - {f.name}")
    
    print(f"\n{'=' * 80}")
    print(f"Starting batch migration of D*.json files")
    print(f"{'=' * 80}\n")
    
    successful = 0
    failed = 0
    shared_images_dir = None
    
    for json_file in json_files:
        print(f"\nProcessing: {json_file.name}")
        print(f"-" * 80)
        
        # Only extract images from D1.json
        extract_images = (json_file.stem == "D1")
        
        try:
            output_json, images_dir = migrate_dataset(
                json_file,
                args.images_dir,
                extract_images=extract_images,
                shared_images_dir=shared_images_dir,
            )
            
            # Save the D1 images directory path to use for other datasets
            if extract_images:
                shared_images_dir = images_dir
            
            # Delete the original JSON file after successful migration
            json_file.unlink()
            print(f"✓ Deleted original file: {json_file.name}")
            
            successful += 1
        except Exception as e:
            print(f"\n✗ Migration failed for {json_file.name}: {e}")
            failed += 1
    
    # Final summary
    print(f"\n{'=' * 80}")
    print(f"Batch Migration Summary")
    print(f"{'=' * 80}")
    print(f"Total files:       {len(json_files)}")
    print(f"Successful:        {successful}")
    print(f"Failed:            {failed}")
    print(f"{'=' * 80}\n")
    
    if failed == 0:
        print(f"✓ All migrations completed successfully!")
    else:
        print(f"⚠ {failed} migration(s) failed. Check errors above.")


if __name__ == "__main__":
    main()
