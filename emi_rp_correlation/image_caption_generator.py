"""
Generate detailed image captions for unique images in the LLavaBench-Shift datasets
using Llama 3.2 Vision (11B) via Ollama.

Outputs a JSON file mapping (dataset_group, question_id) -> caption,
suitable for providing visual context to an LLM judge.
"""

import os
import json
import hashlib
import base64
import argparse
from io import BytesIO
from collections import defaultdict

import requests
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2-vision:11b"
OUTPUT_PATH = "image_captions.json"

CAPTION_PROMPT = (
    "Describe this image in detail for someone who cannot see it. "
    "Include: the main subject(s), their actions and poses, spatial layout, "
    "background elements, notable colors, lighting, text visible in the image, "
    "and any other visually important details. "
    "Be factual and concise — aim for 3-5 sentences."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def image_to_base64(img: Image.Image) -> str:
    """Convert a PIL Image to a base64-encoded PNG string."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def image_hash(img: Image.Image) -> str:
    """Return an MD5 hex-digest for a PIL image (used for deduplication)."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    return hashlib.md5(buf.getvalue()).hexdigest()


def generate_caption(img_b64: str, ollama_url: str = OLLAMA_URL, model: str = MODEL_NAME) -> str:
    """Call Ollama with an image and return the generated caption."""
    payload = {
        "model": model,
        "prompt": CAPTION_PROMPT,
        "images": [img_b64],
        "stream": False,
    }
    resp = requests.post(ollama_url, json=payload, timeout=600)
    resp.raise_for_status()
    return resp.json()["response"].strip()


# ---------------------------------------------------------------------------
# Collect unique images
# ---------------------------------------------------------------------------
def collect_unique_images():
    """
    Walk both HF datasets, deduplicate images by content hash,
    and return:
        unique_images : dict[img_hash] -> PIL.Image
        hash_to_keys  : dict[img_hash] -> list[(dataset_group, question_id)]

    dataset_group is one of:
        - "llava_bench_coco"
        - "llava_bench_in_the_wild"
    """
    print("Loading HuggingFace datasets …")
    ds_natural = load_dataset("changdae/llavabench-shift-natural-v1")
    ds_synthetic = load_dataset("changdae/llavabench-shift-synthetic-v1")

    unique_images: dict[str, Image.Image] = {}
    hash_to_keys: dict[str, list[tuple[str, int]]] = defaultdict(list)

    # --- Natural dataset ---------------------------------------------------
    # Pick one language split per image group (images are the same across langs)
    # COCO images
    coco_split = "llava_bench_coco_English"
    if coco_split in ds_natural:
        print(f"Scanning {coco_split} for unique images …")
        for item in tqdm(ds_natural[coco_split], desc="coco"):
            qid = item["question_id"]
            img = item["image"]
            h = image_hash(img)
            if h not in unique_images:
                unique_images[h] = img
            hash_to_keys[h].append(("llava_bench_coco", qid))

    # In-the-wild images (easy/normal/hard share overlapping qid ranges with
    # different images, so we scan all three difficulties)
    for difficulty in ["easy", "normal", "hard"]:
        wild_split = f"llava_bench_in_the_wild_{difficulty}_English"
        if wild_split not in ds_natural:
            continue
        print(f"Scanning {wild_split} for unique images …")
        for item in tqdm(ds_natural[wild_split], desc=f"wild_{difficulty}"):
            qid = item["question_id"]
            img = item["image"]
            h = image_hash(img)
            if h not in unique_images:
                unique_images[h] = img
            hash_to_keys[h].append((f"llava_bench_in_the_wild_{difficulty}", qid))

    # --- Synthetic dataset -------------------------------------------------
    # The base split "llava_bench_coco" has the clean images; corrupted splits
    # (blur, frost, etc.) have visually degraded versions. We caption only the
    # clean version — the caption describes the *original* scene and the judge
    # can still use it for context regardless of the corruption.
    synth_split = "llava_bench_coco"
    if synth_split in ds_synthetic:
        print(f"Scanning synthetic/{synth_split} for unique images …")
        for item in tqdm(ds_synthetic[synth_split], desc="synthetic_coco"):
            qid = item["question_id"]
            img = item["image"]
            h = image_hash(img)
            if h not in unique_images:
                unique_images[h] = img
            # These have the same images as natural coco, so hash_to_keys
            # entries will already exist.  We don't duplicate.

    print(f"\nTotal unique images collected: {len(unique_images)}")
    return unique_images, hash_to_keys


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate image captions via Llama 3.2 Vision (Ollama)"
    )
    parser.add_argument(
        "--output", "-o", default=OUTPUT_PATH,
        help=f"Output JSON path (default: {OUTPUT_PATH})"
    )
    parser.add_argument(
        "--ollama-url", default=OLLAMA_URL,
        help=f"Ollama API URL (default: {OLLAMA_URL})"
    )
    parser.add_argument(
        "--model", default=MODEL_NAME,
        help=f"Ollama model name (default: {MODEL_NAME})"
    )
    args = parser.parse_args()

    ollama_url = args.ollama_url
    model_name = args.model

    # Load existing captions (for resumability)
    captions: dict = {}
    if os.path.exists(args.output):
        with open(args.output, "r") as f:
            captions = json.load(f)
        print(f"Loaded {len(captions)} existing captions from {args.output}")

    unique_images, hash_to_keys = collect_unique_images()

    # Generate captions for each unique image
    generated = 0
    skipped = 0

    for img_hash, img in tqdm(unique_images.items(), desc="Captioning"):
        # Build a canonical key from the first (dataset_group, qid) pair
        keys = hash_to_keys[img_hash]
        # Use image hash as the primary key — stable across runs
        caption_key = img_hash

        if caption_key in captions:
            skipped += 1
            continue

        try:
            img_b64 = image_to_base64(img)
            caption = generate_caption(img_b64, ollama_url=ollama_url, model=model_name)
            captions[caption_key] = {
                "caption": caption,
                "image_hash": img_hash,
                "question_ids": [
                    {"dataset_group": grp, "question_id": qid}
                    for grp, qid in keys
                ],
            }
            generated += 1

            # Save after every image (for crash resumability)
            with open(args.output, "w") as f:
                json.dump(captions, f, indent=2)

        except Exception as e:
            print(f"\n[ERROR] Failed to caption image {img_hash[:12]}…: {e}")
            continue

    print(f"\nDone! Generated {generated} new captions, skipped {skipped} existing.")
    print(f"Total captions: {len(captions)} → {args.output}")

    # Also build a convenience lookup: (dataset_group, question_id) -> caption
    lookup_path = args.output.replace(".json", "_lookup.json")
    lookup: dict[str, dict[str, str]] = {}
    for entry in captions.values():
        cap = entry["caption"]
        for ref in entry["question_ids"]:
            grp = ref["dataset_group"]
            qid = str(ref["question_id"])
            if grp not in lookup:
                lookup[grp] = {}
            lookup[grp][qid] = cap

    with open(lookup_path, "w") as f:
        json.dump(lookup, f, indent=2)
    print(f"Lookup table saved → {lookup_path}")


if __name__ == "__main__":
    main()
