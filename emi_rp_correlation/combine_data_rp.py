import os
import json
import glob
from datasets import load_dataset
from tqdm import tqdm

# Paths
LOCAL_JSON_DIR = "data/llava-v1.5-13b"
OUTPUT_DIR = "combined_dataset"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load HuggingFace dataset
hf_dataset_natural = load_dataset("changdae/llavabench-shift-natural-v1")
hf_dataset_synthetic = load_dataset("changdae/llavabench-shift-synthetic-v1")

# Build lookup dictionary
hf_lookup = {}

for split in hf_dataset_natural:
    if 'easy' in split or 'hard' in split:
        continue
    print(f"Processing HF Natural Dataset split: {split}")
    hf_split = {}
    for item in tqdm(hf_dataset_natural[split], desc=f"Loading {split}"):
        hf_split[item["question_id"]] = {
            "question": item["question"],
            "reference_answer": item["reference_answer"]
        }
    if 'normal_' in split:
        split = split.replace('normal_', '')
    hf_lookup[split] = hf_split

for split in hf_dataset_synthetic:
    print(f"Processing HF Synthetic Dataset split: {split}")
    hf_split = {}
    for item in tqdm(hf_dataset_synthetic[split], desc=f"Loading {split}"):
        hf_split[item["question_id"]] = {
            "question": item["question"],
            "reference_answer": item["reference_answer"]
        }
    hf_lookup[split] = hf_split

print(f"Loaded {len(hf_lookup)} reference answers from HF Natural & Synthetic dataset.")

# Get all local jsonl files
json_files = glob.glob(os.path.join(LOCAL_JSON_DIR, "*.jsonl"))

for file_path in tqdm(json_files, desc="Processing local json files"):

    split_name = os.path.basename(file_path).replace(".jsonl", "")
    output_path = os.path.join(OUTPUT_DIR, f"{split_name}.jsonl")

    combined_data = []

    # Count lines for tqdm
    with open(file_path, "r") as f:
        total_lines = sum(1 for _ in f)

    with open(file_path, "r") as f:
        for line in tqdm(f, total=total_lines, desc=f"Merging {split_name}"):

            model_entry = json.loads(line)
            qid = model_entry["question_id"]

            if qid not in hf_lookup.get(split_name, {}):
                continue

            combined_entry = {
                "question_id": qid,
                "question": hf_lookup[split_name][qid]["question"],
                "reference_answer": hf_lookup[split_name][qid]["reference_answer"],
                "model_answer": model_entry["text"]
            }

            combined_data.append(combined_entry)

    # Save combined dataset
    with open(output_path, "w") as out:
        for item in combined_data:
            out.write(json.dumps(item) + "\n")

    print(f"Saved {len(combined_data)} samples → {output_path}")