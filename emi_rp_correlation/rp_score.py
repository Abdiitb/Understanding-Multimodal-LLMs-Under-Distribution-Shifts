import os
import json
import glob
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

INPUT_DIR = "combined_dataset"
OUTPUT_DIR = "rp_scores"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "rp_scores_all_splits_captions_llama.json")
CAPTIONS_FILE = "image_captions_lookup.json"

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

NUM_WORKERS = 16   # parallel requests

client = OpenAI(
    base_url="http://10.119.2.98:8000/v1",
    api_key="EMPTY"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------
# Load captions
# ---------------------------------------------------

with open(CAPTIONS_FILE) as f:
    raw = json.load(f)

captions_lookup = {
    group: {str(k): v for k, v in entries.items()}
    for group, entries in raw.items()
}

print("Captions loaded")


# ---------------------------------------------------
# Helpers
# ---------------------------------------------------

def _get_caption_group(split):

    for g in [
        "llava_bench_in_the_wild_easy",
        "llava_bench_in_the_wild_normal",
        "llava_bench_in_the_wild_hard"
    ]:
        if split.startswith(g):
            return g

    if split.startswith("llava_bench_coco"):
        return "llava_bench_coco"

    return "llava_bench_coco"


def get_caption(split, qid):

    group = _get_caption_group(split)

    return captions_lookup.get(group, {}).get(
        str(qid),
        "No image description available."
    )


DEFAULT_PROMPT = """
We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with a few sentences describing the image.

Please rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.

Please first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.

In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.
"""


def build_prompt(context, question, ref, ans):

    return f"""
[Visual Context]
{context}

[Question]
{question}

[Assistant 1]
{ref}

[Assistant 2]
{ans}

[System]
{DEFAULT_PROMPT}
"""


def parse_scores(text):

    for line in text.split("\n"):

        line = re.sub(r'[*#]', '', line).strip()
        line = line.replace(",", " ")

        nums = re.findall(r'\b(\d+(?:\.\d+)?)\b', line)

        if len(nums) >= 2:

            s1 = float(nums[0])
            s2 = float(nums[1])

            if 1 <= s1 <= 10 and 1 <= s2 <= 10:
                return s1, s2

    return None, None


def judge(prompt):

    try:

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=120
        )

        text = response.choices[0].message.content

        ref, model = parse_scores(text)

        if ref is None or ref == 0:
            return None

        return model / ref

    except Exception:
        return None


# ---------------------------------------------------
# Resume support
# ---------------------------------------------------

if os.path.exists(OUTPUT_FILE):

    with open(OUTPUT_FILE) as f:
        results = json.load(f)

else:
    results = {}


json_files = glob.glob(os.path.join(INPUT_DIR, "*.jsonl"))

# ---------------------------------------------------
# Main Loop
# ---------------------------------------------------

for file_path in tqdm(json_files, desc="Processing splits"):

    split = os.path.basename(file_path).replace(".jsonl", "")

    if split in results:
        print("Skipping", split)
        continue

    with open(file_path) as f:
        items = [json.loads(x) for x in f]

    prompts = []

    for item in items:

        caption = get_caption(split, item["question_id"])

        prompts.append(
            build_prompt(
                caption,
                item["question"],
                item["reference_answer"],
                item["model_answer"]
            )
        )

    rp_values = []

    with ThreadPoolExecutor(NUM_WORKERS) as executor:

        futures = [executor.submit(judge, p) for p in prompts]

        for f in tqdm(as_completed(futures), total=len(futures)):

            rp = f.result()

            if rp is not None:
                rp_values.append(rp)

    agg = sum(rp_values) / len(rp_values) if rp_values else 0.0

    if not rp_values:
        print(f"WARNING: No valid scores for {split}, skipping.")
        continue

    results[split] = {
        "num_samples": len(rp_values),
        "rp_score": agg
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(split, "RP:", agg)


print("Finished")