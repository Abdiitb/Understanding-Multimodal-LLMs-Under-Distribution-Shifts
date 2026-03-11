"""
Dataset constants and loading utilities for the Gradio EMI experiment app.
"""

from datasets import load_dataset

# ---------------------------------------------------------------------------
# HuggingFace dataset identifiers
# ---------------------------------------------------------------------------
NATURAL_DS_ID = "changdae/llavabench-shift-natural-v1"
SYNTHETIC_DS_ID = "changdae/llavabench-shift-synthetic-v1"

# ---------------------------------------------------------------------------
# Pre-defined split lists (mirrors main.py conventions)
# ---------------------------------------------------------------------------
NATURAL_SPLITS = [
    "llava_bench_coco_English",
    "llava_bench_coco_German",
    "llava_bench_coco_Chinese",
    "llava_bench_coco_Korean",
    "llava_bench_coco_Greek",
    "llava_bench_coco_Arabic",
    "llava_bench_coco_Hindi",
    "llava_bench_in_the_wild_easy_English",
    "llava_bench_in_the_wild_easy_German",
    "llava_bench_in_the_wild_easy_Chinese",
    "llava_bench_in_the_wild_easy_Korean",
    "llava_bench_in_the_wild_easy_Greek",
    "llava_bench_in_the_wild_easy_Arabic",
    "llava_bench_in_the_wild_easy_Hindi",
    "llava_bench_in_the_wild_normal_English",
    "llava_bench_in_the_wild_normal_German",
    "llava_bench_in_the_wild_normal_Chinese",
    "llava_bench_in_the_wild_normal_Korean",
    "llava_bench_in_the_wild_normal_Greek",
    "llava_bench_in_the_wild_normal_Arabic",
    "llava_bench_in_the_wild_normal_Hindi",
    "llava_bench_in_the_wild_hard_English",
    "llava_bench_in_the_wild_hard_German",
    "llava_bench_in_the_wild_hard_Chinese",
    "llava_bench_in_the_wild_hard_Korean",
    "llava_bench_in_the_wild_hard_Greek",
    "llava_bench_in_the_wild_hard_Arabic",
    "llava_bench_in_the_wild_hard_Hindi",
]

SYNTHETIC_SPLITS = [
    "llava_bench_coco",
    "llava_bench_coco_sr_4",
    "llava_bench_coco_sr_7",
    "llava_bench_coco_KeyboardAug_1",
    "llava_bench_coco_KeyboardAug_3",
    "llava_bench_coco_defocus_blur_1",
    "llava_bench_coco_defocus_blur_1_sr_4",
    "llava_bench_coco_defocus_blur_1_sr_7",
    "llava_bench_coco_defocus_blur_1_KeyboardAug_1",
    "llava_bench_coco_defocus_blur_1_KeyboardAug_3",
    "llava_bench_coco_defocus_blur_3",
    "llava_bench_coco_defocus_blur_3_sr_4",
    "llava_bench_coco_defocus_blur_3_sr_7",
    "llava_bench_coco_defocus_blur_3_KeyboardAug_1",
    "llava_bench_coco_defocus_blur_3_KeyboardAug_3",
    "llava_bench_coco_defocus_blur_5",
    "llava_bench_coco_defocus_blur_5_sr_4",
    "llava_bench_coco_defocus_blur_5_sr_7",
    "llava_bench_coco_defocus_blur_5_KeyboardAug_1",
    "llava_bench_coco_defocus_blur_5_KeyboardAug_3",
    "llava_bench_coco_frost_1",
    "llava_bench_coco_frost_1_sr_4",
    "llava_bench_coco_frost_1_sr_7",
    "llava_bench_coco_frost_1_KeyboardAug_1",
    "llava_bench_coco_frost_1_KeyboardAug_3",
    "llava_bench_coco_frost_3",
    "llava_bench_coco_frost_3_sr_4",
    "llava_bench_coco_frost_3_sr_7",
    "llava_bench_coco_frost_3_KeyboardAug_1",
    "llava_bench_coco_frost_3_KeyboardAug_3",
    "llava_bench_coco_frost_5",
    "llava_bench_coco_frost_5_sr_4",
    "llava_bench_coco_frost_5_sr_7",
    "llava_bench_coco_frost_5_KeyboardAug_1",
    "llava_bench_coco_frost_5_KeyboardAug_3",
]

ALL_SPLITS = NATURAL_SPLITS + SYNTHETIC_SPLITS


def get_split_choices():
    """Return a dict mapping display labels -> split names for dropdowns."""
    natural = {f"[Natural] {s}": s for s in NATURAL_SPLITS}
    synthetic = {f"[Synthetic] {s}": s for s in SYNTHETIC_SPLITS}
    return {**natural, **synthetic}


def resolve_hf_id(split_name: str) -> str:
    """Given a split name, return the HuggingFace dataset id it belongs to."""
    if split_name in NATURAL_SPLITS:
        return NATURAL_DS_ID
    elif split_name in SYNTHETIC_SPLITS:
        return SYNTHETIC_DS_ID
    else:
        raise ValueError(f"Unknown split: {split_name}")


# Cache loaded datasets to avoid re-downloading
_ds_cache: dict = {}


def load_hf_split(split_name: str, num_samples: int | None = None):
    """
    Load a single split from the appropriate HF dataset.
    Returns a HF Dataset object.
    If num_samples is set, returns a subset.
    """
    ds_id = resolve_hf_id(split_name)

    if ds_id not in _ds_cache:
        _ds_cache[ds_id] = load_dataset(ds_id)

    ds = _ds_cache[ds_id][split_name]
    if num_samples is not None and num_samples < len(ds):
        ds = ds.select(range(num_samples))
    return ds
