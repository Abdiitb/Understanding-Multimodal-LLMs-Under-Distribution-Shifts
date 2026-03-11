import json
import os
import sys

LANGUAGES = {"Arabic", "Chinese", "German", "Greek", "Hindi", "Korean", "English"}
SYNTHETIC_KEYWORDS = ["defocus_blur", "frost", "sr_4", "sr_7", "KeyboardAug"]


def is_natural(name):
    """Natural shifts: dataset names ending with a language name."""
    return any(name.endswith("_" + lang) for lang in LANGUAGES)


def is_synthetic(name):
    """Synthetic shifts: dataset names containing corruption keywords."""
    return any(kw in name for kw in SYNTHETIC_KEYWORDS)


def separate(input_path):
    with open(input_path) as f:
        data = json.load(f)

    # Detect structure: flat (qwen) vs nested with RP_Scores/EMI_Scores (llama)
    top_keys = set(data.keys())
    has_sections = "RP_Scores" in top_keys or "EMI_Scores" in top_keys

    if has_sections:
        # Already sectioned (like llama file) — separate within each section
        result = {"natural": {}, "synthetic": {}}
        for section in data:
            result["natural"][section] = {}
            result["synthetic"][section] = {}
            for key, val in data[section].items():
                if is_natural(key):
                    result["natural"][section][key] = val
                elif is_synthetic(key):
                    result["synthetic"][section][key] = val
                else:
                    print(f"  Dropped (neither natural nor synthetic): {key}")
    else:
        # Flat structure — keys are dataset names directly
        result = {"natural": {}, "synthetic": {}}
        for key, val in data.items():
            if is_natural(key):
                result["natural"][key] = val
            elif is_synthetic(key):
                result["synthetic"][key] = val
            else:
                print(f"  Dropped (neither natural nor synthetic): {key}")

    # Summary
    if has_sections:
        for shift_type in ["natural", "synthetic"]:
            for section in result[shift_type]:
                print(f"  {shift_type}/{section}: {len(result[shift_type][section])} entries")
    else:
        for shift_type in ["natural", "synthetic"]:
            print(f"  {shift_type}: {len(result[shift_type])} entries")

    # Write back
    with open(input_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to {input_path}")


if __name__ == "__main__":
    rp_dir = "rp_scores"
    files = sys.argv[1:] if len(sys.argv) > 1 else [
        os.path.join(rp_dir, f)
        for f in os.listdir(rp_dir)
        if f.startswith("rp_scores_all_splits") and f.endswith(".json")
    ]

    for path in files:
        print(f"Processing {path}...")
        separate(path)
