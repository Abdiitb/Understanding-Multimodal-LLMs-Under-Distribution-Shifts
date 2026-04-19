"""
Smoke test for LLaVA yes/no constrained inference.

This script loads a LLaVA model, runs one image-question inference through
`generate_answer`, and checks whether the output is exactly "yes" or "no".
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
import sys
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from gradio_app.model_inference import load_model, generate_answer


def _make_dummy_image(size: int = 224) -> Image.Image:
    return Image.new("RGB", (size, size), color=(255, 255, 255))


def main() -> None:
    parser = argparse.ArgumentParser(description="Test LLaVA yes/no inference prompt")
    parser.add_argument(
        "--model-id",
        type=str,
        default="llava-hf/llava-1.5-13b-hf",
        help="HuggingFace model id (use a LLaVA model)",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="Is there a dog in this image?",
        help="Binary yes/no question to ask",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default="",
        help="Optional path to an image. If omitted, a dummy white image is used.",
    )
    args = parser.parse_args()

    if args.image_path:
        image = Image.open(args.image_path).convert("RGB")
    else:
        image = _make_dummy_image()

    model, processor = load_model(args.model_id)
    answer = generate_answer(model, processor, image, args.question)

    normalized = answer.strip().lower()
    print(f"Model: {args.model_id}")
    print(f"Question: {args.question}")
    print(f"Raw answer: {answer}")

    if normalized not in {"yes", "no"}:
        raise AssertionError(
            "Expected answer to be exactly 'yes' or 'no'. "
            f"Got: {answer!r}"
        )

    print("PASS: Output is binary yes/no.")


if __name__ == "__main__":
    main()
