"""
Train a CLUB estimator on:
1) full POPE dataset (all categories/splits)
2) llava_bench_coco_English split

The trained checkpoint is saved and can be reused by pope_experiment.py.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from PIL import Image

import sys

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from gradio_app.estimator import CLUB, Embedder


CATEGORY_ALIASES = {
    "adversarial": "adversarial",
    "adv": "adversarial",
    "popular": "popular",
    "pop": "popular",
    "random": "random",
    "rand": "random",
}

TARGET_CATEGORIES = ["adversarial", "popular", "random"]


@dataclass
class PopeRecord:
    qid: str
    category: str
    question: str
    reference_answer: str
    image: Image.Image
    image_id: str


def _normalise_category(raw: str | None, fallback: str | None = None) -> str | None:
    value = (raw or fallback or "").strip().lower()
    if not value:
        return None
    return CATEGORY_ALIASES.get(value)


def _read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            return data["data"]
        if "samples" in data and isinstance(data["samples"], list):
            return data["samples"]
    raise ValueError(f"Unsupported JSON structure in {path}")


def _resolve_image(img_value: Any, image_root: Path | None) -> tuple[Image.Image, str]:
    if isinstance(img_value, Image.Image):
        return img_value.convert("RGB"), "in_memory"

    if isinstance(img_value, dict):
        for key in ["path", "file", "image", "img"]:
            if key in img_value:
                img_value = img_value[key]
                break

    if not isinstance(img_value, str):
        raise ValueError(f"Unsupported image field type: {type(img_value)}")

    img_path = Path(img_value)
    if not img_path.is_absolute() and image_root is not None:
        img_path = image_root / img_path

    with Image.open(img_path) as img:
        return img.convert("RGB"), str(img_path)


def _extract_record(raw: dict[str, Any], image_root: Path | None, fallback_category: str | None) -> PopeRecord:
    qid = raw.get("qid") or raw.get("question_id") or raw.get("id") or raw.get("uid")
    question = raw.get("ques") or raw.get("question") or raw.get("prompt")
    answer = raw.get("ans") or raw.get("answer") or raw.get("label")
    image_value = raw.get("img") or raw.get("image") or raw.get("image_path")
    category = _normalise_category(raw.get("category") or raw.get("type") or raw.get("split"), fallback=fallback_category)

    if not question:
        raise ValueError("Missing question field (expected one of: ques/question/prompt)")
    if answer is None:
        raise ValueError("Missing answer field (expected one of: ans/answer/label)")
    if image_value is None:
        raise ValueError("Missing image field (expected one of: img/image/image_path)")
    if category is None:
        raise ValueError("Missing category (expected adversarial/popular/random)")

    image, image_id = _resolve_image(image_value, image_root)
    return PopeRecord(
        qid="" if qid is None else str(qid),
        category=category,
        question=str(question),
        reference_answer=str(answer),
        image=image,
        image_id=image_id,
    )


def load_pope_records(pope_path: Path, image_root: Path | None = None) -> list[PopeRecord]:
    records: list[PopeRecord] = []

    if pope_path.is_file():
        rows = _read_json_or_jsonl(pope_path)
        for idx, row in enumerate(rows):
            rec = _extract_record(row, image_root=image_root, fallback_category=None)
            if rec.qid == "":
                rec.qid = f"pope_{idx}"
            records.append(rec)
        return records

    if not pope_path.is_dir():
        raise ValueError(f"POPE path does not exist: {pope_path}")

    file_candidates = sorted(
        [p for p in pope_path.rglob("*") if p.is_file() and p.suffix.lower() in {".json", ".jsonl"}]
    )
    if not file_candidates:
        raise ValueError(f"No .json/.jsonl files found under {pope_path}")

    for file_path in file_candidates:
        stem = file_path.stem.lower()
        fallback_category = None
        for key in TARGET_CATEGORIES:
            if key in stem:
                fallback_category = key
                break
        rows = _read_json_or_jsonl(file_path)
        for idx, row in enumerate(rows):
            rec = _extract_record(row, image_root=image_root, fallback_category=fallback_category)
            if rec.qid == "":
                rec.qid = f"{file_path.stem}_{idx}"
            records.append(rec)

    return records


def _label_to_yes_no(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return "yes" if int(value) == 1 else "no"
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "present", "positive"}:
        return "yes"
    if text in {"0", "false", "no", "n", "absent", "negative"}:
        return "no"
    return str(value)


def _extract_hf_question(raw: dict[str, Any]) -> str | None:
    for key in ["question", "ques", "prompt", "text"]:
        val = raw.get(key)
        if val is not None and str(val).strip() != "":
            return str(val)
    return None


def _extract_hf_answer(raw: dict[str, Any]) -> str | None:
    for key in ["answer", "ans", "label", "target"]:
        if key in raw and raw[key] is not None:
            return _label_to_yes_no(raw[key])
    return None


def _extract_hf_image(raw: dict[str, Any]) -> tuple[Image.Image, str] | None:
    if "image" in raw and raw["image"] is not None:
        image_value = raw["image"]
    elif "img" in raw and raw["img"] is not None:
        image_value = raw["img"]
    elif "image_path" in raw and raw["image_path"] is not None:
        image_value = raw["image_path"]
    else:
        return None

    if isinstance(image_value, Image.Image):
        return image_value.convert("RGB"), "hf_image"
    if isinstance(image_value, dict):
        if "path" in image_value and image_value["path"]:
            with Image.open(image_value["path"]) as img:
                return img.convert("RGB"), str(image_value["path"])
        if "bytes" in image_value and image_value["bytes"] is not None:
            from io import BytesIO

            with Image.open(BytesIO(image_value["bytes"])) as img:
                return img.convert("RGB"), "hf_image_bytes"
    if isinstance(image_value, str):
        with Image.open(image_value) as img:
            return img.convert("RGB"), image_value

    return None


def load_pope_records_hf(dataset_id: str, split_names: list[str] | None = None) -> list[PopeRecord]:
    ds_dict = load_dataset(dataset_id)

    if split_names:
        active_splits = [s for s in split_names if s in ds_dict]
        missing = [s for s in split_names if s not in ds_dict]
        if missing:
            raise ValueError(f"Requested POPE split(s) not found in {dataset_id}: {missing}")
    else:
        active_splits = list(ds_dict.keys())

    records: list[PopeRecord] = []
    for split_name in active_splits:
        ds = ds_dict[split_name]

        fallback_category = None
        split_lower = split_name.lower()
        for category in TARGET_CATEGORIES:
            if category in split_lower:
                fallback_category = category
                break

        for idx, item in enumerate(ds):
            item_dict = dict(item)
            category = _normalise_category(
                item_dict.get("category") or item_dict.get("type") or item_dict.get("split"),
                fallback=fallback_category,
            )
            if category is None:
                continue

            question = _extract_hf_question(item_dict)
            answer = _extract_hf_answer(item_dict)
            image_pack = _extract_hf_image(item_dict)
            if question is None or answer is None or image_pack is None:
                continue

            image, image_id = image_pack
            qid = item_dict.get("qid") or item_dict.get("question_id") or item_dict.get("id") or item_dict.get("uid")
            if qid is None:
                qid = f"{split_name}_{idx}"
            records.append(
                PopeRecord(
                    qid=str(qid),
                    category=category,
                    question=question,
                    reference_answer=answer,
                    image=image,
                    image_id=image_id,
                )
            )

    if not records:
        raise ValueError(
            f"No usable POPE records were parsed from HF dataset {dataset_id}. "
            "Check schema/columns and category labels."
        )

    return records


def _train_club_on_embeddings(
    club: CLUB,
    embedder: Embedder,
    train_images: list[Image.Image],
    train_questions: list[str],
    train_refs: list[str],
    epochs: int,
    lr: float,
) -> CLUB:
    device = embedder.device
    club = club.to(device)
    optimizer = torch.optim.Adam(club.parameters(), lr=lr)

    with torch.inference_mode():
        z_v, z_t, _, z_y = embedder.encode(train_images, train_questions, train_refs, train_refs)
        z = (z_v + z_t) * 0.5

    z = z.clone()
    z_y = z_y.clone()

    club.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = club.learning_loss(z, z_y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0 or epoch == 0 or epoch + 1 == epochs:
            print(f"Epoch {epoch + 1}/{epochs} - loss: {loss.item():.6f}")

    club.eval()
    return club


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CLUB on full POPE + llava_bench_coco_English and save checkpoint"
    )
    parser.add_argument("--pope-source", type=str, default="hf", choices=["hf", "local"])
    parser.add_argument("--pope-hf-dataset", type=str, default="lmms-lab/POPE")
    parser.add_argument(
        "--pope-hf-splits",
        type=str,
        default="all",
        help="Comma-separated HF split names to use (or 'all')",
    )
    parser.add_argument("--pope-path", type=str, default=None, help="POPE file (.json/.jsonl) or directory")
    parser.add_argument("--image-root", type=str, default=None, help="Root directory for relative POPE image paths")
    parser.add_argument("--club-epochs", type=int, default=500)
    parser.add_argument("--club-lr", type=float, default=1e-4)
    parser.add_argument("--club-hidden-dim", type=int, default=500)
    parser.add_argument("--vision-embedder", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--text-embedder", type=str, default="xlm-roberta-base")
    parser.add_argument("--output-ckpt", type=str, default="estimator_ckpt/club_pope_llava_english.pt")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("[1/4] Loading full POPE records...")
    if args.pope_source == "hf":
        split_names = None
        if args.pope_hf_splits.strip().lower() not in {"", "all"}:
            split_names = [s.strip() for s in args.pope_hf_splits.split(",") if s.strip()]
        pope_records = load_pope_records_hf(dataset_id=args.pope_hf_dataset, split_names=split_names)
    else:
        if not args.pope_path:
            raise ValueError("--pope-path is required when --pope-source local")
        pope_path = Path(args.pope_path)
        image_root = Path(args.image_root) if args.image_root else None
        pope_records = load_pope_records(pope_path=pope_path, image_root=image_root)

    print("[2/4] Loading llava_bench_coco_English split...")
    base_split = load_dataset("changdae/llavabench-shift-natural-v1")["llava_bench_coco_English"]

    train_images = [r.image for r in pope_records] + list(base_split["image"])
    train_questions = [r.question for r in pope_records] + list(base_split["question"])
    train_refs = [r.reference_answer for r in pope_records] + list(base_split["reference_answer"])

    print(f"Training set size: POPE={len(pope_records)}, baseline={len(base_split)}, total={len(train_images)}")

    print("[3/4] Building embedders and training CLUB...")
    embedder = Embedder(v_embedder_name=args.vision_embedder, t_embedder_name=args.text_embedder)
    club = CLUB(768, 768, args.club_hidden_dim)
    club = _train_club_on_embeddings(
        club=club,
        embedder=embedder,
        train_images=train_images,
        train_questions=train_questions,
        train_refs=train_refs,
        epochs=args.club_epochs,
        lr=args.club_lr,
    )

    print("[4/4] Saving checkpoint...")
    ckpt_path = Path(args.output_ckpt)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(club.state_dict(), ckpt_path)
    print(f"Saved CLUB checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
