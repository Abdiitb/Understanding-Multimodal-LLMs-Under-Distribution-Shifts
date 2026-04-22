"""
Run inference on the full POPE dataset using a Hugging Face multimodal model
and save responses as JSON.

Output records include at least:
- qid
- question
- reference_answer
- model_answer

Optional metadata such as split/category/image_id are also saved when available.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForCausalLM, AutoProcessor, AutoTokenizer


logger = logging.getLogger(__name__)

try:
    from transformers import AutoModelForVision2Seq
except Exception:
    AutoModelForVision2Seq = None

try:
    from transformers import LlavaForConditionalGeneration
except Exception:
    LlavaForConditionalGeneration = None

try:
    from transformers import LlavaNextForConditionalGeneration
except Exception:
    LlavaNextForConditionalGeneration = None


def _normalize_yes_no(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return "yes" if int(value) == 1 else "no"

    text = str(value).strip().lower()
    if text in {"yes", "y", "1", "true", "present", "positive"}:
        return "yes"
    if text in {"no", "n", "0", "false", "absent", "negative"}:
        return "no"
    return str(value)


def _extract_question(item: dict[str, Any]) -> str | None:
    for key in ["question", "ques", "prompt", "text"]:
        value = item.get(key)
        if value is not None and str(value).strip() != "":
            return str(value)
    return None


def _extract_reference_answer(item: dict[str, Any]) -> str | None:
    for key in ["reference_answer", "answer", "ans", "label", "target"]:
        if key in item and item[key] is not None:
            return _normalize_yes_no(item[key])
    return None


def _extract_qid(item: dict[str, Any], split_name: str, idx: int) -> str:
    for key in ["qid", "question_id", "id", "uid"]:
        if key in item and item[key] is not None and str(item[key]).strip() != "":
            return str(item[key])
    return f"{split_name}_{idx}"


def _extract_image(item: dict[str, Any]) -> tuple[Image.Image, str] | None:
    image_value = None
    for key in ["image", "img", "image_path"]:
        if key in item and item[key] is not None:
            image_value = item[key]
            break

    if image_value is None:
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


def _from_pretrained_compat(
    model_cls,
    model_id: str,
    dtype: torch.dtype,
    trust_remote_code: bool,
    device_map: str,
):
    common_kwargs = {
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
    }

    try:
        return model_cls.from_pretrained(model_id, dtype=dtype, **common_kwargs)
    except TypeError:
        return model_cls.from_pretrained(model_id, torch_dtype=dtype, **common_kwargs)


def _safe_load_model(model_id: str, dtype: torch.dtype, trust_remote_code: bool, device_map: str):
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model_type = str(getattr(config, "model_type", "")).lower()

    if "llava_next" in model_type and LlavaNextForConditionalGeneration is not None:
        return _from_pretrained_compat(LlavaNextForConditionalGeneration, model_id, dtype, trust_remote_code, device_map)

    if "llava" in model_type and LlavaForConditionalGeneration is not None:
        return _from_pretrained_compat(LlavaForConditionalGeneration, model_id, dtype, trust_remote_code, device_map)

    if AutoModelForVision2Seq is not None:
        try:
            return _from_pretrained_compat(AutoModelForVision2Seq, model_id, dtype, trust_remote_code, device_map)
        except Exception:
            pass

    return _from_pretrained_compat(AutoModelForCausalLM, model_id, dtype, trust_remote_code, device_map)


def load_hf_mllm(model_id: str, trust_remote_code: bool = False):
    low_vram_threshold_gb = 8.0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. CPU fallback is disabled.")

    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if total_vram_gb < low_vram_threshold_gb:
        raise RuntimeError(
            f"GPU has {total_vram_gb:.2f} GB VRAM (< {low_vram_threshold_gb:.1f} GB). "
            "CPU fallback is disabled. Use a smaller model or a higher-memory GPU."
        )

    device = "cuda"
    device_map = "auto"
    dtype = torch.float16

    # Load model first, then processor to avoid tokenizer cache issues
    try:
        model = _safe_load_model(
            model_id=model_id,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
        )
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    def _build_llava_processor_compat(cache_dir: str | None = None):
        logger.warning("Using compatibility LLaVA processor construction path.")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            use_fast=False,
            cache_dir=cache_dir,
        )
        image_processor = AutoImageProcessor.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )

        try:
            from transformers import LlavaProcessor

            return LlavaProcessor(tokenizer=tokenizer, image_processor=image_processor)
        except Exception as llava_ctor_error:
            logger.warning(
                "Could not instantiate LlavaProcessor directly (%s). Falling back to minimal wrapper.",
                llava_ctor_error,
            )

            class _CompatLlavaProcessor:
                def __init__(self, tok, img_proc):
                    self.tokenizer = tok
                    self.image_processor = img_proc

                def __call__(self, text, images, return_tensors="pt"):
                    images_list = images if isinstance(images, list) else [images]
                    text_inputs = self.tokenizer(text=text, return_tensors=return_tensors)
                    image_inputs = self.image_processor(images=images_list, return_tensors=return_tensors)
                    image_inputs.update(text_inputs)
                    return image_inputs

                def decode(self, *args, **kwargs):
                    return self.tokenizer.decode(*args, **kwargs)

                def batch_decode(self, *args, **kwargs):
                    return self.tokenizer.batch_decode(*args, **kwargs)

            return _CompatLlavaProcessor(tokenizer, image_processor)

    # Try to load processor with robust fallbacks for tokenizer/config incompatibilities
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    except Exception as e:
        err_text = str(e)
        is_image_token_error = "unexpected keyword argument 'image_token'" in err_text
        is_fast_tokenizer_error = "ModelWrapper" in err_text or "data did not match" in err_text
        if is_image_token_error:
            processor = _build_llava_processor_compat()
        elif not is_fast_tokenizer_error:
            raise
        else:
            logger.warning("Fast tokenizer load failed (%s). Retrying with slow tokenizer.", err_text)
            try:
                processor = AutoProcessor.from_pretrained(
                    model_id,
                    trust_remote_code=trust_remote_code,
                    use_fast=False,
                )
            except Exception as slow_error:
                slow_err_text = str(slow_error)
                if "unexpected keyword argument 'image_token'" in slow_err_text:
                    processor = _build_llava_processor_compat()
                else:
                    logger.warning("Slow tokenizer retry failed (%s). Retrying with fresh cache.", slow_error)

                    temp_cache = tempfile.mkdtemp(prefix="hf_cache_")
                    try:
                        processor = AutoProcessor.from_pretrained(
                            model_id,
                            trust_remote_code=trust_remote_code,
                            use_fast=False,
                            cache_dir=temp_cache,
                        )
                    except Exception as cache_error:
                        cache_err_text = str(cache_error)
                        if "unexpected keyword argument 'image_token'" in cache_err_text:
                            processor = _build_llava_processor_compat(cache_dir=temp_cache)
                        else:
                            raise
                    os.environ["HF_HOME"] = temp_cache
                    logger.info("Processor loaded successfully using fresh cache directory: %s", temp_cache)
    
    return model, processor


def _build_inputs(processor, image: Image.Image, question: str, device: torch.device):
    if hasattr(processor, "apply_chat_template"):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]
        try:
            chat_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=chat_text, images=[image], return_tensors="pt")
            return {k: v.to(device) for k, v in inputs.items()}
        except Exception:
            pass

    prompt = f"USER: <image>\nAnswer the following question using only one word: yes or no. Use lowercase letters only. Do not add any other text.\nQuestion: {question}\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


def generate_answer(model, processor, image: Image.Image, question: str, max_new_tokens: int) -> str:
    device = next(model.parameters()).device
    inputs = _build_inputs(processor=processor, image=image, question=question, device=device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
    generated_ids = output_ids[0][input_len:]

    if hasattr(processor, "decode"):
        text = processor.decode(generated_ids, skip_special_tokens=True)
    else:
        text = processor.batch_decode(generated_ids.unsqueeze(0), skip_special_tokens=True)[0]
    return text.strip()


def _iter_target_splits(dataset_dict, requested_splits: str) -> list[str]:
    all_splits = list(dataset_dict.keys())
    if requested_splits.strip().lower() in {"", "all"}:
        return all_splits

    split_names = [s.strip() for s in requested_splits.split(",") if s.strip()]
    valid = [s for s in split_names if s in dataset_dict]
    missing = [s for s in split_names if s not in dataset_dict]
    if missing:
        raise ValueError(f"Requested split(s) not found: {missing}. Available: {all_splits}")
    return valid


def run_inference(
    model,
    processor,
    dataset_split: Dataset,
    split_name: str,
    max_new_tokens: int,
    max_samples_per_split: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    limit = min(len(dataset_split), max_samples_per_split) if max_samples_per_split > 0 else len(dataset_split)
    split_start = time.time()

    for idx in range(limit):
        sample_start = time.time()
        logger.info("[%s] starting sample %d/%d", split_name, idx + 1, limit)
        item = dict(dataset_split[idx])

        question = _extract_question(item)
        reference_answer = _extract_reference_answer(item)
        image_pack = _extract_image(item)

        if question is None or reference_answer is None or image_pack is None:
            continue

        image, image_id = image_pack
        qid = _extract_qid(item, split_name=split_name, idx=idx)
        model_answer = generate_answer(
            model=model,
            processor=processor,
            image=image,
            question=question,
            max_new_tokens=max_new_tokens,
        )

        records.append(
            {
                "qid": qid,
                "question": question,
                "reference_answer": reference_answer,
                "model_answer": model_answer,
                "split": split_name,
                "category": item.get("category") or item.get("type") or item.get("split"),
                "image_id": image_id,
            }
        )

        sample_elapsed = time.time() - sample_start
        logger.info("[%s] finished sample %d/%d in %.1fs", split_name, idx + 1, limit, sample_elapsed)

    split_elapsed = time.time() - split_start
    logger.info("[%s] completed %d/%d samples in %.1fs", split_name, len(records), limit, split_elapsed)

    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference on full POPE dataset with a Hugging Face MLLM")
    parser.add_argument("--model-id", type=str, required=True, help="Hugging Face model id")
    parser.add_argument("--pope-dataset", type=str, default="lmms-lab/POPE", help="HF dataset id for POPE")
    parser.add_argument(
        "--pope-splits",
        type=str,
        default="all",
        help="Comma-separated split names or 'all'",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Max generated tokens per sample. Keep low (e.g. 16-64) for quick sanity checks.",
    )
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=5,
        help="Number of samples to run per split for quick sanity-check. Use -1 or 0 for all samples.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="results/pope_responses/pope_model_responses_full.json",
        help="Path to save output JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("[1/4] Loading POPE dataset...")
    dataset_dict = load_dataset(args.pope_dataset)
    split_names = _iter_target_splits(dataset_dict, args.pope_splits)

    logger.info("[2/4] Loading model and processor...")
    model, processor = load_hf_mllm(args.model_id, trust_remote_code=args.trust_remote_code)

    logger.info("[3/4] Running inference...")
    all_records: list[dict[str, Any]] = []
    for split_name in split_names:
        split_size = len(dataset_dict[split_name])
        effective_limit = min(split_size, args.max_samples_per_split) if args.max_samples_per_split > 0 else split_size
        logger.info("Split '%s': running %d/%d samples", split_name, effective_limit, split_size)
        split_records = run_inference(
            model=model,
            processor=processor,
            dataset_split=dataset_dict[split_name],
            split_name=split_name,
            max_new_tokens=args.max_new_tokens,
            max_samples_per_split=args.max_samples_per_split,
        )
        all_records.extend(split_records)
        logger.info("Completed split '%s' with %d usable samples.", split_name, len(split_records))

    logger.info("[4/4] Saving JSON output...")
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_id": args.model_id,
        "pope_dataset": args.pope_dataset,
        "splits": split_names,
        "num_records": len(all_records),
        "records": all_records,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info("Saved %d records to: %s", len(all_records), output_path)


if __name__ == "__main__":
    main()
