"""
Model inference utilities — run a multimodal LLM on HF datasets to produce answers.

Uses HuggingFace transformers for inference.
"""

import torch
from importlib.util import find_spec
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
)

# `AutoModelForVision2Seq` was introduced in newer transformers releases; import
# it only if available to avoid ImportError on older installs.
try:
    from transformers import AutoModelForVision2Seq
except Exception:
    AutoModelForVision2Seq = None

# ---------------------------------------------------------------------------
# Supported models
# ---------------------------------------------------------------------------
SUPPORTED_MODELS = {
    "llava-hf/llava-1.5-7b-hf": "LLaVA 1.5 7B",
    "llava-hf/llava-1.5-13b-hf": "LLaVA 1.5 13B",
    "llava-hf/llava-v1.6-mistral-7b-hf": "LLaVA 1.6 Mistral 7B",
    "Qwen/Qwen2-VL-7B-Instruct": "Qwen2-VL 7B",
    "meta-llama/Llama-3.2-11B-Vision-Instruct": "LLaMA 3.2 11B Vision",
}


def get_model_choices():
    """Return {display_label: model_id} for the UI dropdown."""
    return {v: k for k, v in SUPPORTED_MODELS.items()}


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
_model_cache: dict = {}


def _can_use_device_map_auto() -> bool:
    return find_spec("accelerate") is not None


def _safe_from_pretrained(model_cls, model_id: str, model_kwargs: dict, dtype, device: str):
    try:
        return model_cls.from_pretrained(model_id, **model_kwargs), model_kwargs.get("device_map") == "auto"
    except ValueError as exc:
        message = str(exc).lower()
        if "requires `accelerate`" not in message and "requires accelerate" not in message:
            raise
        print(
            "[model_inference] Warning: accelerate runtime unavailable; retrying without device_map='auto'."
        )
        fallback_kwargs = {"torch_dtype": dtype}
        model = model_cls.from_pretrained(model_id, **fallback_kwargs)
        model = model.to(device)
        return model, False


def load_model(model_id: str):
    """
    Load a multimodal model + processor from HuggingFace.
    Returns (model, processor/tokenizer).
    """
    if model_id in _model_cache:
        return _model_cache[model_id]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    use_device_map = _can_use_device_map_auto()
    model_kwargs = {"torch_dtype": dtype}
    if use_device_map:
        model_kwargs["device_map"] = "auto"
    else:
        print(
            "[model_inference] Warning: 'accelerate' is not installed; loading model without device_map='auto'."
        )

    if "llava-1.5" in model_id:
        model, used_device_map = _safe_from_pretrained(
            LlavaForConditionalGeneration, model_id, model_kwargs, dtype, device
        )
        processor = AutoProcessor.from_pretrained(model_id)
    elif "llava-v1.6" in model_id:
        model, used_device_map = _safe_from_pretrained(
            LlavaNextForConditionalGeneration, model_id, model_kwargs, dtype, device
        )
        processor = AutoProcessor.from_pretrained(model_id)
    elif "Qwen2-VL" in model_id:
        if AutoModelForVision2Seq is not None:
            model, used_device_map = _safe_from_pretrained(
                AutoModelForVision2Seq, model_id, model_kwargs, dtype, device
            )
        else:
            print(
                "[model_inference] Warning: AutoModelForVision2Seq not available in this transformers installation; falling back to AutoModelForCausalLM. Multimodal behavior may not work."
            )
            model, used_device_map = _safe_from_pretrained(
                AutoModelForCausalLM, model_id, model_kwargs, dtype, device
            )
        processor = AutoProcessor.from_pretrained(model_id)
    elif "Llama-3.2" in model_id and "Vision" in model_id:
        if AutoModelForVision2Seq is not None:
            model, used_device_map = _safe_from_pretrained(
                AutoModelForVision2Seq, model_id, model_kwargs, dtype, device
            )
        else:
            print(
                "[model_inference] Warning: AutoModelForVision2Seq not available in this transformers installation; falling back to AutoModelForCausalLM. Multimodal behavior may not work."
            )
            model, used_device_map = _safe_from_pretrained(
                AutoModelForCausalLM, model_id, model_kwargs, dtype, device
            )
        processor = AutoProcessor.from_pretrained(model_id)
    else:
        model, used_device_map = _safe_from_pretrained(
            AutoModelForCausalLM, model_id, model_kwargs, dtype, device
        )
        processor = AutoProcessor.from_pretrained(model_id)

    if not used_device_map:
        model = model.to(device)

    model.eval()
    _model_cache[model_id] = (model, processor)
    return model, processor


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def generate_answer(model, processor, image, question: str, max_new_tokens: int = 512) -> str:
    """
    Generate an answer for a single (image, question) pair.
    Works across different model architectures.
    """
    device = next(model.parameters()).device
    model_name = model.config._name_or_path if hasattr(model.config, "_name_or_path") else ""

    if "llava" in model_name.lower():
        prompt = (
            "USER: <image>\n"
            "Answer the following question using only one word: yes or no. "
            "Use lowercase letters only. Do not add any other text.\n"
            f"Question: {question}\n"
            "ASSISTANT:"
        )
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    elif "qwen2-vl" in model_name.lower():
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=text, images=[image], return_tensors="pt").to(device)
    elif "llama" in model_name.lower() and "vision" in model_name.lower():
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=text, images=[image], return_tensors="pt").to(device)
    else:
        prompt = f"Question: {question}\nAnswer:"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    # Decode only the newly generated tokens
    input_len = inputs.get("input_ids", torch.tensor([])).shape[-1] if "input_ids" in inputs else 0
    generated = output_ids[0][input_len:]
    answer = processor.decode(generated, skip_special_tokens=True).strip()
    if "llava" in model_name.lower():
        answer = answer.lower()
    return answer


def run_inference_on_split(model, processor, hf_split, num_samples: int | None = None):
    """
    Run model inference on every sample in the given HF split.

    Returns list of model_answer strings.
    """
    answers = []
    data = hf_split
    if num_samples is not None and num_samples < len(data):
        data = data.select(range(num_samples))

    for i in range(len(data)):
        item = data[i]
        image = item["image"]
        question = item["question"]
        answer = generate_answer(model, processor, image, question)
        answers.append(answer)

    return answers
