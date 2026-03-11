"""
Model inference utilities — run a multimodal LLM on HF datasets to produce answers.

Uses HuggingFace transformers for inference.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
    AutoModelForVision2Seq,
)

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


def load_model(model_id: str):
    """
    Load a multimodal model + processor from HuggingFace.
    Returns (model, processor/tokenizer).
    """
    if model_id in _model_cache:
        return _model_cache[model_id]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    if "llava-1.5" in model_id:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_id)
    elif "llava-v1.6" in model_id:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_id)
    elif "Qwen2-VL" in model_id:
        model = AutoModelForVision2Seq.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_id)
    elif "Llama-3.2" in model_id and "Vision" in model_id:
        model = AutoModelForVision2Seq.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_id)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_id)

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
        prompt = f"USER: <image>\n{question}\nASSISTANT:"
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
