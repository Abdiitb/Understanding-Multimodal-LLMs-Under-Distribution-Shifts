"""
RP (Relative Performance) score computation using LLaMA 3.1 Vision for
image captioning and LLaMA 3.1 8B as the LLM judge.

Uses HuggingFace transformers for all inference.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, MllamaForConditionalGeneration
from PIL import Image

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
CAPTION_MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
JUDGE_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

CAPTION_PROMPT = (
    "Describe this image in detail for someone who cannot see it. "
    "Include: the main subject(s), their actions and poses, spatial layout, "
    "background elements, notable colors, lighting, text visible in the image, "
    "and any other visually important details. "
    "Be factual and concise — aim for 3-5 sentences."
)

JUDGE_SYSTEM_PROMPT = """You are a fair judge evaluating two AI assistants' responses to a question about an image. You are given a description of the image for context.

Rate each assistant's response on a scale of 1 to 10 based on helpfulness, relevance, accuracy, and level of detail.

Output ONLY a single line with two numbers separated by a space:
score_for_Assistant1 score_for_Assistant2"""


# ---------------------------------------------------------------------------
# Cached model holders
# ---------------------------------------------------------------------------
_caption_model = None
_caption_processor = None
_judge_model = None
_judge_tokenizer = None


def _load_caption_model():
    global _caption_model, _caption_processor
    if _caption_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        _caption_model = MllamaForConditionalGeneration.from_pretrained(
            CAPTION_MODEL_ID, torch_dtype=dtype, device_map="auto"
        )
        _caption_processor = AutoProcessor.from_pretrained(CAPTION_MODEL_ID)
        _caption_model.eval()
    return _caption_model, _caption_processor


def _load_judge_model():
    global _judge_model, _judge_tokenizer
    if _judge_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        _judge_model = AutoModelForCausalLM.from_pretrained(
            JUDGE_MODEL_ID, torch_dtype=dtype, device_map="auto"
        )
        _judge_tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_ID)
        _judge_model.eval()
    return _judge_model, _judge_tokenizer


# ---------------------------------------------------------------------------
# Caption generation
# ---------------------------------------------------------------------------
def generate_caption(image: Image.Image) -> str:
    """Generate a detailed caption for an image using LLaMA 3.2 Vision."""
    model, processor = _load_caption_model()
    device = next(model.parameters()).device

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": CAPTION_PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, images=[image], return_tensors="pt").to(device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=300, do_sample=False)

    input_len = inputs["input_ids"].shape[-1]
    caption = processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
    return caption


# ---------------------------------------------------------------------------
# Judge scoring
# ---------------------------------------------------------------------------
def _parse_scores(text: str):
    """Parse 'score1 score2' from the judge output."""
    try:
        first_line = text.strip().split("\n")[0].replace(",", " ")
        parts = first_line.split()
        return float(parts[0]), float(parts[1])
    except Exception:
        return None, None


def judge_score(question: str, image_caption: str, ref_answer: str, model_answer: str) -> tuple:
    """
    Use LLaMA 3.1 8B to judge ref_answer vs model_answer.
    Returns (ref_score, model_score) or (None, None) on failure.
    """
    model, tokenizer = _load_judge_model()
    device = next(model.parameters()).device

    user_prompt = (
        f"[Image Description]\n{image_caption}\n\n"
        f"[Question]\n{question}\n\n"
        f"[Assistant 1]\n{ref_answer}\n[End of Assistant 1]\n\n"
        f"[Assistant 2]\n{model_answer}\n[End of Assistant 2]"
    )

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

    with torch.inference_mode():
        output_ids = model.generate(input_ids, max_new_tokens=20, do_sample=False)

    generated = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return _parse_scores(generated)


# ---------------------------------------------------------------------------
# RP score for a full split
# ---------------------------------------------------------------------------
def compute_rp_scores(
    questions: list[str],
    ref_answers: list[str],
    model_answers: list[str],
    images: list[Image.Image],
    progress_callback=None,
) -> dict:
    """
    Compute RP scores for a split.

    Steps:
      1. Generate captions for each image using LLaMA 3.2 Vision.
      2. Judge each (ref_answer, model_answer) pair using LLaMA 3.1 8B.
      3. RP = model_score / ref_score (skip if ref_score == 0).

    Returns dict with 'rp_values', 'mean_rp', 'num_scored'.
    """
    rp_values = []
    n = len(questions)

    for i in range(n):
        try:
            caption = generate_caption(images[i])
            ref_score, model_score = judge_score(
                questions[i], caption, ref_answers[i], model_answers[i]
            )
            if ref_score is not None and ref_score > 0:
                rp_values.append(model_score / ref_score)
        except Exception as e:
            print(f"[RP] Error on sample {i}: {e}")
            continue

        if progress_callback and (i + 1) % 10 == 0:
            progress_callback(i + 1, n)

    mean_rp = sum(rp_values) / len(rp_values) if rp_values else 0.0

    return {
        "rp_values": rp_values,
        "mean_rp": mean_rp,
        "num_scored": len(rp_values),
    }
