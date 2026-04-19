## MLLMShift-EMI

Lightweight toolkit and Gradio app for evaluating multimodal LLMs under distribution
shifts using EMI / EMID metrics and RP scoring. This repository builds on and extends
the LLavaBench-Shift datasets and related work by `changdae`.

Key features
- Compute EMI, EMID and an empirical upper bound (EMID_UB) using a CLUB estimator.
- Compute RP (relative performance) scores using an LLM judge and image captions.
- Gradio app to orchestrate experiments: select ID / OOD splits, train or load CLUB,
  run model inference (HuggingFace models), compute metrics and correlations.

Note:
This project is an extension of the LLavaBench-Shift dataset and evaluation
scripts by changdae (see changdae/llavabench-shift-* on HuggingFace, [paper](https://arxiv.org/abs/2304.08485) and [repository](https://github.com/haotian-liu/LLaVA/tree/main)). All dataset
splits, sample mappings and captioning are derived from that prior work.

Quick start
1. Create and activate the Python environment (recommended `mllmshift-emi`):

```bash
conda create -n mllmshift-emi python=3.10 -y
conda activate mllmshift-emi
pip install -r gradio_app/requirements.txt
pip install -e .
```

2. Generate or load image captions (optional):

```bash
python image_caption_generator.py --output image_captions.json
```

3. Run the Gradio experiment app:

```bash
conda run -n mllmshift-emi python -m gradio_app.app
```

Repository layout (important files)
- `main.py` — core CLI for EMI / EMID estimation and pair evaluations.
- `rp_score.py`, `test.py` — scripts for computing RP scores (judge-based).
- `image_caption_generator.py` — generate image captions via Ollama/Llama Vision.
- `gradio_app/` — Gradio UI and modular components for running experiments.
- `combined_dataset/`, `data/` — dataset splits and local combined jsonl files.
