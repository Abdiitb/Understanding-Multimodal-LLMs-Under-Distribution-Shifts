#!/bin/bash
# Wrapper script to run hallucination detection with fresh HuggingFace cache

# Use a temporary cache directory to avoid tokenizer corruption
export HF_HOME="/tmp/hf_cache_$$_$(date +%s)"
mkdir -p "$HF_HOME"

echo "Using HF_HOME: $HF_HOME"

# Run the hallucination detection script
conda activate mllmshift-emi
python hallucination_detection/infer_pope_hf_mllm.py "$@"

# Cleanup
rm -rf "$HF_HOME"
