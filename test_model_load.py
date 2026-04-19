#!/usr/bin/env python
"""Quick test to verify model loading works."""

import sys
sys.path.insert(0, '/home/aryan-badkul/Desktop/College_Study/IE663/Project/mllmshift-emi')

from hallucination_detection.infer_pope_hf_mllm import load_hf_mllm

print("Loading model...")
try:
    model, processor = load_hf_mllm('llava-hf/llava-1.5-7b-hf')
    print("✓ Model loaded successfully!")
    print(f"✓ Processor type: {type(processor).__name__}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
