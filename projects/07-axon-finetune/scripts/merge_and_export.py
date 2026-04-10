#!/usr/bin/env python3
"""
Merge LoRA adapter into base model, export to ONNX, push to HuggingFace.
Can run on CPU — no GPU needed for this step.
"""

import os
import json
import shutil
import subprocess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi, create_repo

# Paths
MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
LORA_PATH = os.environ.get("LORA_PATH", "/workspace/lora-adapter")
MERGED_PATH = os.environ.get("MERGED_PATH", "/workspace/merged")
ONNX_PATH = os.environ.get("ONNX_PATH", "/workspace/onnx")
HF_CACHE = os.environ.get("HF_HOME", "/workspace/hf_cache")
HF_REPO = os.environ.get("HF_REPO", "CelestialCreator/axon-smollm2-360m")
HF_TOKEN = os.environ.get("HF_TOKEN", "")


def merge_lora():
    """Step 1: Merge LoRA weights into base model."""
    print("=" * 60)
    print("Step 1: Merging LoRA adapter into base model")
    print("=" * 60)

    print(f"Loading base model: {MODEL_NAME}")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=HF_CACHE,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE)

    print(f"Loading LoRA adapter from: {LORA_PATH}")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)

    print("Merging LoRA weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {MERGED_PATH}")
    model.save_pretrained(MERGED_PATH)
    tokenizer.save_pretrained(MERGED_PATH)

    # Quick sanity test
    print("\nSanity test (merged model):")
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer("Who is Akshay?", return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"  Q: Who is Akshay?")
    print(f"  A: {response}")

    return model, tokenizer


def export_onnx():
    """Step 2: Export to ONNX for Transformers.js.

    Uses optimum's Python API with device='cuda' to keep the model in VRAM
    and avoid OOM on the 16GB system RAM.
    """
    print("\n" + "=" * 60)
    print("Step 2: Exporting to ONNX (GPU-accelerated)")
    print("=" * 60)

    os.makedirs(ONNX_PATH, exist_ok=True)

    # Load model on GPU to avoid CPU RAM pressure
    print(f"Loading merged model from {MERGED_PATH} onto GPU...")
    model = AutoModelForCausalLM.from_pretrained(
        MERGED_PATH,
        torch_dtype=torch.float32,  # ONNX export needs float32
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(MERGED_PATH)

    # Move to CPU just for the export step (ONNX trace requires CPU)
    # but do it after loading on GPU to avoid loading from disk to CPU RAM
    print("Moving model to CPU for ONNX trace...")
    model = model.to("cpu")
    model.eval()

    # Create dummy input
    dummy_text = "Hello"
    dummy_input = tokenizer(dummy_text, return_tensors="pt")
    input_ids = dummy_input["input_ids"]
    attention_mask = dummy_input["attention_mask"]

    # Export to ONNX
    onnx_model_path = os.path.join(ONNX_PATH, "model.onnx")
    print(f"Exporting to {onnx_model_path}...")

    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        onnx_model_path,
        export_params=True,
        opset_version=14,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"},
        },
    )

    # Free model from memory
    del model
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    # Copy tokenizer files to ONNX dir
    print("Copying tokenizer files...")
    for fname in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                  "vocab.json", "merges.txt", "config.json", "generation_config.json",
                  "chat_template.jinja"]:
        src = os.path.join(MERGED_PATH, fname)
        if os.path.exists(src):
            shutil.copy2(src, ONNX_PATH)

    print(f"\nONNX model saved to: {ONNX_PATH}")
    for f in sorted(os.listdir(ONNX_PATH)):
        size = os.path.getsize(os.path.join(ONNX_PATH, f))
        print(f"  {f}: {size / 1e6:.1f} MB")


def push_to_hub():
    """Step 3: Push to HuggingFace."""
    print("\n" + "=" * 60)
    print(f"Step 3: Pushing to HuggingFace ({HF_REPO})")
    print("=" * 60)

    if not HF_TOKEN:
        print("WARNING: No HF_TOKEN set, skipping push")
        return

    api = HfApi(token=HF_TOKEN)

    # Create repo if it doesn't exist
    try:
        create_repo(HF_REPO, token=HF_TOKEN, exist_ok=True)
        print(f"Repo created/exists: {HF_REPO}")
    except Exception as e:
        print(f"Repo creation warning: {e}")

    # Write model card
    model_card = """---
language: en
license: apache-2.0
library_name: transformers
tags:
  - smollm2
  - lora
  - portfolio-assistant
  - onnx
  - transformers.js
  - webgpu
base_model: HuggingFaceTB/SmolLM2-360M-Instruct
pipeline_tag: text-generation
---

# Axon — SmolLM2-360M Portfolio Assistant

Fine-tuned [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) to serve as a personal portfolio AI assistant. Runs entirely in the browser via WebGPU + Transformers.js.

## What is Axon?

Axon is a client-side AI assistant embedded in a Three.js 3D portfolio website. It answers questions about the portfolio owner (Akshay Mhaskar) — work experience, skills, projects, and contact info. Zero backend, zero API costs, fully private.

## Training

- **Method:** LoRA (rank=32, alpha=64) fine-tuning via TRL SFTTrainer
- **Dataset:** 1,001 curated chat examples (knowledge Q&A, refusal training, edge cases)
- **Hardware:** RTX 2070 SUPER (8GB VRAM)
- **Base model:** SmolLM2-360M-Instruct
- **Precision:** fp16

## Usage (Browser — Transformers.js)

```javascript
import { pipeline } from '@huggingface/transformers';

const generator = await pipeline('text-generation', 'CelestialCreator/axon-smollm2-360m', {
  dtype: 'q4f16',
  device: 'webgpu',
});

const result = await generator([
  { role: 'user', content: 'Who is Akshay?' }
], { max_new_tokens: 150 });
```

## Usage (Python)

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="CelestialCreator/axon-smollm2-360m")
result = pipe([{"role": "user", "content": "Who is Akshay?"}], max_new_tokens=150)
```
"""

    card_path = os.path.join(MERGED_PATH, "README.md")
    with open(card_path, "w") as f:
        f.write(model_card)

    # Upload merged model
    print("Uploading merged model...")
    api.upload_folder(
        folder_path=MERGED_PATH,
        repo_id=HF_REPO,
        commit_message="Add fine-tuned SmolLM2-360M Axon model",
    )

    # Upload ONNX files
    print("Uploading ONNX files...")
    api.upload_folder(
        folder_path=ONNX_PATH,
        repo_id=HF_REPO,
        path_in_repo="onnx",
        commit_message="Add ONNX export for Transformers.js",
    )

    print(f"\nDone! Model available at: https://huggingface.co/{HF_REPO}")


def main():
    if os.path.exists(os.path.join(MERGED_PATH, "model.safetensors")):
        print(f"Merged model already exists at {MERGED_PATH}, skipping merge step.")
    else:
        merge_lora()
    export_onnx()
    push_to_hub()
    print("\n" + "=" * 60)
    print("All steps complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
