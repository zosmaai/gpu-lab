#!/usr/bin/env python3
"""Push merged model to HuggingFace. No model loading — just file uploads."""

import os
from huggingface_hub import HfApi, create_repo

MERGED_PATH = os.environ.get("MERGED_PATH", "/home/akshay/axon-workspace/merged")
HF_REPO = os.environ.get("HF_REPO", "celestialcreator/axon-smollm2-360m")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

MODEL_CARD = """---
language: en
license: apache-2.0
library_name: transformers
tags:
  - smollm2
  - lora
  - portfolio-assistant
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
- **Hardware:** NVIDIA RTX 5090
- **Base model:** SmolLM2-360M-Instruct
- **Precision:** bf16
- **Train loss:** 1.33
- **Eval loss:** 0.94
- **Token accuracy:** 80%
- **Training time:** 6 minutes

## Usage (Browser — Transformers.js)

```javascript
import { pipeline } from '@huggingface/transformers';

const generator = await pipeline('text-generation', 'celestialcreator/axon-smollm2-360m', {
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

pipe = pipeline("text-generation", model="celestialcreator/axon-smollm2-360m")
result = pipe([{"role": "user", "content": "Who is Akshay?"}], max_new_tokens=150)
```
"""


def main():
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set")
        return

    api = HfApi(token=HF_TOKEN)

    # Create repo
    try:
        create_repo(HF_REPO, token=HF_TOKEN, exist_ok=True)
        print(f"Repo: {HF_REPO}")
    except Exception as e:
        print(f"Repo warning: {e}")

    # Write model card to a temp location (merged dir may be root-owned from k8s)
    import tempfile, shutil
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy model files
        for f in os.listdir(MERGED_PATH):
            src = os.path.join(MERGED_PATH, f)
            if os.path.isfile(src):
                shutil.copy2(src, tmpdir)

        # Write model card
        with open(os.path.join(tmpdir, "README.md"), "w") as f:
            f.write(MODEL_CARD)

        # Upload
        print(f"Uploading to {HF_REPO}...")
        api.upload_folder(
            folder_path=tmpdir,
            repo_id=HF_REPO,
            commit_message="Add fine-tuned SmolLM2-360M Axon portfolio assistant",
        )
    print(f"Done! https://huggingface.co/{HF_REPO}")


if __name__ == "__main__":
    main()
