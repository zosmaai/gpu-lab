# Project 07: Axon — Browser AI Portfolio Assistant

**Status:** Done

LoRA fine-tuned [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) to serve as a personal portfolio AI assistant that runs entirely in the browser via WebGPU + Transformers.js. Zero backend, zero API costs.

**Model:** [celestialcreator/axon-smollm2-360m](https://huggingface.co/celestialcreator/axon-smollm2-360m)
**Live site:** [akshay-mhaskar.vercel.app](https://akshay-mhaskar.vercel.app/)

## Results

| Metric | Value |
|--------|-------|
| Train loss | 1.33 |
| Eval loss | 0.94 |
| Token accuracy | 80% |
| Training time | 6 minutes |
| GPU peak VRAM | 1.24 GB |
| Model size (safetensors) | 724 MB |

## What is Axon?

Axon is a client-side AI assistant embedded in a Three.js 3D portfolio website. It answers questions about Akshay Mhaskar — work experience, skills, projects, and contact info. Off-topic questions (code generation, personal questions, jailbreak attempts) are refused.

Previously, Axon used the base SmolLM2-360M-Instruct with a long system prompt stuffed with context. The fine-tune bakes this knowledge directly into the model weights — no system prompt needed, faster inference, better answers.

## Training

- **Base model:** SmolLM2-360M-Instruct (360M parameters)
- **Method:** LoRA (rank=32, alpha=64, dropout=0.1, all linear layers)
- **Precision:** bf16
- **Dataset:** 1,001 chat examples in system/user/assistant format
  - 77% knowledge Q&A about Akshay
  - 14% refusal training (off-topic, jailbreak)
  - 5% behavior/tone (greetings, closings)
  - 5% edge cases (salary, relocation, recruiter questions)
- **Hyperparameters:** lr=2e-4, 3 epochs, batch=2 × grad_accum=8 (effective 16), warmup 10%
- **Hardware:** NVIDIA RTX 5090 (32 GB)
- **Framework:** TRL SFTTrainer + PEFT

## Architecture

```
User's browser (any device with WebGPU)
  → Transformers.js
    → SmolLM2-360M fine-tuned (ONNX, quantized)
      → Responds about Akshay
      → Refuses off-topic
      → No backend, no API keys
```

## Quick Start

### Training (on GPU server)

```bash
# 1. Build Docker image
docker build -t localhost:5000/axon-finetune:latest -f Dockerfile .
docker push localhost:5000/axon-finetune:latest

# 2. Configure secrets
cp .env.example .env
# Edit .env with HF_TOKEN and GPU_UUID

# 3. Run training
./apply.sh k8s/job-train.yaml
kubectl logs -f job/axon-finetune

# 4. Push to HuggingFace
export HF_TOKEN=<your-token>
python3 scripts/push_to_hub.py
```

### Browser deployment

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

## Files

| File | Purpose |
|------|---------|
| `dataset.jsonl` | 1,001 training examples (system/user/assistant format) |
| `Dockerfile` | Container image with torch + peft + trl + optimum |
| `scripts/train.py` | LoRA SFT training with TRL SFTTrainer |
| `scripts/merge_and_export.py` | Merge LoRA → ONNX export → push to HF |
| `scripts/push_to_hub.py` | Lightweight HuggingFace upload (no model loading) |
| `k8s/job-train.yaml` | K8s Job for training + merge + export |
| `k8s/job-export.yaml` | K8s Job for ONNX export only |
| `.env.example` | Template for secrets (HF_TOKEN, GPU_UUID) |
| `apply.sh` | envsubst + kubectl apply helper |

## Lessons Learned

- **SmolLM2-360M trains in minutes** — 6 minutes for 3 epochs on 1,001 examples. The model is small enough that LoRA overhead dominates over compute.
- **bf16 vs fp16 matters** — RTX 2070 SUPER (SM_75) doesn't support bf16. We switched to the RTX 5090 (SM_120) which does. Always check compute capability.
- **ONNX export eats CPU RAM** — `optimum-cli export onnx` loads the model on CPU. On our 16GB system, this OOM-killed the container. The model is small (724MB) but the ONNX graph builder allocates significant extra buffers.
- **K8s GPU scheduling** — the NVIDIA device plugin must be restarted when GPUs are physically added/removed. Lost the RTX 3080 mid-session and had to restart kubelet to re-enumerate.
- **System prompt → fine-tune** — baking context into weights via LoRA is cleaner than stuffing a system prompt. No token overhead, faster first-token latency, and the model can't "forget" the instructions mid-conversation.
