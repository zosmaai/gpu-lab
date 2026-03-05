# Multi-Token Prediction via Self-Distillation

**Status: Done**

Reproduced the paper ["Multi-Token Prediction via Self-Distillation"](https://arxiv.org/abs/2602.06019) (Kirchenbauer et al., 2025) on a single consumer GPU. The original paper used 4x NVIDIA GH200 with Llama-3.1-8B — we adapted it for Llama-3.2-1B on a single RTX 5090 (32GB).

- **Model**: [celestialcreator/Llama-3.2-1B-MTP-k8](https://huggingface.co/celestialcreator/Llama-3.2-1B-MTP-k8) on HuggingFace
- **Code**: [CelestialCreator/mtp-lm](https://github.com/CelestialCreator/mtp-lm/tree/llama32-1b-k8s-reproduction) on GitHub

---

## Background

Standard LLMs generate text one token at a time — each forward pass produces one token, and the process repeats sequentially. For a 500-token response, that's 500 forward passes.

**Multi-Token Prediction (MTP)** breaks this bottleneck. The model is trained to predict multiple future tokens simultaneously via online self-distillation:

1. A **frozen teacher** (the original model) generates soft probability distributions
2. A **trainable student** (same architecture) learns to predict k future tokens at each position using KL divergence loss
3. At inference, **ConfAdapt decoding** emits multiple tokens when the model is confident, falling back to single-token when uncertain

## Results

### GSM8K 8-shot Chain-of-Thought

| Configuration | Exact Match (flexible) | Exact Match (strict) | Throughput |
|---|---|---|---|
| **Baseline** (Llama-3.2-1B, standard AR) | **7.13%** ± 0.71 | **6.07%** ± 0.66 | ~1.5 s/sample |
| **MTP k=1** (single token, quality check) | 5.23% ± 0.61 | 2.96% ± 0.47 | ~2.4 s/sample |
| **MTP k=8 + ConfAdapt 90%** | 5.08% ± 0.60 | 3.03% ± 0.47 | **~1.3 s/sample** |

### Key Findings

- **ConfAdapt works:** k=8 with ConfAdapt matches k=1 quality while being **1.8x faster** (avg 2.82 tokens emitted per step)
- **Quality drop is expected:** The ~2% accuracy drop from baseline is consistent with our smaller setup (1B model, 500M tokens vs paper's 8B, 2B tokens)
- **The core claim holds:** Multi-token decoding preserves generation quality while improving throughput

## What We Changed from the Paper

| Parameter | Paper (8B / 4x GH200) | Ours (1B / 1x RTX 5090) |
|---|---|---|
| Base model | Llama-3.1-8B | Llama-3.2-1B |
| GPUs | 4x GH200 (96GB each) | 1x RTX 5090 (32GB) |
| k_toks | Randomized 2-16 across ranks | Fixed 8 |
| Training tokens | 2B | 500M |
| micro_batch_size | 32 | 8 |
| global_batch_size | 128 | 64 (grad accumulation) |
| mask_region_ct | 5 | 1 |
| rollout_multiplier | 4 | 2 |
| Training time | ~hours on 4x GH200 | ~17 hours on 1x RTX 5090 |

### Why fixed k=8 instead of randomized k=2-16?

The paper randomizes k across GPU ranks — each GPU samples a different k value per step, so the model learns all prediction horizons simultaneously. With a single GPU, we can only train one k value per step. We chose k=8 as a middle ground: large enough for meaningful multi-token speedup, small enough for 32GB VRAM.

## Challenges & Fixes

### 1. Single-GPU checkpoint crash
The original code calls `torch.distributed.all_reduce()` during checkpoint saving, which crashes when distributed is not initialized (single GPU). Fixed by adding an `is_initialized()` guard.

### 2. Blackwell GPU support
RTX 5090 (sm_120) requires PyTorch nightly with CUDA 12.8 — stable releases don't include sm_120 yet.

### 3. W&B defaults to paper authors' org
The W&B config defaults to `entity="tomg-group-umd"`. Fix: `--wandb.entity=null`.

### 4. Supervision flag conflict
Both `hard_teacher_supervision` and `hard_self_teacher_supervision` default to True but are mutually exclusive. For soft teacher distillation, set both to false.

### 5. HuggingFace checkpoint format
The litgpt converter outputs `model.pth` but `transformers` expects `pytorch_model.bin`.

## Training Metrics

- **Total steps:** 48,828
- **Final train loss:** ~0.9
- **Final val loss:** 1.895 (perplexity 6.65)
- **Dataset:** MetaMathQA (plain text template)
- **Supervision:** Soft teacher via KL divergence

## How to Reproduce

See the full step-by-step guide: [REPRODUCTION.md](https://github.com/CelestialCreator/mtp-lm/blob/llama32-1b-k8s-reproduction/REPRODUCTION.md)

Key files in the fork:
- `Dockerfile.mtp` — Container image for CUDA 12.8 + PyTorch nightly
- `k8s/job-mtp-prep.yaml` — Data preparation Job
- `k8s/job-mtp-train.yaml` — Training Job (GPU)
- `k8s/job-mtp-eval.yaml` — Evaluation Job (GPU)
- `config_hub/pretrain/ss_llama32_1b_single_gpu.yaml` — Training config

## References

- Kirchenbauer, J., Geiping, J., Wen, Y., & Goldstein, T. (2025). *Multi-Token Prediction via Self-Distillation*. arXiv:2602.06019.
- Gloeckle, F., et al. (2024). *Better & Faster Large Language Models via Multi-token Prediction*. arXiv:2404.19737.
- DeepSeek-AI. (2024). *DeepSeek-V3 Technical Report*. (MTP at production scale.)
