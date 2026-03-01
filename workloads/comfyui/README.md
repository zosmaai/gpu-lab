# ComfyUI Workload

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) — a node-based UI for diffusion models and other generative AI workflows.

## Deployment

The manifests create:
- [`namespace.yaml`](namespace.yaml) — `comfyui` namespace
- [`comfyui-deployment.yaml`](comfyui-deployment.yaml) — Single replica using `mmartial/comfyui-nvidia-docker`
- [`comfyui-service.yaml`](comfyui-service.yaml) — NodePort exposing the web UI on port **30188**

### Docker Image Selection

Initially tried `ghcr.io/ai-dock/comfyui:pytorch-2.5.1-cu124-runtime` but ran into issues. Switched to [`mmartial/comfyui-nvidia-docker:ubuntu24_cuda12.8-latest`](https://github.com/mmartial/comfyui-nvidia-docker) which provides:
- Ubuntu 24 + CUDA 12.8 base
- `FORCE_CHOWN=yes` env var for fixing volume ownership on startup
- `USE_UV=true` for faster Python package installs
- Built-in venv at `/comfy/mnt/venv`

### Deploying

```bash
kubectl apply -f namespace.yaml
kubectl apply -f comfyui-deployment.yaml
kubectl apply -f comfyui-service.yaml
```

Access at `http://<node-ip>:30188`

## Use Cases

ComfyUI is used for three different projects in this lab:

| Project | What It Does | Custom Nodes |
|---------|-------------|--------------|
| [Dataset Creation](../../projects/02-dataset-creation/) | Qwen Image Edit + Florence2 captioning pipeline | ComfyUI-GGUF, ComfyUI-Florence2, ComfyUI-KJNodes, comfyui_controlnet_aux, rgthree-comfy |
| [Music Generation](../../projects/03-music-generation/) | ACE-Step 1.5 audio generation | ComfyUI_ACE-Step |
| LoRA Inference | Generate images using trained LoRA adapters | (built-in LoraLoader node) |

## Custom Node Installation

All custom nodes were installed via `git clone` into the pod's `/workspace/custom_nodes/` directory, then dependencies installed with the venv pip.

### Batch install (for Qwen Image Edit pipeline)

```bash
POD=$(kubectl -n comfyui get pod -l app=comfyui -o jsonpath='{.items[0].metadata.name}')

kubectl exec -n comfyui $POD -- bash -c '
  cd /workspace/custom_nodes &&
  git clone https://github.com/city96/ComfyUI-GGUF.git &&
  git clone https://github.com/rgthree/rgthree-comfy.git &&
  git clone https://github.com/kijai/ComfyUI-KJNodes.git &&
  git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git
'
```

### Florence2 (for auto-captioning)

```bash
kubectl exec -n comfyui $POD -- bash -c '
  cd /workspace/custom_nodes &&
  git clone https://github.com/kijai/ComfyUI-Florence2.git &&
  cd ComfyUI-Florence2 &&
  /comfy/mnt/venv/bin/pip install -r requirements.txt
'
```

Dependencies: `timm`, `peft`, `accelerate`. The Florence2 model itself (`microsoft/Florence-2-base`) downloads automatically on first workflow run.

### ACE-Step (for music generation)

```bash
kubectl exec -n comfyui $POD -- bash -c '
  cd /workspace/custom_nodes &&
  git clone https://github.com/billwuhao/ComfyUI_ACE-Step.git &&
  cd ComfyUI_ACE-Step &&
  /comfy/mnt/venv/bin/pip install -r requirements.txt
'
```

### ComfyUI Manager

```bash
kubectl exec -n comfyui $POD -- bash -c '
  cd /workspace/custom_nodes &&
  git clone https://github.com/ltdrdata/ComfyUI-Manager.git
'
```

### After installing nodes — restart the pod

```bash
kubectl -n comfyui rollout restart deployment comfyui
```

## Custom Nodes Summary

| Node | Repo | Purpose | Installed For |
|------|------|---------|---------------|
| [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) | city96 | Load GGUF-quantized models | Qwen Image Edit |
| [ComfyUI-Florence2](https://github.com/kijai/ComfyUI-Florence2) | kijai | Florence2 image captioning | Dataset creation |
| [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) | kijai | LoadImagesFromFolder, JoinStrings, ImageResize, StringConstant | Dataset creation |
| [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) | Fannovel16 | ControlNet preprocessors (DWPose, depth) | Qwen Image Edit |
| [rgthree-comfy](https://github.com/rgthree/rgthree-comfy) | rgthree | SetNode/GetNode for clean wiring | Qwen Image Edit |
| [ComfyUI_ACE-Step](https://github.com/billwuhao/ComfyUI_ACE-Step) | billwuhao | ACE-Step music generation | Music generation |
| [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager) | ltdrdata | Package manager UI | General |
| comfyui-videohelpersuite | (pre-installed) | Video processing | Came with Docker image |

## Model Downloads

All models were downloaded manually via `kubectl exec` + `wget` into the pod. Here are the exact commands.

### Qwen Image Edit Models

```bash
# GGUF model (14 GB) → /workspace/models/unet/gguf/
kubectl exec -n comfyui $POD -- bash -c '
  mkdir -p /workspace/models/unet/gguf &&
  wget -q -c -O /workspace/models/unet/gguf/Qwen-Image-Edit-2509-Q5_0.gguf \
    "https://huggingface.co/QuantStack/Qwen-Image-Edit-2509-GGUF/resolve/main/Qwen-Image-Edit-2509-Q5_0.gguf"
'

# CLIP encoder (8.7 GB) → /workspace/models/clip/qwen/
kubectl exec -n comfyui $POD -- bash -c '
  mkdir -p /workspace/models/clip/qwen &&
  wget -q -c -O /workspace/models/clip/qwen/qwen_2.5_vl_7b_fp8_scaled.safetensors \
    "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"
'

# VAE (243 MB) → /workspace/models/vae/qwen/
kubectl exec -n comfyui $POD -- bash -c '
  mkdir -p /workspace/models/vae/qwen &&
  wget -q -c -O /workspace/models/vae/qwen/qwen_image_vae.safetensors \
    "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors"
'

# Lightning LoRA for 4-step generation (811 MB) → /workspace/models/loras/qwen/
kubectl exec -n comfyui $POD -- bash -c '
  mkdir -p /workspace/models/loras/qwen &&
  wget -q -c -O /workspace/models/loras/qwen/Qwen-Image-Lightning-4steps-V2.0.safetensors \
    "https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors"
'
```

### ACE-Step Models

```bash
# ACE-Step 1.5 Turbo AIO checkpoint (9.4 GB) → /workspace/models/checkpoints/
kubectl exec -n comfyui $POD -- bash -c '
  mkdir -p /workspace/models/checkpoints &&
  wget -q -c -O /workspace/models/checkpoints/ace_step_1.5_turbo_aio.safetensors \
    "https://huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files/resolve/main/checkpoints/ace_step_1.5_turbo_aio.safetensors"
'

# ACE-Step v1 checkpoint (7.2 GB) → /workspace/models/checkpoints/
kubectl exec -n comfyui $POD -- bash -c '
  wget -q -c -O /workspace/models/checkpoints/ace_step_v1_3.5b.safetensors \
    "https://huggingface.co/Comfy-Org/ACE-Step_ComfyUI_repackaged/resolve/main/all_in_one/ace_step_v1_3.5b.safetensors"
'
```

### Upscale Model

```bash
# 4x-UltraSharp (67 MB) → /workspace/models/upscale_models/
kubectl exec -n comfyui $POD -- bash -c '
  mkdir -p /workspace/models/upscale_models &&
  wget -q -c -O /workspace/models/upscale_models/4x-UltraSharp.pth \
    "https://huggingface.co/madriss/chkpts/resolve/main/ComfyUI/models/upscale_models/4x-UltraSharp.pth"
'
```

### Model Directory Layout

```
/workspace/models/
├── checkpoints/
│   ├── ace_step_1.5_turbo_aio.safetensors    # 9.4 GB — ACE-Step 1.5 Turbo
│   └── ace_step_v1_3.5b.safetensors          # 7.2 GB — ACE-Step v1
├── clip/qwen/
│   └── qwen_2.5_vl_7b_fp8_scaled.safetensors # 8.7 GB — Qwen CLIP encoder
├── loras/qwen/
│   └── Qwen-Image-Lightning-4steps-V2.0.safetensors  # 811 MB — Lightning LoRA
├── unet/gguf/
│   └── Qwen-Image-Edit-2509-Q5_0.gguf        # 14 GB — Qwen Image Edit GGUF
├── upscale_models/
│   └── 4x-UltraSharp.pth                     # 67 MB — 4x upscaler
└── vae/qwen/
    └── qwen_image_vae.safetensors            # 243 MB — Qwen VAE
```

Total model storage: ~40 GB

## Permission Gotcha

The `mmartial/comfyui-nvidia-docker` image runs as a non-root user (`comfytoo`). When git-cloning custom nodes or downloading models into hostPath volumes, you may hit permission errors. Two fixes:

1. **Set `FORCE_CHOWN=yes`** in the deployment env (already in our manifest) — fixes ownership on pod startup
2. **Manual fix** inside the pod if needed:
   ```bash
   kubectl exec -n comfyui $POD -- sudo chown -R comfytoo:comfytoo /workspace/custom_nodes
   kubectl exec -n comfyui $POD -- sudo chown -R comfytoo:comfytoo /comfy/mnt/venv
   ```

## GPU Assignment

Unlike AI-Toolkit, ComfyUI does **not** use UUID pinning. It requests `nvidia.com/gpu: "1"` and gets whichever GPU the Kubernetes scheduler assigns. This is fine for inference workloads that fit in 8-10 GB VRAM.

## Storage

Uses hostPath volumes:
- `/data/comfyui` → `/workspace` (models, outputs, custom nodes)
- `/data/comfyui/run` → `/comfy/mnt` (runtime data, venv)
- `dshm` emptyDir → `/dev/shm` (8 GiB shared memory)
