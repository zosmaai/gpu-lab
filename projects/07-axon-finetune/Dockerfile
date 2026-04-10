FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv git \
    && rm -rf /var/lib/apt/lists/*

# Create venv to avoid externally-managed-environment error
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch (stable, CUDA 12.8)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install ML dependencies
RUN pip install --no-cache-dir \
    transformers>=4.49.0 \
    peft>=0.15.0 \
    trl>=0.15.0 \
    datasets \
    accelerate \
    huggingface_hub \
    wandb

# Install ONNX export dependencies
RUN pip install --no-cache-dir \
    optimum[onnxruntime] \
    onnxruntime

WORKDIR /opt/axon
