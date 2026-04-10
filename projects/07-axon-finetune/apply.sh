#!/bin/bash
# Source .env and apply k8s manifest with envsubst
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.env"
export HF_TOKEN GPU_UUID
envsubst '${HF_TOKEN} ${GPU_UUID}' < "$1" | kubectl apply -f -
