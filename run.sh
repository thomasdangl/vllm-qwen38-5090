#!/bin/bash

# Change these two.
VLLM_API_KEY="2e5cd1188f2f937901d87ad039eae0e84187a0e6e2629367"
MODEL="$HOME/Models/Minachist--Qwen3.8-27B-INT6-Mixed-AutoRound"

VLLM_IMAGE="docker.io/vllm/vllm-openai:nightly-ba07e4a48fc951300d97eb506217dd530583dea3"
PATCH_CONTAINER_ARGS=()
PATCH_SERVER_ARGS=()
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/patches/apply.sh" "$VLLM_IMAGE" || exit 1

sudo bash <<'NVIDIA_VRAM_SETUP' || exit 1
set -e
trap 'systemctl start nvidia-persistenced.service' EXIT
systemctl stop nvidia-persistenced.service
nvidia-smi -pm 0
modprobe -r nvidia_uvm
modprobe -r nvidia_modeset
modprobe -r nvidia
modprobe nvidia 'NVreg_RegistryDwords=RmGspFirmwareHeapSizeMB=0x58;RMOverrideMaxContextSizeRsvdMemoryMB=0x1'
modprobe -a nvidia_uvm nvidia_modeset
NVIDIA_VRAM_SETUP

if ! sudo podman pod inspect vllm &>/dev/null; then
  sudo podman pod create --name vllm

fi

# Consider setting a reasonable power target:
# sudo nvidia-smi -pm 1
# sudo nvidia-smi -i 0 -pl 480

# You might need to adjust --gpu-memory-utilization if it doesn't fit.
sudo podman run -d \
  --replace \
  --name vllm-patched \
  --pod vllm \
  --restart=unless-stopped \
  --security-opt=label=disable \
  --device=nvidia.com/gpu=all \
  --shm-size=32g \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e CUDA_LAUNCH_BLOCKING=0 \
  -e VLLM_BYPASS_STARTUP_MEMORY_CHECK=1 \
  -v "$MODEL:/model:ro" \
  "${PATCH_CONTAINER_ARGS[@]}" \
  "$VLLM_IMAGE" \
  --model /model \
  --served-model-name "Qwen3.8 27B" \
  --host 0.0.0.0 \
  --port 9001 \
  --api-key "$VLLM_API_KEY" \
  --max-model-len "247808" \
  --max-num-seqs 2 \
  --max-num-batched-tokens 1600 \
  "${PATCH_SERVER_ARGS[@]}" \
  --gpu-memory-utilization 0.986 \
  --limit-mm-per-prompt '{"image":64,"video":0}' \
  --mm-processor-kwargs '{"max_pixels":4194304}' \
  --compilation-config '{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3,4]}' \
  --performance-mode interactivity \
  --attention-backend flashinfer \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --tool-call-parser qwen3_coder \
  --default-chat-template-kwargs '{"preserve_thinking": true}' \
  --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"repetition_penalty":1.0}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
