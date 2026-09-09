#!/usr/bin/env bash
set -euo pipefail

cuda_probe_image="${MNS_CUDA_PROBE_IMAGE:-nvcr.io/nvidia/cuda:12.3.2-base-ubuntu22.04}"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "FAIL: required command '$1' is missing" >&2
    exit 1
  fi
}

echo "LEVEL_0_HOST_GPU"
require_command nvidia-smi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "LEVEL_1_DOCKER_GPU"
require_command docker
docker info --format 'runtimes={{json .Runtimes}} default={{.DefaultRuntime}}' | grep -q 'nvidia'
docker run --rm --gpus all "${cuda_probe_image}" nvidia-smi \
  --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "LEVEL_2_COMPOSE_GRAPHICS_CONFIG"
docker compose --profile simulation config --quiet
capabilities="$({
  docker image inspect multi-nav-scenegraph/simulation:isaac-lab-2.3.1 \
    --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null || true
} | sed -n 's/^NVIDIA_DRIVER_CAPABILITIES=//p' | tail -n 1)"
if [[ -n "${capabilities}" && "${capabilities}" != "all" && "${capabilities}" != *graphics* ]]; then
  echo "FAIL: simulation image lacks NVIDIA graphics capability: ${capabilities}" >&2
  exit 1
fi
echo "graphics_capabilities=${capabilities:-declared-by-compose}"
echo "GPU_PREFLIGHT_PASS"
