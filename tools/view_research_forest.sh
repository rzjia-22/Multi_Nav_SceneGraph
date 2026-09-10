#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${DISPLAY:-}" ]]; then
  echo "DISPLAY is not set; run this target from the NVIDIA host desktop session." >&2
  exit 2
fi

if [[ -z "${XAUTHORITY:-}" || ! -r "${XAUTHORITY}" ]]; then
  echo "XAUTHORITY must point to a readable X11 cookie file." >&2
  exit 2
fi

docker compose --profile simulation run --rm \
  --entrypoint /mns/containers/simulation/dataset_entrypoint.sh simulation \
  --review --mode interactive \
  --scene /mns/research_scenes/dataset_v0/train_scene_000/scene.yaml \
  --assets /mns/config/research_forests/assets.yaml \
  --output-directory /mns/artifacts/dataset_v0_scene_review/train_scene_000 \
  --enable_cameras
