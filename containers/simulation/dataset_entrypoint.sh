#!/usr/bin/env bash
set -euo pipefail

export MNS_PROJECT_ROOT="${MNS_PROJECT_ROOT:-/mns}"
isaac_sim_root="/workspace/isaaclab/_isaac_sim"
export PYTHONPATH="${PYTHONPATH:-}"
export CARB_APP_PATH="${isaac_sim_root}/kit"
export ISAAC_PATH="${isaac_sim_root}"
export EXP_PATH="${isaac_sim_root}/apps"
# shellcheck source=/dev/null
source "${isaac_sim_root}/setup_python_env.sh"
export RESOURCE_NAME="IsaacSim"
export LD_PRELOAD="${isaac_sim_root}/kit/libcarb.so"
export PYTHONPATH="${MNS_PROJECT_ROOT}:${MNS_PROJECT_ROOT}/ros_ws/src/mns_simulation:${PYTHONPATH:-}"

if [[ "${1:-}" == "--tool" ]]; then
  shift
  exec "${isaac_sim_root}/kit/python/bin/python3" -m research_data.cli "$@"
fi

exec "${isaac_sim_root}/kit/python/bin/python3" \
  "${MNS_PROJECT_ROOT}/ros_ws/src/mns_simulation/mns_simulation/research_dataset_runtime.py" "$@"
