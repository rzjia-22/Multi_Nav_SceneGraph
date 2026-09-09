#!/usr/bin/env bash
set -euo pipefail

export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export ROS_DISTRO=jazzy
export MNS_PROJECT_ROOT="${MNS_PROJECT_ROOT:-/mns}"
ros_bridge_root="/isaac-sim/exts/isaacsim.ros2.bridge/jazzy"
export LD_LIBRARY_PATH="${ros_bridge_root}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${ros_bridge_root}/rclpy:${MNS_PROJECT_ROOT}/ros_ws/src/mns_simulation:${PYTHONPATH:-}"

runtime_script="${MNS_PROJECT_ROOT}/ros_ws/src/mns_simulation/mns_simulation/isaac_runtime.py"
if [[ "${1:-}" == "--minimal" ]]; then
  runtime_script="${MNS_PROJECT_ROOT}/ros_ws/src/mns_simulation/mns_simulation/isaac_minimal.py"
  shift
fi

# Both isaaclab.sh and Isaac Sim's python.sh launch Python as a child process
# instead of replacing the shell.  That prevents Docker's stop signal from
# reaching the runtime and turns an otherwise orderly shutdown into exit 137.
# Reproduce python.sh's environment setup here, then make Python PID 1.
isaac_sim_root="/workspace/isaaclab/_isaac_sim"
export CARB_APP_PATH="${isaac_sim_root}/kit"
export ISAAC_PATH="${isaac_sim_root}"
export EXP_PATH="${isaac_sim_root}/apps"
# shellcheck source=/dev/null
source "${isaac_sim_root}/setup_python_env.sh"
export RESOURCE_NAME="IsaacSim"
export LD_PRELOAD="${isaac_sim_root}/kit/libcarb.so"

exec "${isaac_sim_root}/kit/python/bin/python3" "${runtime_script}" "$@"
