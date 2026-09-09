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

exec /workspace/isaaclab/isaaclab.sh -p \
  "${runtime_script}" "$@"
