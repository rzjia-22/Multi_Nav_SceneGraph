#!/usr/bin/env bash
set -euo pipefail

export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export ROS_DISTRO=jazzy
ros_bridge_root="/isaac-sim/exts/isaacsim.ros2.bridge/jazzy"
export LD_LIBRARY_PATH="${ros_bridge_root}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${ros_bridge_root}/rclpy:/workspace/ros_ws/src/mns_simulation:${PYTHONPATH:-}"

exec /workspace/isaaclab/isaaclab.sh -p \
  /workspace/ros_ws/src/mns_simulation/mns_simulation/isaac_runtime.py "$@"
