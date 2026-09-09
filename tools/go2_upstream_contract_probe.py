#!/usr/bin/env python3
"""Run the audited actor inside ForestNavigation's original Go2 environment."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


def parse_args():
    root = Path(os.environ.get("MNS_PROJECT_ROOT", "/mns"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / "models/ForestNavigation")
    parser.add_argument("--checkpoint", type=Path, default=root / "models/go2_locomotion.pt")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--profile", choices=("training", "navigation"), default="training")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


ARGS = parse_args()
APP = AppLauncher(ARGS).app


def policy_tensor(observation):
    if isinstance(observation, tuple):
        observation = observation[0]
    if hasattr(observation, "keys"):
        if "policy" in observation.keys():
            return observation["policy"]
        return next(iter(observation.values()))
    return observation


def main() -> int:
    import gymnasium as gym
    import numpy as np

    source_root = ARGS.source / "source/ForestNavigation"
    local_simulation = Path(os.environ.get("MNS_PROJECT_ROOT", "/mns")) / "ros_ws/src/mns_simulation"
    sys.path[:0] = [str(source_root), str(local_simulation)]
    import ForestNavigation.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from mns_simulation.go2_policy import load_legacy_actor

    if ARGS.steps < 52:
        raise ValueError("steps must be at least 52")
    config = parse_env_cfg(
        "Go2-Agile-Curriculum-Play-v0", device=ARGS.device, num_envs=1, use_fabric=True
    )
    config.scene.camera = None
    config.seed = 0
    environment = gym.make("Go2-Agile-Curriculum-Play-v0", cfg=config)
    actor = load_legacy_actor(str(ARGS.checkpoint), ARGS.device)
    observation, _ = environment.reset()
    heights, tilts, commands, positions, yaws, speeds = [], [], [], [], [], []
    terminations = 0
    for step in range(ARGS.steps):
        current_observation = policy_tensor(observation)
        if ARGS.profile == "navigation":
            elapsed = step * float(environment.unwrapped.step_dt)
            if elapsed < 3.0:
                command = (0.0, 0.0, 0.0)
            elif elapsed < 7.0:
                command = (0.6, 0.0, 0.0)
            elif elapsed < 9.0:
                command = (0.6, 0.0, 0.5)
            elif elapsed < 11.0:
                command = (0.0, 0.0, 0.0)
            elif elapsed < 12.0:
                command = (0.6, 0.0, 0.0)
            else:
                command = (0.0, 0.0, 0.0)
            current_observation[:, 9:12] = current_observation.new_tensor([command])
        commands.append([float(value) for value in current_observation[0, 9:12]])
        actions = actor(current_observation)
        observation, _, terminated, _, _ = environment.step(actions)
        terminations += int(bool(terminated[0]))
        robot = environment.unwrapped.scene["robot"]
        height = float(robot.data.root_pos_w[0, 2])
        gravity_z = float(robot.data.projected_gravity_b[0, 2])
        tilt = math.acos(max(-1.0, min(1.0, -gravity_z)))
        heights.append(height)
        tilts.append(tilt)
        positions.append([float(value) for value in robot.data.root_pos_w[0, :2]])
        w, x, y, z = [float(value) for value in robot.data.root_quat_w[0]]
        yaws.append(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
        speeds.append(float(np.linalg.norm(robot.data.root_lin_vel_b[0, :2].cpu())))
    unwrapped_yaws = np.unwrap(yaws)
    navigation_metrics = {}
    if ARGS.profile == "navigation" and ARGS.steps >= 750:
        heading = yaws[149]
        forward_displacement = (
            (positions[349][0] - positions[149][0]) * math.cos(heading)
            + (positions[349][1] - positions[149][1]) * math.sin(heading)
        )
        navigation_metrics = {
            "forward_displacement_m": forward_displacement,
            "yaw_change_rad": abs(float(unwrapped_yaws[449] - unwrapped_yaws[349])),
            "final_planar_speed_mps": speeds[-1],
        }
    stable = terminations == 0 and min(heights[50:]) > 0.20 and max(tilts[50:]) < 0.70
    if ARGS.profile == "navigation":
        stable = stable and bool(navigation_metrics)
        stable = stable and navigation_metrics["forward_displacement_m"] > 0.30
        stable = stable and navigation_metrics["yaw_change_rad"] > 0.25
        stable = stable and navigation_metrics["final_planar_speed_mps"] < 0.30
    result = {
        "status": "PASS" if stable else "FAIL",
        "profile": ARGS.profile,
        "steps": ARGS.steps,
        "physics_dt": float(environment.unwrapped.physics_dt),
        "minimum_height_m": min(heights[50:]),
        "maximum_tilt_rad": max(tilts[50:]),
        "terminations": terminations,
        "initial_command": commands[0],
        "final_command": commands[-1],
        "final_position_m": [
            float(value) for value in environment.unwrapped.scene["robot"].data.root_pos_w[0]
        ],
        "finite": bool(np.isfinite(heights).all() and np.isfinite(tilts).all()),
        **navigation_metrics,
    }
    print("MNS_GO2_UPSTREAM_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "PASS" and result["finite"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        APP.close(wait_for_replicator=False, skip_cleanup=True)
