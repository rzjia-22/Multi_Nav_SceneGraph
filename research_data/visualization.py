"""Stable, review-oriented Dataset V0 preview figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import file_hash, load_yaml
from .depth import align_depth_to_rgb


def _draw_forest(axis, scene):
    size = scene["extent_m"][0]
    axis.set_xlim(-size / 2.0, size / 2.0)
    axis.set_ylim(-size / 2.0, size / 2.0)
    axis.set_aspect("equal")
    for tree in scene["trees"]:
        axis.add_patch(plt.Circle(tree["position_m"], tree["collision_proxy"]["radius_m"], color="#633b1f", alpha=0.95))
    axis.grid(alpha=0.15)
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")


def visualize(scene_path: Path, plan_path: Path, episode_path: Path, output_dir: Path) -> list[Path]:
    import h5py

    scene = load_yaml(scene_path)
    plan = load_yaml(plan_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    created = []
    figure, axis = plt.subplots(figsize=(8, 8), constrained_layout=True)
    _draw_forest(axis, scene)
    start, goal = plan["start_pose_xyzyaw"], plan["goal_pose_xyz"]
    axis.scatter(start[0], start[1], marker="o", s=90, color="#2878b5", label="start")
    axis.scatter(goal[0], goal[1], marker="*", s=160, color="#d62728", label="goal")
    axis.plot([start[0], goal[0]], [start[1], goal[1]], ":", color="gray", label="blocked direct line")
    axis.set_title(f"{scene['scene_id']} — {scene['factors']['tree_density']} forest / {scene['factors']['terrain']} terrain")
    axis.legend(loc="upper right")
    scene_output = output_dir / "episode_layout.png"
    figure.savefig(scene_output, dpi=150)
    plt.close(figure)
    created.append(scene_output)

    with h5py.File(episode_path, "r") as episode:
        metadata = json.loads(str(episode.attrs["metadata_json"]))
        state_time = np.asarray(episode["state/timestamp_s"])
        executed = np.asarray(episode["state/position_xyz"])
        orientations = np.asarray(episode["state/orientation_xyzw"])
        velocities = np.asarray(episode["state/linear_velocity_xyz"])
        angular_velocities = np.asarray(episode["state/angular_velocity_xyz"])
        commands = np.asarray(episode["state/command_vw"])
        planned = np.asarray(episode["expert/global_path_xyz"])
        rgb = np.asarray(episode["sensors/rgb"])
        depth = align_depth_to_rgb(
            np.asarray(episode["sensors/depth_raw_z16"]),
            float(np.asarray(episode["calibration/depth_scale_m"])),
            np.asarray(episode["calibration/depth_intrinsics"]),
            np.asarray(episode["calibration/rgb_intrinsics"]),
            np.asarray(episode["calibration/depth_to_rgb_translation_m"]),
            np.asarray(episode["calibration/depth_to_rgb_rotation_xyzw"]),
            tuple(np.asarray(episode["calibration/rgb_resolution_wh"], dtype=int)),
        )
    figure, axis = plt.subplots(figsize=(8, 8), constrained_layout=True)
    _draw_forest(axis, scene)
    axis.plot(planned[:, 0], planned[:, 1], "--", linewidth=2.2, color="#ff7f0e", label="planned A* + LOS")
    axis.plot(executed[:, 0], executed[:, 1], "-", linewidth=1.6, color="#1f77b4", label="executed Isaac trajectory")
    axis.scatter(planned[0, 0], planned[0, 1], marker="o", s=90, color="#2878b5")
    axis.scatter(planned[-1, 0], planned[-1, 1], marker="*", s=160, color="#d62728")
    axis.set_title(f"{plan['episode_id']} — planned vs executed")
    axis.legend(loc="upper right")
    trajectory_output = output_dir / "trajectory_overview.png"
    figure.savefig(trajectory_output, dpi=150)
    plt.close(figure)
    created.append(trajectory_output)

    indices = np.unique(np.linspace(0, len(rgb) - 1, 4).round().astype(int))
    figure, axes = plt.subplots(2, len(indices), figsize=(4 * len(indices), 6), constrained_layout=True)
    for column, index in enumerate(indices):
        axes[0, column].imshow(rgb[index])
        axes[0, column].set_title(f"RGB frame {index}")
        axes[0, column].axis("off")
        valid = depth[index][depth[index] > 0]
        maximum = min(10.0, float(np.percentile(valid, 98))) if len(valid) else 10.0
        axes[1, column].imshow(depth[index], cmap="turbo", vmin=0.2, vmax=maximum)
        axes[1, column].set_title("aligned depth [m]")
        axes[1, column].axis("off")
    samples_output = output_dir / "camera_samples.png"
    figure.savefig(samples_output, dpi=140)
    plt.close(figure)
    created.append(samples_output)
    csv_output = output_dir / "executed_trajectory.csv"
    trajectory = np.column_stack((state_time, executed, orientations, velocities, angular_velocities, commands))
    np.savetxt(
        csv_output, trajectory, delimiter=",",
        header="timestamp_s,x_m,y_m,z_m,qx,qy,qz,qw,vx_mps,vy_mps,vz_mps,wx_radps,wy_radps,wz_radps,cmd_v_mps,cmd_w_radps",
        comments="", fmt="%.8f",
    )
    created.append(csv_output)
    summary_output = output_dir / "preview_summary.json"
    summary = {
        "episode_id": metadata["episode_id"], "scene_id": metadata["scene_id"],
        "success": metadata["success"], "planned_path_length_m": metadata["planned_path_length_m"],
        "executed_path_length_m": metadata["executed_path_length_m"],
        "duration_s": float(state_time[-1] - state_time[0]), "rgb_frames": int(len(rgb)),
        "depth_frames": int(len(depth)), "pose_samples": int(len(state_time)),
        "hdf5_size_bytes": episode_path.stat().st_size, "hdf5_sha256": file_hash(episode_path),
        "rgb_value_range": [int(rgb.min()), int(rgb.max())],
        "aligned_depth_valid_fraction": float((depth > 0.0).mean()),
    }
    summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    created.append(summary_output)
    return created
