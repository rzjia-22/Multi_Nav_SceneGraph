"""在 Dataset V0 全部 70 条专家日志上运行官方 NoMaD zero-shot 离线评测。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any

import h5py
import numpy as np

from ..common import ROOT, file_hash, load_yaml
from .metrics import (
    densify_polyline,
    local_to_world,
    maximum_true_streak,
    pairwise_endpoint_diversity,
    point_to_polyline_distances,
    trajectory_clearance,
    world_to_local,
)
from .model_adapter import OfficialNoMaDAdapter
from .reporting import (
    atomic_write_json,
    atomic_write_text,
    build_report,
    episode_csv,
    render_markdown,
)


DEFAULT_CONFIG = ROOT / "config/navigation/nomad_zero_shot_offline.yaml"
MODE_NAMES = ("local_visual_goal", "final_visual_goal", "goal_masked_exploration")


def yaw_from_xyzw(quaternion: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternion, dtype=np.float64)
    x, y, z, w = np.moveaxis(values, -1, 0)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def nearest_timestamp_index(timestamps: np.ndarray, target: float) -> int:
    """返回时间戳最接近 target 的索引。"""

    values = np.asarray(timestamps, dtype=np.float64)
    insertion = int(np.searchsorted(values, target))
    if insertion <= 0:
        return 0
    if insertion >= len(values):
        return len(values) - 1
    before = insertion - 1
    return before if target - values[before] <= values[insertion] - target else insertion


def interpolate_xy_yaw(
    timestamps: np.ndarray,
    positions_xyz: np.ndarray,
    orientations_xyzw: np.ndarray,
    query_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """在状态时间轴上插值得到二维位置和连续 yaw。"""

    source_times = np.asarray(timestamps, dtype=np.float64)
    positions = np.asarray(positions_xyz, dtype=np.float64)
    query = np.asarray(query_times, dtype=np.float64)
    xy = np.stack([
        np.interp(query, source_times, positions[:, axis]) for axis in range(2)
    ], axis=1)
    unwrapped = np.unwrap(yaw_from_xyzw(orientations_xyzw))
    yaw = np.interp(query, source_times, unwrapped)
    return xy, yaw


def executed_polyline(
    state_times: np.ndarray,
    positions_xyz: np.ndarray,
    start_time: float,
    end_time: float,
) -> np.ndarray:
    positions = np.asarray(positions_xyz, dtype=np.float64)
    source_times = np.asarray(state_times, dtype=np.float64)
    interior = positions[
        (state_times > start_time) & (state_times < end_time), :2
    ]
    endpoint_times = np.asarray([start_time, end_time], dtype=np.float64)
    endpoints = np.stack([
        np.interp(endpoint_times, source_times, positions[:, axis]) for axis in range(2)
    ], axis=1)
    return np.concatenate((endpoints[:1], interior, endpoints[1:]), axis=0)


def sample_metrics(
    trajectories_local_m: np.ndarray,
    ground_truth_local_m: np.ndarray,
    ground_truth_polyline_world: np.ndarray,
    origin_world_xy: np.ndarray,
    yaw: float,
    goal_local_m: np.ndarray,
    scene: dict,
    metrics_config: dict,
) -> list[dict[str, Any]]:
    centres = np.asarray([tree["position_m"] for tree in scene["trees"]], dtype=np.float64)
    radii = np.asarray(
        [tree["collision_proxy"]["radius_m"] for tree in scene["trees"]], dtype=np.float64
    )
    robot_radius = float(metrics_config["robot_collision_radius_m"])
    corridor = float(metrics_config["expert_corridor_threshold_m"])
    minimum_progress = float(metrics_config["minimum_goal_progress_m"])
    maximum_direction_error = float(metrics_config["maximum_direction_error_deg"])
    goal_distance = float(np.linalg.norm(goal_local_m))
    results = []
    for trajectory in np.asarray(trajectories_local_m, dtype=np.float64):
        predicted_world = local_to_world(trajectory, origin_world_xy, yaw)
        polyline_world = np.concatenate((origin_world_xy[None], predicted_world), axis=0)
        clearance = trajectory_clearance(
            polyline_world,
            centres,
            radii,
            robot_radius,
            tuple(float(value) for value in scene["extent_m"]),
        )
        errors = np.linalg.norm(trajectory - ground_truth_local_m, axis=1)
        dense_prediction = densify_polyline(
            polyline_world,
            float(metrics_config["expert_corridor_sampling_spacing_m"]),
        )
        cross_track = point_to_polyline_distances(
            dense_prediction, ground_truth_polyline_world
        )
        endpoint = trajectory[-1]
        endpoint_norm = float(np.linalg.norm(endpoint))
        if endpoint_norm <= 1.0e-9 or goal_distance <= 1.0e-9:
            direction_cosine = -1.0
            direction_error_deg = 180.0
        else:
            direction_cosine = float(
                np.clip(np.dot(endpoint, goal_local_m) / (endpoint_norm * goal_distance), -1.0, 1.0)
            )
            direction_error_deg = math.degrees(math.acos(direction_cosine))
        goal_progress = goal_distance - float(np.linalg.norm(goal_local_m - endpoint))
        maximum_cross_track = float(cross_track.max())
        accepted = (
            clearance > 0.0
            and goal_progress >= minimum_progress
            and direction_error_deg <= maximum_direction_error
            and maximum_cross_track <= corridor
        )
        results.append({
            "ADE_m": float(errors.mean()),
            "FDE_m": float(errors[-1]),
            "minimum_clearance_m": clearance,
            "conservative_safe": clearance > float(
                metrics_config["conservative_clearance_margin_m"]
            ),
            "collision_free": clearance > 0.0,
            "mean_expert_cross_track_m": float(cross_track.mean()),
            "maximum_expert_cross_track_m": maximum_cross_track,
            "corridor_adherent": maximum_cross_track <= corridor,
            "goal_progress_m": goal_progress,
            "goal_progress": goal_progress >= minimum_progress,
            "direction_cosine": direction_cosine,
            "direction_error_deg": direction_error_deg,
            "direction_acceptable": direction_error_deg <= maximum_direction_error,
            "accepted_action": accepted,
            "final_displacement_m": endpoint_norm,
            "maximum_waypoint_jump_m": float(
                np.linalg.norm(np.diff(np.concatenate((np.zeros((1, 2)), trajectory), axis=0), axis=0), axis=1).max()
            ),
        })
    return results


def mode_anchor_result(
    trajectories: np.ndarray,
    metrics: list[dict[str, Any]],
    distance_prediction: float | None,
    distance_target: float | None,
    inference_ms: float,
) -> dict[str, Any]:
    first = metrics[0]
    return {
        "sample_0": first,
        "sample_count": len(metrics),
        "sample_safe_probability": float(np.mean([item["collision_free"] for item in metrics])),
        "sample_accepted_probability": float(np.mean([item["accepted_action"] for item in metrics])),
        "any_sample_collision_free": any(item["collision_free"] for item in metrics),
        "any_sample_accepted": any(item["accepted_action"] for item in metrics),
        "best_of_n_ADE_m": min(float(item["ADE_m"]) for item in metrics),
        "best_of_n_FDE_m": min(float(item["FDE_m"]) for item in metrics),
        "endpoint_diversity_m": pairwise_endpoint_diversity(trajectories),
        "distance_prediction_steps": distance_prediction,
        "distance_target_steps": distance_target,
        "distance_absolute_error_steps": (
            abs(float(distance_prediction) - float(distance_target))
            if distance_prediction is not None and distance_target is not None else None
        ),
        "inference_ms": float(inference_ms),
        "sample_0_local_trajectory_m": trajectories[0].tolist(),
    }


def summarize_mode(anchors: list[dict], mode: str) -> dict[str, Any]:
    values = [item["modes"][mode] for item in anchors]
    sample_zero = [item["sample_0"] for item in values]

    def average(records: list[dict], key: str) -> float:
        return float(np.mean([float(item[key]) for item in records]))

    distance_errors = [
        float(item["distance_absolute_error_steps"])
        for item in values if item["distance_absolute_error_steps"] is not None
    ]
    return {
        "anchor_count": len(values),
        "collision_free_fraction": average(sample_zero, "collision_free"),
        "conservative_safe_fraction": average(sample_zero, "conservative_safe"),
        "goal_progress_fraction": average(sample_zero, "goal_progress"),
        "corridor_adherence_fraction": average(sample_zero, "corridor_adherent"),
        "accepted_action_fraction": average(sample_zero, "accepted_action"),
        "sample_safe_probability": float(np.mean([
            item["sample_safe_probability"] for item in values
        ])),
        "any_sample_accepted_fraction": float(np.mean([
            item["any_sample_accepted"] for item in values
        ])),
        "ADE_m": average(sample_zero, "ADE_m"),
        "FDE_m": average(sample_zero, "FDE_m"),
        "best_of_n_ADE_m": float(np.mean([item["best_of_n_ADE_m"] for item in values])),
        "best_of_n_FDE_m": float(np.mean([item["best_of_n_FDE_m"] for item in values])),
        "minimum_clearance_m": min(float(item["minimum_clearance_m"]) for item in sample_zero),
        "endpoint_diversity_m": float(np.mean([item["endpoint_diversity_m"] for item in values])),
        "distance_MAE_steps": float(np.mean(distance_errors)) if distance_errors else None,
        "inference_ms_mean": float(np.mean([item["inference_ms"] for item in values])),
        "inference_ms_p95": float(np.percentile(
            [item["inference_ms"] for item in values], 95
        )),
        "maximum_consecutive_unsafe": maximum_true_streak([
            not bool(item["collision_free"]) for item in sample_zero
        ]),
    }


def route_viability(summary: dict, thresholds: dict) -> dict[str, bool]:
    return {
        level: (
            summary["collision_free_fraction"] >= float(values["collision_free_fraction"])
            and summary["goal_progress_fraction"] >= float(values["goal_progress_fraction"])
            and summary["corridor_adherence_fraction"] >= float(
                values["corridor_adherence_fraction"]
            )
            and summary["accepted_action_fraction"] >= float(
                values["accepted_action_fraction"]
            )
            and summary["maximum_consecutive_unsafe"] <= int(
                values["maximum_consecutive_unsafe"]
            )
        )
        for level, values in thresholds.items()
    }


def episode_anchors(
    sensor_timestamps: np.ndarray,
    config: dict,
    maximum_anchors: int | None,
) -> np.ndarray:
    adapter = config["input_adapter"]
    context_duration = (int(adapter["context_frames"]) - 1) / float(adapter["model_rate_hz"])
    first = float(sensor_timestamps[0]) + context_duration
    last = float(sensor_timestamps[-1]) - float(adapter["minimum_remaining_horizon_s"])
    if last < first:
        return np.empty(0, dtype=np.float64)
    anchors = np.arange(
        first, last + 1.0e-9, float(adapter["evaluation_stride_s"]), dtype=np.float64
    )
    if maximum_anchors is not None:
        anchors = anchors[:maximum_anchors]
    return anchors


def evaluate_episode(
    entry: dict,
    model: OfficialNoMaDAdapter,
    config: dict,
    *,
    episode_ordinal: int,
    raw_stream,
    maximum_anchors: int | None,
) -> dict[str, Any]:
    episode_path = ROOT / entry["relative_path"]
    scene = load_yaml(ROOT / config["dataset"]["scene_root"] / entry["scene_id"] / "scene.yaml")
    robot = load_yaml(ROOT / config["dataset"]["robot_profile"])
    model_rate = float(config["input_adapter"]["model_rate_hz"])
    action_scale_m = float(robot["motion"]["max_forward_speed_mps"]) / model_rate
    context_frames = int(config["input_adapter"]["context_frames"])
    context_offsets = np.arange(-(context_frames - 1), 1, dtype=np.float64) / model_rate
    horizon_times = np.arange(1, model.trajectory_points + 1, dtype=np.float64) / model_rate
    samples = int(config["nomad"]["samples_per_observation"])
    base_seed = int(config["nomad"]["seed"])
    anchors: list[dict[str, Any]] = []

    with h5py.File(episode_path, "r") as source:
        sensor_timestamps = source["sensors/timestamp_s"][:]
        state_timestamps = source["state/timestamp_s"][:]
        state_positions = source["state/position_xyz"][:]
        state_orientations = source["state/orientation_xyzw"][:]
        rgb_source = source["sensors/rgb"]
        anchor_times = episode_anchors(sensor_timestamps, config, maximum_anchors)
        for anchor_ordinal, anchor_time in enumerate(anchor_times):
            context_indices = [
                nearest_timestamp_index(sensor_timestamps, float(anchor_time + offset))
                for offset in context_offsets
            ]
            if len(set(context_indices)) != context_frames:
                raise ValueError(
                    f"{entry['episode_id']}: resampling produced duplicate context frames"
                )
            local_goal_time = anchor_time + float(
                config["input_adapter"]["local_goal_horizon_s"]
            )
            local_goal_index = nearest_timestamp_index(sensor_timestamps, local_goal_time)
            final_goal_index = len(sensor_timestamps) - 1
            context_rgb = rgb_source[context_indices]
            local_goal_rgb = rgb_source[local_goal_index]
            final_goal_rgb = rgb_source[final_goal_index]

            query_times = np.concatenate(([anchor_time], anchor_time + horizon_times))
            state_xy, state_yaw = interpolate_xy_yaw(
                state_timestamps, state_positions, state_orientations, query_times
            )
            origin_xy = state_xy[0]
            origin_yaw = float(state_yaw[0])
            ground_truth_local = world_to_local(state_xy[1:], origin_xy, origin_yaw)
            ground_truth_polyline = executed_polyline(
                state_timestamps,
                state_positions,
                float(anchor_time),
                float(anchor_time + horizon_times[-1]),
            )
            local_goal_xy, _ = interpolate_xy_yaw(
                state_timestamps,
                state_positions,
                state_orientations,
                np.asarray([float(sensor_timestamps[local_goal_index])]),
            )
            final_goal_xy, _ = interpolate_xy_yaw(
                state_timestamps,
                state_positions,
                state_orientations,
                np.asarray([float(sensor_timestamps[final_goal_index])]),
            )
            local_goal_vector = world_to_local(local_goal_xy, origin_xy, origin_yaw)[0]
            final_goal_vector = world_to_local(final_goal_xy, origin_xy, origin_yaw)[0]

            total_started = time.perf_counter()
            goal_conditions = model.encode(
                context_rgb, np.stack((local_goal_rgb, final_goal_rgb)), goal_masked=False
            )
            distance_predictions = model.predict_distance(goal_conditions)
            exploration_condition = model.encode(
                context_rgb, np.zeros_like(local_goal_rgb)[None], goal_masked=True
            )
            mode_inputs = {
                "local_visual_goal": (
                    goal_conditions[0:1],
                    local_goal_vector,
                    float(distance_predictions[0]),
                    min(
                        20.0,
                        max(0.0, float(sensor_timestamps[local_goal_index] - anchor_time) * model_rate),
                    ),
                ),
                "final_visual_goal": (
                    goal_conditions[1:2],
                    final_goal_vector,
                    float(distance_predictions[1]),
                    min(
                        20.0,
                        max(0.0, float(sensor_timestamps[final_goal_index] - anchor_time) * model_rate),
                    ),
                ),
                # 探索模式仍相对未来专家路径计算安全/走廊指标，但不把目标提供给模型。
                "goal_masked_exploration": (
                    exploration_condition,
                    local_goal_vector,
                    None,
                    None,
                ),
            }
            mode_results = {}
            for mode_ordinal, mode in enumerate(MODE_NAMES):
                condition, goal_vector, distance_prediction, distance_target = mode_inputs[mode]
                seed = base_seed + episode_ordinal * 100_000 + anchor_ordinal * 10 + mode_ordinal
                trajectories, diffusion_ms = model.sample_trajectories(
                    condition,
                    sample_count=samples,
                    seed=seed,
                    action_scale_m=action_scale_m,
                )
                metrics = sample_metrics(
                    trajectories,
                    ground_truth_local,
                    ground_truth_polyline,
                    origin_xy,
                    origin_yaw,
                    goal_vector,
                    scene,
                    config["metrics"],
                )
                mode_results[mode] = mode_anchor_result(
                    trajectories,
                    metrics,
                    distance_prediction,
                    distance_target,
                    diffusion_ms,
                )
            if model.device.type == "cuda":
                model.torch.cuda.synchronize(model.device)
            anchor_result = {
                "episode_id": entry["episode_id"],
                "anchor_ordinal": anchor_ordinal,
                "anchor_timestamp_s": float(anchor_time),
                "context_frame_indices": context_indices,
                "local_goal_frame_index": int(local_goal_index),
                "final_goal_frame_index": int(final_goal_index),
                "action_scale_m": action_scale_m,
                "full_anchor_wall_ms": (time.perf_counter() - total_started) * 1000.0,
                "modes": mode_results,
            }
            raw_stream.write(json.dumps(anchor_result, separators=(",", ":")) + "\n")
            raw_stream.flush()
            anchors.append(anchor_result)

    if not anchors:
        raise ValueError(f"{entry['episode_id']}: no valid evaluation anchors")
    modes = {mode: summarize_mode(anchors, mode) for mode in MODE_NAMES}
    primary = modes["local_visual_goal"]
    primary["full_anchor_wall_ms_mean"] = float(np.mean([
        item["full_anchor_wall_ms"] for item in anchors
    ]))
    primary["route_viability_proxy"] = route_viability(
        primary, config["metrics"]["route_viability_thresholds"]
    )
    return {
        "episode_id": entry["episode_id"],
        "split": entry["split"],
        "scene_id": entry["scene_id"],
        "route_bucket": entry["route_bucket"],
        "environment": {
            "terrain": entry["terrain_profile"],
            "ground": entry["ground"],
            "lighting": entry["lighting"],
            "tree_density": entry["tree_density"],
        },
        "source_hdf5_sha256": entry["hdf5_sha256"],
        "source_rgb_frames": entry["rgb_frames"],
        "anchor_count": len(anchors),
        **primary,
        "secondary_modes": {
            key: value for key, value in modes.items() if key != "local_visual_goal"
        },
    }


def validate_inputs(config: dict, maximum_episodes: int | None) -> list[dict]:
    if int(config.get("config_version", 0)) != 1:
        raise ValueError("unsupported NoMaD offline config version")
    index_path = ROOT / config["dataset"]["index"]
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("dataset_version") != config["dataset"]["expected_version"]:
        raise ValueError("Dataset V0 version mismatch")
    entries = list(index["episodes"])
    if len(entries) != int(config["dataset"]["expected_episode_count"]):
        raise ValueError("Dataset V0 must contain exactly 70 episodes")
    if any(item.get("validation_status") != "PASS" for item in entries):
        raise ValueError("every Dataset V0 episode must pass source validation")
    missing = [item["relative_path"] for item in entries if not (ROOT / item["relative_path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"missing Dataset V0 episode files: {missing[:3]}")
    split_order = {"train": 0, "validation": 1, "test": 2}
    entries.sort(key=lambda item: (
        split_order[item["split"]], item["scene_id"], item["episode_id"]
    ))
    return entries[:maximum_episodes] if maximum_episodes is not None else entries


def secondary_mode_summary(episodes: list[dict], mode: str) -> dict[str, Any]:
    values = [item["secondary_modes"][mode] for item in episodes]
    return {
        "episode_count": len(values),
        "mean_collision_free_fraction": float(np.mean([
            item["collision_free_fraction"] for item in values
        ])),
        "mean_goal_progress_fraction": float(np.mean([
            item["goal_progress_fraction"] for item in values
        ])),
        "mean_corridor_adherence_fraction": float(np.mean([
            item["corridor_adherence_fraction"] for item in values
        ])),
        "mean_accepted_action_fraction": float(np.mean([
            item["accepted_action_fraction"] for item in values
        ])),
        "mean_ADE_m": float(np.mean([item["ADE_m"] for item in values])),
        "mean_FDE_m": float(np.mean([item["FDE_m"] for item in values])),
    }


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(arguments.config).resolve()
    config = load_yaml(config_path)
    entries = validate_inputs(config, arguments.maximum_episodes)
    model = OfficialNoMaDAdapter(ROOT, config)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_directory = (
        Path(arguments.output_directory).resolve()
        if arguments.output_directory else ROOT / config["outputs"]["run_root"] / timestamp
    )
    run_directory.mkdir(parents=True, exist_ok=False)
    raw_path = run_directory / "window_metrics.jsonl"
    episodes = []
    started = time.perf_counter()
    with raw_path.open("w", encoding="utf-8") as raw_stream:
        for ordinal, entry in enumerate(entries):
            episode_started = time.perf_counter()
            result = evaluate_episode(
                entry,
                model,
                config,
                episode_ordinal=ordinal,
                raw_stream=raw_stream,
                maximum_anchors=arguments.maximum_anchors_per_episode,
            )
            episodes.append(result)
            print(json.dumps({
                "episode": result["episode_id"],
                "index": ordinal + 1,
                "total": len(entries),
                "anchors": result["anchor_count"],
                "nominal_proxy_success": result["route_viability_proxy"]["nominal"],
                "collision_free_fraction": result["collision_free_fraction"],
                "wall_s": time.perf_counter() - episode_started,
            }, sort_keys=True), flush=True)

    current_commit = subprocess.check_output(
        [
            "git", "-c", f"safe.directory={ROOT}",
            "-C", str(ROOT), "rev-parse", "HEAD",
        ],
        text=True,
    ).strip()
    metadata = {
        "status": "PASS" if len(episodes) == len(entries) else "INCOMPLETE",
        "model_readiness": "OFFLINE_ONLY",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_git_commit": current_commit,
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": file_hash(config_path),
        "dataset_index_sha256": file_hash(ROOT / config["dataset"]["index"]),
        "anchor_count": sum(item["anchor_count"] for item in episodes),
        "wall_time_s": time.perf_counter() - started,
        "nomad": {
            "source_revision": model.source_revision,
            "checkpoint_sha256": model.checkpoint_sha256,
            "parameter_count": model.parameter_count,
            "state_tensor_count": model.state_tensor_count,
            "state_value_count": model.state_value_count,
            "official_model_config": str(config["nomad"]["official_config"]),
            "diffusion_steps": int(config["nomad"]["diffusion_steps"]),
            "samples_per_observation": int(config["nomad"]["samples_per_observation"]),
        },
        "input_adapter": {
            **config["input_adapter"],
            "action_scale_m": float(
                load_yaml(ROOT / config["dataset"]["robot_profile"])["motion"]["max_forward_speed_mps"]
            ) / float(config["input_adapter"]["model_rate_hz"]),
            "depth_used_by_model": False,
        },
        "metrics": config["metrics"],
        "secondary_mode_summary": {
            mode: secondary_mode_summary(episodes, mode)
            for mode in ("final_visual_goal", "goal_masked_exploration")
        },
        "run_directory": str(run_directory.relative_to(ROOT)),
        "raw_window_metrics": str(raw_path.relative_to(ROOT)),
    }
    report = build_report(metadata, episodes)
    atomic_write_json(run_directory / "results.json", report)
    atomic_write_text(run_directory / "summary.md", render_markdown(report))
    atomic_write_text(run_directory / "episode_metrics.csv", episode_csv(episodes))
    if arguments.formal_report:
        formal = ROOT / config["outputs"]["formal_report_directory"]
        atomic_write_json(formal / "results.json", report)
        atomic_write_text(formal / "summary.md", render_markdown(report))
        atomic_write_text(formal / "episode_metrics.csv", episode_csv(episodes))
    return report


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-directory")
    parser.add_argument("--maximum-episodes", type=int)
    parser.add_argument("--maximum-anchors-per-episode", type=int)
    parser.add_argument("--formal-report", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    report = run(arguments)
    print(json.dumps({
        "status": report["status"],
        "episodes": report["episode_count"],
        "anchors": report["anchor_count"],
        "nominal_proxy_success_rate": report["summary"]["route_viability_proxy"]["nominal"]["success_rate"],
    }, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
