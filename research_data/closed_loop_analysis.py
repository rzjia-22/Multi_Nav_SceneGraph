"""Offline analysis for exhaustive NavDiffusion V0 validation runs.

This module consumes only the traces produced by the frozen closed-loop
runtime.  Expert paths and tree proxies are used after each mission for
diagnostics; they are never exposed to the navigator, controller, or Safety.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .common import ROOT, load_yaml


ARTIFACT_ROOT = ROOT / "artifacts/navdiffusion_v0_closed_loop"
RUN_ROOT = ROOT / "runs/navdiffusion_v0_closed_loop"
INDEX_PATH = ROOT / "datasets/dataset_v0/dataset_index.json"
ROBOT_PATH = ROOT / "config/robots/diablo_standing.yaml"
EXPERT_CORRIDOR_THRESHOLD_M = 0.75
EXPERT_CORRIDOR_SUSTAINED_S = 1.0
PROGRESS_EPSILON_M = 0.02
PROGRESS_DEGRADATION_S = 2.0


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validation_index_entries(index_path: Path = INDEX_PATH) -> list[dict]:
    """Return the sealed validation entries after enforcing split cardinality."""

    index = json.loads(index_path.read_text(encoding="utf-8"))
    entries = [item for item in index["episodes"] if item["split"] == "validation"]
    scenes = {item["scene_id"] for item in entries}
    if len(entries) != 10 or len(scenes) != 2:
        raise ValueError("Dataset V0 closed-loop analysis requires exactly 10 validation episodes in 2 scenes")
    if any(item.get("validation_status") != "PASS" for item in entries):
        raise ValueError("all source validation episodes must pass Dataset V0 validation")
    if any(not item["episode_id"].startswith("validation_scene_") for item in entries):
        raise ValueError("non-validation episode found after validation split filtering")
    return entries


def _load_trace_map(run_ids: list[str]) -> dict[str, tuple[dict, Path]]:
    traces: dict[str, tuple[dict, Path]] = {}
    for run_id in run_ids:
        for path in sorted((RUN_ROOT / run_id).glob("validation_scene_*/*/trace.json")):
            trace = json.loads(path.read_text(encoding="utf-8"))
            traces[trace["report"]["episode_id"]] = (trace, path)
    return traces


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    delta = end - start
    denominator = float(np.dot(delta, delta))
    if denominator <= 1.0e-15:
        return float(np.linalg.norm(point - start))
    factor = float(np.clip(np.dot(point - start, delta) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + factor * delta)))


def _polyline_distances(points: np.ndarray, path: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty(0, dtype=np.float64)
    if len(path) < 2:
        return np.linalg.norm(points - path[0], axis=1)
    return np.asarray([
        min(_point_segment_distance(point, start, end) for start, end in zip(path[:-1], path[1:]))
        for point in points
    ], dtype=np.float64)


def _first_sustained_time(times: np.ndarray, mask: np.ndarray, duration_s: float) -> float | None:
    start: float | None = None
    for timestamp, active in zip(times, mask):
        if active and start is None:
            start = float(timestamp)
        elif not active:
            start = None
        if start is not None and float(timestamp) - start >= duration_s:
            return start
    return None


def _expert_metrics(scene: dict, expert_xyz: np.ndarray, planning_radius_m: float) -> dict:
    path = expert_xyz[:, :2]
    clearances = []
    closest_tree = None
    for tree in scene["trees"]:
        center = np.asarray(tree["position_m"], dtype=np.float64)
        centerline_distance = min(
            _point_segment_distance(center, start, end) for start, end in zip(path[:-1], path[1:])
        )
        clearance = centerline_distance - float(tree["collision_proxy"]["radius_m"]) - planning_radius_m
        clearances.append(clearance)
        if closest_tree is None or clearance < closest_tree[0]:
            closest_tree = (clearance, tree["tree_id"])

    segments = np.diff(path, axis=0)
    segment_lengths = np.linalg.norm(segments, axis=1)
    headings = np.arctan2(segments[:, 1], segments[:, 0])
    turns = np.abs(np.arctan2(np.sin(np.diff(headings)), np.cos(np.diff(headings))))
    route_length = float(segment_lengths.sum())
    return {
        "expert_minimum_planning_clearance_m": float(min(clearances)),
        "expert_closest_tree_id": closest_tree[1],
        "expert_total_turn_rad": float(turns.sum()),
        "expert_max_turn_rad": float(turns.max()) if len(turns) else 0.0,
        "expert_turn_burden_rad_per_m": float(turns.sum() / route_length) if route_length else 0.0,
        "expert_waypoint_count": int(len(path)),
    }


def _progress_metrics(state: np.ndarray) -> dict:
    times = state[:, 0]
    distances = state[:, 10]
    if len(state) < 2:
        return {
            "initial_goal_distance_m": float(distances[0]) if len(distances) else None,
            "minimum_goal_distance_m": float(distances.min()) if len(distances) else None,
            "final_goal_distance_m": float(distances[-1]) if len(distances) else None,
            "monotonic_progress_fraction": None,
            "longest_no_progress_duration_s": 0.0,
            "progress_degradation_start_s": None,
        }
    monotonic = float(np.mean(np.diff(distances) <= 0.01))
    best = float(distances[0])
    last_progress_time = float(times[0])
    longest = 0.0
    degradation = None
    for timestamp, distance in zip(times[1:], distances[1:]):
        if float(distance) < best - PROGRESS_EPSILON_M:
            best = float(distance)
            last_progress_time = float(timestamp)
        gap = float(timestamp) - last_progress_time
        longest = max(longest, gap)
        if degradation is None and gap >= PROGRESS_DEGRADATION_S:
            degradation = last_progress_time
    return {
        "initial_goal_distance_m": float(distances[0]),
        "minimum_goal_distance_m": float(distances.min()),
        "final_goal_distance_m": float(distances[-1]),
        "monotonic_progress_fraction": monotonic,
        "longest_no_progress_duration_s": longest,
        "progress_degradation_start_s": degradation,
    }


def _trajectory_proxy_clearance(scene: dict, points: list, robot_radius_m: float) -> float | None:
    path = np.asarray(points, dtype=np.float64)
    if len(path) == 0:
        return None
    if len(path) == 1:
        segments = [(path[0, :2], path[0, :2])]
    else:
        segments = list(zip(path[:-1, :2], path[1:, :2]))
    return min(
        _point_segment_distance(np.asarray(tree["position_m"], dtype=np.float64), start, end)
        - float(tree["collision_proxy"]["radius_m"]) - robot_radius_m
        for tree in scene["trees"]
        for start, end in segments
    )


def _prediction_metrics(planning: list[dict], scene: dict | None = None,
                        robot_radius_m: float | None = None) -> dict:
    passing = [item for item in planning if item.get("status") == "PASS"]
    if scene is not None and robot_radius_m is not None:
        full_clearance = [
            _trajectory_proxy_clearance(scene, item.get("full_world_points", []), robot_radius_m)
            for item in passing
        ]
        control_clearance = [
            _trajectory_proxy_clearance(scene, item.get("control_world_points", []), robot_radius_m)
            for item in passing
        ]
    else:
        full_clearance = [item.get("full_minimum_clearance_m") for item in passing]
        control_clearance = [item.get("control_minimum_clearance_m") for item in passing]
    full_indices = [
        index for index, value in enumerate(full_clearance) if value is not None and float(value) <= 0.0
    ]
    control_indices = [
        index for index, value in enumerate(control_clearance) if value is not None and float(value) <= 0.0
    ]
    return {
        "planning_cycle_count": len(passing),
        "prediction_clearance_method": (
            "continuous_polyline_against_tree_proxy_plus_robot_radius"
            if scene is not None else "runtime_reported_point_clearance"
        ),
        "full_prediction_unsafe_count": len(full_indices),
        "control_prediction_unsafe_count": len(control_indices),
        "full_prediction_unsafe_fraction": len(full_indices) / len(passing) if passing else None,
        "control_prediction_unsafe_fraction": len(control_indices) / len(passing) if passing else None,
        "first_full_prediction_unsafe_s": (
            float(passing[full_indices[0]]["timestamp_from_episode_start_s"]) if full_indices else None
        ),
        "first_control_prediction_unsafe_s": (
            float(passing[control_indices[0]]["timestamp_from_episode_start_s"]) if control_indices else None
        ),
        "minimum_full_prediction_clearance_m": (
            float(min(value for value in full_clearance if value is not None))
            if any(value is not None for value in full_clearance) else None
        ),
        "minimum_control_prediction_clearance_m": (
            float(min(value for value in control_clearance if value is not None))
            if any(value is not None for value in control_clearance) else None
        ),
    }


def _failure_metrics(report: dict, commands: np.ndarray, prediction: dict,
                     progress: dict, deviation_time: float | None) -> dict:
    collision_time = report.get("first_collision_timestamp_s")
    safety_rows = commands[commands[:, -1] > 0.5] if len(commands) else np.empty((0, 8))
    safety_time = float(safety_rows[0, 0]) if len(safety_rows) else None
    full_time = prediction["first_full_prediction_unsafe_s"]
    control_time = prediction["first_control_prediction_unsafe_s"]
    reason = report.get("failure_reason")
    secondary: list[str] = []
    if report.get("success"):
        primary = None
    elif reason == "collision":
        primary = "MODEL" if control_time is not None and (
            collision_time is None or control_time <= collision_time
        ) else "CONTROLLER"
        if safety_time is None:
            secondary.append("SAFETY_NO_INTERVENTION")
        elif collision_time is not None and collision_time - safety_time < 0.5:
            secondary.append("SAFETY_LATE")
    elif reason == "timeout":
        primary = "TIMEOUT"
    elif reason == "stall":
        primary = "STALL"
    elif reason and (reason.startswith("sensor") or reason == "history_timeout"):
        primary = "SENSOR"
    elif reason and reason.startswith("model"):
        primary = "MODEL"
    else:
        primary = report.get("failure_class") or "SIMULATION"
    if not report.get("success") and deviation_time is not None:
        secondary.append("EXPERT_CORRIDOR_DEVIATION")
    if not report.get("success") and progress["progress_degradation_start_s"] is not None:
        secondary.append("PROGRESS_DEGRADATION")

    def delta(event: float | None) -> float | None:
        return float(collision_time - event) if collision_time is not None and event is not None else None

    return {
        "primary_failure_class": primary,
        "secondary_contributors": secondary,
        "timeline_s": {
            "full_prediction_first_unsafe": full_time,
            "control_prediction_first_unsafe": control_time,
            "safety_first_intervention": safety_time,
            "expert_corridor_deviation": deviation_time,
            "progress_degradation": progress["progress_degradation_start_s"],
            "collision": collision_time,
        },
        "collision_lead_time_s": {
            "from_full_prediction_unsafe": delta(full_time),
            "from_control_prediction_unsafe": delta(control_time),
            "from_safety_intervention": delta(safety_time),
        },
    }


def _episode_analysis(entry: dict, trace: dict, scene: dict, plan: dict, robot: dict) -> dict:
    report = trace["report"]
    state = np.asarray(trace["state"], dtype=np.float64)
    commands = np.asarray(trace["commands"], dtype=np.float64)
    expert = np.asarray(trace["expert_reference_path"], dtype=np.float64)
    actual_xy = state[:, 1:3]
    expert_xy = expert[:, :2]
    cross_track = _polyline_distances(actual_xy, expert_xy)
    deviation_time = _first_sustained_time(
        state[:, 0], cross_track > EXPERT_CORRIDOR_THRESHOLD_M, EXPERT_CORRIDOR_SUSTAINED_S
    ) if len(state) else None
    progress = _progress_metrics(state)
    robot_radius = float(robot["surrogate"]["footprint"]["collision_check_radius_m"])
    prediction = _prediction_metrics(trace["planning"], scene, robot_radius)
    mount = report["camera_mount_episode_variation"]
    variation = robot["camera_mount"]["episode_variation"]
    extrema = {
        "height": max(abs(float(value)) for value in variation["height_m"]),
        "pitch": max(abs(float(value)) for value in variation["pitch_deg"]),
        "roll": max(abs(float(value)) for value in variation["roll_deg"]),
    }
    camera_extremeness = max(
        abs(float(mount["height_offset_m"])) / extrema["height"],
        abs(float(mount["pitch_offset_deg"])) / extrema["pitch"],
        abs(float(mount["roll_offset_deg"])) / extrema["roll"],
    )
    result = {
        **report,
        "route_bucket": entry["route_bucket"],
        "environment": {
            **scene["factors"],
            "tree_asset_composition": dict(sorted(Counter(
                tree["asset_id"] for tree in scene["trees"]
            ).items())),
        },
        **_expert_metrics(
            scene, expert, float(robot["surrogate"]["footprint"]["planning_radius_m"])
        ),
        **progress,
        **prediction,
        "mean_expert_cross_track_error_m": float(cross_track.mean()) if len(cross_track) else None,
        "max_expert_cross_track_error_m": float(cross_track.max()) if len(cross_track) else None,
        "expert_corridor_threshold_m": EXPERT_CORRIDOR_THRESHOLD_M,
        "expert_corridor_sustained_s": EXPERT_CORRIDOR_SUSTAINED_S,
        "expert_corridor_deviation_start_s": deviation_time,
        "camera_perturbation_extremeness_fraction": camera_extremeness,
        "expert_used_for_control": False,
    }
    result.update(_failure_metrics(report, commands, prediction, progress, deviation_time))
    return result


def _mean(items: list[dict], key: str) -> float | None:
    values = [float(item[key]) for item in items if item.get(key) is not None]
    return float(np.mean(values)) if values else None


def _summary(items: list[dict]) -> dict:
    return {
        "episode_count": len(items),
        "success_count": sum(bool(item["success"]) for item in items),
        "failure_count": sum(not bool(item["success"]) for item in items),
        "collision_count": sum(bool(item["collision"]) for item in items),
        "success_rate": sum(bool(item["success"]) for item in items) / len(items) if items else None,
        "mean_goal_error_m": _mean(items, "goal_error_m"),
        "mean_minimum_clearance_m": _mean(items, "minimum_clearance_m"),
        "mean_safety_override_fraction": _mean(items, "safety_override_fraction"),
        "mean_full_prediction_unsafe_fraction": _mean(items, "full_prediction_unsafe_fraction"),
        "mean_control_prediction_unsafe_fraction": _mean(items, "control_prediction_unsafe_fraction"),
        "mean_expert_planning_clearance_m": _mean(items, "expert_minimum_planning_clearance_m"),
        "mean_expert_cross_track_error_m": _mean(items, "mean_expert_cross_track_error_m"),
        "mean_simulated_duration_s": _mean(items, "simulated_duration_s"),
        "mean_plan_count": _mean(items, "plan_count"),
    }


def _group_by(items: list[dict], key_function) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        groups[str(key_function(item))].append(item)
    return {key: _summary(values) for key, values in sorted(groups.items())}


def _normalized_trajectory_distance(first: np.ndarray, second: np.ndarray) -> float | None:
    if len(first) < 2 or len(second) < 2:
        return None
    sample = np.linspace(0.0, 1.0, 101)
    first_t = np.linspace(0.0, 1.0, len(first))
    second_t = np.linspace(0.0, 1.0, len(second))
    a = np.column_stack([np.interp(sample, first_t, first[:, axis]) for axis in range(2)])
    b = np.column_stack([np.interp(sample, second_t, second[:, axis]) for axis in range(2)])
    return float(np.linalg.norm(a - b, axis=1).mean())


def _repeatability(current: dict, current_trace: dict) -> dict:
    episode_id = "validation_scene_001_episode_004"
    previous_path = ARTIFACT_ROOT / "gate_b" / episode_id / "report.json"
    result = {"episode_id": episode_id, "previous_evidence_available": previous_path.is_file()}
    if not previous_path.is_file():
        return result
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    gate_report = json.loads((ARTIFACT_ROOT / "gate_b/aggregate_report.json").read_text(encoding="utf-8"))
    previous_traces = _load_trace_map(gate_report.get("run_ids", []))
    trajectory_distance = None
    if episode_id in previous_traces:
        previous_state = np.asarray(previous_traces[episode_id][0]["state"], dtype=np.float64)[:, 1:3]
        current_state = np.asarray(current_trace["state"], dtype=np.float64)[:, 1:3]
        trajectory_distance = _normalized_trajectory_distance(previous_state, current_state)
    previous_time = previous.get("first_collision_timestamp_s")
    current_time = current.get("first_collision_timestamp_s")
    same_failure = previous.get("failure_reason") == current.get("failure_reason") == "collision"
    same_tree = previous.get("collision_tree_id") == current.get("collision_tree_id")
    similar_time = (
        previous_time is not None and current_time is not None
        and abs(float(previous_time) - float(current_time)) <= 2.0
    )
    similar_path = trajectory_distance is not None and trajectory_distance <= 0.75
    result.update({
        "previous": {
            "success": previous["success"],
            "failure_reason": previous.get("failure_reason"),
            "collision_tree_id": previous.get("collision_tree_id"),
            "collision_timestamp_s": previous_time,
            "executed_length_m": previous.get("executed_length_m"),
        },
        "current": {
            "success": current["success"],
            "failure_reason": current.get("failure_reason"),
            "collision_tree_id": current.get("collision_tree_id"),
            "collision_timestamp_s": current_time,
            "executed_length_m": current.get("executed_length_m"),
        },
        "same_failure_mode": same_failure,
        "same_tree": same_tree,
        "collision_timestamp_difference_s": (
            abs(float(previous_time) - float(current_time))
            if previous_time is not None and current_time is not None else None
        ),
        "normalized_trajectory_mean_distance_m": trajectory_distance,
        "similar_timestamp_threshold_s": 2.0,
        "similar_path_threshold_m": 0.75,
        "repeatable_failure": bool(same_failure and same_tree and similar_time and similar_path),
        "interpretation": (
            "The collision repeated on the same tree with similar timing and trajectory."
            if same_failure and same_tree and similar_time and similar_path else
            "The rerun did not reproduce the prior collision under all declared similarity criteria; "
            "this is evidence of closed-loop stochastic or scheduling sensitivity, not a causal diagnosis."
        ),
    })
    return result


def _plot_results(items: list[dict], traces: dict[str, tuple[dict, Path]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = ARTIFACT_ROOT / "plots"
    output.mkdir(parents=True, exist_ok=True)
    ordered = sorted(items, key=lambda item: item["planned_expert_length_m"])
    lengths = [item["planned_expert_length_m"] for item in ordered]
    colors = ["#2ca02c" if item["success"] else "#d62728" for item in ordered]

    figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    axis.scatter(lengths, [int(item["success"]) for item in ordered], c=colors, s=65)
    axis.set_yticks([0, 1], ["FAIL", "PASS"])
    axis.set_xlabel("planned expert path length [m]")
    axis.set_title("NavDiffusion V0 closed-loop outcome vs route length")
    axis.grid(alpha=0.2)
    figure.savefig(output / "success_vs_route_length.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    axis.scatter(lengths, [item["minimum_clearance_m"] for item in ordered], c=colors, s=65)
    axis.axhline(0.0, color="black", linewidth=1.0)
    axis.set_xlabel("planned expert path length [m]")
    axis.set_ylabel("minimum executed conservative clearance [m]")
    axis.set_title("Executed clearance vs route length")
    axis.grid(alpha=0.2)
    figure.savefig(output / "minimum_clearance_vs_route_length.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    labels = [item["episode_id"].replace("validation_scene_", "v").replace("_episode_", "e") for item in items]
    item_colors = ["#2ca02c" if item["success"] else "#d62728" for item in items]
    axis.bar(labels, [item["safety_override_fraction"] for item in items], color=item_colors)
    axis.set_ylabel("Safety override fraction")
    axis.set_title("Safety intervention by validation episode")
    axis.tick_params(axis="x", labelrotation=45)
    axis.grid(axis="y", alpha=0.2)
    figure.savefig(output / "safety_override_fraction.png", dpi=150)
    plt.close(figure)

    for item in items:
        if item["success"]:
            continue
        episode_id = item["episode_id"]
        trace = traces[episode_id][0]
        scene = load_yaml(ROOT / "research_scenes/dataset_v0" / item["scene_id"] / "scene.yaml")
        state = np.asarray(trace["state"], dtype=np.float64)
        expert = np.asarray(trace["expert_reference_path"], dtype=np.float64)
        planning = [value for value in trace["planning"] if value.get("status") == "PASS"]
        indices = {0, max(0, len(planning) // 2), max(0, len(planning) - 1)}
        for index, value in enumerate(planning):
            if value.get("control_minimum_clearance_m") is not None and value["control_minimum_clearance_m"] <= 0:
                indices.update({max(0, index - 1), index})
                break
        figure, axis = plt.subplots(figsize=(8, 8), constrained_layout=True)
        width, height = (float(value) for value in scene["extent_m"])
        axis.set_xlim(-width / 2.0, width / 2.0)
        axis.set_ylim(-height / 2.0, height / 2.0)
        axis.set_aspect("equal")
        for tree in scene["trees"]:
            axis.add_patch(plt.Circle(
                tree["position_m"], float(tree["collision_proxy"]["radius_m"]) + 0.42,
                color="#654321", alpha=0.30,
            ))
        axis.plot(expert[:, 0], expert[:, 1], "--", color="#ff7f0e", label="expert (analysis only)")
        axis.plot(state[:, 1], state[:, 2], color="#1f77b4", linewidth=2.0, label="executed")
        palette = plt.cm.Reds(np.linspace(0.35, 0.95, max(1, len(indices))))
        for color, index in zip(palette, sorted(indices)):
            if not planning:
                break
            points = np.asarray(planning[index]["full_world_points"], dtype=np.float64)
            axis.plot(points[:, 0], points[:, 1], color=color, linewidth=1.2,
                      label=f"prediction t={planning[index]['timestamp_from_episode_start_s']:.1f}s")
        axis.scatter(expert[0, 0], expert[0, 1], s=70, label="start")
        axis.scatter(expert[-1, 0], expert[-1, 1], marker="*", s=150, label="goal")
        if item.get("collision") and len(state):
            axis.scatter(state[-1, 1], state[-1, 2], marker="X", s=130, color="black", label="collision")
        axis.set_title(f"{episode_id}: {item['primary_failure_class']}")
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.grid(alpha=0.15)
        axis.legend(fontsize=7)
        episode_output = ARTIFACT_ROOT / "full_validation/episodes" / episode_id
        episode_output.mkdir(parents=True, exist_ok=True)
        figure.savefig(episode_output / "failure_trajectory.png", dpi=150)
        plt.close(figure)


def _findings(items: list[dict], route: dict, scene: dict) -> list[str]:
    passed = [item for item in items if item["success"]]
    failed = [item for item in items if not item["success"]]
    findings = []
    if failed:
        pass_length = _mean(passed, "planned_expert_length_m")
        fail_length = _mean(failed, "planned_expert_length_m")
        findings.append(
            f"Failed missions averaged {fail_length:.3f} m planned length versus "
            f"{pass_length:.3f} m for successful missions." if pass_length is not None else
            f"Failed missions averaged {fail_length:.3f} m planned length; there were no successful comparators."
        )
        failure_scenes = Counter(item["scene_id"] for item in failed)
        findings.append("Failure counts by scene: " + ", ".join(
            f"{key}={value}" for key, value in sorted(failure_scenes.items())
        ) + ".")
        comparisons = [
            ("expert planning clearance", "expert_minimum_planning_clearance_m", "m"),
            ("Safety override fraction", "safety_override_fraction", ""),
            ("full-prediction unsafe fraction", "full_prediction_unsafe_fraction", ""),
            ("control-prediction unsafe fraction", "control_prediction_unsafe_fraction", ""),
            ("expert cross-track error", "mean_expert_cross_track_error_m", "m"),
            ("simulated duration", "simulated_duration_s", "s"),
            ("camera perturbation extremeness", "camera_perturbation_extremeness_fraction", ""),
        ]
        for label, key, unit in comparisons:
            pass_value, fail_value = _mean(passed, key), _mean(failed, key)
            if pass_value is not None and fail_value is not None:
                findings.append(
                    f"Mean {label}: PASS={pass_value:.4f}{unit}, FAIL={fail_value:.4f}{unit}."
                )
        ordered_buckets = [key for key in ("short", "medium", "long") if key in route]
        findings.append("Route-bucket success: " + ", ".join(
            f"{key}={route[key]['success_count']}/{route[key]['episode_count']}"
            for key in ordered_buckets
        ) + ".")
        findings.append(
            "These ten validation missions support descriptive associations only; they do not establish causality."
        )
    else:
        findings.append("All ten validation missions succeeded; no failure association can be estimated.")
    return findings


def _write_markdown(analysis: dict) -> None:
    def number(value: float | None, digits: int = 3) -> str:
        return "n/a" if value is None else f"{value:.{digits}f}"

    def percent(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.2%}"

    lines = [
        "# NavDiffusion V0 exhaustive closed-loop validation analysis",
        "",
        f"All **{analysis['episode_count']}/10** validation missions were executed without experiment-level fail-fast. "
        f"Navigation succeeded on **{analysis['success_count']}/10**; test was not evaluated and Hydra was disabled.",
        "",
        f"Readiness: **{analysis['readiness']}**",
        "",
        "## Episode overview",
        "",
        "| Episode | Bucket | Expert len [m] | Result | Failure | Goal err [m] | Min clearance [m] | Safety | Full unsafe | Control unsafe |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in analysis["episodes"]:
        lines.append(
            f"| `{item['episode_id']}` | {item['route_bucket']} | {item['planned_expert_length_m']:.3f} | "
            f"{'PASS' if item['success'] else 'FAIL'} | {item['primary_failure_class'] or '—'} | "
            f"{number(item['goal_error_m'])} | {number(item['minimum_clearance_m'])} | "
            f"{percent(item['safety_override_fraction'])} | "
            f"{percent(item['full_prediction_unsafe_fraction'])} | "
            f"{percent(item['control_prediction_unsafe_fraction'])} |"
        )
    lines.extend(["", "## Route buckets", ""])
    for key, value in analysis["route_bucket_summary"].items():
        lines.append(
            f"- {key}: {value['success_count']}/{value['episode_count']} success, "
            f"{value['collision_count']} collision(s), mean goal error {number(value['mean_goal_error_m'])} m, "
            f"mean clearance {number(value['mean_minimum_clearance_m'])} m, "
            f"mean Safety {percent(value['mean_safety_override_fraction'])}."
        )
    lines.extend(["", "## Scenes", ""])
    for key, value in analysis["scene_summary"].items():
        lines.append(
            f"- `{key}`: {value['success_count']}/{value['episode_count']} success, "
            f"{value['collision_count']} collision(s), mean goal error {number(value['mean_goal_error_m'])} m."
        )
    lines.extend(["", "## Failure timelines", ""])
    failures = [item for item in analysis["episodes"] if not item["success"]]
    if not failures:
        lines.append("No failed mission.")
    for item in failures:
        timeline = item["timeline_s"]
        lines.append(
            f"- `{item['episode_id']}` — primary `{item['primary_failure_class']}`; "
            f"secondary {item['secondary_contributors'] or 'none'}; full unsafe {timeline['full_prediction_first_unsafe']}, "
            f"control unsafe {timeline['control_prediction_first_unsafe']}, Safety {timeline['safety_first_intervention']}, "
            f"collision {timeline['collision']} s."
        )
    lines.extend(["", "## Descriptive findings", ""])
    lines.extend(f"- {value}" for value in analysis["findings"])
    repeat = analysis["repeatability_validation_scene_001_episode_004"]
    lines.extend([
        "",
        "## Episode 004 repeatability",
        "",
        f"Repeatable failure under the declared same-mode/same-tree/±2 s/0.75 m trajectory criteria: "
        f"**{repeat.get('repeatable_failure')}**. {repeat.get('interpretation', '')}",
        "",
        "## Scope",
        "",
        "- Split: validation only (2 scenes, 10 episodes)",
        "- Test split evaluated: **NO**",
        "- Hydra/mapping: disabled",
        "- Model, preprocessing, controller, Safety, DIABLO profile, D435i profile and scenes: frozen",
        "- Expert path: post-run analysis only; never used for control",
        "",
    ])
    (ARTIFACT_ROOT / "full_validation_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def analyze_full_validation(full_report: dict, readiness: str) -> dict:
    """Analyze a complete 10-mission report and persist lightweight evidence."""

    entries = validation_index_entries()
    expected = [item["episode_id"] for item in entries]
    if set(full_report["expected_episodes"]) != set(expected):
        raise ValueError("full validation episodes differ from the canonical Dataset V0 validation index")
    if full_report["episode_count"] != 10 or full_report["not_run_episodes"]:
        raise ValueError("analysis requires traces for all ten validation episodes")
    traces = _load_trace_map(full_report["run_ids"])
    if set(traces) != set(expected):
        raise ValueError("full validation trace set does not match the canonical validation split")

    robot = load_yaml(ROBOT_PATH)
    by_entry = {item["episode_id"]: item for item in entries}
    episodes = []
    for episode_id in expected:
        entry = by_entry[episode_id]
        trace = traces[episode_id][0]
        scene = load_yaml(ROOT / "research_scenes/dataset_v0" / entry["scene_id"] / "scene.yaml")
        plan = load_yaml(Path(entry["relative_path"]).parent / "episode_plan.yaml"
                         if Path(entry["relative_path"]).is_absolute() else
                         ROOT / Path(entry["relative_path"]).parent / "episode_plan.yaml")
        if plan["plan_hash"] != entry["plan_hash"] or trace["report"]["plan_hash"] != entry["plan_hash"]:
            raise ValueError(f"plan provenance mismatch for {episode_id}")
        if trace["report"]["scene_hash"] != entry["scene_content_hash"]:
            raise ValueError(f"scene provenance mismatch for {episode_id}")
        if trace["report"].get("test_split_used") or trace["report"].get("mapping_enabled"):
            raise ValueError("test or mapping contamination in full validation trace")
        episodes.append(_episode_analysis(entry, trace, scene, plan, robot))

    failure_classes = Counter(
        item["primary_failure_class"] for item in episodes if item["primary_failure_class"]
    )
    analysis = {
        "analysis_version": 1,
        "scope": "validation_only",
        "episode_count": len(episodes),
        "scene_count": len({item["scene_id"] for item in episodes}),
        "success_count": sum(item["success"] for item in episodes),
        "failure_count": sum(not item["success"] for item in episodes),
        "collision_count": sum(item["collision"] for item in episodes),
        "timeout_count": sum(item["failure_reason"] == "timeout" for item in episodes),
        "stall_count": sum(item["failure_reason"] == "stall" for item in episodes),
        "readiness": readiness,
        "validation_acceptance_pass": readiness == "READY_FOR_LOW_SPEED_REAL_CLOSED_LOOP",
        "test_split_used": False,
        "mapping_enabled": False,
        "episodes": episodes,
        "episodes_by_planned_length": [{
            "episode_id": item["episode_id"],
            "planned_expert_length_m": item["planned_expert_length_m"],
            "success": item["success"],
            "minimum_clearance_m": item["minimum_clearance_m"],
        } for item in sorted(episodes, key=lambda value: value["planned_expert_length_m"])],
        "route_bucket_summary": _group_by(episodes, lambda item: item["route_bucket"]),
        "scene_summary": _group_by(episodes, lambda item: item["scene_id"]),
        "terrain_summary": _group_by(episodes, lambda item: item["environment"]["terrain"]),
        "density_summary": _group_by(episodes, lambda item: item["environment"]["tree_density"]),
        "ground_summary": _group_by(episodes, lambda item: item["environment"]["ground"]),
        "lighting_summary": _group_by(episodes, lambda item: item["environment"]["lighting"]),
        "outcome_summary": {
            "PASS": _summary([item for item in episodes if item["success"]]),
            "FAIL": _summary([item for item in episodes if not item["success"]]),
        },
        "safety_by_outcome": {
            "PASS": _summary([item for item in episodes if item["success"]]),
            "FAIL": _summary([item for item in episodes if not item["success"]]),
        },
        "prediction_risk_by_outcome": {
            "PASS": _summary([item for item in episodes if item["success"]]),
            "FAIL": _summary([item for item in episodes if not item["success"]]),
        },
        "failure_classes": dict(sorted(failure_classes.items())),
        "camera_perturbation_by_episode": [{
            "episode_id": item["episode_id"],
            "success": item["success"],
            **item["camera_mount_episode_variation"],
            "extremeness_fraction": item["camera_perturbation_extremeness_fraction"],
        } for item in episodes],
        "repeatability_validation_scene_001_episode_004": _repeatability(
            next(item for item in episodes if item["episode_id"] == "validation_scene_001_episode_004"),
            traces["validation_scene_001_episode_004"][0],
        ),
    }
    analysis["findings"] = _findings(
        episodes, analysis["route_bucket_summary"], analysis["scene_summary"]
    )
    _write_json(ARTIFACT_ROOT / "full_validation_analysis.json", analysis)
    _write_markdown(analysis)
    _plot_results(episodes, traces)
    return analysis
