"""Navigator-independent offline metrics for Research Forest traces.

Expert paths and privileged tree proxies are consumed only after execution. They are
never exposed to the navigator, controller, or Safety layer.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math

import numpy as np


EXPERT_CORRIDOR_THRESHOLD_M = 0.75
EXPERT_CORRIDOR_SUSTAINED_S = 1.0
PROGRESS_EPSILON_M = 0.02
PROGRESS_DEGRADATION_S = 2.0


def point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    delta = end - start
    denominator = float(np.dot(delta, delta))
    if denominator <= 1.0e-15:
        return float(np.linalg.norm(point - start))
    factor = float(np.clip(np.dot(point - start, delta) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + factor * delta)))


def polyline_distances(points: np.ndarray, path: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty(0, dtype=np.float64)
    if len(path) < 2:
        return np.linalg.norm(points - path[0], axis=1)
    return np.asarray([
        min(point_segment_distance(point, start, end) for start, end in zip(path[:-1], path[1:]))
        for point in points
    ], dtype=np.float64)


def first_sustained_time(times: np.ndarray, mask: np.ndarray, duration_s: float) -> float | None:
    start: float | None = None
    for timestamp, active in zip(times, mask):
        if active and start is None:
            start = float(timestamp)
        elif not active:
            start = None
        if start is not None and float(timestamp) - start >= duration_s:
            return start
    return None


def expert_metrics(scene: dict, expert_xyz: np.ndarray, planning_radius_m: float) -> dict:
    path = expert_xyz[:, :2]
    clearances = []
    closest_tree = None
    for tree in scene["trees"]:
        center = np.asarray(tree["position_m"], dtype=np.float64)
        centerline_distance = min(
            point_segment_distance(center, start, end) for start, end in zip(path[:-1], path[1:])
        )
        clearance = centerline_distance - float(tree["collision_proxy"]["radius_m"]) - planning_radius_m
        clearances.append(clearance)
        if closest_tree is None or clearance < closest_tree[0]:
            closest_tree = (clearance, tree["tree_id"])
    segments = np.diff(path, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    headings = np.arctan2(segments[:, 1], segments[:, 0])
    turns = np.abs(np.arctan2(np.sin(np.diff(headings)), np.cos(np.diff(headings))))
    route_length = float(lengths.sum())
    return {
        "expert_minimum_planning_clearance_m": float(min(clearances)),
        "expert_closest_tree_id": closest_tree[1],
        "expert_total_turn_rad": float(turns.sum()),
        "expert_max_turn_rad": float(turns.max()) if len(turns) else 0.0,
        "expert_turn_burden_rad_per_m": float(turns.sum() / route_length) if route_length else 0.0,
        "expert_waypoint_count": int(len(path)),
    }


def progress_metrics(state: np.ndarray, analysis_start_s: float = 0.0) -> dict:
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
    active = state[times >= analysis_start_s]
    if len(active) < 2:
        active = state
    progress_times = active[:, 0]
    progress_distances = active[:, 10]
    best = float(progress_distances[0])
    last_progress_time = float(progress_times[0])
    longest = 0.0
    for timestamp, distance in zip(progress_times[1:], progress_distances[1:]):
        if float(distance) < best - PROGRESS_EPSILON_M:
            best = float(distance)
            last_progress_time = float(timestamp)
        longest = max(longest, float(timestamp) - last_progress_time)
    final_gap = float(progress_times[-1]) - last_progress_time
    return {
        "initial_goal_distance_m": float(distances[0]),
        "minimum_goal_distance_m": float(distances.min()),
        "final_goal_distance_m": float(distances[-1]),
        "monotonic_progress_fraction": float(np.mean(np.diff(progress_distances) <= 0.01)),
        "longest_no_progress_duration_s": longest,
        "progress_degradation_start_s": last_progress_time if final_gap >= PROGRESS_DEGRADATION_S else None,
    }


def trajectory_proxy_clearance(scene: dict, points: list, robot_radius_m: float) -> float | None:
    path = np.asarray(points, dtype=np.float64)
    if len(path) == 0:
        return None
    segments = [(path[0, :2], path[0, :2])] if len(path) == 1 else list(zip(path[:-1, :2], path[1:, :2]))
    return min(
        point_segment_distance(np.asarray(tree["position_m"], dtype=np.float64), start, end)
        - float(tree["collision_proxy"]["radius_m"]) - robot_radius_m
        for tree in scene["trees"] for start, end in segments
    )


def prediction_metrics(
    planning: list[dict], scene: dict | None = None, robot_radius_m: float | None = None,
) -> dict:
    passing = [item for item in planning if item.get("status") == "PASS"]
    if scene is not None and robot_radius_m is not None:
        full = [trajectory_proxy_clearance(scene, item.get("full_world_points", []), robot_radius_m) for item in passing]
        control = [trajectory_proxy_clearance(scene, item.get("control_world_points", []), robot_radius_m) for item in passing]
        method = "continuous_polyline_against_tree_proxy_plus_robot_radius"
    else:
        full = [item.get("full_minimum_clearance_m") for item in passing]
        control = [item.get("control_minimum_clearance_m") for item in passing]
        method = "runtime_reported_point_clearance"
    full_indices = [index for index, value in enumerate(full) if value is not None and float(value) <= 0.0]
    control_indices = [index for index, value in enumerate(control) if value is not None and float(value) <= 0.0]
    return {
        "planning_cycle_count": len(passing),
        "prediction_clearance_method": method,
        "full_prediction_unsafe_count": len(full_indices),
        "control_prediction_unsafe_count": len(control_indices),
        "full_prediction_unsafe_fraction": len(full_indices) / len(passing) if passing else None,
        "control_prediction_unsafe_fraction": len(control_indices) / len(passing) if passing else None,
        "first_full_prediction_unsafe_s": float(passing[full_indices[0]]["timestamp_from_episode_start_s"]) if full_indices else None,
        "first_control_prediction_unsafe_s": float(passing[control_indices[0]]["timestamp_from_episode_start_s"]) if control_indices else None,
        "minimum_full_prediction_clearance_m": float(min(value for value in full if value is not None)) if any(value is not None for value in full) else None,
        "minimum_control_prediction_clearance_m": float(min(value for value in control if value is not None)) if any(value is not None for value in control) else None,
    }


def failure_metrics(
    report: dict, commands: np.ndarray, prediction: dict, progress: dict,
    deviation_time: float | None,
) -> dict:
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
        primary = "MODEL" if control_time is not None and (collision_time is None or control_time <= collision_time) else "CONTROLLER"
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


def episode_analysis(entry: dict, trace: dict, scene: dict, plan: dict, robot: dict) -> dict:
    report = trace["report"]
    state = np.asarray(trace["state"], dtype=np.float64)
    commands = np.asarray(trace["commands"], dtype=np.float64)
    expert = np.asarray(trace["expert_reference_path"], dtype=np.float64)
    cross_track = polyline_distances(state[:, 1:3], expert[:, :2])
    deviation = first_sustained_time(
        state[:, 0], cross_track > EXPERT_CORRIDOR_THRESHOLD_M, EXPERT_CORRIDOR_SUSTAINED_S
    ) if len(state) else None
    progress = progress_metrics(state, float(report.get("history_ready_time_s") or 0.0))
    robot_radius = float(robot["surrogate"]["footprint"]["collision_check_radius_m"])
    prediction = prediction_metrics(trace["planning"], scene, robot_radius)
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
            "tree_asset_composition": dict(sorted(Counter(tree["asset_id"] for tree in scene["trees"]).items())),
        },
        **expert_metrics(scene, expert, float(robot["surrogate"]["footprint"]["planning_radius_m"])),
        **progress,
        **prediction,
        "mean_expert_cross_track_error_m": float(cross_track.mean()) if len(cross_track) else None,
        "max_expert_cross_track_error_m": float(cross_track.max()) if len(cross_track) else None,
        "expert_corridor_threshold_m": EXPERT_CORRIDOR_THRESHOLD_M,
        "expert_corridor_sustained_s": EXPERT_CORRIDOR_SUSTAINED_S,
        "expert_corridor_deviation_start_s": deviation,
        "camera_perturbation_extremeness_fraction": camera_extremeness,
        "expert_used_for_control": False,
    }
    result.update(failure_metrics(report, commands, prediction, progress, deviation))
    return result


def mean(items: list[dict], key: str) -> float | None:
    values = [float(item[key]) for item in items if item.get(key) is not None]
    return float(np.mean(values)) if values else None


def metric_summary(items: list[dict]) -> dict:
    return {
        "episode_count": len(items),
        "success_count": sum(bool(item["success"]) for item in items),
        "failure_count": sum(not bool(item["success"]) for item in items),
        "collision_count": sum(bool(item["collision"]) for item in items),
        "success_rate": sum(bool(item["success"]) for item in items) / len(items) if items else None,
        "mean_goal_error_m": mean(items, "goal_error_m"),
        "mean_minimum_clearance_m": mean(items, "minimum_clearance_m"),
        "mean_safety_override_fraction": mean(items, "safety_override_fraction"),
        "mean_full_prediction_unsafe_fraction": mean(items, "full_prediction_unsafe_fraction"),
        "mean_control_prediction_unsafe_fraction": mean(items, "control_prediction_unsafe_fraction"),
        "mean_expert_cross_track_error_m": mean(items, "mean_expert_cross_track_error_m"),
    }


def risk_summary(items: list[dict]) -> dict:
    cycles = sum(item["planning_cycle_count"] for item in items)
    full = sum(item["full_prediction_unsafe_count"] for item in items)
    control = sum(item["control_prediction_unsafe_count"] for item in items)
    return {
        "planning_cycle_count": cycles,
        "full_prediction_unsafe_event_count": full,
        "control_prediction_unsafe_event_count": control,
        "full_prediction_unsafe_fraction": full / cycles if cycles else None,
        "control_prediction_unsafe_fraction": control / cycles if cycles else None,
    }


def group_by(items: list[dict], key_function) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        groups[str(key_function(item))].append(item)
    return {key: metric_summary(values) for key, values in sorted(groups.items())}
