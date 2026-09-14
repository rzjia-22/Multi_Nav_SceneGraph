"""Validation-only orchestration and review for NavDiffusion V0 closed loop."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from .common import ROOT, load_yaml
from .closed_loop_analysis import (
    analyze_full_validation,
    dataset_index_entries,
    validation_index_entries,
)


CONFIG = ROOT / "config/navigation/navdiffusion_v0_closed_loop.yaml"
ARTIFACT_ROOT = ROOT / "artifacts/navdiffusion_v0_closed_loop"
RUN_ROOT = ROOT / "runs/navdiffusion_v0_closed_loop"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _scene_id(episode_id: str) -> str:
    return episode_id.rsplit("_episode_", 1)[0]


def _validate_scope(config: dict) -> None:
    if config["scope"] != "validation_only" or config["test_split_allowed"]:
        raise ValueError("closed-loop configuration must remain validation-only")
    checkpoint = ROOT / config["checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    for gate in ("gate_a", "gate_b"):
        episodes = config["gates"][gate]["episodes"]
        if not episodes or any(not value.startswith("validation_scene_") for value in episodes):
            raise ValueError(f"{gate} contains a non-validation episode")
    full = config["gates"]["full_validation"]["scenes"]
    episodes = [episode for values in full.values() for episode in values]
    if len(episodes) != 10 or len(set(episodes)) != 10:
        raise ValueError("full validation must contain ten unique episodes")
    if any(not scene.startswith("validation_scene_") for scene in full):
        raise ValueError("full validation contains a non-validation scene")
    if set(episodes) != {item["episode_id"] for item in validation_index_entries()}:
        raise ValueError("full validation must exactly match the canonical Dataset V0 validation split")


def _compose_session(
    artifact_group: str,
    scene_id: str,
    episode_ids: list[str],
    *,
    capture_review: bool,
    fail_on_episode_failure: bool,
    split: str = "validation",
    run_root: Path = RUN_ROOT,
) -> tuple[int, str]:
    if split not in {"validation", "test"}:
        raise ValueError("closed-loop session accepts validation or test only")
    if any(_scene_id(value) != scene_id for value in episode_ids):
        raise ValueError(f"a scene-batched session may contain only one {split} scene")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{artifact_group}-{scene_id}-{timestamp}"
    environment = os.environ.copy()
    environment.update({
        "MNS_CLOSED_LOOP_SCENE": scene_id,
        "MNS_CLOSED_LOOP_EPISODES": ",".join(episode_ids),
        "MNS_CLOSED_LOOP_RUN_ID": run_id,
        "MNS_CLOSED_LOOP_ARTIFACT_GROUP": artifact_group,
        "MNS_CLOSED_LOOP_SPLIT": split,
        "MNS_CLOSED_LOOP_RUN_ROOT": "/mns/" + str(run_root.relative_to(ROOT)),
        "MNS_CLOSED_LOOP_CAPTURE_FLAG": "--capture-review" if capture_review else "",
        "MNS_CLOSED_LOOP_FAILURE_FLAG": "--fail-on-episode-failure" if fail_on_episode_failure else "",
    })
    command = [
        "docker", "compose", "--profile", "research-navigation", "up",
        "--force-recreate", "--abort-on-container-exit",
        "--exit-code-from", "simulation-navdiffusion-v0-closed-loop",
        "simulation-navdiffusion-v0-closed-loop",
        "robotics-navdiffusion-v0-closed-loop",
    ]
    completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    subprocess.run(
        ["docker", "compose", "--profile", "research-navigation", "rm", "-sf",
         "simulation-navdiffusion-v0-closed-loop", "robotics-navdiffusion-v0-closed-loop"],
        cwd=ROOT, env=environment, check=False,
    )
    return_code = completed.returncode
    session_report = run_root / run_id / scene_id / "session_report.json"
    if not session_report.is_file():
        # Isaac Kit can normalize an exception to exit code zero while closing
        # the application.  A completed scene session must always leave its
        # atomically-written report; absence is an infrastructure failure.
        return_code = return_code or 1
    if fail_on_episode_failure and session_report.is_file():
        session = json.loads(session_report.read_text(encoding="utf-8"))
        if not session.get("all_pass", False):
            # Isaac Kit may normalize its process exit during App.close().
            # Treat the atomically written result as the authoritative gate
            # status so a failed mission cannot leak into the next scene.
            return_code = 2
    return return_code, run_id


def _draw_forest(axis, scene: dict) -> None:
    import matplotlib.pyplot as plt

    width, height = (float(value) for value in scene["extent_m"])
    axis.set_xlim(-width / 2.0, width / 2.0)
    axis.set_ylim(-height / 2.0, height / 2.0)
    axis.set_aspect("equal")
    for tree in scene["trees"]:
        axis.add_patch(plt.Circle(
            tree["position_m"], float(tree["collision_proxy"]["radius_m"]),
            color="#654321", alpha=0.55,
        ))
    axis.grid(alpha=0.15)
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")


def _visualize_episode(run_id: str, group: str, scene_id: str, episode_id: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_directory = RUN_ROOT / run_id / scene_id / episode_id
    trace = json.loads((run_directory / "trace.json").read_text(encoding="utf-8"))
    scene = load_yaml(ROOT / "research_scenes/dataset_v0" / scene_id / "scene.yaml")
    state = np.asarray(trace["state"], dtype=np.float64)
    expert = np.asarray(trace["expert_reference_path"], dtype=np.float64)
    planning = [item for item in trace["planning"] if item.get("status") == "PASS"]
    output = ARTIFACT_ROOT / group / episode_id
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "report.json", trace["report"])

    figure, axis = plt.subplots(figsize=(8, 8), constrained_layout=True)
    _draw_forest(axis, scene)
    axis.plot(expert[:, 0], expert[:, 1], "--", color="#ff7f0e", linewidth=1.8, label="expert reference only")
    if len(state):
        axis.plot(state[:, 1], state[:, 2], color="#1f77b4", linewidth=2.0, label="closed-loop executed")
    if planning:
        indices = sorted(set([0, len(planning) // 2, len(planning) - 1]))
        colors = ["#2ca02c", "#9467bd", "#d62728"]
        for color, index in zip(colors, indices):
            points = np.asarray(planning[index]["full_world_points"], dtype=np.float64)
            axis.plot(points[:, 0], points[:, 1], color=color, alpha=0.8, linewidth=1.1,
                      label=f"prediction t={planning[index]['timestamp_from_episode_start_s']:.1f}s")
    axis.scatter(expert[0, 0], expert[0, 1], marker="o", s=70, color="#2878b5", label="start")
    axis.scatter(expert[-1, 0], expert[-1, 1], marker="*", s=150, color="#d62728", label="goal")
    axis.set_title(f"{episode_id}: NavDiffusion closed loop")
    axis.legend(loc="best", fontsize=8)
    figure.savefig(output / "trajectory.png", dpi=150)
    plt.close(figure)

    review_path = run_directory / "review_frames.npz"
    if review_path.is_file():
        review = np.load(review_path)
        rgb, depth = review["rgb"], review["depth"]
        names = [str(value) for value in review["names"]]
        figure, axes = plt.subplots(2, len(names), figsize=(4.5 * len(names), 6), squeeze=False, constrained_layout=True)
        for column, name in enumerate(names):
            axes[0, column].imshow(rgb[column])
            axes[0, column].set_title(f"{name} RGB")
            axes[0, column].axis("off")
            valid = depth[column][depth[column] > 0.0]
            maximum = min(10.0, float(np.percentile(valid, 98))) if len(valid) else 10.0
            axes[1, column].imshow(depth[column], cmap="turbo", vmin=0.2, vmax=maximum)
            axes[1, column].set_title(f"{name} registered depth [m]")
            axes[1, column].axis("off")
        figure.savefig(output / "sensor_review.png", dpi=140)
        plt.close(figure)

    report = trace["report"]
    if report.get("failure_reason") and planning:
        final_predictions = planning[-min(4, len(planning)) :]
        figure, axis = plt.subplots(figsize=(8, 8), constrained_layout=True)
        _draw_forest(axis, scene)
        axis.plot(expert[:, 0], expert[:, 1], "--", color="#ff7f0e", linewidth=1.8,
                  label="expert reference only")
        if len(state):
            axis.plot(state[:, 1], state[:, 2], color="#1f77b4", linewidth=2.0,
                      label="closed-loop executed")
            axis.scatter(state[-1, 1], state[-1, 2], marker="X", s=120, color="#000000",
                         label="failure position")
        colors = plt.cm.Reds(np.linspace(0.40, 0.95, len(final_predictions)))
        for color, item in zip(colors, final_predictions):
            points = np.asarray(item["full_world_points"], dtype=np.float64)
            axis.plot(points[:, 0], points[:, 1], color=color, linewidth=1.3,
                      label=f"prediction t={item['timestamp_from_episode_start_s']:.2f}s")
        axis.scatter(expert[0, 0], expert[0, 1], marker="o", s=70, color="#2878b5", label="start")
        axis.scatter(expert[-1, 0], expert[-1, 1], marker="*", s=150, color="#d62728", label="goal")
        axis.set_title(f"{episode_id}: final predictions before {report['failure_reason']}")
        axis.legend(loc="best", fontsize=8)
        figure.savefig(output / "failure_predictions.png", dpi=150)
        plt.close(figure)

        full_unsafe = [item for item in planning if item.get("full_minimum_clearance_m") is not None
                       and item["full_minimum_clearance_m"] <= 0.0]
        control_unsafe = [item for item in planning if item.get("control_minimum_clearance_m") is not None
                          and item["control_minimum_clearance_m"] <= 0.0]
        commands = np.asarray(trace["commands"], dtype=np.float64)
        safety_rows = commands[commands[:, -1] > 0.5] if len(commands) else np.empty((0, 8))
        _write_json(output / "failure_analysis.json", {
            "episode_id": episode_id,
            "failure_reason": report["failure_reason"],
            "failure_class": report["failure_class"],
            "collision_tree_id": report.get("collision_tree_id"),
            "collision_timestamp_s": report.get("first_collision_timestamp_s"),
            "first_full_prediction_collision_s": (
                full_unsafe[0]["timestamp_from_episode_start_s"] if full_unsafe else None
            ),
            "first_control_prediction_collision_s": (
                control_unsafe[0]["timestamp_from_episode_start_s"] if control_unsafe else None
            ),
            "first_safety_override_s": float(safety_rows[0, 0]) if len(safety_rows) else None,
            "minimum_executed_clearance_m": report["minimum_clearance_m"],
            "final_prediction_cycles": [{
                "timestamp_s": item["timestamp_from_episode_start_s"],
                "full_minimum_clearance_m": item.get("full_minimum_clearance_m"),
                "control_minimum_clearance_m": item.get("control_minimum_clearance_m"),
                "goal_distance_m": item.get("goal_distance_m"),
                "maximum_consecutive_jump_m": item.get("maximum_consecutive_jump_m"),
                "goal_progress_dot": item.get("goal_progress_dot"),
            } for item in final_predictions],
        })


def _group_report(
    group: str,
    run_ids: list[str],
    expected_episodes: list[str],
    *,
    split: str = "validation",
    run_root: Path = RUN_ROOT,
    artifact_root: Path = ARTIFACT_ROOT,
) -> dict:
    if split not in {"validation", "test"}:
        raise ValueError("closed-loop report accepts validation or test only")
    reports = []
    inference = []
    sessions = []
    for run_id in run_ids:
        run_directory = run_root / run_id
        for session_path in sorted(run_directory.glob(f"{split}_scene_*/session_report.json")):
            session = json.loads(session_path.read_text(encoding="utf-8"))
            episode_reports = session.get("episodes", [])
            sessions.append({
                "run_id": run_id,
                "scene_id": session["scene_id"],
                "app_startup_time_s": (
                    episode_reports[0].get("app_startup_time_s") if episode_reports else None
                ),
                "scene_load_time_s": session.get("scene_load_time_s"),
                "total_wall_time_s": session.get("total_wall_time_s"),
            })
        for trace_path in sorted(run_directory.glob(f"{split}_scene_*/*/trace.json")):
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            reports.append(trace["report"])
            inference.extend(
                float(item["inference_ms"])
                for item in trace["planning"]
                if item.get("status") == "PASS" and item.get("inference_ms") is not None
            )
    by_id = {item["episode_id"]: item for item in reports}
    ordered = [by_id[value] for value in expected_episodes if value in by_id]
    index_by_id = {item["episode_id"]: item for item in dataset_index_entries(split)}
    for item in ordered:
        item.setdefault("route_bucket", index_by_id[item["episode_id"]]["route_bucket"])
    failure_classes: dict[str, int] = defaultdict(int)
    for item in ordered:
        if item.get("failure_class"):
            failure_classes[str(item["failure_class"])] += 1

    def summarize(values: list[dict]) -> dict:
        return {
            "episode_count": len(values),
            "success_count": sum(bool(item["success"]) for item in values),
            "failure_count": sum(not bool(item["success"]) for item in values),
            "collision_count": sum(bool(item["collision"]) for item in values),
            "mean_goal_error_m": float(np.mean([item["goal_error_m"] for item in values])) if values else None,
            "mean_minimum_clearance_m": (
                float(np.mean([item["minimum_clearance_m"] for item in values])) if values else None
            ),
            "mean_safety_override_fraction": (
                float(np.mean([item["safety_override_fraction"] for item in values])) if values else None
            ),
        }

    route_groups: dict[str, list[dict]] = defaultdict(list)
    scene_groups: dict[str, list[dict]] = defaultdict(list)
    for item in ordered:
        route_groups[item["route_bucket"]].append(item)
        scene_groups[item["scene_id"]].append(item)
    result = {
        "report_version": 1,
        "group": group,
        "evaluation_split": split,
        "expected_episodes": expected_episodes,
        "episodes": ordered,
        "episode_count": len(ordered),
        "failure_count": sum(not item["success"] for item in ordered),
        "not_run_episodes": [value for value in expected_episodes if value not in by_id],
        "success_count": sum(item["success"] for item in ordered),
        "success_rate": sum(item["success"] for item in ordered) / len(ordered) if ordered else None,
        "collision_count": sum(item["collision"] for item in ordered),
        "timeout_count": sum(item["failure_reason"] == "timeout" for item in ordered),
        "stall_count": sum(item["failure_reason"] == "stall" for item in ordered),
        "mean_goal_error_m": float(np.mean([item["goal_error_m"] for item in ordered])) if ordered else None,
        "mean_path_efficiency": float(np.mean([item["path_efficiency"] for item in ordered if item["path_efficiency"] is not None])) if ordered else None,
        "minimum_clearance_m": min((item["minimum_clearance_m"] for item in ordered), default=None),
        "inference_ms_mean": float(np.mean(inference)) if inference else None,
        "inference_ms_p95": float(np.percentile(inference, 95)) if inference else None,
        "inference_ms_max": max(inference, default=None),
        "total_safety_override_count": sum(item["safety_override_count"] for item in ordered),
        "total_safety_override_duration_s": sum(item["safety_override_duration_s"] for item in ordered),
        "failure_classes": dict(sorted(failure_classes.items())),
        "mean_safety_override_fraction": float(np.mean([item["safety_override_fraction"] for item in ordered])) if ordered else None,
        "all_pass": len(ordered) == len(expected_episodes) and all(item["success"] for item in ordered),
        "execution_completed": len(ordered) == len(expected_episodes),
        "total_simulated_duration_s": sum(item["simulated_duration_s"] for item in ordered),
        "total_episode_wall_duration_s": sum(item["wall_duration_s"] for item in ordered),
        "total_wall_duration_s": sum(
            float(item["total_wall_time_s"])
            for item in sessions if item["total_wall_time_s"] is not None
        ),
        "total_app_startup_time_s": sum(
            float(item["app_startup_time_s"])
            for item in sessions if item["app_startup_time_s"] is not None
        ),
        "total_scene_load_time_s": sum(
            float(item["scene_load_time_s"])
            for item in sessions if item["scene_load_time_s"] is not None
        ),
        "session_summaries": sessions,
        "total_plan_count": sum(item["plan_count"] for item in ordered),
        "route_bucket_summary": {
            key: summarize(values) for key, values in sorted(route_groups.items())
        },
        "scene_summary": {
            key: summarize(values) for key, values in sorted(scene_groups.items())
        },
        "run_ids": run_ids,
        "test_split_used": split == "test",
        "mapping_enabled": False,
    }
    _write_json(artifact_root / group / "aggregate_report.json", result)
    return result


def _run_group(
    group: str,
    episodes: list[str],
    *,
    fail_fast: bool,
    capture: bool,
    split: str = "validation",
    run_root: Path = RUN_ROOT,
    artifact_root: Path = ARTIFACT_ROOT,
    stop_on_session_error: bool = False,
) -> dict:
    if split not in {"validation", "test"}:
        raise ValueError("closed-loop execution accepts validation or test only")
    grouped: dict[str, list[str]] = defaultdict(list)
    for episode in episodes:
        if not episode.startswith(f"{split}_scene_"):
            raise ValueError(f"only {split} episodes are permitted")
        grouped[_scene_id(episode)].append(episode)
    run_ids = []
    return_codes = []
    for scene_id, scene_episodes in grouped.items():
        code, run_id = _compose_session(
            group, scene_id, scene_episodes,
            capture_review=capture,
            fail_on_episode_failure=fail_fast,
            split=split,
            run_root=run_root,
        )
        run_ids.append(run_id)
        return_codes.append(code)
        for episode in scene_episodes:
            trace = run_root / run_id / scene_id / episode / "trace.json"
            if trace.is_file() and capture:
                _visualize_episode(run_id, group, scene_id, episode)
        if code != 0 and (fail_fast or stop_on_session_error):
            break
    report = _group_report(
        group, run_ids, episodes,
        split=split, run_root=run_root, artifact_root=artifact_root,
    )
    report["compose_return_codes"] = return_codes
    report["execution_completed"] = (
        report["episode_count"] == len(episodes)
        and not report["not_run_episodes"]
        and all(code == 0 for code in return_codes)
    )
    _write_json(artifact_root / group / "aggregate_report.json", report)
    return report


def run_gates(config: dict) -> tuple[dict, dict | None]:
    gate_a = _run_group("gate_a", config["gates"]["gate_a"]["episodes"], fail_fast=True, capture=True)
    if not gate_a["all_pass"]:
        return gate_a, None
    gate_b = _run_group("gate_b", config["gates"]["gate_b"]["episodes"], fail_fast=True, capture=True)
    return gate_a, gate_b


def _existing_gate(name: str, expected: list[str]) -> dict | None:
    path = ARTIFACT_ROOT / name / "aggregate_report.json"
    if not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("expected_episodes") != expected:
        return None
    return report


def _readiness(gate_a: dict, gate_b: dict | None, full: dict | None, config: dict) -> str:
    if not gate_a.get("all_pass"):
        return "NOT_READY"
    if full is not None:
        threshold = config["readiness"]["low_speed_real_closed_loop"]
        model_failures = sum(item.get("failure_class") == "MODEL" for item in full["episodes"])
        if (
            full["success_count"] >= int(threshold["minimum_success_count"])
            and full["collision_count"] <= int(threshold["maximum_collision_count"])
            and full["inference_ms_p95"] is not None and full["inference_ms_p95"] < 500.0
            and full["mean_safety_override_fraction"] <= float(threshold["maximum_mean_safety_override_fraction"])
            and model_failures == 0
        ):
            return "READY_FOR_LOW_SPEED_REAL_CLOSED_LOOP"
        if full["success_count"] >= 7 and full["collision_count"] == 0:
            return "READY_FOR_REAL_INFERENCE_ONLY"
        if full["success_count"] < 5 or full["collision_count"] >= 3:
            return "NOT_READY"
        return "READY_FOR_REAL_SENSOR_ONLY"
    if gate_b is None or not gate_b.get("all_pass"):
        return "READY_FOR_REAL_SENSOR_ONLY"
    return "READY_FOR_REAL_SENSOR_ONLY"


def _write_final_summary(gate_a: dict, gate_b: dict | None, full: dict | None, readiness: str) -> None:
    result = {
        "report_version": 1,
        "checkpoint_sha256": load_yaml(CONFIG)["checkpoint_sha256"],
        "gate_a": gate_a,
        "gate_b": gate_b,
        "full_validation": full,
        "readiness": readiness,
        "validation_acceptance_pass": readiness == "READY_FOR_LOW_SPEED_REAL_CLOSED_LOOP",
        "test_split_used": False,
        "mapping_enabled": False,
    }
    _write_json(ARTIFACT_ROOT / "aggregate_report.json", result)
    lines = [
        "# NavDiffusion V0 closed-loop validation",
        "",
        f"Readiness: **{readiness}**",
        "",
        f"- Gate A: {gate_a['success_count']}/{len(gate_a['expected_episodes'])}",
        f"- Gate B: {gate_b['success_count'] if gate_b else 0}/{len(gate_b['expected_episodes']) if gate_b else 0}",
    ]
    for label, report in (("Gate A", gate_a), ("Gate B", gate_b)):
        if not report:
            continue
        lines.append("")
        lines.append(f"## {label} episodes")
        lines.append("")
        for episode in report["episodes"]:
            status = "PASS" if episode["success"] else f"FAIL ({episode['failure_class']}: {episode['failure_reason']})"
            lines.append(
                f"- `{episode['episode_id']}`: {status}; goal error "
                f"{episode['goal_error_m']:.3f} m; clearance {episode['minimum_clearance_m']:.3f} m; "
                f"Safety {episode['safety_override_fraction']:.3%}."
            )
        if report.get("not_run_episodes"):
            lines.append(f"- Not run after fail-fast: {', '.join(f'`{value}`' for value in report['not_run_episodes'])}.")
    if full:
        lines.extend([
            "",
            "## Full validation",
            "",
            f"- Full validation: {full['success_count']}/10",
            f"- All experiments executed: {full['execution_completed']}",
            f"- Conservative collisions: {full['collision_count']}",
            f"- Timeout/stall: {full['timeout_count']}/{full['stall_count']}",
            f"- Mean goal error: {full['mean_goal_error_m']:.3f} m",
            f"- Mean/p95 inference: {full['inference_ms_mean']:.1f}/{full['inference_ms_p95']:.1f} ms",
            f"- Mean safety override fraction: {full['mean_safety_override_fraction']:.3f}",
            f"- Simulated / complete session wall time: {full['total_simulated_duration_s']:.2f} / "
            f"{full['total_wall_duration_s']:.2f} s",
        ])
        lines.extend(
            f"- {bucket}: {value['success_count']}/{value['episode_count']} success, "
            f"{value['collision_count']} collision(s)."
            for bucket, value in full["route_bucket_summary"].items()
        )
        lines.extend(
            f"- `{scene}`: {value['success_count']}/{value['episode_count']} success, "
            f"{value['collision_count']} collision(s)."
            for scene, value in full["scene_summary"].items()
        )
    else:
        lines.extend(["", "Full ten-mission validation was not run because Gate B failed."])
    lines.extend([
        "",
        "- Dataset split used: validation only",
        "- Test split: untouched",
        "- Hydra/mapping: disabled",
        "",
    ])
    (ARTIFACT_ROOT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("run-gates", "run-validation", "run-analysis", "validate-config")
    )
    args = parser.parse_args()
    config = load_yaml(CONFIG)
    _validate_scope(config)
    if args.command == "validate-config":
        print(json.dumps({"status": "PASS", "scope": config["scope"]}, sort_keys=True))
        return 0

    gate_a_expected = config["gates"]["gate_a"]["episodes"]
    gate_b_expected = config["gates"]["gate_b"]["episodes"]
    gate_a = _existing_gate("gate_a", gate_a_expected)
    gate_b = _existing_gate("gate_b", gate_b_expected)
    if args.command == "run-analysis":
        if gate_a is None:
            raise RuntimeError("historical Gate A evidence is required before exhaustive analysis")
        full_episodes = [item["episode_id"] for item in validation_index_entries()]
        full = _run_group("full_validation", full_episodes, fail_fast=False, capture=False)
        readiness = _readiness(gate_a, gate_b, full, config)
        full["readiness"] = readiness
        analysis = None
        if full["execution_completed"]:
            analysis = analyze_full_validation(full, readiness)
            full["failure_classes"] = analysis["failure_classes"]
            full["route_bucket_summary"] = analysis["route_bucket_summary"]
            full["scene_summary"] = analysis["scene_summary"]
            full["validation_acceptance_pass"] = analysis["validation_acceptance_pass"]
            _write_json(ARTIFACT_ROOT / "full_validation/aggregate_report.json", full)
            _write_json(ARTIFACT_ROOT / "full_validation_report.json", full)
        _write_final_summary(gate_a, gate_b, full, readiness)
        return 0 if full["execution_completed"] and analysis is not None else 2
    if gate_a is None or not gate_a.get("all_pass") or gate_b is None or not gate_b.get("all_pass"):
        gate_a, gate_b = run_gates(config)
    if not gate_a["all_pass"] or gate_b is None or not gate_b["all_pass"]:
        readiness = _readiness(gate_a, gate_b, None, config)
        _write_final_summary(gate_a, gate_b, None, readiness)
        return 2
    if args.command == "run-gates":
        _write_final_summary(gate_a, gate_b, None, _readiness(gate_a, gate_b, None, config))
        return 0

    full_scenes = config["gates"]["full_validation"]["scenes"]
    full_episodes = [episode for episodes in full_scenes.values() for episode in episodes]
    full = _run_group("full_validation", full_episodes, fail_fast=False, capture=False)
    readiness = _readiness(gate_a, gate_b, full, config)
    _write_final_summary(gate_a, gate_b, full, readiness)
    return 0 if full["episode_count"] == 10 else 2


if __name__ == "__main__":
    raise SystemExit(main())
