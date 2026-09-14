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


def _compose_session(
    artifact_group: str,
    scene_id: str,
    episode_ids: list[str],
    *,
    capture_review: bool,
    fail_on_episode_failure: bool,
) -> tuple[int, str]:
    if any(_scene_id(value) != scene_id for value in episode_ids):
        raise ValueError("a scene-batched session may contain only one validation scene")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{artifact_group}-{scene_id}-{timestamp}"
    environment = os.environ.copy()
    environment.update({
        "MNS_CLOSED_LOOP_SCENE": scene_id,
        "MNS_CLOSED_LOOP_EPISODES": ",".join(episode_ids),
        "MNS_CLOSED_LOOP_RUN_ID": run_id,
        "MNS_CLOSED_LOOP_ARTIFACT_GROUP": artifact_group,
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
    return completed.returncode, run_id


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


def _group_report(group: str, run_ids: list[str], expected_episodes: list[str]) -> dict:
    reports = []
    inference = []
    for run_id in run_ids:
        for trace_path in sorted((RUN_ROOT / run_id).glob("validation_scene_*/*/trace.json")):
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            reports.append(trace["report"])
            inference.extend(
                float(item["inference_ms"])
                for item in trace["planning"]
                if item.get("status") == "PASS" and item.get("inference_ms") is not None
            )
    by_id = {item["episode_id"]: item for item in reports}
    ordered = [by_id[value] for value in expected_episodes if value in by_id]
    result = {
        "report_version": 1,
        "group": group,
        "expected_episodes": expected_episodes,
        "episodes": ordered,
        "episode_count": len(ordered),
        "success_count": sum(item["success"] for item in ordered),
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
        "mean_safety_override_fraction": float(np.mean([item["safety_override_fraction"] for item in ordered])) if ordered else None,
        "all_pass": len(ordered) == len(expected_episodes) and all(item["success"] for item in ordered),
        "run_ids": run_ids,
        "test_split_used": False,
        "mapping_enabled": False,
    }
    _write_json(ARTIFACT_ROOT / group / "aggregate_report.json", result)
    return result


def _run_group(group: str, episodes: list[str], *, fail_fast: bool, capture: bool) -> dict:
    grouped: dict[str, list[str]] = defaultdict(list)
    for episode in episodes:
        if not episode.startswith("validation_scene_"):
            raise ValueError("only validation episodes are permitted")
        grouped[_scene_id(episode)].append(episode)
    run_ids = []
    return_codes = []
    for scene_id, scene_episodes in grouped.items():
        code, run_id = _compose_session(
            group, scene_id, scene_episodes,
            capture_review=capture,
            fail_on_episode_failure=fail_fast,
        )
        run_ids.append(run_id)
        return_codes.append(code)
        for episode in scene_episodes:
            trace = RUN_ROOT / run_id / scene_id / episode / "trace.json"
            if trace.is_file() and capture:
                _visualize_episode(run_id, group, scene_id, episode)
        if fail_fast and code != 0:
            break
    report = _group_report(group, run_ids, episodes)
    report["compose_return_codes"] = return_codes
    _write_json(ARTIFACT_ROOT / group / "aggregate_report.json", report)
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
    if gate_b is None or not gate_b.get("all_pass"):
        return "READY_FOR_REAL_SENSOR_ONLY"
    if full is None:
        return "READY_FOR_REAL_SENSOR_ONLY"
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
    return "READY_FOR_REAL_SENSOR_ONLY"


def _write_final_summary(gate_a: dict, gate_b: dict | None, full: dict | None, readiness: str) -> None:
    result = {
        "report_version": 1,
        "checkpoint_sha256": load_yaml(CONFIG)["checkpoint_sha256"],
        "gate_a": gate_a,
        "gate_b": gate_b,
        "full_validation": full,
        "readiness": readiness,
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
    if full:
        lines.extend([
            f"- Full validation: {full['success_count']}/10",
            f"- Conservative collisions: {full['collision_count']}",
            f"- Mean goal error: {full['mean_goal_error_m']:.3f} m",
            f"- Mean/p95 inference: {full['inference_ms_mean']:.1f}/{full['inference_ms_p95']:.1f} ms",
            f"- Mean safety override fraction: {full['mean_safety_override_fraction']:.3f}",
        ])
    lines.extend([
        "- Dataset split used: validation only",
        "- Test split: untouched",
        "- Hydra/mapping: disabled",
        "",
    ])
    (ARTIFACT_ROOT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run-gates", "run-validation", "validate-config"))
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
