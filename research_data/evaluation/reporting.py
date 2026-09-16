"""Stable report IO and aggregation for navigation evaluations."""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from ..common import ROOT, load_yaml
from .metrics import episode_analysis


INDEX_PATH = ROOT / "datasets/dataset_v0/dataset_index.json"
HELD_OUT_SPLITS = frozenset({"validation", "test"})


def atomic_write_json(path: Path, value: Any) -> None:
    """Write a complete report or leave the previous report untouched."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.inprogress")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def scene_id_from_episode(episode_id: str) -> str:
    if "_episode_" not in episode_id:
        raise ValueError(f"invalid episode ID: {episode_id}")
    return episode_id.rsplit("_episode_", 1)[0]


def dataset_index_entries(split: str, index_path: Path = INDEX_PATH) -> list[dict]:
    """Return a validated held-out split without allowing training leakage."""

    if split not in HELD_OUT_SPLITS:
        raise ValueError(f"navigation evaluation accepts {sorted(HELD_OUT_SPLITS)}, not {split!r}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    entries = [item for item in index["episodes"] if item["split"] == split]
    if not entries:
        raise ValueError(f"dataset index has no {split} episodes")
    if any(item.get("validation_status") != "PASS" for item in entries):
        raise ValueError(f"all source {split} episodes must pass Dataset validation")
    if any(scene_id_from_episode(item["episode_id"]) != item["scene_id"] for item in entries):
        raise ValueError(f"{split} episode/scene identity mismatch")
    if len({item["episode_id"] for item in entries}) != len(entries):
        raise ValueError(f"duplicate {split} episode IDs")
    return entries


def load_trace_map(run_ids: list[str], run_root: Path, split: str) -> dict[str, tuple[dict, Path]]:
    traces: dict[str, tuple[dict, Path]] = {}
    for run_id in run_ids:
        for path in sorted((run_root / run_id).glob(f"{split}_scene_*/*/trace.json")):
            trace = json.loads(path.read_text(encoding="utf-8"))
            episode_id = trace["report"]["episode_id"]
            if episode_id in traces:
                raise ValueError(f"duplicate trace for {episode_id}")
            traces[episode_id] = (trace, path)
    return traces


def _mean(items: list[dict], key: str) -> float | None:
    values = [float(item[key]) for item in items if item.get(key) is not None]
    return float(np.mean(values)) if values else None


def summarize_episodes(items: list[dict]) -> dict:
    return {
        "episode_count": len(items),
        "success_count": sum(bool(item["success"]) for item in items),
        "failure_count": sum(not bool(item["success"]) for item in items),
        "collision_count": sum(bool(item["collision"]) for item in items),
        "success_rate": sum(bool(item["success"]) for item in items) / len(items) if items else None,
        "mean_goal_error_m": _mean(items, "goal_error_m"),
        "mean_minimum_clearance_m": _mean(items, "minimum_clearance_m"),
        "mean_safety_override_fraction": _mean(items, "safety_override_fraction"),
    }


def build_evaluation_report(
    evaluation_id: str,
    split: str,
    run_ids: list[str],
    expected_episodes: list[str],
    *,
    run_root: Path,
    robot_profile: Path,
) -> dict:
    """Aggregate scene sessions without interpreting model readiness."""

    entries = {item["episode_id"]: item for item in dataset_index_entries(split)}
    traces = load_trace_map(run_ids, run_root, split)
    ordered: list[dict] = []
    inference: list[float] = []
    sessions: list[dict] = []
    for run_id in run_ids:
        for session_path in sorted((run_root / run_id).glob(f"{split}_scene_*/session_report.json")):
            session = json.loads(session_path.read_text(encoding="utf-8"))
            episode_reports = session.get("episodes", [])
            sessions.append({
                "run_id": run_id,
                "scene_id": session["scene_id"],
                "app_startup_time_s": episode_reports[0].get("app_startup_time_s") if episode_reports else None,
                "scene_load_time_s": session.get("scene_load_time_s"),
                "total_wall_time_s": session.get("total_wall_time_s"),
            })
    for episode_id in expected_episodes:
        if episode_id not in traces:
            continue
        trace = traces[episode_id][0]
        report = trace["report"]
        if report.get("evaluation_split") != split:
            raise ValueError(f"trace split mismatch for {episode_id}")
        if report.get("scene_id") != entries[episode_id]["scene_id"]:
            raise ValueError(f"trace scene mismatch for {episode_id}")
        if report.get("scene_hash") != entries[episode_id]["scene_content_hash"]:
            raise ValueError(f"trace scene hash mismatch for {episode_id}")
        if report.get("plan_hash") != entries[episode_id]["plan_hash"]:
            raise ValueError(f"trace plan hash mismatch for {episode_id}")
        scene = load_yaml(ROOT / "research_scenes/dataset_v0" / report["scene_id"] / "scene.yaml")
        plan = load_yaml(
            ROOT / "datasets/dataset_v0" / split / report["scene_id"]
            / episode_id / "episode_plan.yaml"
        )
        analyzed = episode_analysis(
            entries[episode_id], trace, scene, plan, load_yaml(robot_profile)
        )
        ordered.append(analyzed)
        inference.extend(
            float(item["inference_ms"])
            for item in trace.get("planning", [])
            if item.get("status") == "PASS" and item.get("inference_ms") is not None
        )

    route_groups: dict[str, list[dict]] = defaultdict(list)
    scene_groups: dict[str, list[dict]] = defaultdict(list)
    failure_classes: dict[str, int] = defaultdict(int)
    for item in ordered:
        route_groups[str(item["route_bucket"])].append(item)
        scene_groups[str(item["scene_id"])].append(item)
        if item.get("failure_class"):
            failure_classes[str(item["failure_class"])] += 1

    summary = summarize_episodes(ordered)
    return {
        "report_version": 2,
        "evaluation_id": evaluation_id,
        "evaluation_split": split,
        "expected_episodes": expected_episodes,
        "episodes": ordered,
        **summary,
        "not_run_episodes": [value for value in expected_episodes if value not in traces],
        "execution_completed": len(ordered) == len(expected_episodes),
        "all_navigation_successful": len(ordered) == len(expected_episodes) and all(
            item["success"] for item in ordered
        ),
        "timeout_count": sum(item.get("failure_reason") == "timeout" for item in ordered),
        "stall_count": sum(item.get("failure_reason") == "stall" for item in ordered),
        "minimum_clearance_m": min(
            (float(item["minimum_clearance_m"]) for item in ordered
             if item.get("minimum_clearance_m") is not None),
            default=None,
        ),
        "mean_path_efficiency": _mean(ordered, "path_efficiency"),
        "inference_ms_mean": float(np.mean(inference)) if inference else None,
        "inference_ms_p95": float(np.percentile(inference, 95)) if inference else None,
        "inference_ms_max": max(inference, default=None),
        "total_safety_override_count": sum(item["safety_override_count"] for item in ordered),
        "total_safety_override_duration_s": sum(item["safety_override_duration_s"] for item in ordered),
        "failure_classes": dict(sorted(failure_classes.items())),
        "total_simulated_duration_s": sum(item["simulated_duration_s"] for item in ordered),
        "total_episode_wall_duration_s": sum(item["wall_duration_s"] for item in ordered),
        "total_wall_duration_s": sum(
            float(item["total_wall_time_s"])
            for item in sessions if item["total_wall_time_s"] is not None
        ),
        "session_summaries": sessions,
        "route_bucket_summary": {
            key: summarize_episodes(values) for key, values in sorted(route_groups.items())
        },
        "scene_summary": {
            key: summarize_episodes(values) for key, values in sorted(scene_groups.items())
        },
        "run_ids": run_ids,
        "test_split_used": split == "test",
        "mapping_enabled": False,
    }
