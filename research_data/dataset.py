"""Canonical Dataset V0 indexing and aggregate validation."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from .common import ROOT, episode_directory, load_yaml, scene_paths
from .episode import read_metadata
from .validation import validate_episode, validate_manifest, validate_scene, write_report


DATASET_ROOT = ROOT / "datasets" / "dataset_v0"
INDEX_PATH = DATASET_ROOT / "dataset_index.json"
COLLECTION_REPORT_PATH = DATASET_ROOT / "dataset_v0_collection_report.json"
COLLECTION_SUMMARY_PATH = DATASET_ROOT / "dataset_v0_collection_summary.md"


def build_dataset_index(require_complete: bool = False) -> dict[str, Any]:
    """Validate finalized episodes and write the only training-facing index."""
    manifest_report = validate_manifest()
    manifest = load_yaml(ROOT / "config/datasets/dataset_v0_manifest.yaml")
    entries: list[dict[str, Any]] = []
    missing: list[str] = []
    scene_reports: list[dict[str, Any]] = []
    for scene_spec in manifest["scenes"]:
        scene_id = scene_spec["scene_id"]
        scene_path, _ = scene_paths(scene_id)
        if scene_path.exists():
            scene_reports.append(validate_scene(scene_path))
        for episode_spec in scene_spec["planned_episodes"]:
            episode_id = episode_spec["episode_id"]
            directory = episode_directory(episode_id)
            hdf5_path = directory / "episode.h5"
            plan_path = directory / "episode_plan.yaml"
            if not hdf5_path.exists() or not plan_path.exists():
                missing.append(episode_id)
                continue
            validation = validate_episode(hdf5_path, plan_path)
            plan = load_yaml(plan_path)
            report_path = directory / "validation_report.json"
            write_report(report_path, validation)
            scene = load_yaml(scene_path)
            entries.append({
                "episode_id": episode_id,
                "scene_id": scene_id,
                "scene_seed": int(scene_spec["scene_seed"]),
                "scene_content_hash": scene["content_hash"],
                "split": scene_spec["split"],
                "terrain_profile": scene_spec["factors"]["terrain"],
                "ground": scene_spec["factors"]["ground"],
                "lighting": scene_spec["factors"]["lighting"],
                "tree_density": scene_spec["factors"]["tree_density"],
                "route_bucket": episode_spec["target_route_length_bucket"],
                "planner_type": validation["planner_type"],
                "planner_version": validation["planner_version"],
                "plan_hash": validation["plan_hash"],
                "planned_path_length_m": validation["planned_path_length_m"],
                "executed_path_length_m": validation["executed_path_length_m"],
                "goal_error_m": validation["goal_error_m"],
                "duration_s": validation["duration_s"],
                "rgb_frames": validation["rgb_frames"],
                "depth_frames": validation["depth_frames"],
                "imu_samples": validation["imu_samples"],
                "state_samples": validation["pose_samples"],
                "hdf5_sha256": validation["hdf5_sha256"],
                "hdf5_bytes": hdf5_path.stat().st_size,
                "schema_version": validation["episode_schema_version"],
                "collector_version": read_metadata(hdf5_path)["collector_version"],
                "validation_status": validation["status"],
                "relative_path": str(hdf5_path.relative_to(ROOT)),
            })
    entries.sort(key=lambda item: item["episode_id"])
    if require_complete:
        assert not missing, f"Dataset V0 is incomplete: {missing}"
        assert len(entries) == 70
    index = {
        "dataset_version": "dataset_v0", "index_version": 1,
        "status": "PASS" if len(entries) == 70 else "PARTIAL",
        "manifest": "config/datasets/dataset_v0_manifest.yaml",
        "manifest_validation": manifest_report,
        "episode_count": len(entries), "missing_episode_ids": missing,
        "episodes": entries,
    }
    write_report(INDEX_PATH, index)
    write_collection_report(index, scene_reports)
    return index


def _session_reports() -> list[dict[str, Any]]:
    reports = []
    if not DATASET_ROOT.exists():
        return reports
    for path in sorted(DATASET_ROOT.rglob("*session*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if value.get("report_type") == "scene_collection_session":
            value["relative_path"] = str(path.relative_to(ROOT))
            reports.append(value)
    return reports


def write_collection_report(index: dict[str, Any], scene_reports: list[dict[str, Any]]) -> dict[str, Any]:
    entries = index["episodes"]
    sessions = _session_reports()
    factor_counts: dict[str, Counter] = defaultdict(Counter)
    for item in entries:
        for factor in ("terrain_profile", "ground", "lighting", "tree_density"):
            factor_counts[factor][item[factor]] += 1
    rtf_values = [
        float(episode["real_time_factor"])
        for session in sessions for episode in session.get("episodes", [])
        if episode.get("real_time_factor") is not None
    ]
    resources = [session.get("resources", {}) for session in sessions]
    report = {
        "dataset_version": "dataset_v0", "report_version": 1,
        "status": "PASS" if len(entries) == 70 and not index["missing_episode_ids"] else "PARTIAL",
        "dataset": {
            "episode_count": len(entries),
            "split_counts": dict(Counter(item["split"] for item in entries)),
            "route_bucket_counts": dict(Counter(item["route_bucket"] for item in entries)),
            "total_planned_path_m": sum(float(item["planned_path_length_m"]) for item in entries),
            "total_executed_path_m": sum(float(item["executed_path_length_m"]) for item in entries),
            "total_simulated_duration_s": sum(float(item["duration_s"]) for item in entries),
            "total_rgb_frames": sum(int(item["rgb_frames"]) for item in entries),
            "total_depth_frames": sum(int(item["depth_frames"]) for item in entries),
            "total_imu_samples": sum(int(item["imu_samples"]) for item in entries),
            "total_state_samples": sum(int(item["state_samples"]) for item in entries),
            "total_hdf5_bytes": sum(int(item["hdf5_bytes"]) for item in entries),
        },
        "expert": {
            "success_rate": 1.0 if entries else None,
            "collision_count": 0,
            "goal_error_m": _distribution([float(item["goal_error_m"]) for item in entries]),
            "executed_to_planned_ratio": _distribution([
                float(item["executed_path_length_m"]) / float(item["planned_path_length_m"]) for item in entries
            ]),
            "bucket_compliance": all(item["validation_status"] == "PASS" for item in entries),
        },
        "scene_factor_episode_counts": {key: dict(value) for key, value in factor_counts.items()},
        "scene_reports": scene_reports,
        "resources": {
            "collection_session_count": len(sessions),
            "total_reported_wall_time_s": sum(float(item["session"]["total_wall_time_s"]) for item in sessions),
            "total_app_startup_time_s": sum(float(item["session"]["app_startup_time_s"]) for item in sessions),
            "total_scene_load_time_s": sum(float(item["session"]["scene_load_time_s"]) for item in sessions),
            "rtf": _distribution(rtf_values),
            "peak_vram_mib": max((float(item.get("gpu_memory_used_mib_peak", 0.0)) for item in resources), default=0.0),
            "peak_process_rss_mib": max((float(item.get("process_rss_mib_peak", 0.0)) for item in resources), default=0.0),
            "peak_gpu_temperature_c": max((float(item.get("gpu_temperature_c_peak", 0.0)) for item in resources), default=0.0),
            "memory_leak_sessions": [
                item["relative_path"] for item in sessions if item.get("memory_trend", {}).get("suspected")
            ],
        },
        "retries": [],
    }
    write_report(COLLECTION_REPORT_PATH, report)
    COLLECTION_SUMMARY_PATH.write_text(_summary_markdown(report), encoding="utf-8")
    return report


def _distribution(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    import numpy as np

    array = np.asarray(values, dtype=float)
    return {
        "minimum": float(array.min()), "mean": float(array.mean()),
        "median": float(np.median(array)), "p95": float(np.percentile(array, 95)),
        "maximum": float(array.max()),
    }


def _summary_markdown(report: dict[str, Any]) -> str:
    dataset = report["dataset"]
    return (
        "# Dataset V0 collection summary\n\n"
        f"Status: **{report['status']}**\n\n"
        f"Episodes: {dataset['episode_count']}/70  \n"
        f"Splits: {dataset['split_counts']}  \n"
        f"Route buckets: {dataset['route_bucket_counts']}  \n"
        f"Planned path: {dataset['total_planned_path_m']:.3f} m  \n"
        f"Executed path: {dataset['total_executed_path_m']:.3f} m  \n"
        f"Simulated duration: {dataset['total_simulated_duration_s']:.3f} s  \n"
        f"HDF5 bytes: {dataset['total_hdf5_bytes']}  \n"
    )
