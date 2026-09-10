"""Dataset V0 manifest, scene and episode acceptance checks."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .common import ROOT, load_yaml, scene_paths, stable_hash
from .depth import ALIGNMENT_ALGORITHM, ALIGNMENT_VERSION, align_depth_to_rgb
from .episode import COLLECTOR_VERSION, EPISODE_SCHEMA_VERSION
from .expert import occupancy_grid, path_length
from .forest import generate_scene


def validate_manifest() -> dict[str, Any]:
    manifest = load_yaml(ROOT / "config/datasets/dataset_v0_manifest.yaml")
    scenes = manifest["scenes"]
    assert len(scenes) == 14, "Dataset V0 must contain 14 planned scenes"
    scene_ids = [item["scene_id"] for item in scenes]
    assert len(set(scene_ids)) == 14, "scene IDs are not unique"
    seeds = [int(item["scene_seed"]) for item in scenes]
    assert len(set(seeds)) == 14, "scene seeds are not unique"
    split_counts = {name: sum(item["split"] == name for item in scenes) for name in ("train", "validation", "test")}
    assert split_counts == {"train": 10, "validation": 2, "test": 2}, split_counts
    episodes = [episode for scene in scenes for episode in scene["planned_episodes"]]
    assert len(episodes) == 70, "Dataset V0 must contain 70 planned episodes"
    assert len({item["episode_id"] for item in episodes}) == 70, "episode IDs are not unique"
    buckets = {name: sum(item["target_route_length_bucket"] == name for item in episodes) for name in ("short", "medium", "long")}
    assert buckets == {"short": 21, "medium": 28, "long": 21}, buckets
    memberships: dict[str, set[str]] = {}
    for scene in scenes:
        memberships.setdefault(scene["scene_id"], set()).add(scene["split"])
    assert all(len(splits) == 1 for splits in memberships.values()), "scene-level split leakage"
    return {"status": "PASS", "scene_count": 14, "episode_count": 70, "split_counts": split_counts, "bucket_counts": buckets, "scene_leakage": False}


def validate_scene(scene_path: Path, deterministic: bool = True) -> dict[str, Any]:
    scene = load_yaml(scene_path)
    required = {
        "scene_id", "scene_seed", "split", "terrain", "ground", "lighting",
        "semantics", "trees", "asset_registry", "content_hash",
    }
    assert required <= set(scene), f"scene keys missing: {required - set(scene)}"
    assert scene["schema_version"] == 3
    assert scene["terrain"]["builder"] == "isaaclab.terrains.TerrainImporter"
    assert "waves" not in scene["terrain"]
    statistics = scene["terrain"]["actual_geometry_statistics"]
    assert statistics and statistics["sample_count"] > 0
    assert statistics["slope_max_deg"] >= statistics["slope_p95_deg"] >= statistics["slope_median_deg"]
    assert {"unknown", "ground", "tree_trunk", "foliage", "robot", "other_object", "vegetation"} <= set(scene["semantics"])
    assert scene["semantics"]["robot"] == 6
    assert scene["semantics"]["vegetation"] == scene["semantics"]["other_object"] == 7
    assert len(scene["trees"]) > 0
    assert len({tree["tree_id"] for tree in scene["trees"]}) == len(scene["trees"])
    registry = load_yaml(ROOT / scene["asset_registry"])
    assert all(tree["asset_id"] in registry["trees"] for tree in scene["trees"])
    assert all(not tree["asset_id"].startswith("procedural/") for tree in scene["trees"])
    assert all(tree["collision_proxy"]["shape"] == "hidden_cylinder" for tree in scene["trees"])
    assert scene["ground"]["material_uri"] == registry["ground_materials"][scene["ground"]["profile"]]["uri"]
    stored_hash = scene.pop("content_hash")
    assert stable_hash(scene) == stored_hash, "scene content hash mismatch"
    scene["content_hash"] = stored_hash
    deterministic_match = None
    if deterministic:
        with tempfile.TemporaryDirectory(prefix="mns-scene-check-") as directory:
            generated_yaml = Path(directory) / "scene.yaml"
            regenerated = generate_scene(scene["scene_id"], generated_yaml)
            deterministic_match = stable_hash(regenerated) == stable_hash(scene)
            assert deterministic_match, "scene regeneration differs from versioned scene"
    _, stage_snapshot = scene_paths(scene["scene_id"])
    assert stage_snapshot.is_file(), "Isaac-built scene.usda snapshot is missing"
    stage_text = stage_snapshot.read_text(encoding="utf-8")
    assert "Blue_Berry_Elder.usd" in stage_text or "Gray_Birch.usd" in stage_text
    assert 'def Cylinder "Trunk"' not in stage_text
    assert 'def Sphere "Foliage"' not in stage_text
    return {
        "status": "PASS",
        "scene_id": scene["scene_id"],
        "tree_count": len(scene["trees"]),
        "tree_asset_ids": sorted({tree["asset_id"] for tree in scene["trees"]}),
        "content_hash": stored_hash,
        "terrain_statistics": statistics,
        "deterministic_regeneration": deterministic_match,
        "stage_snapshot": str(stage_snapshot.relative_to(ROOT)),
        "primitive_tree_visuals": False,
    }


def validate_episode(path: Path, plan_path: Path) -> dict[str, Any]:
    import h5py

    plan = load_yaml(plan_path)
    scene_path, _ = scene_paths(plan["scene_id"])
    scene = load_yaml(scene_path)
    robot = load_yaml(ROOT / "config/robots/diablo_standing.yaml")
    manifest = load_yaml(ROOT / "config/datasets/dataset_v0_manifest.yaml")
    manifest_scene = next(item for item in manifest["scenes"] if item["scene_id"] == plan["scene_id"])
    manifest_episode = next(item for item in manifest_scene["planned_episodes"] if item["episode_id"] == plan["episode_id"])
    assert plan["split"] == scene["split"] == manifest_scene["split"], "episode/scene split mismatch"
    assert plan["target_route_length_bucket"] == manifest_episode["target_route_length_bucket"], "episode length bucket differs from manifest"
    with h5py.File(path, "r") as episode:
        assert int(episode.attrs["schema_version"]) == EPISODE_SCHEMA_VERSION
        metadata = json.loads(str(episode.attrs["metadata_json"]))
        assert metadata["schema_version"] == EPISODE_SCHEMA_VERSION
        assert metadata["collector_version"] == COLLECTOR_VERSION
        assert metadata["episode_id"] == plan["episode_id"]
        assert metadata["success"] is True
        assert metadata["scene_schema_version"] == scene["schema_version"]
        assert metadata["scene_content_hash"] == scene["content_hash"] == plan["scene_content_hash"]
        required = (
            "state/timestamp_s", "state/position_xyz", "state/orientation_xyzw",
            "state/linear_velocity_xyz", "state/angular_velocity_xyz", "state/command_vw",
            "imu/timestamp_s", "imu/linear_acceleration_xyz", "imu/angular_velocity_xyz",
            "sensors/timestamp_s", "sensors/rgb", "sensors/depth_raw_z16",
            "sensors/camera_pose_xyz_xyzw",
            "calibration/rgb_intrinsics", "calibration/depth_intrinsics",
            "calibration/depth_to_rgb_translation_m", "calibration/depth_to_rgb_rotation_xyzw",
            "expert/global_path_xyz",
            "calibration/depth_scale_m", "calibration/rgb_resolution_wh",
        )
        for name in required:
            assert name in episode, f"missing HDF5 dataset: {name}"
            assert not np.isnan(np.asarray(episode[name])).any(), f"NaN in {name}"
        state_time = np.asarray(episode["state/timestamp_s"])
        imu_time = np.asarray(episode["imu/timestamp_s"])
        sensor_time = np.asarray(episode["sensors/timestamp_s"])
        assert all(np.all(np.diff(values) > 0.0) for values in (state_time, imu_time, sensor_time)), "timestamps must be strictly increasing"
        rgb_count = episode["sensors/rgb"].shape[0]
        assert "sensors/depth_raw_m" not in episode
        assert "sensors/depth_aligned_to_rgb_m" not in episode
        assert episode["sensors/depth_raw_z16"].shape[0] == rgb_count
        assert episode["sensors/depth_raw_z16"].dtype == np.dtype("uint16")
        rgb = np.asarray(episode["sensors/rgb"])
        depth_z16 = np.asarray(episode["sensors/depth_raw_z16"])
        depth_scale = float(np.asarray(episode["calibration/depth_scale_m"]))
        assert 0.0 < depth_scale <= 0.001
        assert episode["calibration"].attrs["alignment_algorithm"] == ALIGNMENT_ALGORITHM
        assert int(episode["calibration"].attrs["alignment_version"]) == ALIGNMENT_VERSION
        assert episode["calibration"].attrs["invalid_depth_convention"] == "zero_is_invalid"
        aligned_depth = align_depth_to_rgb(
            depth_z16, depth_scale,
            np.asarray(episode["calibration/depth_intrinsics"]),
            np.asarray(episode["calibration/rgb_intrinsics"]),
            np.asarray(episode["calibration/depth_to_rgb_translation_m"]),
            np.asarray(episode["calibration/depth_to_rgb_rotation_xyzw"]),
            tuple(np.asarray(episode["calibration/rgb_resolution_wh"], dtype=int)),
        )
        assert int(rgb.max()) - int(rgb.min()) >= 80, "RGB dynamic range is suspiciously low"
        assert float(np.abs(rgb[-1].astype(float) - rgb[0].astype(float)).mean()) >= 2.0, "RGB frames appear stale"
        depth_valid_fraction = float((aligned_depth > 0.0).mean())
        assert 0.05 <= depth_valid_fraction <= 0.98, f"implausible aligned depth valid fraction: {depth_valid_fraction}"
        assert episode["calibration/rgb_intrinsics"].shape == (3, 3)
        assert episode["calibration/depth_intrinsics"].shape == (3, 3)
        positions = np.asarray(episode["state/position_xyz"])
        orientations = np.asarray(episode["state/orientation_xyzw"])
        assert float(np.max(np.abs(orientations[:, :2]))) > 1.0e-4, "body roll/pitch never followed terrain normal"
        executed_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
        goal_error = float(np.linalg.norm(positions[-1, :2] - np.asarray(plan["goal_pose_xyz"][:2])))
        assert goal_error <= float(plan["controller"]["goal_tolerance_m"]) + 0.08, f"goal error {goal_error:.3f} m"
        assert executed_length <= 10.5, f"executed path exceeds V0 limit: {executed_length:.3f} m"
        grid = occupancy_grid(scene, robot, clearance_key="collision_check_radius_m")
        for point in positions[:, :2]:
            cell = grid.world_to_cell(tuple(point))
            assert not grid.occupied[cell[1], cell[0]], f"executed trajectory intersects inflated obstacle at {point}"
        return {
            "status": "PASS", "episode_id": metadata["episode_id"], "success": True,
            "duration_s": float(state_time[-1] - state_time[0]), "rgb_frames": int(rgb_count),
            "depth_frames": int(rgb_count), "imu_samples": int(len(imu_time)), "pose_samples": int(len(state_time)),
            "planned_path_length_m": float(plan["planned_path_length_m"]), "executed_path_length_m": executed_length,
            "goal_error_m": goal_error, "collision_free_in_privileged_map": True,
            "rgb_value_range": [int(rgb.min()), int(rgb.max())],
            "aligned_depth_valid_fraction": depth_valid_fraction,
            "episode_schema_version": EPISODE_SCHEMA_VERSION,
            "depth_dtype": "uint16",
            "depth_scale_m": depth_scale,
        }


def validate_runtime_contract() -> dict[str, Any]:
    """Use Python syntax structure to guard the single active scene path."""
    import ast

    runtime_path = ROOT / "ros_ws/src/mns_simulation/mns_simulation/research_dataset_runtime.py"
    tree = ast.parse(runtime_path.read_text(encoding="utf-8"))
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            calls.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            calls.append(node.func.attr)
    assert "build_research_forest" in calls, "collector does not call the shared Research Forest builder"
    forbidden = {"terrain_height", "Mesh", "Cylinder", "Sphere"}
    active_forbidden = sorted(forbidden.intersection(calls))
    assert not active_forbidden, f"obsolete collector construction calls remain: {active_forbidden}"
    return {"status": "PASS", "shared_builder_call": True, "obsolete_construction_calls": []}


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
