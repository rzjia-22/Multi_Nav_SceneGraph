#!/usr/bin/env python3
"""Fast static validation for repository contracts and ROS metadata."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import yaml


ROOT = Path(__file__).resolve().parents[1]


def fail(message: str) -> None:
    raise AssertionError(message)


def validate_yaml() -> None:
    for path in sorted(ROOT.glob("config/**/*.yaml")):
        with path.open("r", encoding="utf-8") as stream:
            if yaml.safe_load(stream) is None:
                fail(f"empty YAML: {path.relative_to(ROOT)}")


def validate_python() -> None:
    for base in (ROOT / "ros_ws" / "src", ROOT / "research_data", ROOT / "tools", ROOT / "tests"):
        for path in sorted(base.rglob("*.py")):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def validate_packages() -> None:
    packages = []
    for path in sorted((ROOT / "ros_ws" / "src").glob("*/package.xml")):
        tree = ET.parse(path)
        name = tree.findtext("name")
        if not name or name in packages:
            fail(f"missing or duplicate package name in {path}")
        packages.append(name)
        if tree.findtext("license") != "BSD-3-Clause":
            fail(f"unexpected license in {name}")
    expected = {
        "mns_interfaces", "mns_core", "mns_navigation", "mns_mapping",
        "mns_motion", "mns_multi_robot", "mns_simulation", "mns_bringup",
    }
    if set(packages) != expected:
        fail(f"package set differs: {set(packages) ^ expected}")


def validate_configs() -> None:
    sys.path.insert(0, str(ROOT / "ros_ws" / "src" / "mns_core"))
    from mns_core.models import SystemSpec
    from mns_core.naming import RobotNames

    phase1 = SystemSpec.load(ROOT / "config" / "robots" / "phase1.yaml")
    phase2 = SystemSpec.load(ROOT / "config" / "robots" / "phase2.yaml")
    motion_acceptance = SystemSpec.load(
        ROOT / "config" / "robots" / "phase1_motion_acceptance.yaml"
    )
    if len(phase1.robots) != 1:
        fail("Phase 1 must contain exactly one robot")
    if [robot.kind.value for robot in phase2.robots].count("go2") != 2:
        fail("Phase 2 must contain two Go2")
    if [robot.kind.value for robot in phase2.robots].count("uav") != 2:
        fail("Phase 2 must contain two UAV")
    if len(motion_acceptance.robots) != 1 or motion_acceptance.robots[0].kind.value != "go2":
        fail("motion acceptance must contain exactly one Go2")
    all_frames = []
    for robot in phase2.robots:
        names = RobotNames(robot.robot_id)
        all_frames.extend([names.odom_frame, names.base_frame, names.camera_link_frame, names.optical_frame, names.map_frame])
    if len(all_frames) != len(set(all_frames)):
        fail("Phase 2 frame IDs collide")
    scene_path = ROOT / "config" / "simulation" / "forest.yaml"
    with scene_path.open("r", encoding="utf-8") as stream:
        scene = yaml.safe_load(stream)
    if min(
        float(scene["ground_size"]),
        float(scene["trunk_radius"]),
        float(scene["trunk_height"]),
        float(scene["foliage_radius"]),
    ) <= 0:
        fail("forest dimensions must be positive")
    tree_ids = [str(item["id"]) for item in scene["trees"]]
    if not tree_ids or len(tree_ids) != len(set(tree_ids)):
        fail("forest tree IDs must be non-empty and unique")
    for item in scene["trees"]:
        if len(item.get("position", [])) != 2:
            fail(f"invalid tree position: {item}")


def validate_dataset_v0() -> None:
    sys.path.insert(0, str(ROOT))
    from research_data.validation import validate_manifest

    report = validate_manifest()
    if report["scene_leakage"] or report["scene_count"] != 14 or report["episode_count"] != 70:
        fail("Dataset V0 manifest contract failed")
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    if "*.h5 filter=lfs" not in attributes or "*.usd filter=lfs" not in attributes:
        fail("Dataset V0 binary artifacts are not covered by Git LFS")


def validate_navdiffusion_v0_baseline() -> None:
    path = ROOT / "artifacts/baselines/navdiffusion_v0/results.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "baseline_id", "baseline_schema_version", "dataset_version", "training",
        "closed_loop_validation", "closed_loop_test", "checkpoint", "data_governance",
        "final_readiness",
    }
    if required - set(record):
        fail(f"NavDiffusion V0 baseline record lacks {sorted(required - set(record))}")
    if record["baseline_id"] != "navdiffusion_v0" or record["final_readiness"] != "NOT_READY":
        fail("unexpected NavDiffusion V0 baseline identity/readiness")
    checkpoint = ROOT / record["checkpoint"]["path"]
    if not checkpoint.is_file() or checkpoint.stat().st_size != record["checkpoint"]["size_bytes"]:
        fail("NavDiffusion V0 checkpoint reference is missing or has the wrong size")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != record["checkpoint"]["sha256"]:
        fail("NavDiffusion V0 checkpoint SHA256 differs from the frozen baseline")
    if not record["data_governance"]["test_not_untouched_for_future_v1"]:
        fail("Dataset V0 test-split governance is not recorded")


def validate_nomad_zero_shot_baseline() -> None:
    path = ROOT / "artifacts/baselines/nomad_zero_shot_offline/results.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("evaluation_id") != "nomad_zero_shot_offline_dataset_v0":
        fail("unexpected NoMaD offline baseline identity")
    if record.get("status") != "PASS" or record.get("model_readiness") != "OFFLINE_ONLY":
        fail("unexpected NoMaD offline status/readiness")
    episodes = record.get("episodes", [])
    if len(episodes) != 70 or len({item["episode_id"] for item in episodes}) != 70:
        fail("NoMaD offline baseline must contain 70 unique tasks")
    if sum(int(item["anchor_count"]) for item in episodes) != record.get("anchor_count"):
        fail("NoMaD offline anchor count is inconsistent")
    split_counts = {
        split: sum(item["split"] == split for item in episodes)
        for split in ("train", "validation", "test")
    }
    if split_counts != {"train": 50, "validation": 10, "test": 10}:
        fail("NoMaD offline split counts are inconsistent")
    if record["interpretation"]["closed_loop_success_rate"] is not None:
        fail("NoMaD offline result must not claim closed-loop success")
    if record["summary"]["route_viability_proxy"]["nominal"]["success_count"] != 20:
        fail("unexpected NoMaD full nominal proxy result")
    held_out = record["held_out_validation_and_test"]
    if held_out["episode_count"] != 20:
        fail("NoMaD held-out comparison slice must contain 20 tasks")
    if held_out["route_viability_proxy"]["nominal"]["success_count"] != 3:
        fail("unexpected NoMaD held-out nominal proxy result")
    config_path = ROOT / record["config_path"]
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if digest != record["config_sha256"]:
        fail("NoMaD result configuration hash differs")


def main() -> int:
    validate_yaml()
    validate_python()
    validate_packages()
    validate_configs()
    validate_dataset_v0()
    validate_navdiffusion_v0_baseline()
    validate_nomad_zero_shot_baseline()
    print("repository validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
