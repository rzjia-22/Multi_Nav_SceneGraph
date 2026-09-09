#!/usr/bin/env python3
"""Fast static validation for repository contracts and ROS metadata."""

from __future__ import annotations

import ast
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
    for base in (ROOT / "ros_ws" / "src", ROOT / "tools", ROOT / "tests"):
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
    if len(phase1.robots) != 1:
        fail("Phase 1 must contain exactly one robot")
    if [robot.kind.value for robot in phase2.robots].count("go2") != 2:
        fail("Phase 2 must contain two Go2")
    if [robot.kind.value for robot in phase2.robots].count("uav") != 2:
        fail("Phase 2 must contain two UAV")
    all_frames = []
    for robot in phase2.robots:
        names = RobotNames(robot.robot_id)
        all_frames.extend([names.odom_frame, names.base_frame, names.camera_link_frame, names.optical_frame, names.map_frame])
    if len(all_frames) != len(set(all_frames)):
        fail("Phase 2 frame IDs collide")


def main() -> int:
    validate_yaml()
    validate_python()
    validate_packages()
    validate_configs()
    print("repository validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

