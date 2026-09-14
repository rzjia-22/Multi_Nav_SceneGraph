from pathlib import Path
import ast
import json

import numpy as np
import pytest
import yaml

from mns_simulation.research_robot import episode_mount_variation, research_rig_pose


ROOT = Path(__file__).resolve().parents[1]


class FlatSurface:
    def height_and_normal(self, _x, _y):
        return 0.25, np.asarray([0.0, 0.0, 1.0])


def test_closed_loop_scope_is_validation_only_and_uses_frozen_contracts():
    config = yaml.safe_load((ROOT / "config/navigation/navdiffusion_v0_closed_loop.yaml").read_text())
    assert config["scope"] == "validation_only"
    assert config["test_split_allowed"] is False
    assert config["robot_id"] == "diablo_1"
    assert config["checkpoint_sha256"] == "7b3bca67af26c2d839555e9b27a7a420762b572fa88664a227986174d9a68ae0"
    assert config["navigation"]["prediction_waypoints"] == 32
    assert config["navigation"]["control_waypoints"] == 8
    assert config["navigation"]["planning_rate_hz"] == 2.0
    full = config["gates"]["full_validation"]["scenes"]
    episodes = [episode for values in full.values() for episode in values]
    assert len(episodes) == 10
    assert len(set(episodes)) == 10
    assert all(value.startswith("validation_scene_") for value in episodes)


def test_research_camera_pose_uses_shared_deterministic_mount_and_surface():
    robot = yaml.safe_load((ROOT / "config/robots/diablo_standing.yaml").read_text())
    sensor = yaml.safe_load((ROOT / "config/sensors/d435i_navigation_v0.yaml").read_text())
    first = episode_mount_variation(robot, 51011, "validation_scene_000_episode_000")
    second = episode_mount_variation(robot, 51011, "validation_scene_000_episode_000")
    assert first == second
    rig = research_rig_pose(FlatSurface(), 1.0, -2.0, 0.4, 3.0, robot, sensor, first)
    assert rig.ground_z == 0.25
    np.testing.assert_allclose(rig.body_rotation[:, 2], [0.0, 0.0, 1.0], atol=1.0e-12)
    assert rig.rgb_position[2] > rig.ground_z + 0.4
    assert np.linalg.norm(rig.depth_position - rig.rgb_position) == pytest.approx(0.015)


def test_d435i_closed_loop_dimensions_keep_distinct_raw_camera_models():
    sensor = yaml.safe_load((ROOT / "config/sensors/d435i_navigation_v0.yaml").read_text())
    profile = sensor["application_profile"]
    assert profile["rgb"]["resolution"] == [640, 360]
    assert profile["depth_raw"]["resolution"] == [848, 480]
    assert profile["depth_aligned_to_rgb"]["resolution"] == [640, 360]
    assert profile["rgb"]["fov_deg"] != profile["depth_raw"]["fov_deg"]


def test_isaac_runtime_keeps_standard_ros_boundary_and_shared_forest_builder():
    runtime = ROOT / "ros_ws/src/mns_simulation/mns_simulation/research_navigation_runtime.py"
    tree = ast.parse(runtime.read_text(encoding="utf-8"))
    imports = set()
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
            imported_names.update(alias.name for alias in node.names)
    assert not any(name.startswith("mns_interfaces") for name in imports)
    assert "build_research_forest" in imported_names


def test_recorded_closed_loop_gate_is_fail_fast_and_test_sealed():
    path = ROOT / "artifacts/navdiffusion_v0_closed_loop/aggregate_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["test_split_used"] is False
    assert report["mapping_enabled"] is False
    assert report["gate_a"]["all_pass"] is True
    assert report["gate_a"]["success_count"] == 1
    assert report["gate_b"]["all_pass"] is False
    assert report["gate_b"]["compose_return_codes"] == [2]
    assert report["gate_b"]["episode_count"] == 2
    assert report["gate_b"]["collision_count"] == 1
    assert report["full_validation"] is None
    assert report["readiness"] == "READY_FOR_REAL_SENSOR_ONLY"
    failed = next(item for item in report["gate_b"]["episodes"] if not item["success"])
    assert failed["failure_class"] == "MODEL"
    assert failed["collision_tree_id"] == "tree_021"
