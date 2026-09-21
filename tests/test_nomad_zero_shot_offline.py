from pathlib import Path

import numpy as np
import pytest
import yaml

from research_data.nomad_zero_shot.metrics import (
    densify_polyline,
    local_to_world,
    maximum_true_streak,
    pairwise_endpoint_diversity,
    point_to_polyline_distances,
    trajectory_clearance,
    world_to_local,
)


ROOT = Path(__file__).resolve().parents[1]


def test_nomad_offline_config_freezes_official_assets_and_primary_input_adapter():
    config = yaml.safe_load(
        (ROOT / "config/navigation/nomad_zero_shot_offline.yaml").read_text(encoding="utf-8")
    )
    assert config["dataset"]["expected_episode_count"] == 70
    assert config["nomad"]["source_revision"] == "dca79815b704e5aa9c6bdc3082351f9e3b2848c2"
    assert config["nomad"]["checkpoint_sha256"] == (
        "70f79b8262527e20e56ced64a3e3d7ef91855bc9e7c3fa348d78edcb83c6a333"
    )
    assert config["input_adapter"]["context_frames"] == 4
    assert config["input_adapter"]["model_rate_hz"] == 4.0
    assert config["input_adapter"]["local_goal_horizon_s"] == 2.0
    assert config["nomad"]["samples_per_observation"] == 8


def test_local_world_coordinate_transform_round_trip():
    local = np.asarray([[1.0, 0.2], [0.3, -0.8], [-0.2, 0.1]])
    origin = np.asarray([3.2, -1.7])
    world = local_to_world(local, origin, yaw=0.71)
    assert np.allclose(world_to_local(world, origin, yaw=0.71), local, atol=1.0e-12)


def test_continuous_clearance_detects_collision_between_waypoints():
    # 两个离散路点都离树较远，但连接线穿过树干；指标必须按连续折线判碰撞。
    trajectory = np.asarray([[-1.0, 0.0], [1.0, 0.0]])
    clearance = trajectory_clearance(
        trajectory,
        tree_centres_xy=np.asarray([[0.0, 0.0]]),
        tree_radii_m=np.asarray([0.2]),
        robot_radius_m=0.42,
        scene_extent_m=(10.0, 10.0),
    )
    assert clearance == pytest.approx(-0.62)


def test_polyline_distance_uses_segments_not_only_vertices():
    path = np.asarray([[0.0, 0.0], [2.0, 0.0]])
    distances = point_to_polyline_distances(np.asarray([[1.0, 0.3]]), path)
    assert distances[0] == pytest.approx(0.3)


def test_corridor_densification_limits_segment_spacing():
    dense = densify_polyline(np.asarray([[0.0, 0.0], [0.12, 0.0]]), 0.05)
    assert np.all(np.linalg.norm(np.diff(dense, axis=0), axis=1) <= 0.05 + 1.0e-12)
    assert np.array_equal(dense[[0, -1]], np.asarray([[0.0, 0.0], [0.12, 0.0]]))


def test_sequence_and_multimodal_metrics_are_deterministic():
    assert maximum_true_streak([False, True, True, False, True]) == 2
    trajectories = np.asarray([
        [[0.0, 0.0], [1.0, 0.0]],
        [[0.0, 0.0], [0.0, 1.0]],
        [[0.0, 0.0], [-1.0, 0.0]],
    ])
    expected = (np.sqrt(2.0) + 2.0 + np.sqrt(2.0)) / 3.0
    assert pairwise_endpoint_diversity(trajectories) == pytest.approx(expected)
