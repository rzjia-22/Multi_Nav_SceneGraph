import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from mns_navigation.navdiffusion_v0.data import (
    NavDiffusionWindowDataset,
    _episode_windows,
)
from mns_navigation.navdiffusion_v0.preprocessing import (
    NavDiffusionPreprocessor,
    local_coordinates,
    world_coordinates,
)


ROOT = Path(__file__).resolve().parents[1]


def test_body_planar_coordinates_are_exact_inverses():
    points = np.asarray([[2.0, 4.0], [-1.0, 3.0], [0.2, -0.7]])
    origin = np.asarray([0.4, 1.1])
    yaw = 0.73
    local = local_coordinates(points, origin, yaw)
    np.testing.assert_allclose(world_coordinates(local, origin, yaw), points, atol=1.0e-12)


def test_window_contract_drops_short_history_and_pads_only_future_tail():
    sensor_time = np.arange(0.0, 1.01, 0.1)
    state_time = np.arange(0.0, 1.001, 0.02)
    position = np.column_stack((state_time, np.zeros_like(state_time), np.zeros_like(state_time)))
    orientation = np.tile(np.asarray([0.0, 0.0, 0.0, 1.0]), (len(state_time), 1))
    windows = _episode_windows(
        sensor_time,
        state_time,
        position,
        orientation,
        np.asarray([2.0, 0.0]),
        history=5,
        horizon=4,
        future_step_s=0.1,
        goal_scale_m=10.0,
    )
    assert windows["anchor_indices"].tolist() == list(range(4, 11))
    np.testing.assert_allclose(windows["future_local_m"][0, :, 0], [0.1, 0.2, 0.3, 0.4])
    assert windows["future_padding_points"][-1] == 4
    np.testing.assert_allclose(windows["future_local_m"][-1], 0.0, atol=1.0e-12)


def test_preprocessing_contract_is_frozen_and_round_trips_metadata():
    raw = yaml.safe_load((ROOT / "config/models/navigation_input_v0.yaml").read_text())
    assert raw["inputs"]["history_frames"] == 5
    assert raw["inputs"]["image_size_wh"] == [160, 90]
    assert raw["inputs"]["depth"]["invalid_policy"] == "fill_far"
    assert raw["inputs"]["depth"]["maximum_m"] == 10.0
    assert raw["augmentation"]["enabled"] is False
    preprocessor = NavDiffusionPreprocessor(
        width=160,
        height=90,
        history_length=5,
        rgb_mean=(0.485, 0.456, 0.406),
        rgb_std=(0.229, 0.224, 0.225),
        depth_max_m=10.0,
        depth_invalid_fill_m=10.0,
        depth_mean=0.5,
        depth_std=0.2,
        goal_scale_m=10.0,
        trajectory_bound_m=2.5,
    )
    assert NavDiffusionPreprocessor.from_dict(preprocessor.to_dict()) == preprocessor


def test_training_dataset_refuses_test_split_before_accessing_cache():
    with pytest.raises(ValueError, match="never test"):
        NavDiffusionWindowDataset(
            "test",
            ROOT / "config/models/navigation_input_v0.yaml",
            ROOT / "config/models/navdiffusion_v0.yaml",
        )


def test_goal_and_trajectory_use_same_anchor_body_frame():
    yaw = math.pi / 2
    origin = np.asarray([3.0, 4.0])
    future = np.asarray([[3.0, 5.0], [2.0, 5.0]])
    local = local_coordinates(future, origin, yaw)
    np.testing.assert_allclose(local, [[1.0, 0.0], [1.0, 1.0]], atol=1.0e-12)
