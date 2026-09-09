from pathlib import Path

import pytest

from mns_core.models import SystemSpec
from mns_core.naming import RobotNames
from mns_core.sensor_contract import ImageContract, StreamWindow


ROOT = Path(__file__).resolve().parents[1]


def test_phase_rosters_and_frame_isolation():
    phase1 = SystemSpec.load(ROOT / "config/robots/phase1.yaml")
    phase2 = SystemSpec.load(ROOT / "config/robots/phase2.yaml")
    assert len(phase1.robots) == 1
    assert [robot.kind.value for robot in phase2.robots].count("go2") == 2
    assert [robot.kind.value for robot in phase2.robots].count("uav") == 2
    frames = []
    for robot in phase2.robots:
        names = RobotNames(robot.robot_id)
        frames += [names.odom_frame, names.base_frame, names.optical_frame, names.map_frame]
    assert len(frames) == len(set(frames))


def test_sensor_contract_and_rate():
    contract = ImageContract("rgb8", "32FC1", "16UC1", "go2_1/camera_optical_frame", "go2_1/camera_optical_frame", "go2_1/camera_optical_frame")
    contract.validate("go2_1/camera_optical_frame")
    with pytest.raises(ValueError):
        ImageContract("bgr8", "32FC1", "16UC1", "a", "a", "a").validate("a")
    window = StreamWindow()
    for index in range(11):
        window.observe(index / 10)
    assert window.rate_hz == pytest.approx(10.0)

