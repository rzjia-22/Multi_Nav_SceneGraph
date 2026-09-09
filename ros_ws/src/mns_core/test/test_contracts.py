import pytest

from mns_core.models import Pose, RobotKind, RobotSpec
from mns_core.naming import RobotNames


def test_robot_configuration_produces_isolated_names_and_backend_default():
    robot = RobotSpec("go2_1", RobotKind.GO2, "coverage", Pose(0, 0, 0.45, 0), "mission")
    names = RobotNames(robot.robot_id)
    assert names.sensor_topics["depth"] == "/go2_1/camera/depth/image_rect"
    assert names.base_frame == "go2_1/base_link"
    assert robot.motion_backend == "isaac_rl"
    with pytest.raises(ValueError):
        RobotNames("invalid-id")
