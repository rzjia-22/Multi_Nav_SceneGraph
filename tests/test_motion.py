import pytest

from mns_motion.arbiter import Command, CommandArbiter
from mns_motion.go2_adapter import adapt_go2_command


def test_priority_timeout_and_bounds():
    arbiter = CommandArbiter(max_linear_speed=1.0, max_angular_speed=1.0)
    arbiter.update("navigation", Command(2.0, 0.0, 2.0), 1.0)
    assert arbiter.select(1.1) == ("navigation", Command(1.0, 0.0, 1.0))
    arbiter.update("safety", Command(-0.2, 0.0, 0.5), 1.1)
    assert arbiter.select(1.15)[0] == "safety"
    assert arbiter.select(2.0) == ("deadman", Command.zero())


def test_go2_gait_adapter():
    command = adapt_go2_command(Command(0.01, 0.0, 1.0), minimum_gait_speed=0.1, minimum_turn_radius=0.5)
    assert command.x == pytest.approx(0.1)
    assert command.yaw == pytest.approx(0.2)

