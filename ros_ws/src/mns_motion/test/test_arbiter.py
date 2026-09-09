from mns_motion.arbiter import Command, CommandArbiter


def test_deadman_and_safety_priority():
    arbiter = CommandArbiter(navigation_timeout=0.3, safety_timeout=0.2)
    arbiter.update("navigation", Command(0.5, 0.0, 0.0), 1.0)
    arbiter.update("safety", Command(-0.2, 0.0, 0.4), 1.1)
    assert arbiter.select(1.15)[0] == "safety"
    assert arbiter.select(2.0) == ("deadman", Command.zero())
