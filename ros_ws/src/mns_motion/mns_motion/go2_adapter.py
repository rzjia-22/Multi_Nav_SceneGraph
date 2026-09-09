"""Pure velocity adapters for Go2 locomotion policy command inputs."""

from __future__ import annotations

import math

from .arbiter import Command


def adapt_go2_command(
    command: Command,
    *,
    minimum_gait_speed: float = 0.1,
    minimum_turn_radius: float = 0.25,
) -> Command:
    if minimum_gait_speed < 0 or minimum_turn_radius <= 0:
        raise ValueError("invalid Go2 adapter limits")
    planar = math.hypot(command.x, command.y)
    x, y = command.x, command.y
    if 1e-6 < planar < minimum_gait_speed:
        x *= minimum_gait_speed / planar
        y *= minimum_gait_speed / planar
        planar = minimum_gait_speed
    yaw_limit = planar / minimum_turn_radius if planar > 1e-6 else abs(command.yaw)
    return Command(x, y, max(-yaw_limit, min(yaw_limit, command.yaw)))

