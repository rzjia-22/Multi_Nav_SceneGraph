"""Pure pursuit over a common high-level velocity interface."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from .coverage import Point2D


@dataclass(frozen=True)
class MotionCommand:
    linear_x: float
    linear_y: float
    angular_z: float

    @classmethod
    def zero(cls) -> "MotionCommand":
        return cls(0.0, 0.0, 0.0)


def wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


class PurePursuitFollower:
    def __init__(
        self,
        lookahead_distance: float = 0.8,
        max_linear_speed: float = 0.7,
        max_angular_speed: float = 1.0,
        goal_tolerance: float = 0.25,
    ) -> None:
        if min(lookahead_distance, max_linear_speed, max_angular_speed, goal_tolerance) <= 0:
            raise ValueError("follower parameters must be positive")
        self.lookahead_distance = lookahead_distance
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.goal_tolerance = goal_tolerance

    def command(
        self,
        position: Point2D,
        yaw: float,
        trajectory: Sequence[Point2D],
        *,
        stop_at_goal: bool = True,
    ) -> MotionCommand:
        if len(trajectory) < 2:
            return MotionCommand.zero()
        goal = trajectory[-1]
        if stop_at_goal and math.hypot(goal.x - position.x, goal.y - position.y) <= self.goal_tolerance:
            return MotionCommand.zero()
        closest = min(
            range(len(trajectory)),
            key=lambda i: math.hypot(trajectory[i].x - position.x, trajectory[i].y - position.y),
        )
        remaining = self.lookahead_distance
        target = trajectory[-1]
        for start, end in zip(trajectory[closest:-1], trajectory[closest + 1 :]):
            length = math.hypot(end.x - start.x, end.y - start.y)
            if length <= 1e-9:
                continue
            if remaining <= length:
                alpha = remaining / length
                target = Point2D(
                    start.x + alpha * (end.x - start.x),
                    start.y + alpha * (end.y - start.y),
                )
                break
            remaining -= length
        bearing = math.atan2(target.y - position.y, target.x - position.x)
        error = wrap_angle(bearing - yaw)
        angular = max(-self.max_angular_speed, min(self.max_angular_speed, 1.2 * error))
        alignment = max(0.0, math.cos(error))
        turn_scale = 1.0 - 0.8 * min(1.0, abs(angular) / self.max_angular_speed)
        return MotionCommand(self.max_linear_speed * alignment * turn_scale, 0.0, angular)

