"""Local velocity safety and bounded command-response recovery."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

from .path_follower import MotionCommand


@dataclass(frozen=True)
class SafetyDecision:
    state: str
    command: MotionCommand | None
    detail: str = ""


class DepthSafetyController:
    def __init__(
        self,
        stop_distance: float = 0.7,
        release_distance: float = 0.9,
        reverse_speed: float = 0.0,
        turn_speed: float = 0.5,
    ) -> None:
        if not 0 < stop_distance < release_distance:
            raise ValueError("stop distance must be positive and below release distance")
        self.stop_distance = stop_distance
        self.release_distance = release_distance
        self.reverse_speed = abs(reverse_speed)
        self.turn_speed = abs(turn_speed)
        self._active = False
        self._turn_sign = 1.0

    def decide(self, left_distance: float, center_distance: float, right_distance: float) -> SafetyDecision:
        nearest = min(left_distance, center_distance, right_distance)
        if self._active and nearest >= self.release_distance:
            self._active = False
            return SafetyDecision("released", None)
        if not self._active and nearest > self.stop_distance:
            return SafetyDecision("clear", None)
        self._active = True
        self._turn_sign = -1.0 if left_distance < right_distance else 1.0
        # The migrated Go2 actor was trained for forward velocity tracking and
        # is not stable under combined reverse/yaw commands.  The conservative
        # default therefore stops translation and turns the camera away from
        # the obstacle. Backends with validated reverse motion may opt in by
        # constructing the controller with a non-zero reverse_speed.
        reverse = -self.reverse_speed if center_distance < self.stop_distance * 0.7 else 0.0
        return SafetyDecision(
            "avoidance",
            MotionCommand(reverse, 0.0, self._turn_sign * self.turn_speed),
            f"nearest depth {nearest:.3f} m",
        )


class StallRecovery:
    """Detect little odometric response to commanded motion and recover."""

    def __init__(
        self,
        trigger_window: float = 5.0,
        minimum_motion: float = 0.05,
        reverse_duration: float = 1.5,
        turn_duration: float = 2.0,
        escape_duration: float = 1.5,
    ) -> None:
        if min(trigger_window, minimum_motion, reverse_duration, turn_duration, escape_duration) <= 0:
            raise ValueError("recovery parameters must be positive")
        self.trigger_window = trigger_window
        self.minimum_motion = minimum_motion
        self.reverse_duration = reverse_duration
        self.turn_duration = turn_duration
        self.escape_duration = escape_duration
        self._history: deque[tuple[float, float, float]] = deque()
        self._start: float | None = None
        self._turn_sign = 1.0

    def update(self, now: float, x: float, y: float, commanded: bool) -> SafetyDecision:
        self._history.append((now, x, y))
        while len(self._history) > 1 and self._history[1][0] <= now - self.trigger_window:
            self._history.popleft()
        if self._start is not None:
            elapsed = now - self._start
            if elapsed < self.reverse_duration:
                return SafetyDecision("stall_reverse", MotionCommand(-0.18, 0.0, 0.0))
            if elapsed < self.reverse_duration + self.turn_duration:
                return SafetyDecision("stall_turn", MotionCommand(0.0, 0.0, self._turn_sign * 0.55))
            if elapsed < self.reverse_duration + self.turn_duration + self.escape_duration:
                return SafetyDecision("stall_escape", MotionCommand(0.18, 0.0, self._turn_sign * 0.15))
            self._start = None
            self._history.clear()
            self._history.append((now, x, y))
            return SafetyDecision("recovered", None)
        if not commanded or len(self._history) < 2:
            return SafetyDecision("monitoring", None)
        first = self._history[0]
        if now - first[0] >= self.trigger_window * 0.95 and math.hypot(x - first[1], y - first[2]) < self.minimum_motion:
            self._start = now
            self._turn_sign *= -1.0
            return SafetyDecision("stall_reverse", MotionCommand(-0.18, 0.0, 0.0))
        return SafetyDecision("monitoring", None)
