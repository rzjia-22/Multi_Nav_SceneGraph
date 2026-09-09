"""Deterministic source-priority arbitration with a dead-man timeout."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Command:
    x: float
    y: float
    yaw: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.x, self.y, self.yaw)):
            raise ValueError("velocity command values must be finite")

    @classmethod
    def zero(cls) -> "Command":
        return cls(0.0, 0.0, 0.0)


@dataclass
class _StampedCommand:
    command: Command
    stamp: float


class CommandArbiter:
    def __init__(
        self,
        navigation_timeout: float = 0.3,
        safety_timeout: float = 0.2,
        max_linear_speed: float = 1.0,
        max_angular_speed: float = 1.5,
    ) -> None:
        if min(navigation_timeout, safety_timeout, max_linear_speed, max_angular_speed) <= 0:
            raise ValueError("arbiter limits and timeouts must be positive")
        self.navigation_timeout = navigation_timeout
        self.safety_timeout = safety_timeout
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self._navigation: _StampedCommand | None = None
        self._safety: _StampedCommand | None = None

    def update(self, source: str, command: Command, stamp: float) -> None:
        item = _StampedCommand(command, float(stamp))
        if source == "navigation":
            self._navigation = item
        elif source == "safety":
            self._safety = item
        else:
            raise ValueError(f"unknown command source: {source}")

    def select(self, now: float) -> tuple[str, Command]:
        if self._safety is not None and now - self._safety.stamp <= self.safety_timeout:
            return "safety", self._bounded(self._safety.command)
        if self._navigation is not None and now - self._navigation.stamp <= self.navigation_timeout:
            return "navigation", self._bounded(self._navigation.command)
        return "deadman", Command.zero()

    def _bounded(self, command: Command) -> Command:
        speed = math.hypot(command.x, command.y)
        scale = min(1.0, self.max_linear_speed / speed) if speed > 0 else 1.0
        return Command(
            command.x * scale,
            command.y * scale,
            max(-self.max_angular_speed, min(self.max_angular_speed, command.yaw)),
        )

