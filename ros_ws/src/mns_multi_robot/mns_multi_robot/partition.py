"""Deterministic spatial assignment, adapted from ForestNavigation two-team planning."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Bounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def __post_init__(self) -> None:
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("bounds must have positive area")


def partition_quadrants(bounds: Bounds, gap: float = 0.0) -> tuple[Bounds, Bounds, Bounds, Bounds]:
    if gap < 0.0 or gap >= min(bounds.x_max - bounds.x_min, bounds.y_max - bounds.y_min):
        raise ValueError("invalid partition gap")
    mid_x = (bounds.x_min + bounds.x_max) / 2.0
    mid_y = (bounds.y_min + bounds.y_max) / 2.0
    half = gap / 2.0
    return (
        Bounds(bounds.x_min, mid_x - half, bounds.y_min, mid_y - half),
        Bounds(mid_x + half, bounds.x_max, bounds.y_min, mid_y - half),
        Bounds(bounds.x_min, mid_x - half, mid_y + half, bounds.y_max),
        Bounds(mid_x + half, bounds.x_max, mid_y + half, bounds.y_max),
    )

