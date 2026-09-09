"""Sensor-observed free-space coverage and residual route planning.

Adapted from ForestNavigation's ``sensor_coverage`` planner at revision
0b29c399754f510499bfe9cc9d592cba215a7161.  The implementation is kept
ROS-independent so simulation and future real sensor adapters share it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
import math
from typing import Iterable, Sequence

import numpy as np

from .coverage import CoveragePath, Point2D


GridCell = tuple[int, int]


@dataclass(frozen=True)
class CoverageStats:
    target_cells: int
    observed_cells: int
    inaccessible_cells: int

    @property
    def ratio(self) -> float:
        return self.observed_cells / max(self.target_cells, 1)


@dataclass(frozen=True)
class ResidualCoveragePlan:
    path: CoveragePath
    goals: tuple[Point2D, ...]
    residual_cells: int
    component_count: int


def _rotate_wxyz(quaternion: Sequence[float], vectors: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if quaternion.shape != (4,) or norm <= 1.0e-12:
        raise ValueError("camera quaternion must contain four non-zero values")
    quaternion /= norm
    axis = quaternion[1:]
    first_cross = np.cross(np.broadcast_to(axis, vectors.shape), vectors)
    return vectors + 2.0 * (
        quaternion[0] * first_cross
        + np.cross(np.broadcast_to(axis, vectors.shape), first_cross)
    )


class SensorCoverageGrid:
    """Track reachable free cells observed by sampled RGB-D rays."""

    def __init__(
        self,
        bounds: Iterable[float],
        resolution: float,
        obstacles: Iterable[Point2D] = (),
        *,
        clearance: float = 0.55,
        boundary_margin: float = 0.2,
        start: Point2D | None = None,
    ) -> None:
        self.bounds = tuple(float(value) for value in bounds)
        if len(self.bounds) != 4:
            raise ValueError("bounds must be [x_min, x_max, y_min, y_max]")
        x_min, x_max, y_min, y_max = self.bounds
        self.resolution = float(resolution)
        if x_min >= x_max or y_min >= y_max:
            raise ValueError("bounds must have positive area")
        if self.resolution <= 0.0 or clearance <= 0.0:
            raise ValueError("resolution and clearance must be positive")
        if boundary_margin < 0.0 or 2.0 * boundary_margin >= min(
            x_max - x_min, y_max - y_min
        ):
            raise ValueError("boundary_margin leaves no target area")

        self.width = int(math.ceil((x_max - x_min) / self.resolution))
        self.height = int(math.ceil((y_max - y_min) / self.resolution))
        x_centers = x_min + (np.arange(self.width) + 0.5) * self.resolution
        y_centers = y_min + (np.arange(self.height) + 0.5) * self.resolution
        xx, yy = np.meshgrid(x_centers, y_centers)
        inside = (
            (xx >= x_min + boundary_margin)
            & (xx <= x_max - boundary_margin)
            & (yy >= y_min + boundary_margin)
            & (yy <= y_max - boundary_margin)
        )
        navigation_free = np.ones_like(inside, dtype=bool)
        for obstacle in obstacles:
            navigation_free &= (
                (xx - obstacle.x) ** 2 + (yy - obstacle.y) ** 2 >= clearance**2
            )
        free_targets = inside & navigation_free
        if not np.any(free_targets):
            raise RuntimeError("coverage grid has no free target cells")
        desired_start = start or Point2D(
            x_min + boundary_margin, y_min + boundary_margin
        )
        start_cell = self.nearest_cell(desired_start, navigation_free)
        reachable = self._component(start_cell, navigation_free)
        self.navigation_mask = navigation_free & reachable
        self.target_mask = free_targets & reachable
        self.inaccessible_mask = free_targets & ~reachable
        self.observed_mask = np.zeros_like(self.target_mask)

    @property
    def stats(self) -> CoverageStats:
        return CoverageStats(
            target_cells=int(np.count_nonzero(self.target_mask)),
            observed_cells=int(np.count_nonzero(self.observed_mask & self.target_mask)),
            inaccessible_cells=int(np.count_nonzero(self.inaccessible_mask)),
        )

    @property
    def ratio(self) -> float:
        return self.stats.ratio

    @property
    def residual_mask(self) -> np.ndarray:
        return self.target_mask & ~self.observed_mask

    def contains(self, cell: GridCell) -> bool:
        return 0 <= cell[0] < self.width and 0 <= cell[1] < self.height

    def world_to_cell(self, point: Point2D | Sequence[float]) -> GridCell:
        x = float(point.x) if hasattr(point, "x") else float(point[0])
        y = float(point.y) if hasattr(point, "y") else float(point[1])
        x_min, _, y_min, _ = self.bounds
        return (
            min(max(math.floor((x - x_min) / self.resolution), 0), self.width - 1),
            min(max(math.floor((y - y_min) / self.resolution), 0), self.height - 1),
        )

    def cell_to_world(self, cell: GridCell) -> Point2D:
        x_min, _, y_min, _ = self.bounds
        return Point2D(
            x_min + (cell[0] + 0.5) * self.resolution,
            y_min + (cell[1] + 0.5) * self.resolution,
        )

    def nearest_cell(self, point: Point2D | Sequence[float], mask: np.ndarray) -> GridCell:
        desired = self.world_to_cell(point)
        if mask[desired[1], desired[0]]:
            return desired
        rows, columns = np.nonzero(mask)
        if rows.size == 0:
            raise RuntimeError("no valid cells are available")
        distances = (columns - desired[0]) ** 2 + (rows - desired[1]) ** 2
        index = int(np.argmin(distances))
        return int(columns[index]), int(rows[index])

    def _component(self, start: GridCell, mask: np.ndarray) -> np.ndarray:
        result = np.zeros_like(mask, dtype=bool)
        pending = deque([start])
        result[start[1], start[0]] = True
        while pending:
            x, y = pending.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = x + dx, y + dy
                if (
                    self.contains(neighbor)
                    and mask[neighbor[1], neighbor[0]]
                    and not result[neighbor[1], neighbor[0]]
                ):
                    result[neighbor[1], neighbor[0]] = True
                    pending.append(neighbor)
        return result

    def mark_ray(
        self,
        origin: Point2D | Sequence[float],
        endpoint: Point2D | Sequence[float],
        max_range: float,
    ) -> None:
        if max_range <= 0.0:
            raise ValueError("max_range must be positive")
        ox = float(origin.x) if hasattr(origin, "x") else float(origin[0])
        oy = float(origin.y) if hasattr(origin, "y") else float(origin[1])
        ex = float(endpoint.x) if hasattr(endpoint, "x") else float(endpoint[0])
        ey = float(endpoint.y) if hasattr(endpoint, "y") else float(endpoint[1])
        dx, dy = ex - ox, ey - oy
        distance = math.hypot(dx, dy)
        if distance <= 1.0e-9:
            return
        scale = min(1.0, max_range / distance)
        dx, dy = dx * scale, dy * scale
        steps = max(1, math.ceil(math.hypot(dx, dy) / (0.5 * self.resolution)))
        for index in range(steps + 1):
            alpha = index / steps
            cell = self.world_to_cell((ox + alpha * dx, oy + alpha * dy))
            if self.target_mask[cell[1], cell[0]]:
                self.observed_mask[cell[1], cell[0]] = True

    def observe_depth(
        self,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        camera_position: Sequence[float],
        camera_quaternion_wxyz: Sequence[float],
        *,
        max_range: float,
        min_range: float = 0.15,
        pixel_stride: int = 24,
        row_fraction: tuple[float, float] = (0.3, 0.9),
    ) -> int:
        """Project sampled optical-frame depth rays into the world XY grid."""
        if not 0.0 <= min_range < max_range or pixel_stride <= 0:
            raise ValueError("depth range and pixel stride are invalid")
        depth = np.squeeze(np.asarray(depth, dtype=np.float64))
        intrinsics = np.squeeze(np.asarray(intrinsics, dtype=np.float64))
        if depth.ndim != 2 or intrinsics.shape != (3, 3):
            raise ValueError("expected a depth image and 3x3 intrinsics")
        row_start, row_end = row_fraction
        if not 0.0 <= row_start < row_end <= 1.0:
            raise ValueError("row_fraction must be inside [0, 1]")
        height, width = depth.shape
        rows = np.arange(int(height * row_start), max(int(height * row_end), 1), pixel_stride)
        columns = np.arange(0, width, pixel_stride)
        uu, vv = np.meshgrid(columns, rows)
        ranges = depth[vv, uu]
        valid = np.isfinite(ranges) & (ranges >= min_range)
        if not np.any(valid):
            return 0
        uu, vv = uu[valid], vv[valid]
        ranges = np.minimum(ranges[valid], max_range)
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        rays = np.column_stack(((uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)))
        endpoints = (
            _rotate_wxyz(camera_quaternion_wxyz, rays * ranges[:, None])
            + np.asarray(camera_position, dtype=np.float64)[None, :]
        )
        before = int(np.count_nonzero(self.observed_mask))
        origin = camera_position[:2]
        for endpoint in endpoints:
            self.mark_ray(origin, endpoint[:2], max_range)
        return int(np.count_nonzero(self.observed_mask)) - before


def _components(mask: np.ndarray) -> list[set[GridCell]]:
    remaining = {(int(x), int(y)) for y, x in zip(*np.nonzero(mask))}
    result: list[set[GridCell]] = []
    while remaining:
        seed = remaining.pop()
        component = {seed}
        pending = deque([seed])
        while pending:
            x, y = pending.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = x + dx, y + dy
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    pending.append(neighbor)
        result.append(component)
    return result


def _astar(source: GridCell, target: GridCell, mask: np.ndarray) -> list[GridCell]:
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
    frontier = [(0.0, source)]
    cost = {source: 0.0}
    parent: dict[GridCell, GridCell | None] = {source: None}
    height, width = mask.shape
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == target:
            break
        for dx, dy in directions:
            neighbor = current[0] + dx, current[1] + dy
            if not (0 <= neighbor[0] < width and 0 <= neighbor[1] < height):
                continue
            if not mask[neighbor[1], neighbor[0]]:
                continue
            if dx and dy and (
                not mask[current[1], neighbor[0]] or not mask[neighbor[1], current[0]]
            ):
                continue
            candidate = cost[current] + math.hypot(dx, dy)
            if candidate >= cost.get(neighbor, math.inf):
                continue
            cost[neighbor] = candidate
            parent[neighbor] = current
            heuristic = math.hypot(target[0] - neighbor[0], target[1] - neighbor[1])
            heapq.heappush(frontier, (candidate + heuristic, neighbor))
    if target not in parent:
        raise RuntimeError("residual goal is not reachable")
    path: list[GridCell] = []
    cursor: GridCell | None = target
    while cursor is not None:
        path.append(cursor)
        cursor = parent[cursor]
    return list(reversed(path))


def plan_residual_coverage(
    grid: SensorCoverageGrid,
    start: Point2D,
    *,
    goal_spacing: float = 0.8,
    min_component_area: float = 0.12,
    max_goals: int = 80,
) -> ResidualCoveragePlan | None:
    """Connect representative cells from meaningful unobserved regions."""
    if goal_spacing <= 0.0 or min_component_area < 0.0 or max_goals <= 0:
        raise ValueError("residual planner parameters are invalid")
    start_cell = grid.nearest_cell(start, grid.navigation_mask)
    residual = grid.residual_mask & grid.navigation_mask
    minimum_cells = max(1, math.ceil(min_component_area / grid.resolution**2))
    components = [item for item in _components(residual) if len(item) >= minimum_cells]
    if not components:
        return None
    candidates: list[GridCell] = []
    spacing_sq = (goal_spacing / grid.resolution) ** 2
    for component in sorted(components, key=len, reverse=True):
        selected: list[GridCell] = []
        for cell in sorted(component, key=lambda item: (item[1], item[0])):
            if all((cell[0] - other[0]) ** 2 + (cell[1] - other[1]) ** 2 >= spacing_sq for other in selected):
                selected.append(cell)
        candidates.extend(selected or [next(iter(component))])
    if len(candidates) > max_goals:
        indices = np.linspace(0, len(candidates) - 1, max_goals, dtype=int)
        candidates = [candidates[int(index)] for index in indices]
    ordered: list[GridCell] = []
    current = start_cell
    remaining = set(candidates)
    while remaining:
        current = min(remaining, key=lambda cell: (cell[0] - current[0]) ** 2 + (cell[1] - current[1]) ** 2)
        ordered.append(current)
        remaining.remove(current)
    points = [start]
    current = start_cell
    for target in ordered:
        for cell in _astar(current, target, grid.navigation_mask)[1:]:
            point = grid.cell_to_world(cell)
            if point != points[-1]:
                points.append(point)
        current = target
    if len(points) < 2:
        return None
    return ResidualCoveragePlan(
        path=CoveragePath.from_points(points),
        goals=tuple(grid.cell_to_world(cell) for cell in ordered),
        residual_cells=int(np.count_nonzero(residual)),
        component_count=len(components),
    )
