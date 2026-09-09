"""Coverage routes with obstacle-aware A* connectors.

Behavior is derived from ForestNavigation's zigzag and connected coverage
planners at revision 0b29c399754f510499bfe9cc9d592cba215a7161. The API and
implementation are reorganized for a ROS-independent mission layer.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float


@dataclass(frozen=True)
class CoveragePath:
    waypoints: tuple[Point2D, ...]
    cumulative_lengths: tuple[float, ...]
    total_length: float

    @classmethod
    def from_points(cls, points: Sequence[Point2D]) -> "CoveragePath":
        if len(points) < 2:
            raise ValueError("a route requires at least two points")
        cumulative = [0.0]
        for start, end in zip(points[:-1], points[1:]):
            cumulative.append(
                cumulative[-1] + math.hypot(end.x - start.x, end.y - start.y)
            )
        return cls(tuple(points), tuple(cumulative), cumulative[-1])

    def point_at(self, distance: float) -> Point2D:
        distance = min(max(float(distance), 0.0), self.total_length)
        for index in range(len(self.waypoints) - 1):
            end_distance = self.cumulative_lengths[index + 1]
            if distance <= end_distance or index == len(self.waypoints) - 2:
                start = self.waypoints[index]
                end = self.waypoints[index + 1]
                segment = end_distance - self.cumulative_lengths[index]
                alpha = 0.0 if segment <= 1e-12 else (
                    distance - self.cumulative_lengths[index]
                ) / segment
                return Point2D(
                    start.x + alpha * (end.x - start.x),
                    start.y + alpha * (end.y - start.y),
                )
        return self.waypoints[-1]

    def sampled(self, spacing: float) -> tuple[Point2D, ...]:
        if spacing <= 0.0:
            raise ValueError("spacing must be positive")
        distances = [index * spacing for index in range(int(self.total_length / spacing) + 1)]
        if not math.isclose(distances[-1], self.total_length):
            distances.append(self.total_length)
        return tuple(self.point_at(value) for value in distances)


def _validated_bounds(bounds: Iterable[float]) -> tuple[float, float, float, float]:
    values = tuple(float(value) for value in bounds)
    if len(values) != 4:
        raise ValueError("bounds must be [x_min, x_max, y_min, y_max]")
    x_min, x_max, y_min, y_max = values
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("bounds must have positive area")
    return values


def plan_zigzag(bounds: Iterable[float], lane_spacing: float) -> CoveragePath:
    x_min, x_max, y_min, y_max = _validated_bounds(bounds)
    if lane_spacing <= 0.0:
        raise ValueError("lane_spacing must be positive")
    lanes = max(1, math.ceil((y_max - y_min) / lane_spacing))
    ys = [min(y_min + index * lane_spacing, y_max) for index in range(lanes + 1)]
    ys[-1] = y_max
    points: list[Point2D] = []
    for index, y in enumerate(ys):
        xs = (x_min, x_max) if index % 2 == 0 else (x_max, x_min)
        for x in xs:
            point = Point2D(x, y)
            if not points or point != points[-1]:
                points.append(point)
    return CoveragePath.from_points(points)


GridCell = tuple[int, int]


def plan_connected_coverage(
    bounds: Iterable[float],
    lane_spacing: float,
    obstacles: Iterable[Point2D],
    *,
    resolution: float = 0.2,
    clearance: float = 0.7,
    boundary_margin: float = 0.5,
    start: Point2D | None = None,
) -> CoveragePath:
    """Cover horizontal free intervals and connect them through an inflated grid."""
    x_min, x_max, y_min, y_max = _validated_bounds(bounds)
    if min(lane_spacing, resolution, clearance) <= 0.0:
        raise ValueError("spacing, resolution, and clearance must be positive")
    if boundary_margin < 0.0 or 2 * boundary_margin >= min(
        x_max - x_min, y_max - y_min
    ):
        raise ValueError("boundary_margin leaves no usable area")
    obstacles = tuple(obstacles)
    nx = int(math.floor((x_max - x_min) / resolution)) + 1
    ny = int(math.floor((y_max - y_min) / resolution)) + 1

    def world(cell: GridCell) -> Point2D:
        return Point2D(x_min + cell[0] * resolution, y_min + cell[1] * resolution)

    def cell(point: Point2D) -> GridCell:
        return (
            min(max(round((point.x - x_min) / resolution), 0), nx - 1),
            min(max(round((point.y - y_min) / resolution), 0), ny - 1),
        )

    def is_free(item: GridCell) -> bool:
        point = world(item)
        return all(
            math.hypot(point.x - obstacle.x, point.y - obstacle.y) >= clearance
            for obstacle in obstacles
        )

    free = {(ix, iy) for ix in range(nx) for iy in range(ny) if is_free((ix, iy))}
    if not free:
        raise RuntimeError("obstacle inflation removed all free space")
    desired_start = start or Point2D(x_min + boundary_margin, y_min + boundary_margin)
    start_cell = min(
        free,
        key=lambda item: math.hypot(
            world(item).x - desired_start.x, world(item).y - desired_start.y
        ),
    )
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))

    reachable = {start_cell}
    pending = [start_cell]
    while pending:
        current = pending.pop()
        for dx, dy in directions[:4]:
            neighbor = current[0] + dx, current[1] + dy
            if neighbor in free and neighbor not in reachable:
                reachable.add(neighbor)
                pending.append(neighbor)

    def astar(source: GridCell, target: GridCell) -> list[GridCell]:
        frontier = [(0.0, source)]
        cost = {source: 0.0}
        parent: dict[GridCell, GridCell | None] = {source: None}
        while frontier:
            _, current = heapq.heappop(frontier)
            if current == target:
                break
            for dx, dy in directions:
                neighbor = current[0] + dx, current[1] + dy
                if neighbor not in reachable:
                    continue
                if dx and dy and (
                    (current[0] + dx, current[1]) not in reachable
                    or (current[0], current[1] + dy) not in reachable
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
            raise RuntimeError("coverage segments are not connected")
        route: list[GridCell] = []
        cursor: GridCell | None = target
        while cursor is not None:
            route.append(cursor)
            cursor = parent[cursor]
        return list(reversed(route))

    y = y_min + boundary_margin
    row_indices: list[int] = []
    while y < y_max - boundary_margin:
        row_indices.append(cell(Point2D(x_min, y))[1])
        y += lane_spacing
    row_indices.append(cell(Point2D(x_min, y_max - boundary_margin))[1])
    segments: list[tuple[GridCell, GridCell]] = []
    for row_order, iy in enumerate(dict.fromkeys(row_indices)):
        xs = [
            ix for ix in range(nx)
            if (ix, iy) in reachable
            and x_min + boundary_margin <= world((ix, iy)).x <= x_max - boundary_margin
        ]
        runs: list[tuple[int, int]] = []
        if xs:
            begin = previous = xs[0]
            for ix in xs[1:] + [xs[-1] + 2]:
                if ix != previous + 1:
                    if previous > begin:
                        runs.append((begin, previous))
                    begin = ix
                previous = ix
        if row_order % 2:
            runs.reverse()
        for run_index, (begin, end) in enumerate(runs):
            forward = (row_order + run_index) % 2 == 0
            segments.append(((begin, iy), (end, iy)) if forward else ((end, iy), (begin, iy)))
    if not segments:
        raise RuntimeError("no coverable lane segments")

    cells = [start_cell]
    for begin, end in segments:
        connector = astar(cells[-1], begin)
        cells.extend(connector[1:])
        step = 1 if end[0] >= begin[0] else -1
        cells.extend((ix, begin[1]) for ix in range(begin[0] + step, end[0] + step, step))
    points: list[Point2D] = []
    for item in cells:
        point = world(item)
        if not points or point != points[-1]:
            points.append(point)
    return CoveragePath.from_points(points)


class ProgressTracker:
    """Forward-only route projection, bounded to avoid parallel-lane jumps."""

    def __init__(self, path: CoveragePath, max_advance: float = 3.0) -> None:
        if max_advance <= 0.0:
            raise ValueError("max_advance must be positive")
        self.path = path
        self.max_advance = max_advance
        self.distance = 0.0

    @property
    def fraction(self) -> float:
        return 1.0 if self.path.total_length <= 0 else self.distance / self.path.total_length

    def update(self, x: float, y: float) -> float:
        end = min(self.path.total_length, self.distance + self.max_advance)
        best = self.distance
        best_error = math.inf
        sample = self.distance
        step = min(0.05, self.max_advance / 20.0)
        while sample <= end + 1e-9:
            point = self.path.point_at(sample)
            error = (point.x - x) ** 2 + (point.y - y) ** 2
            if error < best_error:
                best_error, best = error, sample
            sample += step
        self.distance = max(self.distance, best)
        return self.distance

