"""Privileged-map A* expert and deterministic Dataset V0 task sampler."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from pathlib import Path
from typing import Any

import numpy as np

from .common import ROOT, dump_yaml, load_yaml, stable_hash, terrain_height


@dataclass(frozen=True)
class Grid:
    half_extent: float
    resolution: float
    occupied: np.ndarray

    def world_to_cell(self, point: tuple[float, float]) -> tuple[int, int]:
        x, y = point
        return (
            int(round((x + self.half_extent) / self.resolution)),
            int(round((y + self.half_extent) / self.resolution)),
        )

    def cell_to_world(self, cell: tuple[int, int]) -> tuple[float, float]:
        x, y = cell
        return x * self.resolution - self.half_extent, y * self.resolution - self.half_extent


def occupancy_grid(
    scene: dict[str, Any], robot: dict[str, Any], resolution: float = 0.10,
    clearance_key: str = "planning_radius_m",
) -> Grid:
    half = min(float(value) for value in scene["extent_m"]) / 2.0
    size = int(round(2.0 * half / resolution)) + 1
    occupied = np.zeros((size, size), dtype=bool)
    clearance = float(robot["surrogate"]["footprint"][clearance_key])
    xs = np.linspace(-half, half, size)
    ys = np.linspace(-half, half, size)
    occupied[0, :] = occupied[-1, :] = True
    occupied[:, 0] = occupied[:, -1] = True
    for tree in scene["trees"]:
        tx, ty = tree["position_m"]
        radius = float(tree["trunk_radius_m"]) + clearance
        xmask = np.abs(xs - tx) <= radius
        ymask = np.abs(ys - ty) <= radius
        occupied[np.ix_(ymask, xmask)] |= (
            (xs[xmask][None, :] - tx) ** 2 + (ys[ymask][:, None] - ty) ** 2 <= radius**2
        )
    return Grid(half_extent=half, resolution=resolution, occupied=occupied)


def line_free(grid: Grid, first: tuple[float, float], second: tuple[float, float]) -> bool:
    distance = math.dist(first, second)
    samples = max(2, int(math.ceil(distance / (grid.resolution * 0.45))))
    for fraction in np.linspace(0.0, 1.0, samples):
        point = (first[0] + fraction * (second[0] - first[0]), first[1] + fraction * (second[1] - first[1]))
        x, y = grid.world_to_cell(point)
        if x < 0 or y < 0 or y >= grid.occupied.shape[0] or x >= grid.occupied.shape[1] or grid.occupied[y, x]:
            return False
    return True


def astar(grid: Grid, start: tuple[float, float], goal: tuple[float, float]) -> list[tuple[float, float]]:
    start_cell, goal_cell = grid.world_to_cell(start), grid.world_to_cell(goal)
    if grid.occupied[start_cell[1], start_cell[0]] or grid.occupied[goal_cell[1], goal_cell[0]]:
        raise ValueError("start or goal is occupied")
    queue = [(0.0, start_cell)]
    cost = {start_cell: 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    motions = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
    while queue:
        _, current = heapq.heappop(queue)
        if current == goal_cell:
            break
        for dx, dy in motions:
            neighbor = current[0] + dx, current[1] + dy
            if not (0 <= neighbor[0] < grid.occupied.shape[1] and 0 <= neighbor[1] < grid.occupied.shape[0]):
                continue
            if grid.occupied[neighbor[1], neighbor[0]]:
                continue
            step = grid.resolution * (math.sqrt(2.0) if dx and dy else 1.0)
            new_cost = cost[current] + step
            if new_cost < cost.get(neighbor, math.inf):
                cost[neighbor] = new_cost
                parent[neighbor] = current
                heuristic = grid.resolution * math.dist(neighbor, goal_cell)
                heapq.heappush(queue, (new_cost + heuristic, neighbor))
    if goal_cell not in cost:
        raise RuntimeError("A* found no route")
    cells = [goal_cell]
    while cells[-1] != start_cell:
        cells.append(parent[cells[-1]])
    cells.reverse()
    raw = [grid.cell_to_world(cell) for cell in cells]
    # Greedy visibility smoothing keeps privileged collision constraints while avoiding grid jitter.
    smooth = [raw[0]]
    index = 0
    while index < len(raw) - 1:
        candidate = len(raw) - 1
        while candidate > index + 1 and not line_free(grid, raw[index], raw[candidate]):
            candidate -= 1
        smooth.append(raw[candidate])
        index = candidate
    return smooth


def path_length(path: list[tuple[float, float]] | list[list[float]]) -> float:
    return sum(math.dist(path[index - 1][:2], path[index][:2]) for index in range(1, len(path)))


def sample_preview_plan(scene: dict[str, Any], episode_id: str) -> dict[str, Any]:
    robot = load_yaml(ROOT / "config/robots/diablo_standing.yaml")
    sensor = load_yaml(ROOT / "config/sensors/d435i_navigation_v0.yaml")
    grid = occupancy_grid(scene, robot)
    rng = np.random.Generator(np.random.PCG64(int(scene["scene_seed"]) + 100_003))
    half = grid.half_extent - 1.2
    best = None
    for _ in range(15000):
        start = tuple(float(value) for value in rng.uniform(-half, half, size=2))
        direction = float(rng.uniform(-math.pi, math.pi))
        euclidean_target = float(rng.uniform(5.2, 7.4))
        goal = (start[0] + euclidean_target * math.cos(direction), start[1] + euclidean_target * math.sin(direction))
        if not (-half <= goal[0] <= half and -half <= goal[1] <= half):
            continue
        try:
            route = astar(grid, start, goal)
        except (ValueError, RuntimeError):
            continue
        route_length = path_length(route)
        euclidean = math.dist(start, goal)
        if 5.0 <= route_length <= 9.0 and route_length / euclidean >= 1.07 and len(route) >= 3 and not line_free(grid, start, goal):
            best = start, goal, route, route_length, euclidean
            break
    if best is None:
        raise RuntimeError("could not sample a medium, non-trivial preview route")
    start, goal, route, route_length, euclidean = best
    start_yaw = math.atan2(route[1][1] - route[0][1], route[1][0] - route[0][0])
    path = [[round(x, 6), round(y, 6), round(terrain_height(scene, x, y), 6)] for x, y in route]
    plan = {
        "schema_version": 1,
        "dataset_version": "dataset_v0",
        "episode_id": episode_id,
        "episode_type": "nominal_expert",
        "split": scene["split"],
        "scene_id": scene["scene_id"],
        "scene_seed": scene["scene_seed"],
        "robot_profile": robot["profile_id"],
        "sensor_profile": sensor["profile_id"],
        "planner": {"type": "privileged_grid_astar", "grid_resolution_m": grid.resolution, "smoothing": "greedy_line_of_sight"},
        "controller": {"type": "pure_pursuit", "lookahead_m": 0.32, "goal_tolerance_m": 0.18},
        "target_route_length_bucket": "medium",
        "start_pose_xyzyaw": [round(start[0], 6), round(start[1], 6), path[0][2], round(start_yaw, 6)],
        "goal_pose_xyz": [round(goal[0], 6), round(goal[1], 6), path[-1][2]],
        "straight_line_distance_m": round(euclidean, 6),
        "planned_path_length_m": round(route_length, 6),
        "direct_line_collision_free": False,
        "planned_path": path,
        "configuration_hashes": {
            "scene": scene["content_hash"],
            "robot": stable_hash(robot),
            "sensor": stable_hash(sensor),
        },
    }
    plan["plan_hash"] = stable_hash(plan)
    return plan


def write_preview_plan(scene: dict[str, Any], output: Path) -> dict[str, Any]:
    plan = sample_preview_plan(scene, "train_scene_000_episode_000")
    dump_yaml(output, plan)
    return plan
