"""Privileged-map A* expert and deterministic Dataset V0 task sampler."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .common import ROOT, dump_yaml, load_yaml, stable_hash


PLANNER_TYPE = "privileged_grid_astar"
PLANNER_VERSION = 2
DIAGONAL_POLICY = "require_both_orthogonal_cells_free"
SMOOTHING_POLICY = "greedy_line_of_sight"
ROUTE_RANGES = {
    "short": (3.0, 5.0),
    "medium": (5.0, 8.0),
    "long": (8.0, 10.0),
}


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
        radius = float(tree["collision_proxy"]["radius_m"]) + clearance
        xmask = np.abs(xs - tx) <= radius
        ymask = np.abs(ys - ty) <= radius
        occupied[np.ix_(ymask, xmask)] |= (
            (xs[xmask][None, :] - tx) ** 2 + (ys[ymask][:, None] - ty) ** 2 <= radius**2
        )
    return Grid(half_extent=half, resolution=resolution, occupied=occupied)


def line_free(grid: Grid, first: tuple[float, float], second: tuple[float, float]) -> bool:
    distance = math.dist(first, second)
    samples = max(2, int(math.ceil(distance / (grid.resolution * 0.45))))
    previous_cell = None
    for fraction in np.linspace(0.0, 1.0, samples):
        point = (first[0] + fraction * (second[0] - first[0]), first[1] + fraction * (second[1] - first[1]))
        x, y = grid.world_to_cell(point)
        if x < 0 or y < 0 or y >= grid.occupied.shape[0] or x >= grid.occupied.shape[1] or grid.occupied[y, x]:
            return False
        cell = (x, y)
        if previous_cell is not None and cell != previous_cell and not grid_transition_free(grid, previous_cell, cell):
            return False
        previous_cell = cell
    return True


def grid_transition_free(
    grid: Grid, current: tuple[int, int], neighbor: tuple[int, int]
) -> bool:
    """Return whether an 8-connected grid transition is collision-free.

    A diagonal move is only valid when both adjacent orthogonal cells are free.
    This prevents the inflated robot footprint from slipping through an obstacle
    corner even though the diagonal destination cell itself is free.
    """
    dx, dy = neighbor[0] - current[0], neighbor[1] - current[1]
    if abs(dx) > 1 or abs(dy) > 1 or (dx == 0 and dy == 0):
        return False
    height, width = grid.occupied.shape
    if not (0 <= current[0] < width and 0 <= current[1] < height):
        return False
    if not (0 <= neighbor[0] < width and 0 <= neighbor[1] < height):
        return False
    if grid.occupied[neighbor[1], neighbor[0]]:
        return False
    if dx and dy:
        if grid.occupied[current[1], current[0] + dx]:
            return False
        if grid.occupied[current[1] + dy, current[0]]:
            return False
    return True


def path_collision_free(grid: Grid, path: list[tuple[float, float]] | list[list[float]]) -> bool:
    """Check every smoothed path segment against the conservative grid."""
    return all(
        line_free(grid, tuple(path[index - 1][:2]), tuple(path[index][:2]))
        for index in range(1, len(path))
    )


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
            if not grid_transition_free(grid, current, neighbor):
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
    if not path_collision_free(grid, smooth):
        raise RuntimeError("A* smoothing produced a colliding segment")
    return smooth


def path_length(path: list[tuple[float, float]] | list[list[float]]) -> float:
    return sum(math.dist(path[index - 1], path[index]) for index in range(1, len(path)))


def plan_hash(plan: dict[str, Any]) -> str:
    """Hash the exact authoritative plan, excluding its self-referential field."""
    return stable_hash({key: value for key, value in plan.items() if key != "plan_hash"})


def validate_plan(scene: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    """Validate the complete planner-v2 contract without starting RTX sensors."""
    robot = load_yaml(ROOT / "config/robots/diablo_standing.yaml")
    sensor = load_yaml(ROOT / "config/sensors/d435i_navigation_v0.yaml")
    manifest = load_yaml(ROOT / "config/datasets/dataset_v0_manifest.yaml")
    manifest_scene = next(item for item in manifest["scenes"] if item["scene_id"] == scene["scene_id"])
    episode_spec = next(item for item in manifest_scene["planned_episodes"] if item["episode_id"] == plan["episode_id"])
    assert plan["schema_version"] == 2
    assert plan["dataset_version"] == "dataset_v0"
    assert plan["scene_id"] == scene["scene_id"]
    assert plan["scene_seed"] == scene["scene_seed"]
    assert plan["scene_content_hash"] == scene["content_hash"]
    assert plan["split"] == scene["split"] == manifest_scene["split"]
    assert plan["target_route_length_bucket"] == episode_spec["target_route_length_bucket"]
    assert plan["planner"] == {
        "type": PLANNER_TYPE,
        "version": PLANNER_VERSION,
        "grid_resolution_m": 0.1,
        "diagonal_policy": DIAGONAL_POLICY,
        "smoothing": SMOOTHING_POLICY,
    }
    assert plan["configuration_hashes"]["scene"] == scene["content_hash"]
    assert plan["configuration_hashes"]["robot"] == stable_hash(robot)
    assert plan["configuration_hashes"]["sensor"] == stable_hash(sensor)
    assert plan["plan_hash"] == plan_hash(plan), "plan hash mismatch"
    route = plan["planned_path"]
    assert len(route) >= 3, "formal route must remain non-trivial after smoothing"
    grid = occupancy_grid(scene, robot, resolution=float(plan["planner"]["grid_resolution_m"]))
    assert path_collision_free(grid, route), "planned path violates strict conservative occupancy"
    assert not line_free(grid, tuple(plan["start_pose_xyzyaw"][:2]), tuple(plan["goal_pose_xyz"][:2]))
    assert plan["direct_line_collision_free"] is False
    start_cell = grid.world_to_cell(tuple(plan["start_pose_xyzyaw"][:2]))
    goal_cell = grid.world_to_cell(tuple(plan["goal_pose_xyz"][:2]))
    assert not grid.occupied[start_cell[1], start_cell[0]], "start is occupied"
    assert not grid.occupied[goal_cell[1], goal_cell[0]], "goal is occupied"
    measured_length = path_length(route)
    assert math.isclose(measured_length, float(plan["planned_path_length_m"]), abs_tol=2.0e-6)
    lower, upper = ROUTE_RANGES[plan["target_route_length_bucket"]]
    assert lower <= measured_length <= upper
    straight = float(plan["straight_line_distance_m"])
    assert measured_length / straight >= 1.07
    return {
        "status": "PASS",
        "episode_id": plan["episode_id"],
        "planner_type": PLANNER_TYPE,
        "planner_version": PLANNER_VERSION,
        "diagonal_policy": DIAGONAL_POLICY,
        "plan_hash": plan["plan_hash"],
        "route_bucket": plan["target_route_length_bucket"],
        "start_pose_xyzyaw": plan["start_pose_xyzyaw"],
        "goal_pose_xyz": plan["goal_pose_xyz"],
        "planned_path_length_m": measured_length,
        "straight_line_distance_m": straight,
        "smoothed_waypoint_count": len(route),
        "strict_corner_cut_validation": True,
        "conservative_planning_collision_free": True,
    }


def sample_episode_plan(
    scene: dict[str, Any], episode_id: str, surface_height: Callable[[float, float], float]
) -> dict[str, Any]:
    robot = load_yaml(ROOT / "config/robots/diablo_standing.yaml")
    sensor = load_yaml(ROOT / "config/sensors/d435i_navigation_v0.yaml")
    manifest = load_yaml(ROOT / "config/datasets/dataset_v0_manifest.yaml")
    manifest_scene = next(item for item in manifest["scenes"] if item["scene_id"] == scene["scene_id"])
    episode_spec = next(item for item in manifest_scene["planned_episodes"] if item["episode_id"] == episode_id)
    bucket = episode_spec["target_route_length_bucket"]
    ranges = {
        "short": {"route": ROUTE_RANGES["short"], "euclidean": (3.0, 4.6)},
        "medium": {"route": ROUTE_RANGES["medium"], "euclidean": (4.8, 7.4)},
        "long": {"route": ROUTE_RANGES["long"], "euclidean": (7.2, 9.3)},
    }[bucket]
    grid = occupancy_grid(scene, robot)
    episode_index = int(episode_id.rsplit("_", 1)[-1])
    rng = np.random.Generator(np.random.PCG64(int(scene["scene_seed"]) + 100_003 + episode_index))
    half = grid.half_extent - 1.2
    best = None
    for _ in range(15000):
        start = tuple(float(value) for value in rng.uniform(-half, half, size=2))
        direction = float(rng.uniform(-math.pi, math.pi))
        euclidean_target = float(rng.uniform(*ranges["euclidean"]))
        goal = (start[0] + euclidean_target * math.cos(direction), start[1] + euclidean_target * math.sin(direction))
        if not (-half <= goal[0] <= half and -half <= goal[1] <= half):
            continue
        try:
            route = astar(grid, start, goal)
        except (ValueError, RuntimeError):
            continue
        route_length_xy = path_length(route)
        euclidean = math.dist(start, goal)
        if ranges["route"][0] <= route_length_xy <= ranges["route"][1] and route_length_xy / euclidean >= 1.07 and len(route) >= 3 and not line_free(grid, start, goal):
            surface_path = [[round(float(x), 6), round(float(y), 6), round(float(surface_height(x, y)), 6)] for x, y in route]
            if not path_collision_free(grid, surface_path):
                # Serialization may move a waypoint lying extremely close to a
                # cell boundary. Only accept the exact path that will be saved.
                continue
            route_length_surface = path_length(surface_path)
            if not ranges["route"][0] <= route_length_surface <= ranges["route"][1]:
                continue
            best = start, goal, route, surface_path, euclidean, route_length_surface
            break
    if best is None:
        raise RuntimeError(f"could not sample a non-trivial {bucket} route")
    start, goal, route, surface_path, euclidean, route_length = best
    start_yaw = math.atan2(route[1][1] - route[0][1], route[1][0] - route[0][0])
    path = surface_path
    plan = {
        "schema_version": 2,
        "dataset_version": "dataset_v0",
        "episode_id": episode_id,
        "episode_type": "nominal_expert",
        "split": scene["split"],
        "scene_id": scene["scene_id"],
        "scene_seed": scene["scene_seed"],
        "scene_schema_version": scene["schema_version"],
        "scene_content_hash": scene["content_hash"],
        "robot_profile": robot["profile_id"],
        "sensor_profile": sensor["profile_id"],
        "planner": {
            "type": PLANNER_TYPE,
            "version": PLANNER_VERSION,
            "grid_resolution_m": grid.resolution,
            "diagonal_policy": DIAGONAL_POLICY,
            "smoothing": SMOOTHING_POLICY,
        },
        "controller": {"type": "pure_pursuit", "lookahead_m": 0.32, "goal_tolerance_m": 0.18},
        "target_route_length_bucket": bucket,
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
    plan["plan_hash"] = plan_hash(plan)
    validate_plan(scene, plan)
    return plan


def write_episode_plan(
    scene: dict[str, Any], output: Path, surface_height: Callable[[float, float], float]
) -> dict[str, Any]:
    plan = sample_episode_plan(scene, "train_scene_000_episode_000", surface_height)
    dump_yaml(output, plan)
    return plan
