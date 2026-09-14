import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from mns_navigation.coverage import Point2D, ProgressTracker, plan_connected_coverage, plan_zigzag
from mns_navigation.diffusion import DiffusionPlanner, RGBDHistory
from mns_navigation.path_follower import PurePursuitFollower
from mns_navigation.safety import DepthSafetyController, StallRecovery, depth_sector_distances
from mns_navigation.sensor_coverage import SensorCoverageGrid, plan_residual_coverage


def test_zigzag_and_progress_are_monotonic():
    path = plan_zigzag((0, 4, 0, 2), 1.0)
    assert path.waypoints[0] == Point2D(0, 0)
    assert path.waypoints[-1] == Point2D(4, 2)
    tracker = ProgressTracker(path, max_advance=1.0)
    values = [tracker.update(x, 0.0) for x in (0.0, 0.5, 1.0, 0.8)]
    assert values == sorted(values)


def test_connected_route_avoids_inflated_obstacle():
    obstacle = Point2D(2.0, 1.0)
    path = plan_connected_coverage((0, 4, 0, 2), 0.8, [obstacle], resolution=0.2, clearance=0.45, boundary_margin=0.2)
    assert path.total_length > 0
    assert min(math.hypot(point.x - obstacle.x, point.y - obstacle.y) for point in path.waypoints) >= 0.44


def test_phase1_known_map_route_avoids_acceptance_forest():
    root = Path(__file__).resolve().parents[1]
    with (root / "config/simulation/forest.yaml").open(encoding="utf-8") as stream:
        scene = yaml.safe_load(stream)
    with (root / "config/missions/coverage.yaml").open(encoding="utf-8") as stream:
        mission = yaml.safe_load(stream)["missions"]["phase1_coverage"]
    obstacles = [Point2D(*item["position"]) for item in scene["trees"]]
    path = plan_connected_coverage(
        mission["bounds"], mission["lane_spacing"], obstacles, clearance=0.7
    )
    for waypoint in path.waypoints:
        assert min(math.hypot(waypoint.x - tree.x, waypoint.y - tree.y) for tree in obstacles) >= 0.69


def test_pure_pursuit_turns_then_advances():
    follower = PurePursuitFollower(max_linear_speed=1.0, max_angular_speed=1.0)
    command = follower.command(Point2D(0, 0), 0.0, (Point2D(0, 0), Point2D(0, 2)))
    assert command.angular_z > 0.9
    assert command.linear_x < 0.1


def test_diffusion_is_trajectory_only_and_transforms_local_output():
    history = RGBDHistory(2)
    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    depth = np.ones((4, 5), dtype=np.float32)
    history.append(rgb, depth)
    history.append(rgb, depth)
    predictor = lambda _rgb, _depth, _goal: np.asarray([[1.0, 0.0], [2.0, 0.0]])
    planner = DiffusionPlanner(predictor, history)
    points = planner.trajectory(Point2D(3, 4), math.pi / 2, Point2D(3, 8))
    assert points[0].x == pytest.approx(3.0)
    assert points[0].y == pytest.approx(5.0)
    np.testing.assert_allclose(planner.last_local_trajectory, [[1.0, 0.0], [2.0, 0.0]])
    planner.reset()
    assert not history.ready
    assert planner.last_local_trajectory.shape == (0, 2)


def test_registered_depth_safety_ignores_zero_registration_holes():
    depth = np.full((9, 12), 2.0, dtype=np.float32)
    depth[3:6, ::2] = 0.0
    left, centre, right = depth_sector_distances(depth)
    assert left == pytest.approx(2.0)
    assert centre == pytest.approx(2.0)
    assert right == pytest.approx(2.0)


def test_safety_hysteresis_and_stall_recovery():
    safety = DepthSafetyController()
    assert safety.decide(2, 2, 2).command is None
    avoidance = safety.decide(0.6, 0.5, 0.8).command
    assert avoidance is not None
    assert avoidance.linear_x == 0.0
    assert avoidance.angular_z != 0.0
    assert safety.decide(0.8, 0.8, 0.8).command is not None
    assert safety.decide(1.0, 1.0, 1.0).state == "released"
    recovery = StallRecovery(trigger_window=1.0)
    recovery.update(0.0, 0.0, 0.0, True)
    decision = recovery.update(1.0, 0.01, 0.0, True)
    assert decision.state == "stall_reverse"


def test_sensor_coverage_projects_depth_in_front_of_camera():
    grid = SensorCoverageGrid((-1.0, 4.0, -2.0, 2.0), 0.1, clearance=0.2, boundary_margin=0.0)
    depth = np.full((9, 9), 2.0, dtype=np.float32)
    intrinsics = np.array(
        [[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    marked = grid.observe_depth(
        depth,
        intrinsics,
        (0.0, 0.0, 0.5),
        (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0),
        max_range=3.0,
        pixel_stride=2,
        row_fraction=(0.0, 1.0),
    )
    forward = grid.world_to_cell((1.5, 0.0))
    behind = grid.world_to_cell((-0.8, 0.0))
    assert marked > 10
    assert grid.observed_mask[forward[1], forward[0]]
    assert not grid.observed_mask[behind[1], behind[0]]


def test_residual_coverage_connects_unobserved_components_around_obstacle():
    obstacle = Point2D(0.0, 0.0)
    grid = SensorCoverageGrid(
        (-2.0, 2.0, -2.0, 2.0),
        0.1,
        [obstacle],
        clearance=0.45,
        boundary_margin=0.1,
        start=Point2D(-1.5, -1.5),
    )
    grid.observed_mask[:, :] = grid.target_mask
    for center in ((-1.0, 1.0), (1.0, -1.0)):
        column, row = grid.world_to_cell(center)
        grid.observed_mask[row - 2 : row + 3, column - 2 : column + 3] = False
    plan = plan_residual_coverage(
        grid,
        Point2D(-1.5, -1.5),
        goal_spacing=0.4,
        min_component_area=0.02,
    )
    assert plan is not None
    assert plan.component_count >= 2
    assert len(plan.goals) >= 2
    for point in plan.path.sampled(0.025):
        assert math.hypot(point.x - obstacle.x, point.y - obstacle.y) >= 0.42


def test_residual_coverage_is_empty_after_full_observation():
    grid = SensorCoverageGrid((-1.0, 1.0, -1.0, 1.0), 0.1, clearance=0.2, boundary_margin=0.0)
    grid.observed_mask[:, :] = grid.target_mask
    assert plan_residual_coverage(grid, Point2D(-0.8, -0.8)) is None
