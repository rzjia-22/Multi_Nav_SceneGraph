"""NoMaD 离线评测的几何指标。

这些函数只依赖 NumPy，便于在主机单元测试中核验。所有二维局部坐标均采用
``x 向前、y 向左`` 的机器人坐标约定。
"""

from __future__ import annotations

import math

import numpy as np


def local_to_world(points_local_xy: np.ndarray, origin_xy: np.ndarray, yaw: float) -> np.ndarray:
    """把局部二维点转换到世界坐标。"""

    local = np.asarray(points_local_xy, dtype=np.float64)
    cosine, sine = math.cos(float(yaw)), math.sin(float(yaw))
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    return local @ rotation.T + np.asarray(origin_xy, dtype=np.float64)


def world_to_local(points_world_xy: np.ndarray, origin_xy: np.ndarray, yaw: float) -> np.ndarray:
    """把世界二维点转换到局部机器人坐标。"""

    world = np.asarray(points_world_xy, dtype=np.float64)
    cosine, sine = math.cos(float(yaw)), math.sin(float(yaw))
    rotation = np.asarray([[cosine, sine], [-sine, cosine]], dtype=np.float64)
    return (world - np.asarray(origin_xy, dtype=np.float64)) @ rotation.T


def point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    """点到闭线段的欧氏距离。"""

    point = np.asarray(point, dtype=np.float64)
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    delta = end - start
    denominator = float(np.dot(delta, delta))
    if denominator <= 1.0e-15:
        return float(np.linalg.norm(point - start))
    factor = float(np.clip(np.dot(point - start, delta) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + factor * delta)))


def point_to_polyline_distances(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """计算每个点到折线的连续距离。"""

    query = np.asarray(points, dtype=np.float64)
    path = np.asarray(polyline, dtype=np.float64)
    if len(query) == 0:
        return np.empty(0, dtype=np.float64)
    if len(path) == 0:
        raise ValueError("polyline must contain at least one point")
    if len(path) == 1:
        return np.linalg.norm(query - path[0], axis=1)
    return np.asarray([
        min(point_segment_distance(point, start, end) for start, end in zip(path[:-1], path[1:]))
        for point in query
    ], dtype=np.float64)


def densify_polyline(polyline: np.ndarray, maximum_spacing_m: float = 0.05) -> np.ndarray:
    """按最大点间距加密折线，供连续走廊近似计算。"""

    path = np.asarray(polyline, dtype=np.float64)
    if len(path) == 0:
        raise ValueError("polyline must contain at least one point")
    if maximum_spacing_m <= 0.0:
        raise ValueError("maximum_spacing_m must be positive")
    dense = [path[0]]
    for start, end in zip(path[:-1], path[1:]):
        distance = float(np.linalg.norm(end - start))
        intervals = max(1, int(math.ceil(distance / maximum_spacing_m)))
        dense.extend(
            start + (end - start) * (step / intervals)
            for step in range(1, intervals + 1)
        )
    return np.asarray(dense, dtype=np.float64)


def trajectory_clearance(
    points_world_xy: np.ndarray,
    tree_centres_xy: np.ndarray,
    tree_radii_m: np.ndarray,
    robot_radius_m: float,
    scene_extent_m: tuple[float, float],
) -> float:
    """返回预测折线相对树木碰撞代理和场景边界的最小净空。"""

    points = np.asarray(points_world_xy, dtype=np.float64)
    if len(points) == 0:
        raise ValueError("trajectory must contain at least one point")
    centres = np.asarray(tree_centres_xy, dtype=np.float64)
    radii = np.asarray(tree_radii_m, dtype=np.float64)
    segments = [(points[0], points[0])] if len(points) == 1 else list(zip(points[:-1], points[1:]))
    tree_clearance = min(
        point_segment_distance(centre, start, end) - float(radius) - float(robot_radius_m)
        for centre, radius in zip(centres, radii)
        for start, end in segments
    )
    half_width, half_height = (float(value) / 2.0 for value in scene_extent_m)
    boundary_clearance = min(
        half_width - abs(float(point[0])) - float(robot_radius_m)
        for point in points
    )
    boundary_clearance = min(
        boundary_clearance,
        *(half_height - abs(float(point[1])) - float(robot_radius_m) for point in points),
    )
    return float(min(tree_clearance, boundary_clearance))


def maximum_true_streak(mask: np.ndarray) -> int:
    """返回布尔序列中最长连续 True 长度。"""

    longest = 0
    current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def pairwise_endpoint_diversity(trajectories: np.ndarray) -> float:
    """返回多样本末端点两两距离均值。"""

    values = np.asarray(trajectories, dtype=np.float64)
    if len(values) < 2:
        return 0.0
    endpoints = values[:, -1, :2]
    distances = [
        float(np.linalg.norm(endpoints[first] - endpoints[second]))
        for first in range(len(endpoints))
        for second in range(first + 1, len(endpoints))
    ]
    return float(np.mean(distances))
