"""Shared DIABLO surrogate and D435i rig geometry for research runtimes.

Dataset collection and closed-loop validation must render the same camera pose
for a given scene, episode and simulated timestamp.  This module deliberately
contains no ROS or model code so both Isaac entry points can reuse it.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def rotation_matrix_to_xyzw(matrix: np.ndarray) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix to a normalized xyzw quaternion."""
    matrix = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w, x = 0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x, y, z = (
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
            )
        elif axis == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x, y, z = (
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x, y, z = (
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
            )
    quaternion = np.asarray([x, y, z, w], dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion)


def surface_frame(surface, x: float, y: float, yaw: float) -> tuple[float, np.ndarray]:
    """Return terrain height and body x-forward/y-left/z-normal frame."""
    height, normal = surface.height_and_normal(x, y)
    normal = np.asarray(normal, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    heading = np.asarray([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float64)
    forward = heading - normal * float(np.dot(heading, normal))
    forward /= np.linalg.norm(forward)
    left = np.cross(normal, forward)
    left /= np.linalg.norm(left)
    forward = np.cross(left, normal)
    return float(height), np.column_stack((forward, left, normal))


def make_camera(sim_utils, Camera, CameraCfg, path: str, cfg: dict, sensor_rate: float, data_types: list[str]):
    """Create one pinhole sensor directly from the frozen D435i V0 profile."""
    width, height = (int(value) for value in cfg["resolution"])
    focal = 18.0
    horizontal_aperture = 2.0 * focal * math.tan(
        math.radians(float(cfg["fov_deg"]["horizontal"])) / 2.0
    )
    vertical_aperture = 2.0 * focal * math.tan(
        math.radians(float(cfg["fov_deg"]["vertical"])) / 2.0
    )
    clipping = cfg.get("clipping_range_m", cfg.get("valid_range_m"))
    return Camera(CameraCfg(
        prim_path=path,
        update_period=1.0 / sensor_rate,
        height=height,
        width=width,
        data_types=data_types,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal,
            horizontal_aperture=horizontal_aperture,
            vertical_aperture=vertical_aperture,
            clipping_range=(float(clipping[0]), float(clipping[1])),
        ),
    ))


def episode_mount_variation(robot: dict, scene_seed: int, episode_id: str) -> dict[str, float]:
    """Return the same deterministic mount variation used by Dataset V0."""
    episode_number = int(episode_id.rsplit("_", 1)[-1])
    rng = np.random.Generator(np.random.PCG64(int(scene_seed) + episode_number + 700_001))
    variation = robot["camera_mount"]["episode_variation"]
    return {
        "height_offset_m": float(rng.uniform(*variation["height_m"])),
        "pitch_offset_deg": float(rng.uniform(*variation["pitch_deg"])),
        "roll_offset_deg": float(rng.uniform(*variation["roll_deg"])),
    }


@dataclass(frozen=True)
class ResearchRigPose:
    ground_z: float
    body_rotation: np.ndarray
    body_xyzw: np.ndarray
    rgb_position: np.ndarray
    depth_position: np.ndarray
    optical_rotation: np.ndarray
    optical_xyzw: np.ndarray
    optical_wxyz: np.ndarray
    body_to_optical_translation: np.ndarray
    body_to_optical_xyzw: np.ndarray


def research_rig_pose(
    surface,
    x: float,
    y: float,
    yaw: float,
    sim_time: float,
    robot: dict,
    sensor: dict,
    mount_variation: dict[str, float],
) -> ResearchRigPose:
    """Compose terrain attitude, yaw, mount and correlated camera motion."""
    ground_z, body_rotation = surface_frame(surface, x, y, yaw)
    body_xyzw = rotation_matrix_to_xyzw(body_rotation)
    mount = robot["camera_mount"]
    correlation = mount["correlated_motion"]
    oscillation = (
        math.sin(2.0 * math.pi * float(correlation["frequency_hz"]) * sim_time)
        if correlation["enabled"] else 0.0
    )
    camera_height = (
        float(mount["nominal_height_m"])
        + float(mount_variation["height_offset_m"])
        + float(correlation["vertical_amplitude_m"]) * oscillation
    )
    pitch = math.radians(
        float(mount["nominal_pitch_deg"])
        + float(mount_variation["pitch_offset_deg"])
        + float(correlation["pitch_amplitude_deg"]) * oscillation
    )
    roll = math.radians(
        float(mount["nominal_roll_deg"])
        + float(mount_variation["roll_offset_deg"])
        + float(correlation["roll_amplitude_deg"]) * oscillation
    )
    body_forward, body_left, body_up = body_rotation.T
    optical_forward = math.cos(pitch) * body_forward + math.sin(pitch) * body_up
    optical_right_zero = -body_left
    optical_down_zero = np.cross(optical_forward, optical_right_zero)
    optical_down_zero /= np.linalg.norm(optical_down_zero)
    optical_right = math.cos(roll) * optical_right_zero + math.sin(roll) * optical_down_zero
    optical_down = -math.sin(roll) * optical_right_zero + math.cos(roll) * optical_down_zero
    optical_rotation = np.column_stack((optical_right, optical_down, optical_forward))
    optical_xyzw = rotation_matrix_to_xyzw(optical_rotation)
    optical_wxyz = np.asarray([optical_xyzw[3], *optical_xyzw[:3]])
    body_position = np.asarray([x, y, ground_z], dtype=np.float64)
    rgb_position = body_position + body_rotation @ np.asarray([
        float(mount["forward_offset_m"]),
        float(mount["lateral_offset_m"]),
        camera_height,
    ])
    depth_to_rgb_translation = np.asarray(
        sensor["calibration"]["depth_to_rgb"]["translation_m"], dtype=np.float64
    )
    depth_position = rgb_position - optical_rotation @ depth_to_rgb_translation
    body_to_optical_rotation = body_rotation.T @ optical_rotation
    return ResearchRigPose(
        ground_z=ground_z,
        body_rotation=body_rotation,
        body_xyzw=body_xyzw,
        rgb_position=rgb_position,
        depth_position=depth_position,
        optical_rotation=optical_rotation,
        optical_xyzw=optical_xyzw,
        optical_wxyz=optical_wxyz,
        body_to_optical_translation=body_rotation.T @ (rgb_position - body_position),
        body_to_optical_xyzw=rotation_matrix_to_xyzw(body_to_optical_rotation),
    )
