"""Single preprocessing contract shared by Dataset V0 training and ROS inference."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def yaw_from_xyzw(quaternion: np.ndarray) -> np.ndarray:
    """Return ZYX yaw for one or more xyzw quaternions."""
    q = np.asarray(quaternion, dtype=np.float64)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def local_coordinates(points_xy: np.ndarray, origin_xy: np.ndarray, yaw: float) -> np.ndarray:
    """Transform world XY into the anchor robot planar frame."""
    delta = np.asarray(points_xy, dtype=np.float64) - np.asarray(origin_xy, dtype=np.float64)
    cosine, sine = math.cos(float(yaw)), math.sin(float(yaw))
    rotation = np.asarray([[cosine, sine], [-sine, cosine]], dtype=np.float64)
    return delta @ rotation.T


def world_coordinates(points_local_xy: np.ndarray, origin_xy: np.ndarray, yaw: float) -> np.ndarray:
    """Inverse of :func:`local_coordinates`."""
    local = np.asarray(points_local_xy, dtype=np.float64)
    cosine, sine = math.cos(float(yaw)), math.sin(float(yaw))
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    return local @ rotation.T + np.asarray(origin_xy, dtype=np.float64)


@dataclass(frozen=True)
class NavDiffusionPreprocessor:
    width: int
    height: int
    history_length: int
    rgb_mean: tuple[float, float, float]
    rgb_std: tuple[float, float, float]
    depth_max_m: float
    depth_invalid_fill_m: float
    depth_mean: float
    depth_std: float
    goal_scale_m: float
    trajectory_bound_m: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "history_length": self.history_length,
            "rgb_mean": list(self.rgb_mean),
            "rgb_std": list(self.rgb_std),
            "depth_max_m": self.depth_max_m,
            "depth_invalid_fill_m": self.depth_invalid_fill_m,
            "depth_mean": self.depth_mean,
            "depth_std": self.depth_std,
            "goal_scale_m": self.goal_scale_m,
            "trajectory_bound_m": self.trajectory_bound_m,
            "depth_invalid_policy": "fill_far",
            "rgb_interpolation": "bilinear",
            "depth_interpolation": "nearest",
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "NavDiffusionPreprocessor":
        keys = (
            "width", "height", "history_length", "rgb_mean", "rgb_std",
            "depth_max_m", "depth_invalid_fill_m", "depth_mean", "depth_std",
            "goal_scale_m", "trajectory_bound_m",
        )
        missing = [key for key in keys if key not in value]
        if missing:
            raise ValueError(f"checkpoint preprocessing metadata is incomplete: {missing}")
        return cls(
            width=int(value["width"]), height=int(value["height"]),
            history_length=int(value["history_length"]),
            rgb_mean=tuple(float(v) for v in value["rgb_mean"]),
            rgb_std=tuple(float(v) for v in value["rgb_std"]),
            depth_max_m=float(value["depth_max_m"]),
            depth_invalid_fill_m=float(value["depth_invalid_fill_m"]),
            depth_mean=float(value["depth_mean"]), depth_std=float(value["depth_std"]),
            goal_scale_m=float(value["goal_scale_m"]),
            trajectory_bound_m=float(value["trajectory_bound_m"]),
        )

    @classmethod
    def from_files(
        cls,
        config_path: str | Path,
        report_path: str | Path | None = None,
        *,
        require_statistics: bool = True,
    ) -> "NavDiffusionPreprocessor":
        with Path(config_path).open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        report: dict[str, Any] = {}
        if report_path is not None and Path(report_path).is_file():
            import json

            report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        depth = config["inputs"]["depth"]
        mean = report.get("depth", {}).get("mean", depth.get("mean"))
        std = report.get("depth", {}).get("std", depth.get("std"))
        bound = report.get("trajectory", {}).get(
            "selected_symmetric_bound_m",
            config["target"]["normalization"]["default_symmetric_bound_m"],
        )
        if require_statistics and (mean is None or std is None):
            raise ValueError("training-split depth statistics are required")
        width, height = config["inputs"]["image_size_wh"]
        return cls(
            width=int(width),
            height=int(height),
            history_length=int(config["inputs"]["history_frames"]),
            rgb_mean=tuple(float(v) for v in config["inputs"]["rgb"]["mean"]),
            rgb_std=tuple(float(v) for v in config["inputs"]["rgb"]["std"]),
            depth_max_m=float(depth["maximum_m"]),
            depth_invalid_fill_m=float(depth["invalid_fill_m"]),
            depth_mean=float(mean if mean is not None else 0.0),
            depth_std=float(std if std is not None else 1.0),
            goal_scale_m=float(config["goal"]["scale_m"]),
            trajectory_bound_m=float(bound),
        )

    def resize_rgb_depth(self, rgb: np.ndarray, aligned_depth_m: np.ndarray):
        """Return standardized RGB and unstandardized normalized depth.

        The depth result is exactly the population used to calculate train-only
        mean/std. Runtime calls the same method before applying those statistics.
        """
        import torch
        import torch.nn.functional as functional

        rgb_array = np.asarray(rgb)
        depth_array = np.asarray(aligned_depth_m, dtype=np.float32)
        if rgb_array.ndim == 3:
            rgb_array = rgb_array[None, ...]
        if depth_array.ndim == 2:
            depth_array = depth_array[None, ...]
        if rgb_array.ndim != 4 or rgb_array.shape[-1] != 3:
            raise ValueError("RGB must have shape [N,H,W,3]")
        if depth_array.shape != rgb_array.shape[:3]:
            raise ValueError("aligned depth must have shape [N,H,W]")
        rgb_tensor = torch.as_tensor(rgb_array, dtype=torch.float32).permute(0, 3, 1, 2)
        if rgb_array.dtype == np.uint8 or float(rgb_tensor.max()) > 1.5:
            rgb_tensor = rgb_tensor / 255.0
        rgb_tensor = functional.interpolate(
            rgb_tensor, size=(self.height, self.width), mode="bilinear", align_corners=False
        )
        mean = torch.tensor(self.rgb_mean, dtype=torch.float32)[None, :, None, None]
        std = torch.tensor(self.rgb_std, dtype=torch.float32)[None, :, None, None]
        rgb_tensor = (rgb_tensor - mean) / std

        valid = np.isfinite(depth_array) & (depth_array > 0.0)
        depth_clean = np.where(valid, depth_array, self.depth_invalid_fill_m)
        depth_clean = np.clip(depth_clean, 0.0, self.depth_max_m) / self.depth_max_m
        depth_tensor = torch.as_tensor(depth_clean, dtype=torch.float32)[:, None]
        depth_tensor = functional.interpolate(
            depth_tensor, size=(self.height, self.width), mode="nearest"
        )[:, 0]
        return rgb_tensor.numpy(), depth_tensor.numpy(), float((~valid).mean())

    def prepare_history(self, rgb: np.ndarray, aligned_depth_m: np.ndarray):
        """Create one model history tensor with shape [T,4,H,W]."""
        import torch

        rgb_ready, depth_unit, _ = self.resize_rgb_depth(rgb, aligned_depth_m)
        if len(rgb_ready) != self.history_length:
            raise ValueError(f"history must contain exactly {self.history_length} frames")
        depth_model = (depth_unit - self.depth_mean) / self.depth_std
        return torch.cat(
            (torch.from_numpy(rgb_ready), torch.from_numpy(depth_model[:, None])), dim=1
        ).float()

    def normalize_goal(self, goal_local_m: np.ndarray):
        return np.asarray(goal_local_m, dtype=np.float32) / self.goal_scale_m

    def normalize_trajectory(self, trajectory_local_m: np.ndarray):
        trajectory = np.asarray(trajectory_local_m, dtype=np.float32)
        if np.max(np.abs(trajectory), initial=0.0) > self.trajectory_bound_m + 1.0e-6:
            raise ValueError("trajectory label exceeds the selected non-clipping bound")
        return trajectory / self.trajectory_bound_m

    def inverse_trajectory(self, trajectory_normalized: np.ndarray):
        return np.asarray(trajectory_normalized, dtype=np.float32) * self.trajectory_bound_m
