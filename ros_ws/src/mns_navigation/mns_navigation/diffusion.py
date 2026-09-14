"""Runtime adapter for the original goal-conditioned NavDiffusion model.

The model remains a trajectory provider. It neither evaluates the Go2 gait nor
publishes joint actions. Checkpoint-specific imports are delayed so coverage
and Nav2 installations do not depend on PyTorch.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import sys
import types
from typing import Callable

import numpy as np
import yaml

from .coverage import Point2D


class RGBDHistory:
    def __init__(self, length: int = 5) -> None:
        if length <= 0:
            raise ValueError("history length must be positive")
        self.length = length
        self._rgb: deque[np.ndarray] = deque(maxlen=length)
        self._depth: deque[np.ndarray] = deque(maxlen=length)

    def append(self, rgb: np.ndarray, depth_m: np.ndarray) -> None:
        rgb = np.asarray(rgb)
        depth = np.asarray(depth_m)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError("RGB frame must have shape [H,W,3]")
        if depth.ndim == 2:
            depth = depth[..., None]
        if depth.ndim != 3 or depth.shape[-1] != 1 or depth.shape[:2] != rgb.shape[:2]:
            raise ValueError("registered depth must have shape [H,W,1]")
        clean_depth = np.nan_to_num(depth.astype(np.float32), nan=50.0, posinf=50.0, neginf=0.0)
        self._rgb.append(rgb.astype(np.float32) / (255.0 if rgb.dtype == np.uint8 else 1.0))
        self._depth.append(clean_depth)

    @property
    def ready(self) -> bool:
        return len(self._rgb) == self.length

    def clear(self) -> None:
        self._rgb.clear()
        self._depth.clear()

    def __len__(self) -> int:
        return len(self._rgb)

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.ready:
            raise RuntimeError(f"history needs {self.length} synchronized frames")
        return np.stack(self._rgb), np.stack(self._depth)


@dataclass(frozen=True)
class DiffusionConfig:
    checkpoint: Path
    model_config: Path
    source_path: Path | None = None
    device: str = "cuda"
    output_waypoints: int = 8


class OriginalNavDiffusionPredictor:
    """Load the upstream Lightning checkpoint without coupling it to ROS."""

    def __init__(self, config: DiffusionConfig) -> None:
        if not config.checkpoint.is_file():
            raise FileNotFoundError(f"diffusion checkpoint missing: {config.checkpoint}")
        if not config.model_config.is_file():
            raise FileNotFoundError(f"diffusion config missing: {config.model_config}")
        if config.source_path is not None:
            package_root = str(config.source_path.resolve())
            if package_root not in sys.path:
                sys.path.insert(0, package_root)
        try:
            import torch
            # The upstream class only needs Lightning's nn.Module conveniences
            # during inference.  Loading Lightning 1.8 on Jazzy's Python 3.12
            # is neither maintainable nor necessary, so provide a deliberately
            # tiny compatibility base and load tensor weights in safe mode.
            lightning = types.ModuleType("pytorch_lightning")

            class InferenceLightningModule(torch.nn.Module):
                def save_hyperparameters(self, *args, **kwargs) -> None:
                    return None

                @property
                def device(self):
                    parameter = next(self.parameters(), None)
                    return parameter.device if parameter is not None else torch.device("cpu")

            lightning.LightningModule = InferenceLightningModule
            sys.modules.setdefault("pytorch_lightning", lightning)
            warmup = types.ModuleType("warmup_scheduler")
            warmup.GradualWarmupScheduler = object
            sys.modules.setdefault("warmup_scheduler", warmup)
            from navdiffusion import NavDiffsionLightning
        except ImportError as error:
            raise RuntimeError(
                "original navdiffusion package and its pinned ML dependencies "
                "must be installed in the robotics image"
            ) from error
        self.torch = torch
        self.device = torch.device(
            config.device if config.device != "cuda" or torch.cuda.is_available() else "cpu"
        )
        checkpoint = torch.load(
            str(config.checkpoint), map_location="cpu", weights_only=True
        )
        hyperparameters = checkpoint.get("hyper_parameters", {})
        if "model_params" not in hyperparameters or "train_params" not in hyperparameters:
            raise ValueError("diffusion checkpoint lacks expected hyper_parameters")
        self.model = NavDiffsionLightning(
            hyperparameters["model_params"], hyperparameters["train_params"]
        )
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.eval().to(self.device)
        with config.model_config.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
        transform = raw["data_params"]["tranform"]
        self.history_length = int(raw["data_params"]["time_horizon"])
        self.size = tuple(int(value) for value in transform["resize"]["w_h"])
        self.mean = np.asarray(transform["norm"]["mean"], dtype=np.float32)
        self.std = np.asarray(transform["norm"]["std"], dtype=np.float32)
        self.output_waypoints = config.output_waypoints

    def __call__(self, rgb: np.ndarray, depth_m: np.ndarray, goal_local: np.ndarray) -> np.ndarray:
        torch = self.torch
        if len(rgb) != self.history_length:
            raise ValueError("history length does not match training configuration")
        rgbd = np.concatenate([rgb, np.clip(depth_m, 0.0, 50.0) / 50.0], axis=-1)
        tensor = torch.as_tensor(rgbd, dtype=torch.float32, device=self.device).permute(0, 3, 1, 2)
        tensor = torch.nn.functional.interpolate(
            tensor, size=(self.size[1], self.size[0]), mode="bilinear", align_corners=False
        )
        mean = torch.as_tensor(self.mean, device=self.device)[None, :, None, None]
        std = torch.as_tensor(self.std, device=self.device)[None, :, None, None]
        tensor = ((tensor - mean) / std).unsqueeze(0)
        goal = torch.as_tensor(goal_local[:2], dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            result = self.model.predict((tensor, goal, None))[0]
        return result[: self.output_waypoints].detach().cpu().numpy()


class DiffusionPlanner:
    def __init__(self, predictor: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray], history: RGBDHistory) -> None:
        self.predictor = predictor
        self.history = history
        self.last_local_trajectory = np.empty((0, 2), dtype=np.float32)
        self.last_goal_local = np.empty((0,), dtype=np.float32)

    def reset(self) -> None:
        self.history.clear()
        self.last_local_trajectory = np.empty((0, 2), dtype=np.float32)
        self.last_goal_local = np.empty((0,), dtype=np.float32)

    def trajectory(
        self, position: Point2D, yaw: float, goal: Point2D
    ) -> tuple[Point2D, ...]:
        rgb, depth = self.history.arrays()
        dx, dy = goal.x - position.x, goal.y - position.y
        cosine, sine = np.cos(yaw), np.sin(yaw)
        goal_local = np.asarray([cosine * dx + sine * dy, -sine * dx + cosine * dy])
        local = np.asarray(self.predictor(rgb, depth, goal_local), dtype=np.float32)
        if local.ndim != 2 or local.shape[1] != 2:
            raise ValueError("diffusion model output must have shape [N,2]")
        if not np.all(np.isfinite(local)):
            raise ValueError("diffusion model output contains NaN or Inf")
        self.last_goal_local = goal_local.astype(np.float32, copy=True)
        self.last_local_trajectory = local.copy()
        return tuple(
            Point2D(
                position.x + cosine * float(x) - sine * float(y),
                position.y + sine * float(x) + cosine * float(y),
            )
            for x, y in local
        )
