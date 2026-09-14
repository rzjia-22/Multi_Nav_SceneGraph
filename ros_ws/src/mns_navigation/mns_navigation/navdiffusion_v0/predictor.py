"""Checkpoint-only inference adapter for MNS NavDiffusion V0."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

from .preprocessing import NavDiffusionPreprocessor


@dataclass(frozen=True)
class MNSNavDiffusionConfig:
    checkpoint: Path
    device: str = "cuda"
    output_waypoints: int = 8
    seed: int = 20260914


class MNSNavDiffusionPredictor:
    """Load a project-owned checkpoint without any training-only state/code."""

    def __init__(self, config: MNSNavDiffusionConfig) -> None:
        import torch
        from .model import NavDiffusionV0

        if not config.checkpoint.is_file():
            raise FileNotFoundError(f"MNS NavDiffusion checkpoint missing: {config.checkpoint}")
        self.torch = torch
        self.device = torch.device(
            config.device if config.device != "cuda" or torch.cuda.is_available() else "cpu"
        )
        checkpoint = torch.load(config.checkpoint, map_location="cpu", weights_only=True)
        if int(checkpoint.get("checkpoint_version", -1)) != 1:
            raise ValueError("unsupported MNS NavDiffusion checkpoint version")
        self.preprocessor = NavDiffusionPreprocessor.from_dict(checkpoint["preprocessing"])
        self.model = NavDiffusionV0(checkpoint["architecture_config"], pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval().to(self.device)
        self.output_waypoints = int(config.output_waypoints)
        self.seed = int(config.seed)
        self.last_inference_ms = 0.0
        self.last_full_prediction = np.empty((0, 2), dtype=np.float32)
        self.checkpoint_metadata = {
            key: checkpoint.get(key)
            for key in (
                "checkpoint_version", "model_name", "dataset_version",
                "dataset_index_sha256", "git_commit", "training_seed", "epoch",
                "epoch_index", "best_validation_ADE", "validation_FDE",
            )
        }

    def __call__(self, rgb: np.ndarray, depth_m: np.ndarray, goal_local: np.ndarray) -> np.ndarray:
        torch = self.torch
        images = self.preprocessor.prepare_history(rgb, np.asarray(depth_m)[..., 0]).unsqueeze(0).to(self.device)
        goal = torch.from_numpy(self.preprocessor.normalize_goal(np.asarray(goal_local)[:2])).unsqueeze(0).to(self.device)
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        started = time.perf_counter()
        with torch.inference_mode():
            normalized = self.model.sample_normalized(images, goal, generator=generator)[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.last_inference_ms = (time.perf_counter() - started) * 1000.0
        trajectory = self.preprocessor.inverse_trajectory(normalized.float().cpu().numpy())
        self.last_full_prediction = np.asarray(trajectory, dtype=np.float32).copy()
        return trajectory[: self.output_waypoints]
