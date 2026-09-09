#!/usr/bin/env python3
"""Load the pinned NavDiffusion checkpoint and execute one deterministic inference."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mns_navigation.diffusion import DiffusionConfig, OriginalNavDiffusionPredictor
from mns_simulation.go2_policy import load_legacy_actor


def main() -> None:
    predictor = OriginalNavDiffusionPredictor(
        DiffusionConfig(
            checkpoint=Path("/workspace/models/navdiffusion.ckpt"),
            model_config=Path("/workspace/config/navigation/diffusion.yaml"),
            source_path=Path("/opt/forestnavigation/forest_nav"),
            device="cpu",
        )
    )
    predictor.torch.manual_seed(7)
    output = predictor(
        np.zeros((5, 120, 160, 3), dtype=np.float32),
        np.full((5, 120, 160, 1), 3.0, dtype=np.float32),
        np.asarray([1.5, 0.0], dtype=np.float32),
    )
    if output.shape != (8, 2):
        raise RuntimeError(f"unexpected trajectory shape: {output.shape}")
    if not np.isfinite(output).all():
        raise RuntimeError("trajectory contains non-finite values")
    actor = load_legacy_actor("/workspace/models/go2_locomotion.pt", "cpu")
    actor_output = actor(predictor.torch.zeros((2, 48)))
    if tuple(actor_output.shape) != (2, 12):
        raise RuntimeError(f"unexpected Go2 actor shape: {tuple(actor_output.shape)}")
    if not predictor.torch.isfinite(actor_output).all():
        raise RuntimeError("Go2 actor output contains non-finite values")
    print(json.dumps({
        "checkpoint": "navdiffusion.ckpt",
        "device": str(predictor.device),
        "shape": list(output.shape),
        "finite": True,
        "minimum": float(output.min()),
        "maximum": float(output.max()),
        "go2_actor_shape": list(actor_output.shape),
        "go2_actor_finite": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
