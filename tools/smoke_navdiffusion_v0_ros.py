#!/usr/bin/env python3
"""Load the project checkpoint through the real ROS DiffusionNavigatorNode."""

from __future__ import annotations

import json
from pathlib import Path
import time

import rclpy

from mns_navigation.diffusion_node import DiffusionNavigatorNode
from mns_navigation.navdiffusion_v0.predictor import MNSNavDiffusionPredictor


ROOT = Path("/workspace")
CHECKPOINT = ROOT / "models/trained/navdiffusion_v0/best.pt"
REPORT = ROOT / "models/trained/navdiffusion_v0/ros_runtime_smoke.json"


def main() -> int:
    rclpy.init(args=[
        "--ros-args",
        "-p", "model_backend:=mns_v0",
        "-p", f"checkpoint:={CHECKPOINT}",
        "-p", "device:=cpu",
    ])
    started = time.perf_counter()
    node = None
    try:
        node = DiffusionNavigatorNode()
        predictor = node.planner.predictor
        if not isinstance(predictor, MNSNavDiffusionPredictor):
            raise AssertionError(f"unexpected predictor type: {type(predictor)}")
        report = {
            "status": "PASS",
            "node": node.get_fully_qualified_name(),
            "model_backend": node.model_backend,
            "predictor": type(predictor).__name__,
            "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
            "checkpoint_metadata": predictor.checkpoint_metadata,
            "history_length": node.history.length,
            "output_waypoints": predictor.output_waypoints,
            "device": str(predictor.device),
            "load_wall_time_s": time.perf_counter() - started,
            "test_split_used": False,
        }
        REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
