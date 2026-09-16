"""Lightweight, navigator-independent trajectory review figures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..common import ROOT, load_yaml


def _draw_forest(axis, scene: dict, robot_radius_m: float = 0.0) -> None:
    import matplotlib.pyplot as plt

    width, height = (float(value) for value in scene["extent_m"])
    axis.set_xlim(-width / 2.0, width / 2.0)
    axis.set_ylim(-height / 2.0, height / 2.0)
    axis.set_aspect("equal")
    for tree in scene["trees"]:
        axis.add_patch(plt.Circle(
            tree["position_m"], float(tree["collision_proxy"]["radius_m"]) + robot_radius_m,
            color="#654321", alpha=0.35,
        ))
    axis.grid(alpha=0.15)
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")


def render_episode_trace(trace_path: Path, output: Path, *, robot_radius_m: float = 0.0) -> Path:
    """Render one compact post-run trajectory view from an ignored raw trace."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    report = trace["report"]
    scene = load_yaml(ROOT / "research_scenes/dataset_v0" / report["scene_id"] / "scene.yaml")
    state = np.asarray(trace["state"], dtype=np.float64)
    expert = np.asarray(trace["expert_reference_path"], dtype=np.float64)
    planning = [item for item in trace["planning"] if item.get("status") == "PASS"]
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 8), constrained_layout=True)
    _draw_forest(axis, scene, robot_radius_m)
    axis.plot(expert[:, 0], expert[:, 1], "--", color="#ff7f0e", label="expert (analysis only)")
    if len(state):
        axis.plot(state[:, 1], state[:, 2], color="#1f77b4", linewidth=2.0, label="executed")
    if planning:
        indices = sorted({0, len(planning) // 2, len(planning) - 1})
        for color, index in zip(("#2ca02c", "#9467bd", "#d62728"), indices):
            points = np.asarray(planning[index]["full_world_points"], dtype=np.float64)
            axis.plot(points[:, 0], points[:, 1], color=color, linewidth=1.2,
                      label=f"prediction t={planning[index]['timestamp_from_episode_start_s']:.1f}s")
    axis.scatter(expert[0, 0], expert[0, 1], s=70, label="start")
    axis.scatter(expert[-1, 0], expert[-1, 1], marker="*", s=150, label="goal")
    if report.get("collision") and len(state):
        axis.scatter(state[-1, 1], state[-1, 2], marker="X", s=130, color="black", label="collision")
    axis.set_title(f"{report['episode_id']}: {'PASS' if report['success'] else report['failure_reason']}")
    axis.legend(fontsize=7)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output
