"""Non-photorealistic geometry review artifacts for Research Forest scenes."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any


def write_scene_layout(scene: dict[str, Any], output: Path) -> Path:
    """Write a planner-style XY schematic; this is not a rendered screenshot."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    colors = {
        "nvidia_blue_berry_elder": "#2f6b3b",
        "nvidia_gray_birch": "#78a663",
    }
    figure, axis = plt.subplots(figsize=(8, 8), constrained_layout=True)
    for tree in scene["trees"]:
        x, y = tree["position_m"]
        radius = float(tree["collision_proxy"]["radius_m"])
        axis.add_patch(
            Circle(
                (x, y),
                radius,
                facecolor=colors.get(tree["asset_id"], "#527a4a"),
                edgecolor="#183d20",
                linewidth=0.7,
                alpha=0.9,
            )
        )
    half_x, half_y = (float(value) / 2.0 for value in scene["extent_m"])
    axis.set(xlim=(-half_x, half_x), ylim=(-half_y, half_y), xlabel="x (m)", ylabel="y (m)")
    axis.set_aspect("equal")
    axis.grid(True, alpha=0.18)
    counts = Counter(tree["asset_id"] for tree in scene["trees"])
    distribution = ", ".join(f"{key}: {value}" for key, value in sorted(counts.items()))
    axis.set_title(
        f"{scene['scene_id']} geometry layout (schematic, not RTX)\n"
        f"seed={scene['scene_seed']}; {distribution}"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output
