"""Deterministic Research Forest scene specification generator.

This module deliberately owns specification, not rendering geometry. The
Isaac-dependent builder consumes the YAML and uses Isaac Lab terrain plus
NVIDIA USD assets to populate the stage.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .common import ROOT, dump_yaml, load_yaml, scene_paths, stable_hash


def _scene_plan(scene_id: str) -> dict[str, Any]:
    manifest = load_yaml(ROOT / "config/datasets/dataset_v0_manifest.yaml")
    matches = [item for item in manifest["scenes"] if item["scene_id"] == scene_id]
    if len(matches) != 1:
        raise ValueError(f"scene {scene_id!r} must appear exactly once in Dataset V0 manifest")
    return matches[0]


def _weighted_asset(rng: np.random.Generator, tree_assets: dict[str, Any]) -> str:
    asset_ids = list(tree_assets)
    weights = np.asarray([float(tree_assets[asset_id]["weight"]) for asset_id in asset_ids], dtype=float)
    weights /= weights.sum()
    return str(rng.choice(asset_ids, p=weights))


def generate_scene(scene_id: str, output_yaml: Path | None = None) -> dict[str, Any]:
    """Realize one version-controlled scene specification from the V0 manifest."""
    plan = _scene_plan(scene_id)
    profiles = load_yaml(ROOT / "config/research_forests/profiles.yaml")
    registry = load_yaml(ROOT / "config/research_forests/assets.yaml")
    seed = int(plan["scene_seed"])
    rng = np.random.Generator(np.random.PCG64(seed))
    factors = plan["factors"]

    density_cfg = profiles["trees"][factors["tree_density"]]
    tree_assets = profiles["trees"]["assets"]
    half = float(profiles["scene_size_m"]) / 2.0
    margin = float(profiles["trees"]["boundary_margin_m"])
    min_spacing = float(density_cfg["minimum_spacing_m"])
    trees: list[dict[str, Any]] = []
    attempts = 0
    while len(trees) < int(density_cfg["count"]) and attempts < int(density_cfg["count"]) * 500:
        attempts += 1
        x, y = rng.uniform(-half + margin, half - margin, size=2)
        if any(math.hypot(x - item["position_m"][0], y - item["position_m"][1]) < min_spacing for item in trees):
            continue
        asset_id = _weighted_asset(rng, tree_assets)
        asset = registry["trees"][asset_id]
        scale_multiplier = float(rng.uniform(*tree_assets[asset_id]["scale_multiplier_range"]))
        trees.append(
            {
                "tree_id": f"tree_{len(trees):03d}",
                "asset_id": asset_id,
                "position_m": [round(float(x), 6), round(float(y), 6)],
                "ground_height_policy": "terrain_raycast_top_surface",
                "yaw_deg": round(float(rng.uniform(0.0, 360.0)), 6),
                "scale_multiplier": round(scale_multiplier, 6),
                "spawn_scale": [round(float(value) * scale_multiplier, 8) for value in asset["spawn_scale"]],
                "collision_proxy": {
                    "shape": "hidden_cylinder",
                    "radius_m": round(float(asset["collision_proxy"]["radius_m"]) * scale_multiplier, 6),
                    "height_m": round(float(asset["collision_proxy"]["height_m"]) * scale_multiplier, 6),
                },
            }
        )
    if len(trees) != int(density_cfg["count"]):
        raise RuntimeError(f"could place only {len(trees)} of {density_cfg['count']} trees")

    terrain = profiles["terrain"][factors["terrain"]]
    ground = registry["ground_materials"][factors["ground"]]
    lighting_profile = profiles["lighting"][factors["lighting"]]
    sky = registry["sky"][lighting_profile["sky_asset_id"]]
    scene = {
        "schema_version": 2,
        "dataset_version": "dataset_v0",
        "scene_id": scene_id,
        "scene_seed": seed,
        "split": plan["split"],
        "generator": {
            "name": "mns_research_forest",
            "version": 2,
            "rng": "numpy.PCG64",
            "isaac_sim": "5.1.0",
            "isaac_lab": "2.3.1",
        },
        "extent_m": [float(profiles["scene_size_m"]), float(profiles["scene_size_m"])],
        "factors": factors,
        "asset_registry": "config/research_forests/assets.yaml",
        "terrain": {
            "profile": factors["terrain"],
            "builder": "isaaclab.terrains.TerrainImporter",
            "generator": terrain,
            "seed": seed,
        },
        "ground": {
            "profile": factors["ground"],
            "material_asset_id": factors["ground"],
            "material_uri": ground["uri"],
            "texture_scale": ground["texture_scale"],
        },
        "lighting": {
            "profile": factors["lighting"],
            **lighting_profile,
            "sky_uri": sky["uri"],
        },
        "physics": profiles["physics"],
        "semantics": profiles["semantics"],
        "tree_count": len(trees),
        "trees": trees,
        "review_views": profiles["review_views"],
    }
    scene["content_hash"] = stable_hash(scene)
    default_yaml, _ = scene_paths(scene_id)
    dump_yaml(output_yaml or default_yaml, scene)
    return scene
