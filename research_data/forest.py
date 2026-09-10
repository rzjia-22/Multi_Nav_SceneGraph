"""Deterministic, asset-independent Research Forest scene generator."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .common import ROOT, dump_yaml, load_yaml, scene_paths, stable_hash, terrain_height


def _scene_plan(scene_id: str) -> dict[str, Any]:
    manifest = load_yaml(ROOT / "config/datasets/dataset_v0_manifest.yaml")
    matches = [item for item in manifest["scenes"] if item["scene_id"] == scene_id]
    if len(matches) != 1:
        raise ValueError(f"scene {scene_id!r} must appear exactly once in Dataset V0 manifest")
    return matches[0]


def generate_scene(scene_id: str, output_yaml: Path | None = None, output_usda: Path | None = None) -> dict[str, Any]:
    plan = _scene_plan(scene_id)
    profiles = load_yaml(ROOT / "config/research_forests/profiles.yaml")
    seed = int(plan["scene_seed"])
    rng = np.random.Generator(np.random.PCG64(seed))
    factors = plan["factors"]
    terrain_cfg = profiles["terrain"][factors["terrain"]]
    amplitude = float(terrain_cfg["amplitude_m"])
    waves = []
    if amplitude > 0.0:
        for fraction, angle_offset in ((0.62, 0.0), (0.28, 67.0), (0.10, 131.0)):
            waves.append(
                {
                    "amplitude_m": round(amplitude * fraction, 6),
                    "wavelength_m": round(float(terrain_cfg["wavelength_m"]) * float(rng.uniform(0.85, 1.15)), 6),
                    "direction_deg": round(float((rng.uniform(0.0, 360.0) + angle_offset) % 360.0), 6),
                    "phase_rad": round(float(rng.uniform(0.0, 2.0 * math.pi)), 6),
                }
            )
    tree_cfg = profiles["trees"]
    density_cfg = tree_cfg[factors["tree_density"]]
    half = float(profiles["scene_size_m"]) / 2.0
    margin = float(tree_cfg["boundary_margin_m"])
    min_spacing = float(density_cfg["minimum_spacing_m"])
    type_names = list(tree_cfg["types"])
    weights = np.asarray([tree_cfg["types"][name]["weight"] for name in type_names], dtype=float)
    weights /= weights.sum()
    trees: list[dict[str, Any]] = []
    attempts = 0
    while len(trees) < int(density_cfg["count"]) and attempts < int(density_cfg["count"]) * 500:
        attempts += 1
        x, y = rng.uniform(-half + margin, half - margin, size=2)
        if any(math.hypot(x - item["position_m"][0], y - item["position_m"][1]) < min_spacing for item in trees):
            continue
        tree_type = str(rng.choice(type_names, p=weights))
        radius = float(rng.uniform(*tree_cfg["trunk_radius_m"]))
        height = float(rng.uniform(*tree_cfg["trunk_height_m"]))
        scale = float(rng.uniform(*tree_cfg["scale"]))
        trees.append(
            {
                "tree_id": f"tree_{len(trees):03d}",
                "type": tree_type,
                "position_m": [round(float(x), 6), round(float(y), 6)],
                "ground_height_m": 0.0,
                "trunk_radius_m": round(radius * scale, 6),
                "trunk_height_m": round(height * scale, 6),
                "foliage_radius_m": round(height * float(tree_cfg["types"][tree_type]["foliage_radius_ratio"]) * scale, 6),
                "foliage_color": list(tree_cfg["types"][tree_type]["foliage_color"]),
                "yaw_deg": round(float(rng.uniform(0.0, 360.0)), 6),
                "scale": round(scale, 6),
                "asset_id": f"procedural/{tree_type}/v0",
            }
        )
    if len(trees) != int(density_cfg["count"]):
        raise RuntimeError(f"could place only {len(trees)} of {density_cfg['count']} trees")
    scene = {
        "schema_version": 1,
        "dataset_version": "dataset_v0",
        "scene_id": scene_id,
        "scene_seed": seed,
        "split": plan["split"],
        "generator": {"name": "mns_research_forest", "version": 1, "rng": "numpy.PCG64"},
        "extent_m": [float(profiles["scene_size_m"]), float(profiles["scene_size_m"])],
        "factors": factors,
        "terrain": {
            "profile": factors["terrain"],
            "waves": waves,
            "roughness_m": float(terrain_cfg["roughness_m"]),
            "max_slope_deg": float(terrain_cfg["max_slope_deg"]),
            "mesh_resolution_m": 0.5,
        },
        "ground": {"profile": factors["ground"], **profiles["ground"][factors["ground"]]},
        "lighting": {"profile": factors["lighting"], **profiles["lighting"][factors["lighting"]]},
        "physics": profiles["physics"],
        "semantics": profiles["semantics"],
        "trees": trees,
    }
    for tree in scene["trees"]:
        x, y = tree["position_m"]
        tree["ground_height_m"] = round(terrain_height(scene, x, y), 6)
    scene["content_hash"] = stable_hash(scene)
    default_yaml, default_usda = scene_paths(scene_id)
    output_yaml = output_yaml or default_yaml
    output_usda = output_usda or default_usda
    dump_yaml(output_yaml, scene)
    _write_usda(output_usda, scene)
    return scene


def _write_usda(path: Path, scene: dict[str, Any]) -> None:
    """Write a compact deterministic ASCII USD suitable for review and re-import."""
    extent_x, extent_y = scene["extent_m"]
    resolution = float(scene["terrain"]["mesh_resolution_m"])
    xs = np.arange(-extent_x / 2.0, extent_x / 2.0 + resolution / 2.0, resolution)
    ys = np.arange(-extent_y / 2.0, extent_y / 2.0 + resolution / 2.0, resolution)
    points = [(x, y, terrain_height(scene, float(x), float(y))) for y in ys for x in xs]
    width = len(xs)
    indices = []
    counts = []
    for row in range(len(ys) - 1):
        for column in range(width - 1):
            a = row * width + column
            indices.extend((a, a + 1, a + width + 1, a, a + width + 1, a + width))
            counts.extend((3, 3))
    point_text = ",\n            ".join(f"({x:.6f}, {y:.6f}, {z:.6f})" for x, y, z in points)
    index_text = ", ".join(str(value) for value in indices)
    count_text = ", ".join(str(value) for value in counts)
    ground_color = scene["ground"]["rgb"]
    lines = [
        "#usda 1.0",
        "(",
        "    defaultPrim = \"World\"",
        "    metersPerUnit = 1",
        "    upAxis = \"Z\"",
        ")",
        "",
        'def Xform "World" {',
        '    def Mesh "Terrain" (prepend apiSchemas = ["PhysicsCollisionAPI", "SemanticsAPI:Semantics"]) {',
        '        uniform token subdivisionScheme = "none"',
        f"        color3f[] primvars:displayColor = [({ground_color[0]}, {ground_color[1]}, {ground_color[2]})]",
        '        uniform token semantics:Semantics:semanticType = "class"',
        '        uniform string semantics:Semantics:semanticData = "ground"',
        f"        int[] faceVertexCounts = [{count_text}]",
        f"        int[] faceVertexIndices = [{index_text}]",
        f"        point3f[] points = [{point_text}]",
        "    }",
        '    def Xform "Forest" {',
    ]
    for tree in scene["trees"]:
        x, y = tree["position_m"]
        z = tree["ground_height_m"]
        radius = tree["trunk_radius_m"]
        height = tree["trunk_height_m"]
        foliage = tree["foliage_radius_m"]
        foliage_color = tree["foliage_color"]
        lines.extend(
            [
                f'        def Xform "{tree["tree_id"]}" {{',
                '            def Cylinder "Trunk" (prepend apiSchemas = ["PhysicsCollisionAPI", "SemanticsAPI:Semantics"]) {',
                '                uniform token axis = "Z"',
                f"                double radius = {radius}",
                f"                double height = {height}",
                "                color3f[] primvars:displayColor = [(0.30, 0.12, 0.04)]",
                '                uniform token semantics:Semantics:semanticType = "class"',
                '                uniform string semantics:Semantics:semanticData = "tree_trunk"',
                f"                double3 xformOp:translate = ({x}, {y}, {z + height / 2.0})",
                '                uniform token[] xformOpOrder = ["xformOp:translate"]',
                "            }",
                '            def Sphere "Foliage" (prepend apiSchemas = ["SemanticsAPI:Semantics"]) {',
                f"                double radius = {foliage}",
                f"                color3f[] primvars:displayColor = [({foliage_color[0]}, {foliage_color[1]}, {foliage_color[2]})]",
                '                uniform token semantics:Semantics:semanticType = "class"',
                '                uniform string semantics:Semantics:semanticData = "foliage"',
                f"                double3 xformOp:translate = ({x}, {y}, {z + height - 0.25})",
                '                uniform token[] xformOpOrder = ["xformOp:translate"]',
                "            }",
                "        }",
            ]
        )
    lines.extend(("    }", "}", ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
