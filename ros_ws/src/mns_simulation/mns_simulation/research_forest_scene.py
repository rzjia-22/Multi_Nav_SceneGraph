"""Shared Isaac Lab builder for every Dataset V0 Research Forest runtime."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any


_ACTIVE_TERRAIN_SURFACE = None


class TerrainSurfaceQuery:
    """Exact XY queries over the top-facing triangles authored by Isaac Lab.

    The query is built from the actual USD mesh after ``TerrainImporter`` has
    generated it.  It is therefore shared ground truth for rendering, tree
    placement, the kinematic surrogate, cameras, and stored terrain metrics.
    """

    def __init__(self, points, faces, extent_m, horizontal_scale_m):
        import numpy as np

        self._np = np
        self.points = np.asarray(points, dtype=np.float64)
        triangles = self.points[np.asarray(faces, dtype=np.int64)]
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        norms = np.linalg.norm(cross, axis=1)
        usable = norms > 1.0e-10
        cross[usable] /= norms[usable, None]
        cross[cross[:, 2] < 0.0] *= -1.0
        # Side walls and degenerate caps are not terrain surfaces.
        usable &= cross[:, 2] > 1.0e-4
        self.triangles = triangles[usable]
        self.normals = cross[usable]
        self.extent_m = tuple(float(value) for value in extent_m)
        self.bin_size_m = max(float(horizontal_scale_m), 0.05)
        self._bins: dict[tuple[int, int], list[int]] = {}
        for index, triangle in enumerate(self.triangles):
            minimum = np.floor(triangle[:, :2].min(axis=0) / self.bin_size_m).astype(int)
            maximum = np.floor(triangle[:, :2].max(axis=0) / self.bin_size_m).astype(int)
            for ix in range(int(minimum[0]), int(maximum[0]) + 1):
                for iy in range(int(minimum[1]), int(maximum[1]) + 1):
                    self._bins.setdefault((ix, iy), []).append(index)
        if not self._bins:
            raise RuntimeError("Isaac terrain mesh has no queryable top-facing triangles")

    def height_and_normal(self, x: float, y: float):
        """Return the highest containing triangle's height and upward normal."""
        np = self._np
        key = (math.floor(float(x) / self.bin_size_m), math.floor(float(y) / self.bin_size_m))
        candidates = self._bins.get(key, ())
        matches = []
        query = np.asarray([float(x), float(y)], dtype=np.float64)
        for index in candidates:
            triangle = self.triangles[index]
            a, b, c = triangle[:, :2]
            denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            if abs(float(denominator)) < 1.0e-12:
                continue
            first = ((b[1] - c[1]) * (query[0] - c[0]) + (c[0] - b[0]) * (query[1] - c[1])) / denominator
            second = ((c[1] - a[1]) * (query[0] - c[0]) + (a[0] - c[0]) * (query[1] - c[1])) / denominator
            third = 1.0 - first - second
            if min(first, second, third) >= -1.0e-7:
                z = first * triangle[0, 2] + second * triangle[1, 2] + third * triangle[2, 2]
                matches.append((float(z), self.normals[index]))
        if not matches:
            raise ValueError(f"point ({x:.3f}, {y:.3f}) is outside the imported terrain surface")
        height, normal = max(matches, key=lambda item: item[0])
        return height, normal.copy()

    def height(self, x: float, y: float) -> float:
        return self.height_and_normal(x, y)[0]

    def normal(self, x: float, y: float):
        return self.height_and_normal(x, y)[1]

    def statistics(self) -> dict[str, float | int]:
        """Summarize actual top-surface geometry, excluding the outer rim."""
        np = self._np
        half_x, half_y = self.extent_m[0] / 2.0, self.extent_m[1] / 2.0
        centers = self.triangles.mean(axis=1)
        margin = self.bin_size_m * 1.01
        interior = (
            (np.abs(centers[:, 0]) < half_x - margin)
            & (np.abs(centers[:, 1]) < half_y - margin)
        )
        triangles = self.triangles[interior]
        normals = self.normals[interior]
        if len(triangles) == 0:
            raise RuntimeError("no interior terrain triangles available for statistics")
        elevations = triangles[:, :, 2].reshape(-1)
        slopes = np.degrees(np.arccos(np.clip(normals[:, 2], 0.0, 1.0)))
        return {
            "sample_count": int(len(slopes)),
            "elevation_min_m": float(elevations.min()),
            "elevation_max_m": float(elevations.max()),
            "elevation_range_m": float(elevations.max() - elevations.min()),
            "elevation_std_m": float(elevations.std()),
            "slope_mean_deg": float(slopes.mean()),
            "slope_median_deg": float(np.median(slopes)),
            "slope_p90_deg": float(np.percentile(slopes, 90)),
            "slope_p95_deg": float(np.percentile(slopes, 95)),
            "slope_max_deg": float(slopes.max()),
        }


@dataclass
class BuiltResearchForest:
    terrain: Any
    surface: TerrainSurfaceQuery
    terrain_statistics: dict[str, float | int]
    resolved_assets: dict[str, str]
    tree_ground_heights_m: dict[str, float]
    missing_registry_assets: list[str]


def _resolve_uri(uri: str) -> str:
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR, NUCLEUS_ASSET_ROOT_DIR

    return uri.format(
        ISAAC_NUCLEUS_DIR=ISAAC_NUCLEUS_DIR,
        NVIDIA_NUCLEUS_DIR=NVIDIA_NUCLEUS_DIR,
        NUCLEUS_ASSET_ROOT_DIR=NUCLEUS_ASSET_ROOT_DIR,
    )


def _add_semantics(stage, path: str, label: str) -> None:
    from pxr import Semantics

    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"semantic target does not exist: {path}")
    semantic = Semantics.SemanticsAPI.Apply(prim, "Semantics")
    semantic.CreateSemanticTypeAttr().Set("class")
    semantic.CreateSemanticDataAttr().Set(label)


def _terrain_generator_cfg(scene: dict[str, Any]):
    from isaaclab.terrains import TerrainGeneratorCfg
    from isaaclab.terrains.height_field.hf_terrains_cfg import HfRandomUniformTerrainCfg, HfWaveTerrainCfg

    cfg = scene["terrain"]["generator"]
    common = {"proportion": 1.0, "border_width": 0.0}
    if cfg["subterrain_type"] == "hf_wave":
        subterrain = HfWaveTerrainCfg(
            **common,
            amplitude_range=tuple(float(value) for value in cfg["amplitude_range_m"]),
            num_waves=int(cfg["num_waves"]),
        )
    elif cfg["subterrain_type"] == "hf_random_uniform":
        subterrain = HfRandomUniformTerrainCfg(
            **common,
            noise_range=tuple(float(value) for value in cfg["noise_range_m"]),
            noise_step=float(cfg["noise_step_m"]),
            downsampled_scale=float(cfg["downsampled_scale_m"]),
        )
    else:
        raise ValueError(f"unsupported official terrain profile: {cfg['subterrain_type']}")
    return TerrainGeneratorCfg(
        seed=int(scene["terrain"]["seed"]),
        curriculum=False,
        size=tuple(float(value) for value in scene["extent_m"]),
        border_width=0.0,
        num_rows=1,
        num_cols=1,
        horizontal_scale=float(cfg["horizontal_scale_m"]),
        vertical_scale=float(cfg["vertical_scale_m"]),
        slope_threshold=math.tan(math.radians(float(cfg["maximum_design_slope_deg"]))),
        difficulty_range=tuple(float(value) for value in cfg["difficulty_range"]),
        color_scheme="none",
        use_cache=False,
        sub_terrains={cfg["subterrain_type"]: subterrain},
    )


def _sun_orientation(scene: dict[str, Any]) -> tuple[float, float, float, float]:
    pitch = math.radians(90.0 - float(scene["lighting"]["sun_elevation_deg"]))
    yaw = math.radians(float(scene["lighting"]["sun_azimuth_deg"]))
    return (
        math.cos(pitch / 2.0) * math.cos(yaw / 2.0),
        -math.sin(pitch / 2.0) * math.sin(yaw / 2.0),
        math.sin(pitch / 2.0) * math.cos(yaw / 2.0),
        math.cos(pitch / 2.0) * math.sin(yaw / 2.0),
    )


def sample_terrain_height(x: float, y: float) -> float:
    """Compatibility facade for review cameras; backed by the active mesh."""
    if _ACTIVE_TERRAIN_SURFACE is None:
        raise RuntimeError("terrain surface query is not initialized")
    return _ACTIVE_TERRAIN_SURFACE.height(x, y)


def build_research_forest(simulation, scene: dict[str, Any], registry: dict[str, Any]) -> BuiltResearchForest:
    """Build terrain, materials, vegetation, lighting, collision, and semantics."""
    import omni.client
    import omni.usd
    from pxr import Gf, UsdGeom

    import isaaclab.sim as sim_utils
    from isaaclab.terrains import TerrainImporter, TerrainImporterCfg

    global _ACTIVE_TERRAIN_SURFACE
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage, "/World/ResearchForest")

    resolved_assets: dict[str, str] = {}
    for asset_id, asset in registry["trees"].items():
        resolved_assets[asset_id] = _resolve_uri(asset["uri"])
    resolved_assets[scene["ground"]["material_asset_id"]] = _resolve_uri(scene["ground"]["material_uri"])
    resolved_assets[scene["lighting"]["sky_asset_id"]] = _resolve_uri(scene["lighting"]["sky_uri"])
    missing = [asset_id for asset_id, uri in resolved_assets.items() if omni.client.stat(uri)[0] != omni.client.Result.OK]
    if missing:
        raise RuntimeError(f"registry assets unavailable: {missing}")

    ground_material = sim_utils.MdlFileCfg(
        mdl_path=resolved_assets[scene["ground"]["material_asset_id"]],
        project_uvw=True,
        texture_scale=tuple(float(value) for value in scene["ground"]["texture_scale"]),
    )
    physics = scene["physics"]
    terrain = TerrainImporter(
        TerrainImporterCfg(
            prim_path="/World/ResearchForest/Terrain",
            terrain_type="generator",
            terrain_generator=_terrain_generator_cfg(scene),
            collision_group=-1,
            visual_material=ground_material,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=float(physics["static_friction"]),
                dynamic_friction=float(physics["dynamic_friction"]),
                restitution=float(physics["restitution"]),
            ),
            debug_vis=False,
        )
    )
    _add_semantics(stage, "/World/ResearchForest/Terrain/terrain", "ground")
    terrain_mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/ResearchForest/Terrain/terrain/mesh"))
    if not terrain_mesh:
        raise RuntimeError("Isaac Lab TerrainImporter did not create the expected terrain mesh")
    import numpy as np

    local_points = terrain_mesh.GetPointsAttr().Get()
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(terrain_mesh.GetPrim())
    points = np.asarray([transform.Transform(point) for point in local_points], dtype=np.float64)
    counts = list(terrain_mesh.GetFaceVertexCountsAttr().Get())
    indices = list(terrain_mesh.GetFaceVertexIndicesAttr().Get())
    faces = []
    cursor = 0
    for count in counts:
        polygon = indices[cursor : cursor + count]
        cursor += count
        for offset in range(1, count - 1):
            faces.append((polygon[0], polygon[offset], polygon[offset + 1]))
    surface = TerrainSurfaceQuery(
        points,
        faces,
        scene["extent_m"],
        scene["terrain"]["generator"]["horizontal_scale_m"],
    )
    _ACTIVE_TERRAIN_SURFACE = surface

    dome_cfg = sim_utils.DomeLightCfg(
        intensity=float(scene["lighting"]["dome_intensity"]),
        texture_file=resolved_assets[scene["lighting"]["sky_asset_id"]],
        texture_format="latlong",
        visible_in_primary_ray=True,
    )
    dome_cfg.func("/World/ResearchForest/Sky", dome_cfg)
    sun_cfg = sim_utils.DistantLightCfg(
        intensity=float(scene["lighting"]["sun_intensity"]),
        angle=0.53,
        color=(1.0, 0.94, 0.84),
    )
    sun_cfg.func(
        "/World/ResearchForest/Sun",
        sun_cfg,
        translation=(0.0, 0.0, 10.0),
        orientation=_sun_orientation(scene),
    )

    simulation.reset()
    for _ in range(3):
        simulation.step(render=False)

    UsdGeom.Xform.Define(stage, "/World/ResearchForest/Trees")
    ground_heights: dict[str, float] = {}
    for tree in scene["trees"]:
        asset = registry["trees"][tree["asset_id"]]
        x, y = (float(value) for value in tree["position_m"])
        ground_z = surface.height(x, y)
        ground_heights[tree["tree_id"]] = ground_z
        root_path = f"/World/ResearchForest/Trees/{tree['tree_id']}"
        root = UsdGeom.Xform.Define(stage, root_path)
        root.AddTranslateOp().Set(Gf.Vec3d(x, y, ground_z - 0.04))
        root.AddRotateZOp().Set(float(tree["yaw_deg"]))

        visual_path = root_path + "/Visual"
        tree_cfg = sim_utils.UsdFileCfg(
            usd_path=resolved_assets[tree["asset_id"]],
            scale=tuple(float(value) for value in tree["spawn_scale"]),
            semantic_tags=[("class", asset["semantic_policy"]["root"])],
        )
        tree_cfg.func(visual_path, tree_cfg)
        for relative_path, semantic_label in asset["semantic_policy"].get("prim_overrides", {}).items():
            _add_semantics(stage, f"{visual_path}/{relative_path}", semantic_label)

        proxy = tree["collision_proxy"]
        proxy_cfg = sim_utils.CylinderCfg(
            radius=float(proxy["radius_m"]),
            height=float(proxy["height_m"]),
            visible=False,
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
        proxy_cfg.func(
            root_path + "/CollisionProxy",
            proxy_cfg,
            translation=(0.0, 0.0, float(proxy["height_m"]) / 2.0),
        )

    simulation.reset()
    return BuiltResearchForest(
        terrain=terrain,
        surface=surface,
        terrain_statistics=surface.statistics(),
        resolved_assets=resolved_assets,
        tree_ground_heights_m=ground_heights,
        missing_registry_assets=missing,
    )


def export_stage_snapshot(path: Path) -> None:
    import omni.usd

    path.parent.mkdir(parents=True, exist_ok=True)
    stage = omni.usd.get_context().get_stage()
    # Export only the authored root layer. Usd.Stage.Export() flattens every
    # referenced tree and turns a compact reproducible composition into a
    # multi-gigabyte file that would also duplicate NVIDIA content.
    if not stage.GetRootLayer().Export(str(path)):
        raise RuntimeError(f"failed to export Research Forest stage to {path}")
