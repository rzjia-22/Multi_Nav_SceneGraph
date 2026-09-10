"""Measure Dataset V0 terrain profiles from Isaac Lab generated meshes."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


ARGS = parse_args()
APP = AppLauncher(ARGS).app


def main() -> int:
    from isaaclab.terrains import TerrainGenerator

    from research_data.common import dump_yaml, load_yaml
    from mns_simulation.research_forest_scene import TerrainSurfaceQuery, _terrain_generator_cfg

    scene = load_yaml(ARGS.scene)
    profiles = load_yaml(ARGS.profiles)
    report = {
        "schema_version": 1,
        "source": "IsaacLab TerrainGenerator.terrain_mesh",
        "isaac_sim_version": "5.1.0",
        "isaac_lab_version": "2.3.1",
        "calibration_seed": int(scene["scene_seed"]),
        "extent_m": [float(value) for value in scene["extent_m"]],
        "profiles": {},
    }
    for profile_name in ("flat", "gentle", "moderate"):
        candidate = copy.deepcopy(scene)
        candidate["terrain"]["profile"] = profile_name
        candidate["terrain"]["generator"] = copy.deepcopy(profiles["terrain"][profile_name])
        generator = TerrainGenerator(_terrain_generator_cfg(candidate), device="cpu")
        mesh = generator.terrain_mesh
        surface = TerrainSurfaceQuery(
            mesh.vertices,
            mesh.faces,
            candidate["extent_m"],
            candidate["terrain"]["generator"]["horizontal_scale_m"],
        )
        report["profiles"][profile_name] = {
            "generator": copy.deepcopy(candidate["terrain"]["generator"]),
            "actual_geometry_statistics": surface.statistics(),
        }
    dump_yaml(ARGS.output, report)
    print("MNS_TERRAIN_CALIBRATION_RESULT=" + json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except BaseException:
        traceback.print_exc()
    finally:
        APP.close()
    sys.exit(code)
