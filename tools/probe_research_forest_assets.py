"""Probe candidate NVIDIA assets from inside the pinned Isaac runtime.

This is a diagnostics-only tool.  It does not generate a scene or download
assets into the repository.
"""

from __future__ import annotations

import argparse
import json

from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


ARGS = parse_args()
APP = AppLauncher(ARGS).app


def main() -> None:
    import omni.client
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR, NUCLEUS_ASSET_ROOT_DIR
    from pxr import Usd, UsdGeom

    candidates = [
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Vegetation/Trees/Blue_Berry_Elder.usd",
        f"{NVIDIA_NUCLEUS_DIR}/Assets/Vegetation/Trees/Blue_Berry_Elder.usd",
        f"{NUCLEUS_ASSET_ROOT_DIR}/Materials/2023_1/Base/Natural/Dirt.mdl",
        f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Natural/Dirt.mdl",
        f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
    ]
    results = {}
    for uri in candidates:
        result, entry = omni.client.stat(uri)
        results[uri] = {
            "result": str(result),
            "size": getattr(entry, "size", None) if entry is not None else None,
        }
    tree_directory = f"{NVIDIA_NUCLEUS_DIR}/Assets/Vegetation/Trees"
    list_result, entries = omni.client.list(tree_directory)
    tree_entries = sorted(entry.relative_path for entry in entries) if list_result == omni.client.Result.OK else []
    natural_material_directory = f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Natural"
    material_list_result, material_entries = omni.client.list(natural_material_directory)
    natural_material_entries = (
        sorted(entry.relative_path for entry in material_entries)
        if material_list_result == omni.client.Result.OK
        else []
    )
    inspected_trees = {}
    for filename in ("Blue_Berry_Elder.usd", "Gray_Birch.usd", "White_Pine.usd"):
        uri = f"{tree_directory}/{filename}"
        stage = Usd.Stage.Open(uri)
        if stage is None:
            inspected_trees[filename] = {"opened": False}
            continue
        default_prim = stage.GetDefaultPrim()
        bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(default_prim)
        aligned = bbox.ComputeAlignedBox()
        inspected_trees[filename] = {
            "opened": True,
            "default_prim": default_prim.GetPath().pathString,
            "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
            "range_min": list(aligned.GetMin()),
            "range_max": list(aligned.GetMax()),
            "prim_names": [prim.GetName() for prim in stage.Traverse()][:40],
        }
    print(
        "MNS_RESEARCH_ASSET_PROBE="
        + json.dumps(
            {
                "asset_root": NUCLEUS_ASSET_ROOT_DIR,
                "candidates": results,
                "tree_directory": tree_directory,
                "tree_directory_result": str(list_result),
                "tree_entries": tree_entries,
                "natural_material_directory": natural_material_directory,
                "natural_material_directory_result": str(material_list_result),
                "natural_material_entries": natural_material_entries,
                "inspected_trees": inspected_trees,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        APP.close(wait_for_replicator=False, skip_cleanup=True)
