"""Interactive and fixed-view RTX review for one Dataset V0 forest scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--mode", choices=("interactive", "capture"), default="interactive")
    parser.add_argument("--warmup-frames", type=int, default=16)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


ARGS = parse_args()
APP = AppLauncher(ARGS).app


def _gpu_snapshot() -> dict:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, timeout=5).strip().split(", ")
        return {
            "name": output[0],
            "memory_used_mib": int(output[1]),
            "memory_total_mib": int(output[2]),
            "utilization_percent": int(output[3]),
        }
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {"available": False}


def main() -> int:
    import numpy as np
    import torch
    from PIL import Image

    import isaaclab.sim as sim_utils
    from isaaclab.sensors import Camera, CameraCfg
    from isaaclab.sim import SimulationContext

    from research_data.common import load_yaml
    from mns_simulation.research_forest_scene import (
        build_research_forest,
        export_stage_snapshot,
        sample_terrain_height,
    )

    scene = load_yaml(ARGS.scene)
    registry = load_yaml(ARGS.assets)
    output_directory = ARGS.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    simulation = SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 60.0, render_interval=1, device=ARGS.device)
    )
    initial_eye = scene["review_views"]["aerial"]["eye_m"]
    initial_target = scene["review_views"]["aerial"]["target_m"]
    simulation.set_camera_view(initial_eye, initial_target)

    start = time.monotonic()
    built = build_research_forest(simulation, scene, registry)
    load_time = time.monotonic() - start
    snapshot_path = ARGS.scene.parent / "scene.usda"
    export_stage_snapshot(snapshot_path)

    base_report = {
        "status": "PASS",
        "scene_id": scene["scene_id"],
        "scene_content_hash": scene["content_hash"],
        "isaac_sim_version": "5.1.0",
        "isaac_lab_version": "2.3.1",
        "terrain_builder": scene["terrain"]["builder"],
        "tree_count": len(scene["trees"]),
        "resolved_assets": built.resolved_assets,
        "missing_registry_assets": built.missing_registry_assets,
        "scene_load_time_s": load_time,
        "stage_snapshot": str(snapshot_path),
    }

    if ARGS.mode == "interactive":
        print("MNS_RESEARCH_FOREST_READY=" + json.dumps(base_report, sort_keys=True), flush=True)
        while APP.is_running():
            simulation.step(render=True)
        return 0

    if not ARGS.enable_cameras:
        raise ValueError("capture mode requires --enable_cameras")
    camera = Camera(
        CameraCfg(
            prim_path="/World/ResearchForest/ReviewCamera",
            update_period=0.0,
            height=int(ARGS.height),
            width=int(ARGS.width),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0,
                horizontal_aperture=25.0,
                clipping_range=(0.05, 100.0),
            ),
        )
    )
    simulation.reset()

    views = scene["review_views"]
    ground = views["ground"]
    ground_xy = ground["eye_xy_height_m"][:2]
    ground_target_xy = ground["target_xy_height_m"][:2]
    ground_eye = [*ground_xy, sample_terrain_height(*ground_xy) + float(ground["eye_xy_height_m"][2])]
    ground_target = [
        *ground_target_xy,
        sample_terrain_height(*ground_target_xy) + float(ground["target_xy_height_m"][2]),
    ]
    mid = views["mid_height"]
    mid_xy = mid["eye_xy_height_m"][:2]
    mid_target_xy = mid["target_xy_height_m"][:2]
    mid_eye = [*mid_xy, sample_terrain_height(*mid_xy) + float(mid["eye_xy_height_m"][2])]
    mid_target = [
        *mid_target_xy,
        sample_terrain_height(*mid_target_xy) + float(mid["target_xy_height_m"][2]),
    ]
    close = views["closeup"]
    anchor = next(tree for tree in scene["trees"] if tree["tree_id"] == close["anchor_tree_id"])
    anchor_x, anchor_y = anchor["position_m"]
    angle = np.deg2rad(float(anchor["yaw_deg"]) + 180.0)
    close_eye_xy = [
        float(anchor_x) + float(close["distance_m"]) * float(np.cos(angle)),
        float(anchor_y) + float(close["distance_m"]) * float(np.sin(angle)),
    ]
    close_eye = [
        *close_eye_xy,
        sample_terrain_height(*close_eye_xy) + float(close["camera_height_m"]),
    ]
    close_target = [
        float(anchor_x),
        float(anchor_y),
        built.tree_ground_heights_m[anchor["tree_id"]] + float(close["target_height_m"]),
    ]
    resolved_views = {
        "isaac_aerial.png": (views["aerial"]["eye_m"], views["aerial"]["target_m"]),
        "isaac_ground_view.png": (ground_eye, ground_target),
        "isaac_mid_height.png": (mid_eye, mid_target),
        "isaac_tree_closeup.png": (close_eye, close_target),
    }

    captured = {}
    render_start = time.monotonic()
    total_frames = 0
    for filename, (eye, target) in resolved_views.items():
        camera.set_world_poses_from_view(
            torch.tensor([eye], dtype=torch.float32, device=simulation.device),
            torch.tensor([target], dtype=torch.float32, device=simulation.device),
        )
        for _ in range(int(ARGS.warmup_frames)):
            simulation.step(render=True)
            camera.update(simulation.get_physics_dt())
            total_frames += 1
        rgb = camera.data.output["rgb"][0].detach().cpu().numpy()[..., :3].astype(np.uint8)
        path = output_directory / filename
        Image.fromarray(rgb).save(path)
        captured[filename] = {
            "path": str(path),
            "eye_m": [float(value) for value in eye],
            "target_m": [float(value) for value in target],
            "mean_rgb": float(rgb.mean()),
            "std_rgb": float(rgb.std()),
        }
    render_seconds = time.monotonic() - render_start
    base_report.update(
        {
            "mode": "capture",
            "resolution": [int(ARGS.width), int(ARGS.height)],
            "review_images": captured,
            "rendered_frames": total_frames,
            "render_wall_time_s": render_seconds,
            "average_render_fps": total_frames / render_seconds,
            "gpu": _gpu_snapshot(),
        }
    )
    report_path = output_directory / "isaac_review_report.json"
    report_path.write_text(json.dumps(base_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("MNS_RESEARCH_FOREST_RESULT=" + json.dumps(base_report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except BaseException:
        traceback.print_exc()
    finally:
        APP.close(wait_for_replicator=False, skip_cleanup=True)
    sys.exit(code)
