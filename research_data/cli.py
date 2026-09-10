"""Small, composable command-line entry points for Dataset V0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import ROOT, load_yaml, scene_paths
from .forest import generate_scene
from .manifest import create_manifest
from .validation import validate_manifest, validate_scene


PREVIEW_SCENE = "train_scene_000"
PREVIEW_EPISODE = "train_scene_000_episode_000"


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init-manifest")
    scene_parser = commands.add_parser("generate-scene")
    scene_parser.add_argument("--scene-id", default=PREVIEW_SCENE)
    commands.add_parser("plan-preview")
    validate_parser = commands.add_parser("validate")
    commands.add_parser("visualize")
    args = parser.parse_args()
    if args.command == "init-manifest":
        create_manifest()
    elif args.command == "generate-scene":
        generate_scene(args.scene_id)
    elif args.command == "plan-preview":
        raise RuntimeError("expert planning is blocked until train_scene_000 passes human visual review")
    elif args.command == "validate":
        report = {"manifest": validate_manifest()}
        scene_path, _ = scene_paths(PREVIEW_SCENE)
        report["scene"] = validate_scene(scene_path)
        report["status"] = "PASS"
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "visualize":
        from .review import write_scene_layout

        scene_path, _ = scene_paths(PREVIEW_SCENE)
        output = ROOT / "artifacts/dataset_v0_scene_review" / PREVIEW_SCENE / "scene_layout.png"
        print(write_scene_layout(load_yaml(scene_path), output).relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
