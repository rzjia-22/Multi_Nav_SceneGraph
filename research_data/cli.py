"""Small, composable command-line entry points for Dataset V0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import ROOT, episode_directory, load_yaml, scene_paths
from .expert import write_preview_plan
from .forest import generate_scene
from .manifest import create_manifest
from .validation import validate_episode, validate_manifest, validate_scene, write_report


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
    validate_parser.add_argument("--with-episode", action="store_true")
    commands.add_parser("visualize")
    args = parser.parse_args()
    if args.command == "init-manifest":
        create_manifest()
    elif args.command == "generate-scene":
        generate_scene(args.scene_id)
    elif args.command == "plan-preview":
        scene_path, _ = scene_paths(PREVIEW_SCENE)
        output = episode_directory(PREVIEW_EPISODE) / "episode_plan.yaml"
        write_preview_plan(load_yaml(scene_path), output)
    elif args.command == "validate":
        report = {"manifest": validate_manifest()}
        scene_path, _ = scene_paths(PREVIEW_SCENE)
        report["scene"] = validate_scene(scene_path)
        if args.with_episode:
            directory = episode_directory(PREVIEW_EPISODE)
            report["episode"] = validate_episode(directory / "episode.h5", directory / "episode_plan.yaml")
        report["status"] = "PASS"
        if args.with_episode:
            report_path = episode_directory(PREVIEW_EPISODE) / "validation_report.json"
            write_report(report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "visualize":
        from .visualization import visualize

        scene_path, _ = scene_paths(PREVIEW_SCENE)
        directory = episode_directory(PREVIEW_EPISODE)
        for path in visualize(scene_path, directory / "episode_plan.yaml", directory / "episode.h5", directory):
            print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
