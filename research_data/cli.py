"""Small, composable command-line entry points for Dataset V0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import ROOT, episode_directory, load_yaml, scene_paths
from .dataset import build_dataset_index
from .expert import validate_plan
from .forest import generate_scene
from .manifest import create_manifest
from .validation import validate_episode, validate_manifest, validate_runtime_contract, validate_scene, write_report


PREVIEW_SCENE = "train_scene_000"
PREVIEW_EPISODE = "train_scene_000_episode_000"


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init-manifest")
    scene_parser = commands.add_parser("generate-scene")
    scene_parser.add_argument("--scene-id", default=PREVIEW_SCENE)
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("--require-complete", action="store_true")
    episode_validate_parser = commands.add_parser("validate-episode")
    episode_validate_parser.add_argument("--episode-id", required=True)
    commands.add_parser("aggregate-plan-preflight")
    commands.add_parser("visualize")
    visualize_episode_parser = commands.add_parser("visualize-episode")
    visualize_episode_parser.add_argument("--episode-id", default=PREVIEW_EPISODE)
    args = parser.parse_args()
    if args.command == "init-manifest":
        create_manifest()
    elif args.command == "generate-scene":
        generate_scene(args.scene_id)
    elif args.command == "validate":
        report = {
            "manifest": validate_manifest(), "runtime_contract": validate_runtime_contract(),
            "dataset_index": build_dataset_index(require_complete=args.require_complete),
        }
        report["status"] = "PASS" if report["dataset_index"]["episode_count"] == 70 else "PARTIAL"
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "validate-episode":
        directory = episode_directory(args.episode_id)
        report = validate_episode(directory / "episode.h5", directory / "episode_plan.yaml")
        write_report(directory / "validation_report.json", report)
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "aggregate-plan-preflight":
        manifest = load_yaml(ROOT / "config/datasets/dataset_v0_manifest.yaml")
        plans = []
        failures = []
        for scene_spec in manifest["scenes"]:
            scene_path, _ = scene_paths(scene_spec["scene_id"])
            scene = load_yaml(scene_path)
            for episode_spec in scene_spec["planned_episodes"]:
                plan_path = episode_directory(episode_spec["episode_id"]) / "episode_plan.yaml"
                try:
                    result = validate_plan(scene, load_yaml(plan_path))
                    result["plan_path"] = str(plan_path.relative_to(ROOT))
                    plans.append(result)
                except Exception as error:
                    failures.append({"episode_id": episode_spec["episode_id"], "error": str(error)})
        report = {
            "status": "PASS" if len(plans) == 70 and not failures else "FAIL",
            "plan_count": len(plans), "failure_count": len(failures), "failures": failures,
            "bucket_counts": {
                name: sum(item["route_bucket"] == name for item in plans)
                for name in ("short", "medium", "long")
            },
            "planner_version": 2, "plans": plans,
        }
        write_report(ROOT / "datasets/dataset_v0/dataset_v0_plan_preflight.json", report)
        assert report["status"] == "PASS", failures
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "visualize":
        from .review import write_scene_layout

        scene_path, _ = scene_paths(PREVIEW_SCENE)
        output = ROOT / "artifacts/dataset_v0_scene_review" / PREVIEW_SCENE / "scene_layout.png"
        print(write_scene_layout(load_yaml(scene_path), output).relative_to(ROOT))
    elif args.command == "visualize-episode":
        from .visualization import visualize

        directory = episode_directory(args.episode_id)
        plan = load_yaml(directory / "episode_plan.yaml")
        scene_path, _ = scene_paths(plan["scene_id"])
        created = visualize(
            scene_path,
            directory / "episode_plan.yaml",
            directory / "episode.h5",
            directory,
        )
        print("\n".join(str(path.relative_to(ROOT)) for path in created))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
