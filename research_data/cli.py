"""Small, composable command-line entry points for Dataset V0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import ROOT, load_yaml, scene_paths
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
    commands.add_parser("visualize")
    commands.add_parser("visualize-episode")
    args = parser.parse_args()
    if args.command == "init-manifest":
        create_manifest()
    elif args.command == "generate-scene":
        generate_scene(args.scene_id)
    elif args.command == "validate":
        report = {"manifest": validate_manifest(), "runtime_contract": validate_runtime_contract()}
        scene_path, _ = scene_paths(PREVIEW_SCENE)
        report["scene"] = validate_scene(scene_path)
        episode_directory = ROOT / "artifacts/dataset_v0_preview" / PREVIEW_EPISODE
        episode_path = episode_directory / "episode.h5"
        plan_path = episode_directory / "episode_plan.yaml"
        report["episode"] = validate_episode(episode_path, plan_path) if episode_path.exists() else {"status": "NOT_GENERATED"}
        benchmark_path = episode_directory / "benchmark_report.json"
        if benchmark_path.exists():
            benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
            assert benchmark["scene_content_hash"] == report["scene"]["content_hash"]
            assert benchmark["episode_schema_version"] == 2
            assert benchmark["episode"]["alignment_round_trip"]["valid_pixel_agreement"] == 1.0
            assert benchmark["episode"]["alignment_round_trip"]["maximum_depth_difference_m"] == 0.0
            assert benchmark["episode"]["depth_quantization"]["saturation_count"] == 0
            for name, value in report["scene"]["terrain_statistics"].items():
                measured = benchmark["terrain_statistics"][name]
                tolerance = 1.0e-5 if isinstance(value, float) else 0
                assert abs(measured - value) <= tolerance, f"scene/episode terrain statistic mismatch: {name}"
            report["benchmark"] = {
                "status": "PASS",
                "alignment_round_trip": benchmark["episode"]["alignment_round_trip"],
                "depth_quantization": benchmark["episode"]["depth_quantization"],
                "hardware_recommendation": benchmark["hardware_recommendation"],
            }
        report["status"] = "PASS"
        if episode_path.exists():
            write_report(episode_directory / "validation_report.json", report)
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "visualize":
        from .review import write_scene_layout

        scene_path, _ = scene_paths(PREVIEW_SCENE)
        output = ROOT / "artifacts/dataset_v0_scene_review" / PREVIEW_SCENE / "scene_layout.png"
        print(write_scene_layout(load_yaml(scene_path), output).relative_to(ROOT))
    elif args.command == "visualize-episode":
        from .visualization import visualize

        scene_path, _ = scene_paths(PREVIEW_SCENE)
        episode_directory = ROOT / "artifacts/dataset_v0_preview" / PREVIEW_EPISODE
        created = visualize(
            scene_path,
            episode_directory / "episode_plan.yaml",
            episode_directory / "episode.h5",
            episode_directory,
        )
        print("\n".join(str(path.relative_to(ROOT)) for path in created))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
