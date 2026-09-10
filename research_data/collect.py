"""Single-worker, resumable Dataset V0 collection orchestration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

from .common import ROOT, dataset_scene_directory, episode_directory, load_yaml, scene_paths
from .forest import generate_scene


ENTRYPOINT = "/mns/containers/simulation/dataset_entrypoint.sh"
ASSETS = "/mns/config/research_forests/assets.yaml"


def _manifest_scenes() -> list[dict]:
    return load_yaml(ROOT / "config/datasets/dataset_v0_manifest.yaml")["scenes"]


def _container_path(path: Path) -> str:
    return "/mns/" + str(path.relative_to(ROOT))


def _run_scene(scene_spec: dict, episode_ids: list[str], plan_only: bool, report_name: str) -> None:
    scene_id = scene_spec["scene_id"]
    scene_path, _ = scene_paths(scene_id)
    output_directory = dataset_scene_directory(scene_id)
    output_directory.mkdir(parents=True, exist_ok=True)
    command = [
        "docker", "compose", "--profile", "simulation", "run", "--rm",
        "--entrypoint", ENTRYPOINT, "simulation",
        "--scene", _container_path(scene_path), "--assets", ASSETS,
        "--output-directory", _container_path(output_directory),
        "--benchmark-output", _container_path(output_directory / report_name),
    ]
    for episode_id in episode_ids:
        command.extend(("--episode-id", episode_id))
    if plan_only:
        command.append("--plan-only")
    else:
        command.append("--enable_cameras")
    command.append("--headless")
    subprocess.run(command, cwd=ROOT, check=True)


def _run_tool(*arguments: str) -> None:
    subprocess.run([
        "docker", "compose", "--profile", "simulation", "run", "--rm",
        "--entrypoint", ENTRYPOINT, "simulation", "--tool", *arguments,
    ], cwd=ROOT, check=True)


def regenerate_pilot() -> None:
    scene = _manifest_scenes()[0]
    assert scene["scene_id"] == "train_scene_000"
    generate_scene(scene["scene_id"])
    episode_id = "train_scene_000_episode_000"
    directory = episode_directory(episode_id)
    if directory.exists() and (directory / "episode.h5").exists():
        _run_tool("validate-episode", "--episode-id", episode_id)
        print(f"MNS_DATASET_RESUME_SKIP={episode_id}", flush=True)
        return
    _run_scene(scene, [episode_id], True, "pilot_plan_preflight_session.json")
    _run_scene(scene, [episode_id], False, "pilot_collection_session.json")
    _run_tool("validate-episode", "--episode-id", episode_id)
    _run_tool("visualize-episode", "--episode-id", episode_id)
    obsolete = ROOT / "artifacts/dataset_v0_preview/train_scene_000_episode_000"
    if obsolete.exists():
        shutil.rmtree(obsolete)


def batch_gate() -> None:
    scene = _manifest_scenes()[0]
    episode_ids = [item["episode_id"] for item in scene["planned_episodes"]][1:]
    _run_scene(scene, episode_ids, True, "batch_gate_plan_preflight_session.json")
    _run_scene(scene, episode_ids, False, "batch_gate_collection_session.json")
    for episode_id in episode_ids:
        _run_tool("validate-episode", "--episode-id", episode_id)
    _run_tool("validate")


def all_plan_preflight() -> None:
    started = time.monotonic()
    for scene in _manifest_scenes():
        generate_scene(scene["scene_id"])
        episode_ids = [item["episode_id"] for item in scene["planned_episodes"]]
        _run_scene(scene, episode_ids, True, "plan_preflight_session.json")
    _run_tool("aggregate-plan-preflight")
    print(json.dumps({"status": "PASS", "wall_time_s": time.monotonic() - started}), flush=True)


def collect_scenes(scene_ids: list[str] | None = None) -> None:
    preflight = ROOT / "datasets/dataset_v0/dataset_v0_plan_preflight.json"
    if not preflight.exists() or json.loads(preflight.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("70/70 plan preflight must pass before bulk RGB-D collection")
    selected = [scene for scene in _manifest_scenes() if scene["scene_id"] != "train_scene_000"]
    if scene_ids:
        requested = set(scene_ids)
        selected = [scene for scene in selected if scene["scene_id"] in requested]
        missing = requested - {scene["scene_id"] for scene in selected}
        if missing:
            raise ValueError(f"unknown or gate scene IDs: {sorted(missing)}")
    for scene in selected:
        episode_ids = [item["episode_id"] for item in scene["planned_episodes"]]
        _run_scene(scene, episode_ids, False, "collection_session.json")
    _run_tool("validate", *( ["--require-complete"] if scene_ids is None else [] ))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("regenerate-pilot")
    commands.add_parser("batch-gate")
    commands.add_parser("plan-preflight")
    collect = commands.add_parser("collect")
    collect.add_argument("--scene-id", action="append")
    args = parser.parse_args()
    if args.command == "regenerate-pilot":
        regenerate_pilot()
    elif args.command == "batch-gate":
        batch_gate()
    elif args.command == "plan-preflight":
        all_plan_preflight()
    elif args.command == "collect":
        collect_scenes(args.scene_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
