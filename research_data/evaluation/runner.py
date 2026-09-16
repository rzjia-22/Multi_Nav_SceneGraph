"""Scene-batched Research Forest navigation evaluation runner."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

from ..common import ROOT, load_yaml
from .reporting import (
    atomic_write_json,
    build_evaluation_report,
    dataset_index_entries,
    scene_id_from_episode,
)
from .visualization import render_episode_trace


DEFAULT_CONFIG = ROOT / "config/navigation/research_navigation_eval.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_config(config: dict, split: str, *, allow_test: bool = False) -> list[dict]:
    if int(config.get("config_version", 0)) != 1:
        raise ValueError("unsupported evaluation config version")
    allowed = set(config["evaluation"]["allowed_splits"])
    if split not in allowed or split not in {"validation", "test"}:
        raise ValueError(f"split {split!r} is not allowed for held-out evaluation")
    if split == "test" and config["evaluation"].get("test_requires_explicit_opt_in", True) and not allow_test:
        raise ValueError("test evaluation requires --allow-test")
    if config.get("mapping_enabled"):
        raise ValueError("Research Navigation evaluation must keep mapping disabled")
    checkpoint = ROOT / config["navigator"]["checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    expected = str(config["navigator"]["checkpoint_sha256"])
    if _sha256(checkpoint) != expected:
        raise ValueError("navigator checkpoint SHA256 mismatch")
    return dataset_index_entries(split)


def compose_session(
    config: dict,
    evaluation_id: str,
    scene_id: str,
    episode_ids: list[str],
    *,
    split: str,
    run_root: Path,
    capture_review: bool,
) -> tuple[int, str]:
    if any(scene_id_from_episode(value) != scene_id for value in episode_ids):
        raise ValueError("a scene-batched session may contain only one scene")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{evaluation_id}-{scene_id}-{timestamp}"
    navigator = config["navigator"]
    runtime = config["runtime"]
    environment = os.environ.copy()
    environment.update({
        "MNS_EVAL_SCENE": scene_id,
        "MNS_EVAL_EPISODES": ",".join(episode_ids),
        "MNS_EVAL_RUN_ID": run_id,
        "MNS_EVAL_ID": evaluation_id,
        "MNS_EVAL_SPLIT": split,
        "MNS_EVAL_RUN_ROOT": "/mns/" + str(run_root.relative_to(ROOT)),
        "MNS_EVAL_CAPTURE_FLAG": "--capture-review" if capture_review else "",
        "MNS_EVAL_ROBOT_ID": str(config["robot_id"]),
        "MNS_EVAL_NAVIGATOR_ID": str(navigator["navigator_id"]),
        "MNS_EVAL_NAVIGATOR_PACKAGE": str(navigator["package"]),
        "MNS_EVAL_NAVIGATOR_EXECUTABLE": str(navigator["executable"]),
        "MNS_EVAL_MODEL_BACKEND": str(navigator["model_backend"]),
        "MNS_EVAL_CHECKPOINT": "/workspace/" + str(navigator["checkpoint"]),
        "MNS_EVAL_CHECKPOINT_SHA256": str(navigator["checkpoint_sha256"]),
        "MNS_EVAL_DEVICE": str(navigator["device"]),
        "MNS_EVAL_HISTORY_LENGTH": str(navigator["history_length"]),
        "MNS_EVAL_PLANNING_RATE_HZ": str(navigator["planning_rate_hz"]),
        "MNS_EVAL_PREDICTION_WAYPOINTS": str(navigator["prediction_waypoints"]),
        "MNS_EVAL_CONTROL_WAYPOINTS": str(navigator["control_waypoints"]),
        "MNS_EVAL_GOAL_TOLERANCE_M": str(navigator["goal_tolerance_m"]),
        "MNS_EVAL_MAXIMUM_DURATION_S": str(runtime["maximum_episode_duration_s"]),
        "MNS_EVAL_COMMAND_TIMEOUT_S": str(runtime["command_timeout_s"]),
    })
    services = ["simulation-research-navigation", "robotics-research-navigation"]
    completed = subprocess.run(
        ["docker", "compose", "--profile", "research-navigation", "up", "--force-recreate",
         "--abort-on-container-exit", "--exit-code-from", services[0], *services],
        cwd=ROOT, env=environment, check=False,
    )
    subprocess.run(
        ["docker", "compose", "--profile", "research-navigation", "rm", "-sf", *services],
        cwd=ROOT, env=environment, check=False,
    )
    session_report = run_root / run_id / scene_id / "session_report.json"
    return_code = completed.returncode
    if not session_report.is_file():
        return_code = return_code or 1
    return return_code, run_id


def run_evaluation(
    config: dict,
    entries: list[dict],
    *,
    split: str,
    capture_review: bool = False,
) -> dict:
    evaluation_id = str(config["evaluation"]["evaluation_id"])
    run_root = ROOT / config["evaluation"]["run_root"]
    grouped: dict[str, list[str]] = defaultdict(list)
    for entry in entries:
        grouped[entry["scene_id"]].append(entry["episode_id"])
    run_ids: list[str] = []
    return_codes: list[int] = []
    for scene_id, episode_ids in grouped.items():
        code, run_id = compose_session(
            config, evaluation_id, scene_id, episode_ids,
            split=split, run_root=run_root, capture_review=capture_review,
        )
        run_ids.append(run_id)
        return_codes.append(code)
        if capture_review:
            for episode_id in episode_ids:
                trace = run_root / run_id / scene_id / episode_id / "trace.json"
                if trace.is_file():
                    render_episode_trace(
                        trace,
                        run_root / evaluation_id / "review" / f"{episode_id}.png",
                    )
        if code != 0:
            break
    report = build_evaluation_report(
        evaluation_id, split, run_ids, [item["episode_id"] for item in entries], run_root=run_root,
        robot_profile=ROOT / config["robot_profile"],
    )
    report["session_return_codes"] = return_codes
    report["infrastructure_completed"] = report["execution_completed"] and all(code == 0 for code in return_codes)
    output = run_root / evaluation_id / "aggregate_report.json"
    atomic_write_json(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate-config", "run"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split", choices=("validation", "test"))
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--episodes", nargs="*")
    parser.add_argument("--capture-review", action="store_true")
    arguments = parser.parse_args()
    config = load_yaml(arguments.config)
    split = arguments.split or str(config["evaluation"]["default_split"])
    entries = validate_config(config, split, allow_test=arguments.allow_test)
    if arguments.episodes:
        allowed = {item["episode_id"]: item for item in entries}
        unknown = sorted(set(arguments.episodes) - set(allowed))
        if unknown:
            raise ValueError(f"episodes do not belong to {split}: {unknown}")
        entries = [allowed[value] for value in arguments.episodes]
    if arguments.command == "validate-config":
        print(json.dumps({
            "status": "PASS",
            "evaluation_id": config["evaluation"]["evaluation_id"],
            "split": split,
            "scene_count": len({item["scene_id"] for item in entries}),
            "episode_count": len(entries),
            "mapping_enabled": False,
        }, sort_keys=True))
        return 0
    report = run_evaluation(config, entries, split=split, capture_review=arguments.capture_review)
    print(json.dumps({
        "status": "EXECUTION_COMPLETE" if report["infrastructure_completed"] else "INFRASTRUCTURE_FAILURE",
        "episodes": report["episode_count"],
        "successes": report["success_count"],
        "collisions": report["collision_count"],
    }, sort_keys=True))
    return 0 if report["infrastructure_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
