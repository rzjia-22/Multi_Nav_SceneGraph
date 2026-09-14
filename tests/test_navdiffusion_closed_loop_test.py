from pathlib import Path
import ast
import inspect

import pytest
import yaml

from research_data.closed_loop import _group_report, _run_group
from research_data.closed_loop_analysis import dataset_index_entries
from research_data.closed_loop_test import CHECKPOINT_SHA256, _validate_config


ROOT = Path(__file__).resolve().parents[1]


def test_final_test_contract_is_exactly_two_scenes_and_ten_episodes():
    config = yaml.safe_load((
        ROOT / "config/navigation/navdiffusion_v0_test.yaml"
    ).read_text(encoding="utf-8"))
    base, entries = _validate_config(config)
    assert base["navigation"] == {
        "model_backend": "mns_v0",
        "history_length": 5,
        "planning_rate_hz": 2.0,
        "prediction_waypoints": 32,
        "control_waypoints": 8,
        "goal_tolerance_m": 0.35,
        "diffusion_seed": 20260914,
    }
    assert len(entries) == 10
    assert {item["scene_id"] for item in entries} == {
        "test_scene_000", "test_scene_001"
    }
    assert all(item["split"] == "test" for item in entries)
    assert config["frozen_safety"] == {
        "stop_distance_m": 0.7,
        "release_distance_m": 0.9,
        "reverse_speed_mps": 0.0,
        "turn_speed_radps": 0.5,
    }
    assert config["checkpoint_sha256"] == CHECKPOINT_SHA256


@pytest.mark.parametrize("split", ["train", "development", ""])
def test_closed_loop_index_rejects_non_held_out_splits(split):
    with pytest.raises(ValueError):
        dataset_index_entries(split)


def test_test_index_does_not_contain_train_or_validation_episodes():
    entries = dataset_index_entries("test")
    assert len(entries) == 10
    assert len({item["scene_id"] for item in entries}) == 2
    assert all(item["episode_id"].startswith("test_scene_") for item in entries)
    assert not any("train_scene_" in item["episode_id"] for item in entries)
    assert not any("validation_scene_" in item["episode_id"] for item in entries)


def test_shared_group_runner_keeps_explicit_split_and_infrastructure_controls():
    signature = inspect.signature(_run_group)
    assert signature.parameters["split"].default == "validation"
    assert signature.parameters["stop_on_session_error"].default is False
    report_signature = inspect.signature(_group_report)
    assert report_signature.parameters["split"].default == "validation"
    with pytest.raises(ValueError):
        _run_group(
            "invalid",
            ["train_scene_000_episode_000"],
            fail_fast=False,
            capture=False,
            split="train",
        )


def test_isaac_runtime_rejects_train_and_uses_selected_split_for_plan_path():
    runtime = ROOT / "ros_ws/src/mns_simulation/mns_simulation/research_navigation_runtime.py"
    tree = ast.parse(runtime.read_text(encoding="utf-8"))
    choices = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            for keyword in node.keywords:
                if keyword.arg == "choices" and isinstance(keyword.value, (ast.Tuple, ast.List)):
                    choices.append([value.value for value in keyword.value.elts])
    assert ["validation", "test"] in choices
    source = runtime.read_text(encoding="utf-8")
    assert '"datasets/dataset_v0" / ARGS.split' in source
    assert 'scene["split"] != ARGS.split' in source
    assert '"test_split_used": ARGS.split == "test"' in source


def test_compose_passes_explicit_split_and_separate_test_run_root():
    compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    assert "--split ${MNS_CLOSED_LOOP_SPLIT:-validation}" in compose
    assert "--run-root ${MNS_CLOSED_LOOP_RUN_ROOT:-/mns/runs/navdiffusion_v0_closed_loop}" in compose


def test_validation_evidence_remains_sealed_after_generic_runner_refactor():
    report = yaml.safe_load((
        ROOT / "artifacts/navdiffusion_v0_closed_loop/aggregate_report.json"
    ).read_text(encoding="utf-8"))
    assert report["test_split_used"] is False
    assert report["full_validation"]["test_split_used"] is False
    assert report["full_validation"]["episode_count"] == 10
