from pathlib import Path
import ast
import json
import inspect

import pytest
import yaml

from research_data.closed_loop import _compose_session, _group_report, _run_group
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


def test_missing_scene_session_report_is_an_infrastructure_failure(monkeypatch):
    class Completed:
        returncode = 0

    monkeypatch.setattr("research_data.closed_loop.subprocess.run", lambda *args, **kwargs: Completed())
    code, _ = _compose_session(
        "test",
        "test_scene_000",
        ["test_scene_000_episode_000"],
        capture_review=False,
        fail_on_episode_failure=False,
        split="test",
        run_root=ROOT / "runs/nonexistent_closed_loop_unit_test",
    )
    assert code == 1


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
    assert 'failure_reason = "model_terrain_exit"' in source
    assert '"terrain_exit_candidate_xy": terrain_exit_candidate' in source


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


def test_final_test_evidence_is_complete_frozen_and_one_pass():
    artifact_root = ROOT / "artifacts/navdiffusion_v0_test"
    report = json.loads((artifact_root / "test_report.json").read_text(encoding="utf-8"))
    analysis = json.loads((artifact_root / "test_analysis.json").read_text(encoding="utf-8"))
    execution = json.loads((artifact_root / "test_execution.json").read_text(encoding="utf-8"))

    assert report["execution_completed"] is True
    assert report["evaluation_split"] == "test"
    assert report["test_split_used"] is True
    assert report["train_split_executed"] is False
    assert report["validation_split_executed"] is False
    assert report["mapping_enabled"] is False
    assert report["checkpoint_sha256"] == CHECKPOINT_SHA256
    assert report["episode_count"] == 10
    assert report["success_count"] == 3
    assert report["collision_count"] == 6
    assert report["timeout_count"] == 0
    assert report["stall_count"] == 0
    assert report["not_run_episodes"] == []
    assert report["compose_return_codes"] == [0, 0]

    assert analysis["scope"] == "test_only_final_one_pass"
    assert analysis["terrain_exit_count"] == 1
    assert analysis["failure_classes"] == {"MODEL": 7}
    assert all(item["checkpoint_sha256"] == CHECKPOINT_SHA256 for item in analysis["episodes"])
    assert all(item["moving_stale_rgb_frames"] == 0 for item in analysis["episodes"])
    assert all(item["full_and_control_same_sample"] for item in analysis["episodes"])
    assert execution["status"] == "complete"
    assert execution["episode_count"] == 10
    assert execution["mid_test_tuning"] is False
    assert execution["per_episode_reruns"] == 0


def test_validation_test_comparison_preserves_governance_and_failure_signature():
    comparison = json.loads((
        ROOT / "artifacts/navdiffusion_v0_test/validation_vs_test.json"
    ).read_text(encoding="utf-8"))
    assert comparison["validation"]["success_count"] == 4
    assert comparison["validation"]["collision_count"] == 6
    assert comparison["test"]["success_count"] == 3
    assert comparison["test"]["collision_count"] == 6
    assert comparison["held_out_generalization_classification"] == "similar"
    assert comparison["test_collisions_with_prior_full_unsafe_prediction"] == 6
    assert comparison["test_collisions_with_prior_control_unsafe_prediction"] == 6
    assert comparison["unsafe_prediction_failure_signature_reproduced"] is True
    assert comparison["future_v1_requires_new_final_test_scenes"] is True
