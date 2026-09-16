from pathlib import Path
import ast
import hashlib
import json

import numpy as np
import pytest
import yaml

from research_data.evaluation.metrics import failure_metrics, prediction_metrics, progress_metrics
from research_data.evaluation.reporting import atomic_write_json, dataset_index_entries
from research_data.evaluation.runner import validate_config


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_SHA256 = "7b3bca67af26c2d839555e9b27a7a420762b572fa88664a227986174d9a68ae0"


def _config() -> dict:
    return yaml.safe_load(
        (ROOT / "config/navigation/research_navigation_eval.yaml").read_text(encoding="utf-8")
    )


def test_evaluation_index_is_split_safe_and_complete():
    validation = dataset_index_entries("validation")
    test = dataset_index_entries("test")
    assert len(validation) == 10
    assert len(test) == 10
    assert len({item["scene_id"] for item in validation}) == 2
    assert len({item["scene_id"] for item in test}) == 2
    assert all(item["split"] == "validation" for item in validation)
    assert all(item["split"] == "test" for item in test)
    with pytest.raises(ValueError):
        dataset_index_entries("train")


def test_generic_evaluation_defaults_to_validation_and_requires_test_opt_in():
    config = _config()
    assert config["evaluation"]["default_split"] == "validation"
    assert config["mapping_enabled"] is False
    assert len(validate_config(config, "validation")) == 10
    with pytest.raises(ValueError, match="requires --allow-test"):
        validate_config(config, "test")
    assert len(validate_config(config, "test", allow_test=True)) == 10


def test_prediction_and_failure_metrics_preserve_collision_timeline():
    planning = [
        {"status": "PASS", "timestamp_from_episode_start_s": 1.0,
         "full_minimum_clearance_m": 0.2, "control_minimum_clearance_m": 0.4},
        {"status": "PASS", "timestamp_from_episode_start_s": 2.0,
         "full_minimum_clearance_m": -0.1, "control_minimum_clearance_m": 0.1},
        {"status": "PASS", "timestamp_from_episode_start_s": 3.0,
         "full_minimum_clearance_m": -0.2, "control_minimum_clearance_m": -0.1},
    ]
    prediction = prediction_metrics(planning)
    assert prediction["full_prediction_unsafe_count"] == 2
    assert prediction["control_prediction_unsafe_count"] == 1
    commands = np.asarray([[2.5, 0.4, 0, 0, 0, 0, 0.5, 1]], dtype=np.float64)
    failure = failure_metrics(
        {"success": False, "failure_reason": "collision", "first_collision_timestamp_s": 3.2},
        commands,
        prediction,
        {"progress_degradation_start_s": 1.5},
        1.7,
    )
    assert failure["primary_failure_class"] == "MODEL"
    assert failure["timeline_s"]["full_prediction_first_unsafe"] == 2.0
    assert failure["timeline_s"]["control_prediction_first_unsafe"] == 3.0
    assert failure["collision_lead_time_s"]["from_control_prediction_unsafe"] == pytest.approx(0.2)


def test_progress_metrics_detect_sustained_non_improvement():
    state = np.zeros((6, 11), dtype=np.float64)
    state[:, 0] = [0, 1, 2, 3, 4, 5]
    state[:, 10] = [5.0, 4.8, 4.8, 4.81, 4.82, 4.83]
    metrics = progress_metrics(state)
    assert metrics["minimum_goal_distance_m"] == 4.8
    assert metrics["longest_no_progress_duration_s"] == 4.0
    assert metrics["progress_degradation_start_s"] == 1.0


def test_atomic_report_replaces_inprogress_file(tmp_path):
    output = tmp_path / "report.json"
    atomic_write_json(output, {"status": "PASS"})
    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "PASS"}
    assert not (tmp_path / ".report.json.inprogress").exists()


def test_runtime_keeps_shared_forest_ros_boundary_and_episode_reset_contract():
    path = ROOT / "ros_ws/src/mns_simulation/mns_simulation/research_navigation_runtime.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "build_research_forest" in imported_names
    assert "mns_navigation" not in source
    assert 'choices=("validation", "test")' in source
    assert "endpoint.clear_episode()" in source
    assert "rgb_camera.reset()" in source
    assert "depth_camera.reset()" in source
    assert 'failure_reason = "collision"' in source
    assert 'failure_reason = "stall"' in source
    assert 'failure_reason = "timeout"' in source


def test_compose_uses_generic_research_navigation_services():
    source = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    assert "robotics-research-navigation:" in source
    assert "simulation-research-navigation:" in source
    assert "--evaluation-id ${MNS_EVAL_ID:-manual}" in source
    assert "MNS_CLOSED_LOOP" not in source


def test_compact_baseline_record_is_authoritative_and_checkpoint_is_unchanged():
    path = ROOT / "artifacts/baselines/navdiffusion_v0/results.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["baseline_schema_version"] == 1
    assert record["baseline_id"] == "navdiffusion_v0"
    assert record["dataset_version"] == "dataset_v0"
    assert record["closed_loop_validation"]["success"] == 4
    assert record["closed_loop_test"]["success"] == 3
    assert record["closed_loop_test"]["terrain_exit_count"] == 1
    assert record["final_readiness"] == "NOT_READY"
    checkpoint = ROOT / record["checkpoint"]["path"]
    assert checkpoint.stat().st_size == record["checkpoint"]["size_bytes"]
    hasher = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    assert digest == CHECKPOINT_SHA256 == record["checkpoint"]["sha256"]
    assert record["data_governance"]["test_not_untouched_for_future_v1"] is True
