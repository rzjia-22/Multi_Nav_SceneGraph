"""One-pass final held-out closed-loop evaluation for NavDiffusion V0."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from .closed_loop import _run_group, _write_json
from .closed_loop_analysis import (
    _episode_analysis,
    _group_by,
    _load_trace_map,
    _plot_results,
    _risk_summary,
    _safety_summary,
    _summary,
    dataset_index_entries,
)
from .common import ROOT, load_yaml


CONFIG = ROOT / "config/navigation/navdiffusion_v0_test.yaml"
VALIDATION_ANALYSIS = ROOT / "artifacts/navdiffusion_v0_closed_loop/full_validation_analysis.json"
TEST_ARTIFACT_ROOT = ROOT / "artifacts/navdiffusion_v0_test"
TEST_RUN_ROOT = ROOT / "runs/navdiffusion_v0_test"
ATTEMPT_RECORD = TEST_ARTIFACT_ROOT / "test_execution.json"
ROBOT_PATH = ROOT / "config/robots/diablo_standing.yaml"
CHECKPOINT_SHA256 = "7b3bca67af26c2d839555e9b27a7a420762b572fa88664a227986174d9a68ae0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _validate_config(config: dict[str, Any]) -> tuple[dict, list[dict]]:
    if config.get("scope") != "test_only" or config.get("evaluation_split") != "test":
        raise ValueError("the final evaluation configuration must be explicitly test-only")
    if not config.get("one_pass_policy"):
        raise ValueError("the final test must retain its one-pass policy")
    if config.get("allow_train_split") or config.get("allow_validation_split"):
        raise ValueError("the final test command may not accept train or validation episodes")
    if config.get("mapping_enabled"):
        raise ValueError("Hydra/mapping must remain disabled for the final V0 test")
    if config.get("artifact_root") != "artifacts/navdiffusion_v0_test":
        raise ValueError("the canonical test artifact root changed")
    if config.get("run_root") != "runs/navdiffusion_v0_test":
        raise ValueError("the canonical test run root changed")
    expected_safety = {
        "stop_distance_m": 0.7,
        "release_distance_m": 0.9,
        "reverse_speed_mps": 0.0,
        "turn_speed_radps": 0.5,
    }
    if config.get("frozen_safety") != expected_safety:
        raise ValueError("the frozen DepthSafety contract changed")
    for relative_path, expected_hash in config.get("frozen_file_sha256", {}).items():
        path = ROOT / relative_path
        if not path.is_file() or _sha256(path) != expected_hash:
            raise ValueError(f"frozen test dependency changed: {relative_path}")
    base = load_yaml(ROOT / config["base_runtime_config"])
    if base["scope"] != "validation_only" or base["mapping_enabled"]:
        raise ValueError("the frozen validation runtime contract is invalid")
    expected_navigation = {
        "model_backend": "mns_v0",
        "history_length": 5,
        "planning_rate_hz": 2.0,
        "prediction_waypoints": 32,
        "control_waypoints": 8,
        "goal_tolerance_m": 0.35,
        "diffusion_seed": 20260914,
    }
    if base["navigation"] != expected_navigation:
        raise ValueError("the frozen NavDiffusion navigation contract changed")
    if base["checkpoint_sha256"] != CHECKPOINT_SHA256:
        raise ValueError("the frozen validation checkpoint declaration changed")
    checkpoint = ROOT / config["checkpoint"]
    if _sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("the frozen checkpoint content hash changed")
    entries = dataset_index_entries("test")
    if len(entries) != int(config["expected_episode_count"]):
        raise ValueError("the canonical test episode count changed")
    if len({item["scene_id"] for item in entries}) != int(config["expected_scene_count"]):
        raise ValueError("the canonical test scene count changed")
    return base, entries


def _mean(items: list[dict], key: str) -> float | None:
    values = [float(item[key]) for item in items if item.get(key) is not None]
    return float(np.mean(values)) if values else None


def _write_test_markdown(analysis: dict) -> None:
    def number(value: float | None, digits: int = 3) -> str:
        return "n/a" if value is None else f"{value:.{digits}f}"

    def percent(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.2%}"

    lines = [
        "# NavDiffusion V0 final held-out closed-loop test",
        "",
        f"All **{analysis['episode_count']}/10** canonical test missions were executed exactly once. "
        f"Navigation succeeded on **{analysis['success_count']}/10** and recorded "
        f"**{analysis['collision_count']}** conservative collision(s).",
        "",
        "## Episode overview",
        "",
        "| Episode | Scene | Bucket | Expert len [m] | Result | Failure | Goal err [m] | Min clearance [m] | Safety | Full unsafe | Control unsafe |",
        "| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in analysis["episodes"]:
        lines.append(
            f"| `{item['episode_id']}` | `{item['scene_id']}` | {item['route_bucket']} | "
            f"{item['planned_expert_length_m']:.3f} | {'PASS' if item['success'] else 'FAIL'} | "
            f"{item['primary_failure_class'] or '—'} | {number(item['goal_error_m'])} | "
            f"{number(item['minimum_clearance_m'])} | {percent(item['safety_override_fraction'])} | "
            f"{percent(item['full_prediction_unsafe_fraction'])} | "
            f"{percent(item['control_prediction_unsafe_fraction'])} |"
        )
    lines.extend(["", "## Route buckets", ""])
    for key, value in analysis["route_bucket_summary"].items():
        lines.append(
            f"- {key}: {value['success_count']}/{value['episode_count']} success, "
            f"{value['collision_count']} collision(s), mean goal error "
            f"{number(value['mean_goal_error_m'])} m, mean clearance "
            f"{number(value['mean_minimum_clearance_m'])} m, mean Safety "
            f"{percent(value['mean_safety_override_fraction'])}, full/control unsafe "
            f"{percent(value['mean_full_prediction_unsafe_fraction'])}/"
            f"{percent(value['mean_control_prediction_unsafe_fraction'])}."
        )
    lines.extend(["", "## Test scenes", ""])
    for key, value in analysis["scene_summary"].items():
        environment = analysis["scene_environment"][key]
        lines.append(
            f"- `{key}`: {value['success_count']}/{value['episode_count']} success, "
            f"{value['collision_count']} collision(s), mean goal error "
            f"{number(value['mean_goal_error_m'])} m; {environment['terrain']} terrain, "
            f"{environment['tree_density']} density, {environment['ground']}, "
            f"{environment['lighting']} lighting."
        )
    lines.extend(["", "## Failure timelines", ""])
    failures = [item for item in analysis["episodes"] if not item["success"]]
    if not failures:
        lines.append("No failed mission.")
    for item in failures:
        timeline = item["timeline_s"]
        lead = item["collision_lead_time_s"]
        lines.append(
            f"- `{item['episode_id']}`: primary `{item['primary_failure_class']}`, "
            f"secondary {item['secondary_contributors'] or 'none'}; full/control/Safety/collision "
            f"at {timeline['full_prediction_first_unsafe']}/{timeline['control_prediction_first_unsafe']}/"
            f"{timeline['safety_first_intervention']}/{timeline['collision']} s; collision lead "
            f"{lead['from_full_prediction_unsafe']}/{lead['from_control_prediction_unsafe']}/"
            f"{lead['from_safety_intervention']} s."
        )
    lines.extend([
        "",
        "## Scope and governance",
        "",
        "- Evaluation split: test only (2 scenes, 10 episodes)",
        "- Each canonical mission was run once; no per-episode reruns",
        "- Train and validation missions were not executed by this command",
        "- Hydra/mapping: disabled",
        "- Expert path: post-run analysis only; never used for control",
        "- Model, checkpoint, preprocessing, controller, Safety, DIABLO, D435i and environments: frozen",
        "- Dataset V0 test split was first opened for NavDiffusion V0 final closed-loop evaluation",
        "- Any future V1 informed by this result requires new held-out final-test scenes",
        "",
    ])
    (TEST_ARTIFACT_ROOT / "test_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def _comparison_payload(validation: dict, test: dict) -> dict:
    delta = int(test["success_count"]) - int(validation["success_count"])
    if delta >= 3:
        generalization = "test_better"
    elif delta <= -3:
        generalization = "test_worse"
    else:
        generalization = "similar"
    test_failures = [item for item in test["episodes"] if not item["success"]]
    collision_failures = [item for item in test_failures if item["collision"]]
    full_before_collision = sum(
        item["timeline_s"]["full_prediction_first_unsafe"] is not None
        for item in collision_failures
    )
    control_before_collision = sum(
        item["timeline_s"]["control_prediction_first_unsafe"] is not None
        for item in collision_failures
    )
    return {
        "comparison_version": 1,
        "validation": {
            "episode_count": validation["episode_count"],
            "success_count": validation["success_count"],
            "collision_count": validation["collision_count"],
            "route_bucket_summary": validation["route_bucket_summary"],
            "scene_summary": validation["scene_summary"],
            "scene_environment": validation["scene_environment"],
            "prediction_risk_by_outcome": validation["prediction_risk_by_outcome"],
            "safety_by_outcome": validation["safety_by_outcome"],
            "environment_summary": {
                "terrain": validation["terrain_summary"],
                "density": validation["density_summary"],
                "ground": validation["ground_summary"],
                "lighting": validation["lighting_summary"],
            },
        },
        "test": {
            "episode_count": test["episode_count"],
            "success_count": test["success_count"],
            "collision_count": test["collision_count"],
            "route_bucket_summary": test["route_bucket_summary"],
            "scene_summary": test["scene_summary"],
            "scene_environment": test["scene_environment"],
            "prediction_risk_by_outcome": test["prediction_risk_by_outcome"],
            "safety_by_outcome": test["safety_by_outcome"],
            "environment_summary": {
                "terrain": test["terrain_summary"],
                "density": test["density_summary"],
                "ground": test["ground_summary"],
                "lighting": test["lighting_summary"],
            },
        },
        "test_minus_validation_success_count": delta,
        "held_out_generalization_classification": generalization,
        "test_collision_failure_count": len(collision_failures),
        "test_collisions_with_prior_full_unsafe_prediction": full_before_collision,
        "test_collisions_with_prior_control_unsafe_prediction": control_before_collision,
        "unsafe_prediction_failure_signature_reproduced": bool(
            collision_failures
            and full_before_collision == len(collision_failures)
            and control_before_collision == len(collision_failures)
        ),
        "test_split_first_opened_for": "NavDiffusion V0 final closed-loop evaluation",
        "future_v1_requires_new_final_test_scenes": True,
        "descriptive_only": True,
    }


def _write_comparison_markdown(comparison: dict) -> None:
    def percent(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.2%}"

    validation, test = comparison["validation"], comparison["test"]
    lines = [
        "# NavDiffusion V0 validation versus final test",
        "",
        "| Split | Success | Collisions |",
        "| --- | ---: | ---: |",
        f"| Validation | {validation['success_count']}/{validation['episode_count']} | {validation['collision_count']} |",
        f"| Test | {test['success_count']}/{test['episode_count']} | {test['collision_count']} |",
        "",
        f"Held-out generalization classification: **{comparison['held_out_generalization_classification']}** "
        f"(test minus validation success count: {comparison['test_minus_validation_success_count']:+d}).",
        "",
        "## Route buckets",
        "",
        "| Bucket | Validation success | Test success | Validation collisions | Test collisions |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for bucket in ("short", "medium", "long"):
        v, t = validation["route_bucket_summary"][bucket], test["route_bucket_summary"][bucket]
        lines.append(
            f"| {bucket} | {v['success_count']}/{v['episode_count']} | "
            f"{t['success_count']}/{t['episode_count']} | {v['collision_count']} | {t['collision_count']} |"
        )
    lines.extend(["", "## Prediction risk", ""])
    for outcome in ("PASS", "FAIL"):
        v = validation["prediction_risk_by_outcome"][outcome]
        t = test["prediction_risk_by_outcome"][outcome]
        lines.append(
            f"- {outcome}: validation full/control unsafe "
            f"{percent(v['full_prediction_unsafe_fraction'])}/"
            f"{percent(v['control_prediction_unsafe_fraction'])}; test "
            f"{percent(t['full_prediction_unsafe_fraction'])}/"
            f"{percent(t['control_prediction_unsafe_fraction'])}."
        )
    lines.extend(["", "## Safety", ""])
    for outcome in ("PASS", "FAIL"):
        v = validation["safety_by_outcome"][outcome]
        t = test["safety_by_outcome"][outcome]
        lines.append(
            f"- {outcome}: Safety intervened in validation {v['episodes_with_override']}/"
            f"{v['episode_count']} and test {t['episodes_with_override']}/{t['episode_count']} episodes; "
            f"mean override fraction {percent(v['mean_override_fraction'])}/"
            f"{percent(t['mean_override_fraction'])}; mean first intervention "
            f"{v['mean_first_intervention_s'] if v['mean_first_intervention_s'] is not None else 'n/a'}/"
            f"{t['mean_first_intervention_s'] if t['mean_first_intervention_s'] is not None else 'n/a'} s."
        )
    lines.extend(["", "## Environment factors", ""])
    for dimension in ("terrain", "density", "ground", "lighting"):
        lines.append(f"### {dimension.title()}")
        lines.append("")
        for split_name, payload in (("Validation", validation), ("Test", test)):
            values = payload["environment_summary"][dimension]
            description = ", ".join(
                f"{name}={summary['success_count']}/{summary['episode_count']} success, "
                f"{summary['collision_count']} collision(s)"
                for name, summary in values.items()
            )
            lines.append(f"- {split_name}: {description}.")
        lines.append("")
    lines.extend([
        "",
        "## Failure signature",
        "",
        f"- Test collision failures: {comparison['test_collision_failure_count']}",
        f"- Full prediction unsafe before collision: "
        f"{comparison['test_collisions_with_prior_full_unsafe_prediction']}/"
        f"{comparison['test_collision_failure_count']}",
        f"- Control prediction unsafe before collision: "
        f"{comparison['test_collisions_with_prior_control_unsafe_prediction']}/"
        f"{comparison['test_collision_failure_count']}",
        f"- Validation unsafe-trajectory failure signature reproduced: "
        f"**{comparison['unsafe_prediction_failure_signature_reproduced']}**",
        "",
        "## Held-out scene governance",
        "",
        "Dataset V0 test was first opened for this NavDiffusion V0 final closed-loop evaluation. "
        "Validation influenced early stopping, checkpoint selection and development interpretation; test did not.",
        "",
        "Any future NavDiffusion V1 design informed by these results must use new held-out scenes for its final evaluation. "
        "The scene and environment comparisons here are descriptive and do not establish causality.",
        "",
    ])
    (TEST_ARTIFACT_ROOT / "validation_vs_test.md").write_text("\n".join(lines), encoding="utf-8")


def _analyze_test(report: dict) -> dict:
    entries = dataset_index_entries("test")
    expected = [item["episode_id"] for item in entries]
    if report["expected_episodes"] != expected:
        raise ValueError("test report does not preserve canonical Dataset V0 order")
    if report["episode_count"] != 10 or report["not_run_episodes"]:
        raise ValueError("final test analysis requires all ten canonical traces")
    traces = _load_trace_map(report["run_ids"], run_root=TEST_RUN_ROOT, split="test")
    if set(traces) != set(expected):
        raise ValueError("test trace set differs from the canonical Dataset V0 test split")

    robot = load_yaml(ROBOT_PATH)
    by_id = {item["episode_id"]: item for item in entries}
    episodes = []
    for episode_id in expected:
        entry = by_id[episode_id]
        trace = traces[episode_id][0]
        source_report = trace["report"]
        scene = load_yaml(ROOT / "research_scenes/dataset_v0" / entry["scene_id"] / "scene.yaml")
        episode_directory = ROOT / Path(entry["relative_path"]).parent
        plan = load_yaml(episode_directory / "episode_plan.yaml")
        if source_report.get("evaluation_split") != "test" or not source_report.get("test_split_used"):
            raise ValueError(f"test scope was not explicit for {episode_id}")
        if source_report.get("mapping_enabled") or source_report.get("expert_path_used_for_control"):
            raise ValueError(f"mapping or expert-control contamination for {episode_id}")
        if source_report["checkpoint_sha256"] != CHECKPOINT_SHA256:
            raise ValueError(f"checkpoint mismatch for {episode_id}")
        if source_report["scene_hash"] != entry["scene_content_hash"]:
            raise ValueError(f"scene mismatch for {episode_id}")
        if source_report["plan_hash"] != entry["plan_hash"] or plan["plan_hash"] != entry["plan_hash"]:
            raise ValueError(f"plan mismatch for {episode_id}")
        if source_report.get("discarded_mismatched_planning_diagnostics") != 0:
            raise ValueError(f"cross-episode diagnostics were observed for {episode_id}")
        if source_report.get("history_ready_time_s") is not None and source_report["history_ready_time_s"] < 0:
            raise ValueError(f"negative history-ready time for {episode_id}")
        episodes.append(_episode_analysis(entry, trace, scene, plan, robot))

    failures = Counter(
        item["primary_failure_class"] for item in episodes if item["primary_failure_class"]
    )
    analysis = {
        "analysis_version": 1,
        "scope": "test_only_final_one_pass",
        "episode_count": len(episodes),
        "scene_count": len({item["scene_id"] for item in episodes}),
        "success_count": sum(item["success"] for item in episodes),
        "failure_count": sum(not item["success"] for item in episodes),
        "collision_count": sum(item["collision"] for item in episodes),
        "timeout_count": sum(item["failure_reason"] == "timeout" for item in episodes),
        "stall_count": sum(item["failure_reason"] == "stall" for item in episodes),
        "test_split_used": True,
        "mapping_enabled": False,
        "one_pass_policy": True,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "episodes": episodes,
        "route_bucket_summary": _group_by(episodes, lambda item: item["route_bucket"]),
        "scene_summary": _group_by(episodes, lambda item: item["scene_id"]),
        "terrain_summary": _group_by(episodes, lambda item: item["environment"]["terrain"]),
        "density_summary": _group_by(episodes, lambda item: item["environment"]["tree_density"]),
        "ground_summary": _group_by(episodes, lambda item: item["environment"]["ground"]),
        "lighting_summary": _group_by(episodes, lambda item: item["environment"]["lighting"]),
        "outcome_summary": {
            "PASS": _summary([item for item in episodes if item["success"]]),
            "FAIL": _summary([item for item in episodes if not item["success"]]),
        },
        "prediction_risk_by_outcome": {
            "ALL": _risk_summary(episodes),
            "PASS": _risk_summary([item for item in episodes if item["success"]]),
            "FAIL": _risk_summary([item for item in episodes if not item["success"]]),
        },
        "safety_by_outcome": {
            "PASS": _safety_summary([item for item in episodes if item["success"]]),
            "FAIL": _safety_summary([item for item in episodes if not item["success"]]),
        },
        "failure_classes": dict(sorted(failures.items())),
        "scene_environment": {
            scene_id: next(item["environment"] for item in episodes if item["scene_id"] == scene_id)
            for scene_id in sorted({item["scene_id"] for item in episodes})
        },
        "descriptive_findings": {
            "planned_length_mean_pass_m": _mean([item for item in episodes if item["success"]], "planned_expert_length_m"),
            "planned_length_mean_fail_m": _mean([item for item in episodes if not item["success"]], "planned_expert_length_m"),
            "expert_clearance_mean_pass_m": _mean([item for item in episodes if item["success"]], "expert_minimum_planning_clearance_m"),
            "expert_clearance_mean_fail_m": _mean([item for item in episodes if not item["success"]], "expert_minimum_planning_clearance_m"),
            "expert_turn_burden_mean_pass_rad_per_m": _mean([item for item in episodes if item["success"]], "expert_turn_burden_rad_per_m"),
            "expert_turn_burden_mean_fail_rad_per_m": _mean([item for item in episodes if not item["success"]], "expert_turn_burden_rad_per_m"),
            "camera_extremeness_mean_pass": _mean([item for item in episodes if item["success"]], "camera_perturbation_extremeness_fraction"),
            "camera_extremeness_mean_fail": _mean([item for item in episodes if not item["success"]], "camera_perturbation_extremeness_fraction"),
        },
    }
    _write_json(TEST_ARTIFACT_ROOT / "test_analysis.json", analysis)
    _write_test_markdown(analysis)
    _plot_results(
        episodes, traces,
        artifact_root=TEST_ARTIFACT_ROOT,
        episode_directory="episodes",
        scope_label="test",
    )
    validation = json.loads(VALIDATION_ANALYSIS.read_text(encoding="utf-8"))
    comparison = _comparison_payload(validation, analysis)
    _write_json(TEST_ARTIFACT_ROOT / "validation_vs_test.json", comparison)
    _write_comparison_markdown(comparison)
    return analysis


def _write_summary(report: dict, analysis: dict, comparison: dict) -> None:
    lines = [
        "# NavDiffusion V0 final test summary",
        "",
        "Final status: **NOT_READY**",
        "",
        f"- Test experiments executed: {report['episode_count']}/10",
        f"- Test navigation success: {analysis['success_count']}/10",
        f"- Test conservative collisions: {analysis['collision_count']}",
        f"- Test timeout/stall: {analysis['timeout_count']}/{analysis['stall_count']}",
        f"- Validation baseline: 4/10 success, 6 collisions",
        f"- Held-out classification: {comparison['held_out_generalization_classification']}",
        f"- Test wall/simulated duration: {report['total_wall_duration_s']:.2f}/"
        f"{report['total_simulated_duration_s']:.2f} s",
        f"- Inference mean/p95/max: {report['inference_ms_mean']:.1f}/"
        f"{report['inference_ms_p95']:.1f}/{report['inference_ms_max']:.1f} ms",
        "- Checkpoint changed: NO",
        "- Mid-test tuning: NO",
        "- Hydra/mapping: disabled",
        "- Dataset V0 test split is now opened and cannot be claimed untouched for a future V1",
        "",
    ]
    (TEST_ARTIFACT_ROOT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_test(config: dict, entries: list[dict]) -> dict:
    if (TEST_ARTIFACT_ROOT / "test_report.json").exists():
        raise RuntimeError("the canonical one-pass Dataset V0 test has already been finalized")
    invalid_attempt_history = []
    if ATTEMPT_RECORD.exists():
        previous = json.loads(ATTEMPT_RECORD.read_text(encoding="utf-8"))
        if previous.get("status") == "started":
            raise RuntimeError("an unfinished final-test attempt exists and must be audited first")
        if previous.get("status") == "invalid" and previous.get("system_git_commit") == _git_commit():
            raise RuntimeError("this code revision already produced an invalid test attempt")
        if previous.get("status") == "invalid":
            invalid_attempt_history = list(previous.get("invalid_attempt_history", []))
            invalid_attempt_history.append({
                "system_git_commit": previous.get("system_git_commit"),
                "reason": previous.get("reason"),
                "executed_episode_count": previous.get("executed_episode_count", 0),
            })
    _write_json(ATTEMPT_RECORD, {
        "status": "started",
        "evaluation_split": "test",
        "expected_episode_ids": [item["episode_id"] for item in entries],
        "system_git_commit": _git_commit(),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "mid_test_tuning_allowed": False,
        "per_episode_rerun_allowed": False,
        "invalid_attempt_history": invalid_attempt_history,
    })
    episodes = [item["episode_id"] for item in entries]
    report = _run_group(
        "test",
        episodes,
        fail_fast=False,
        capture=False,
        split="test",
        run_root=TEST_RUN_ROOT,
        artifact_root=TEST_ARTIFACT_ROOT,
        stop_on_session_error=True,
    )
    report.update({
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "one_pass_policy": True,
        "system_git_commit": _git_commit(),
        "validation_split_executed": False,
        "train_split_executed": False,
        "v0_final_status": "NOT_READY",
    })
    if not report["execution_completed"]:
        _write_json(ATTEMPT_RECORD, {
            "status": "invalid",
            "reason": "test infrastructure did not execute all ten missions",
            "report": report,
            "system_git_commit": _git_commit(),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "executed_episode_count": report.get("episode_count", 0),
            "invalid_attempt_history": invalid_attempt_history,
        })
        raise RuntimeError("test infrastructure did not execute all ten missions; the attempt is invalid")
    analysis = _analyze_test(report)
    report["failure_classes"] = analysis["failure_classes"]
    report["route_bucket_summary"] = analysis["route_bucket_summary"]
    report["scene_summary"] = analysis["scene_summary"]
    _write_json(TEST_ARTIFACT_ROOT / "test_report.json", report)
    comparison = json.loads((TEST_ARTIFACT_ROOT / "validation_vs_test.json").read_text(encoding="utf-8"))
    _write_summary(report, analysis, comparison)
    _write_json(ATTEMPT_RECORD, {
        "status": "complete",
        "evaluation_split": "test",
        "episode_count": 10,
        "run_ids": report["run_ids"],
        "system_git_commit": report["system_git_commit"],
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "mid_test_tuning": False,
        "per_episode_reruns": 0,
        "invalid_attempt_history": invalid_attempt_history,
    })
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate-config", "run-test"))
    args = parser.parse_args()
    config = load_yaml(CONFIG)
    _, entries = _validate_config(config)
    if args.command == "validate-config":
        print(json.dumps({
            "status": "PASS",
            "scope": "test_only",
            "scene_count": len({item["scene_id"] for item in entries}),
            "episode_count": len(entries),
            "checkpoint_sha256": CHECKPOINT_SHA256,
        }, sort_keys=True))
        return 0
    report = run_test(config, entries)
    print(json.dumps({
        "status": "EXECUTION_COMPLETE",
        "episodes": report["episode_count"],
        "successes": report["success_count"],
        "collisions": report["collision_count"],
        "v0_final_status": report["v0_final_status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
