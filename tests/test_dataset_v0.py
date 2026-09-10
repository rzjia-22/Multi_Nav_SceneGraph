from pathlib import Path
import tempfile

from research_data.common import ROOT, load_yaml, stable_hash
from research_data.expert import line_free, occupancy_grid, path_length
from research_data.forest import generate_scene
from research_data.validation import validate_manifest


def test_manifest_has_exact_scene_level_split_and_length_distribution():
    report = validate_manifest()
    assert report["split_counts"] == {"train": 10, "validation": 2, "test": 2}
    assert report["bucket_counts"] == {"short": 21, "medium": 28, "long": 21}
    assert report["scene_leakage"] is False


def test_preview_scene_regenerates_deterministically():
    checked_in = load_yaml(ROOT / "research_scenes/dataset_v0/train_scene_000/scene.yaml")
    with tempfile.TemporaryDirectory() as directory:
        generated = generate_scene(
            "train_scene_000", Path(directory) / "scene.yaml", Path(directory) / "scene.usda"
        )
    assert stable_hash(generated) == stable_hash(checked_in)


def test_preview_expert_is_nontrivial_and_collision_free():
    scene = load_yaml(ROOT / "research_scenes/dataset_v0/train_scene_000/scene.yaml")
    plan = load_yaml(ROOT / "artifacts/dataset_v0_preview/train_scene_000_episode_000/episode_plan.yaml")
    robot = load_yaml(ROOT / "config/robots/diablo_standing.yaml")
    grid = occupancy_grid(scene, robot)
    path = plan["planned_path"]
    assert 5.0 <= path_length(path) <= 9.0
    assert not line_free(grid, tuple(path[0][:2]), tuple(path[-1][:2]))
    assert all(line_free(grid, tuple(first[:2]), tuple(second[:2])) for first, second in zip(path, path[1:]))
