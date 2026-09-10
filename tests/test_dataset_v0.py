from pathlib import Path
import tempfile

from research_data.common import ROOT, load_yaml, stable_hash
from research_data.forest import generate_scene
from research_data.validation import validate_manifest, validate_scene


def test_manifest_has_exact_scene_level_split_and_length_distribution():
    report = validate_manifest()
    assert report["split_counts"] == {"train": 10, "validation": 2, "test": 2}
    assert report["bucket_counts"] == {"short": 21, "medium": 28, "long": 21}
    assert report["scene_leakage"] is False


def test_preview_scene_regenerates_deterministically():
    checked_in = load_yaml(ROOT / "research_scenes/dataset_v0/train_scene_000/scene.yaml")
    with tempfile.TemporaryDirectory() as directory:
        generated = generate_scene(
            "train_scene_000", Path(directory) / "scene.yaml"
        )
    assert stable_hash(generated) == stable_hash(checked_in)


def test_preview_scene_uses_real_assets_and_official_terrain():
    scene = load_yaml(ROOT / "research_scenes/dataset_v0/train_scene_000/scene.yaml")
    assert scene["terrain"]["builder"] == "isaaclab.terrains.TerrainImporter"
    assert {tree["asset_id"] for tree in scene["trees"]} == {
        "nvidia_blue_berry_elder",
        "nvidia_gray_birch",
    }
    assert all(tree["ground_height_policy"] == "terrain_raycast_top_surface" for tree in scene["trees"])
    report = validate_scene(ROOT / "research_scenes/dataset_v0/train_scene_000/scene.yaml")
    assert report["primitive_tree_visuals"] is False
