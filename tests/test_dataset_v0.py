from pathlib import Path
import tempfile

from research_data.common import ROOT, load_yaml, stable_hash
from research_data.depth import align_depth_to_rgb, dequantize_depth_z16, quantize_depth_z16
from research_data.forest import generate_scene
from research_data.validation import validate_manifest, validate_runtime_contract, validate_scene


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
    assert scene["schema_version"] == 3
    assert scene["terrain"]["actual_geometry_statistics"]["slope_p90_deg"] > 2.0


def test_z16_quantization_preserves_invalids_and_millimeter_tolerance():
    import numpy as np

    source = np.asarray([[0.1, 0.2, 1.2344, 9.9996, 10.1, np.inf]], dtype=np.float32)
    encoded, report = quantize_depth_z16(source, 0.001, [0.2, 10.0])
    restored = dequantize_depth_z16(encoded, 0.001)
    assert encoded.dtype == np.uint16
    assert encoded[0, 0] == encoded[0, 4] == encoded[0, 5] == 0
    assert report["saturation_count"] == 0
    assert report["maximum_absolute_error_m"] <= 0.000501
    assert np.allclose(restored[encoded > 0], source[encoded > 0], atol=0.000501)


def test_offline_alignment_is_deterministic():
    import numpy as np

    raw = np.asarray([[1000, 1000], [0, 2000]], dtype=np.uint16)
    intrinsics = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    args = (raw, 0.001, intrinsics, intrinsics, np.zeros(3), np.asarray([0.0, 0.0, 0.0, 1.0]), [2, 2])
    first = align_depth_to_rgb(*args)
    second = align_depth_to_rgb(*args)
    assert np.array_equal(first, second)
    assert first[0, 0] == 1.0
    assert first[1, 1] == 2.0


def test_collector_uses_only_shared_research_forest_builder():
    report = validate_runtime_contract()
    assert report["shared_builder_call"] is True
