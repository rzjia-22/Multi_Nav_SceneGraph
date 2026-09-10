"""Creation of the single authoritative 14-scene/70-episode V0 plan."""

from __future__ import annotations

from pathlib import Path

from .common import ROOT, dump_yaml


SCENES = [
    ("train_scene_000", 41001, "train", "gentle", "grass", "normal", "medium"),
    ("train_scene_001", 41017, "train", "flat", "bare_soil", "bright", "low"),
    ("train_scene_002", 41039, "train", "moderate", "grass", "dim", "medium"),
    ("train_scene_003", 41057, "train", "gentle", "bare_soil", "bright", "high"),
    ("train_scene_004", 41077, "train", "flat", "grass", "normal", "high"),
    ("train_scene_005", 41101, "train", "moderate", "bare_soil", "normal", "low"),
    ("train_scene_006", 41131, "train", "gentle", "grass", "dim", "high"),
    ("train_scene_007", 41149, "train", "flat", "bare_soil", "normal", "medium"),
    ("train_scene_008", 41183, "train", "moderate", "grass", "bright", "low"),
    ("train_scene_009", 41201, "train", "gentle", "bare_soil", "dim", "medium"),
    ("validation_scene_000", 51011, "validation", "flat", "grass", "dim", "medium"),
    ("validation_scene_001", 51047, "validation", "moderate", "bare_soil", "bright", "high"),
    ("test_scene_000", 61001, "test", "gentle", "grass", "bright", "low"),
    ("test_scene_001", 61043, "test", "moderate", "bare_soil", "normal", "high"),
]


def create_manifest(path: Path = ROOT / "config/datasets/dataset_v0_manifest.yaml") -> None:
    scenes = []
    # Seven A and seven B patterns give exactly 21 short, 28 medium and 21 long episodes.
    patterns = (
        ["short", "medium", "medium", "long", "long"],
        ["short", "short", "medium", "medium", "long"],
    )
    for scene_index, (scene_id, seed, split, terrain, ground, lighting, density) in enumerate(SCENES):
        episodes = [
            {
                "episode_id": f"{scene_id}_episode_{episode_index:03d}",
                "episode_type": "nominal_expert",
                "target_route_length_bucket": bucket,
            }
            for episode_index, bucket in enumerate(patterns[scene_index % 2])
        ]
        scenes.append(
            {
                "scene_id": scene_id,
                "scene_seed": seed,
                "split": split,
                "factors": {
                    "terrain": terrain,
                    "ground": ground,
                    "lighting": lighting,
                    "tree_density": density,
                },
                "planned_episodes": episodes,
            }
        )
    dump_yaml(
        path,
        {
            "dataset_version": "dataset_v0",
            "status": "pilot_plan",
            "split_policy": "scene_level_only",
            "scene_size_m": 24.0,
            "route_length_buckets_m": {"short": [3.0, 5.0], "medium": [5.0, 8.0], "long": [8.0, 10.0]},
            "profiles": {
                "robot": "config/robots/diablo_standing.yaml",
                "sensor": "config/sensors/d435i_navigation_v0.yaml",
                "forest": "config/research_forests/profiles.yaml",
            },
            "scenes": scenes,
        },
    )

