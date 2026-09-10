"""Shared configuration and geometry helpers for Dataset V0."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping in {path}")
    return value


def dump_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(value, stream, sort_keys=False, allow_unicode=True)


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def camera_intrinsics(width: int, height: int, horizontal_fov_deg: float, vertical_fov_deg: float):
    fx = width / (2.0 * math.tan(math.radians(horizontal_fov_deg) / 2.0))
    fy = height / (2.0 * math.tan(math.radians(vertical_fov_deg) / 2.0))
    return [[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]]


def scene_paths(scene_id: str) -> tuple[Path, Path]:
    directory = ROOT / "research_scenes" / "dataset_v0" / scene_id
    return directory / "scene.yaml", directory / "scene.usda"


def episode_directory(episode_id: str) -> Path:
    return ROOT / "artifacts" / "dataset_v0_preview" / episode_id
