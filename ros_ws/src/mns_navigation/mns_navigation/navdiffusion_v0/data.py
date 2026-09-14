"""Dataset V0 window construction and disposable deterministic cache."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import yaml

from .preprocessing import NavDiffusionPreprocessor, local_coordinates, yaw_from_xyzw


PROJECT_ROOT = Path(__file__).resolve().parents[5]
CACHE_FORMAT_VERSION = 1


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping in {path}")
    return value


def _episode_windows(
    sensor_timestamps: np.ndarray,
    state_timestamps: np.ndarray,
    state_positions: np.ndarray,
    state_orientations: np.ndarray,
    goal_xy: np.ndarray,
    history: int,
    horizon: int,
    future_step_s: float,
    goal_scale_m: float,
) -> dict[str, np.ndarray]:
    anchors = np.arange(history - 1, len(sensor_timestamps), dtype=np.int32)
    state_yaw = np.unwrap(yaw_from_xyzw(state_orientations))
    anchor_times = sensor_timestamps[anchors]
    anchor_x = np.interp(anchor_times, state_timestamps, state_positions[:, 0])
    anchor_y = np.interp(anchor_times, state_timestamps, state_positions[:, 1])
    anchor_yaw = np.interp(anchor_times, state_timestamps, state_yaw)
    future_local = np.empty((len(anchors), horizon, 2), dtype=np.float32)
    goals = np.empty((len(anchors), 2), dtype=np.float32)
    padding = np.empty(len(anchors), dtype=np.int16)
    offsets = np.arange(1, horizon + 1, dtype=np.float64) * future_step_s
    for output_index, (time_value, x, y, yaw) in enumerate(
        zip(anchor_times, anchor_x, anchor_y, anchor_yaw)
    ):
        targets = time_value + offsets
        future_world = np.column_stack((
            np.interp(targets, state_timestamps, state_positions[:, 0]),
            np.interp(targets, state_timestamps, state_positions[:, 1]),
        ))
        future_local[output_index] = local_coordinates(future_world, [x, y], yaw)
        goals[output_index] = local_coordinates(np.asarray(goal_xy)[None], [x, y], yaw)[0] / goal_scale_m
        padding[output_index] = int(np.count_nonzero(targets > state_timestamps[-1]))
    return {
        "anchor_indices": anchors,
        "anchor_timestamp_s": anchor_times.astype(np.float64),
        "goal_normalized": goals,
        "future_local_m": future_local,
        "future_padding_points": padding,
    }


def _calibration(handle) -> dict[str, Any]:
    group = handle["calibration"]
    return {name: group[name][()] for name in group.keys()}


def _write_episode_cache(
    entry: dict[str, Any],
    cache_root: Path,
    config: dict[str, Any],
    preprocessor: NavDiffusionPreprocessor,
) -> dict[str, Any]:
    import h5py
    from research_data.depth import align_depth_to_rgb

    episode_path = PROJECT_ROOT / entry["relative_path"]
    episode_id = entry["episode_id"]
    output = cache_root / "episodes" / episode_id
    output.mkdir(parents=True, exist_ok=True)
    with h5py.File(episode_path, "r") as source:
        if int(source.attrs["schema_version"]) != int(config["raw_source"]["episode_schema_version"]):
            raise ValueError(f"{episode_id}: unsupported HDF5 schema")
        rgb_source = source["sensors/rgb"]
        depth_source = source["sensors/depth_raw_z16"]
        frame_count = int(rgb_source.shape[0])
        if frame_count != int(depth_source.shape[0]):
            raise ValueError(f"{episode_id}: RGB/depth frame mismatch")
        rgb_cache = np.lib.format.open_memmap(
            output / "rgb_standardized.npy", mode="w+", dtype=np.float32,
            shape=(frame_count, 3, preprocessor.height, preprocessor.width),
        )
        depth_cache = np.lib.format.open_memmap(
            output / "depth_unit.npy", mode="w+", dtype=np.float32,
            shape=(frame_count, preprocessor.height, preprocessor.width),
        )
        calibration = _calibration(source)
        invalid_count = 0
        pixel_count = 0
        for start in range(0, frame_count, 8):
            stop = min(start + 8, frame_count)
            z16 = depth_source[start:stop]
            aligned = align_depth_to_rgb(
                z16,
                float(calibration["depth_scale_m"]),
                calibration["depth_intrinsics"],
                calibration["rgb_intrinsics"],
                calibration["depth_to_rgb_translation_m"],
                calibration["depth_to_rgb_rotation_xyzw"],
                calibration["rgb_resolution_wh"],
            )
            rgb_ready, depth_ready, _ = preprocessor.resize_rgb_depth(rgb_source[start:stop], aligned)
            rgb_cache[start:stop] = rgb_ready
            depth_cache[start:stop] = depth_ready
            invalid_count += int(np.count_nonzero(depth_ready >= 1.0))
            pixel_count += int(depth_ready.size)
        rgb_cache.flush()
        depth_cache.flush()
        plan_path = episode_path.with_name("episode_plan.yaml")
        plan = load_yaml(plan_path)
        windows = _episode_windows(
            source["sensors/timestamp_s"][:],
            source["state/timestamp_s"][:],
            source["state/position_xyz"][:],
            source["state/orientation_xyzw"][:],
            np.asarray(plan["goal_pose_xyz"][:2], dtype=np.float64),
            preprocessor.history_length,
            int(config["target"]["horizon_points"]),
            float(config["target"]["step_s"]),
            preprocessor.goal_scale_m,
        )
        for name, value in windows.items():
            np.save(output / f"{name}.npy", value, allow_pickle=False)
    metadata = {
        "episode_id": episode_id,
        "split": entry["split"],
        "hdf5_sha256": entry["hdf5_sha256"],
        "plan_hash": entry["plan_hash"],
        "frame_count": frame_count,
        "window_count": int(len(windows["anchor_indices"])),
        "history_dropped_anchors": min(preprocessor.history_length - 1, frame_count),
        "invalid_depth_pixels_after_resize": invalid_count,
        "depth_pixel_count_after_resize": pixel_count,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(values.min()),
        "p1": float(np.percentile(values, 1)),
        "p5": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "maximum": float(values.max()),
    }


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def prepare_cache(
    preprocessing_path: Path,
    model_config_path: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Build/reuse a cache from train and validation only; never open test HDF5."""
    config = load_yaml(preprocessing_path)
    model_config = load_yaml(model_config_path)
    index_path = PROJECT_ROOT / config["raw_source"]["dataset_index"]
    index_sha = file_sha256(index_path)
    preprocessing_sha = file_sha256(preprocessing_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    entries = [item for item in index["episodes"] if item["split"] in {"train", "validation"}]
    if any(item["split"] == "test" for item in entries):
        raise AssertionError("test split entered cache input")
    cache_root = PROJECT_ROOT / model_config["outputs"]["cache_root"]
    manifest_path = cache_root / "cache_manifest.json"
    if manifest_path.is_file() and not force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        current_hdf5 = {item["episode_id"]: item["hdf5_sha256"] for item in entries}
        if (
            existing.get("dataset_index_sha256") == index_sha
            and existing.get("preprocessing_config_sha256") == preprocessing_sha
            and existing.get("episode_hdf5_sha256") == current_hdf5
            and existing.get("status") == "PASS"
        ):
            return existing["data_report"]
    if cache_root.exists():
        shutil.rmtree(cache_root)
    cache_root.mkdir(parents=True)
    base_preprocessor = NavDiffusionPreprocessor.from_files(
        preprocessing_path, require_statistics=False
    )
    metadata = [
        _write_episode_cache(entry, cache_root, config, base_preprocessor)
        for entry in entries
    ]

    train_metadata = [item for item in metadata if item["split"] == "train"]
    validation_metadata = [item for item in metadata if item["split"] == "validation"]
    depth_sum = 0.0
    depth_square_sum = 0.0
    depth_count = 0
    trajectories = []
    goals = []
    padding = []
    for item in train_metadata:
        directory = cache_root / "episodes" / item["episode_id"]
        depth = np.load(directory / "depth_unit.npy", mmap_mode="r")
        depth_sum += float(np.sum(depth, dtype=np.float64))
        depth_square_sum += float(np.sum(np.square(depth, dtype=np.float64), dtype=np.float64))
        depth_count += int(depth.size)
        trajectories.append(np.load(directory / "future_local_m.npy"))
        goals.append(np.load(directory / "goal_normalized.npy") * base_preprocessor.goal_scale_m)
        padding.append(np.load(directory / "future_padding_points.npy"))
    depth_mean = depth_sum / depth_count
    depth_variance = max(depth_square_sum / depth_count - depth_mean * depth_mean, 0.0)
    depth_std = math.sqrt(depth_variance)
    if depth_std < 1.0e-6:
        raise ValueError("train depth standard deviation is degenerate")
    all_trajectories = np.concatenate(trajectories)
    all_goals = np.concatenate(goals)
    all_padding = np.concatenate(padding)
    maximum_absolute = float(np.max(np.abs(all_trajectories)))
    normalization = config["target"]["normalization"]
    default_bound = float(normalization["default_symmetric_bound_m"])
    margin = float(normalization["margin_fraction"])
    increment = float(normalization["increment_m"])
    selected_bound = max(
        default_bound,
        math.ceil(maximum_absolute * (1.0 + margin) / increment) * increment,
    )
    outlier_count = int(np.count_nonzero(np.abs(all_trajectories) > default_bound))
    report = {
        "report_version": 1,
        "status": "PASS",
        "dataset_index_sha256": index_sha,
        "preprocessing_config_sha256": preprocessing_sha,
        "train_trajectory_count": len(train_metadata),
        "validation_trajectory_count": len(validation_metadata),
        "train_window_count": sum(item["window_count"] for item in train_metadata),
        "validation_window_count": sum(item["window_count"] for item in validation_metadata),
        "history_dropped_anchors": {
            "train": sum(item["history_dropped_anchors"] for item in train_metadata),
            "validation": sum(item["history_dropped_anchors"] for item in validation_metadata),
        },
        "future_padding": {
            "sample_ratio": float(np.mean(all_padding > 0)),
            "mean_points_per_window": float(np.mean(all_padding)),
            "fully_padded_proportion": float(np.mean(all_padding == int(config["target"]["horizon_points"]))),
            "total_padding_points": int(all_padding.sum()),
            "total_target_points": int(all_padding.size * int(config["target"]["horizon_points"])),
        },
        "goal_local_m": {
            "x": _distribution(all_goals[:, 0]),
            "y": _distribution(all_goals[:, 1]),
        },
        "trajectory": {
            "x_m": _distribution(all_trajectories[:, :, 0].reshape(-1)),
            "y_m": _distribution(all_trajectories[:, :, 1].reshape(-1)),
            "absolute_m": _distribution(np.abs(all_trajectories).reshape(-1)),
            "maximum_absolute_m": maximum_absolute,
            "default_bound_outlier_count": outlier_count,
            "selected_symmetric_bound_m": selected_bound,
            "labels_clipped": False,
        },
        "depth": {
            "mean": depth_mean,
            "std": depth_std,
            "sample_count": depth_count,
            "frame_count": sum(item["frame_count"] for item in train_metadata),
            "invalid_fraction_after_resize": sum(item["invalid_depth_pixels_after_resize"] for item in train_metadata)
            / sum(item["depth_pixel_count_after_resize"] for item in train_metadata),
            "statistics_split": "train",
        },
        "cache": {
            "root": str(cache_root.relative_to(PROJECT_ROOT)),
            "format_version": CACHE_FORMAT_VERSION,
            "size_bytes": 0,
            "episode_count": len(metadata),
            "contains_test_split": False,
        },
    }
    report_path = PROJECT_ROOT / model_config["outputs"]["formal_model_directory"] / "navdiffusion_v0_data_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "status": "PASS",
        "cache_format_version": CACHE_FORMAT_VERSION,
        "dataset_index_sha256": index_sha,
        "preprocessing_config_sha256": preprocessing_sha,
        "episode_hdf5_sha256": {item["episode_id"]: item["hdf5_sha256"] for item in entries},
        "splits": sorted({item["split"] for item in entries}),
        "data_report": report,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["cache"]["size_bytes"] = _directory_bytes(cache_root)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["data_report"] = report
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


class NavDiffusionWindowDataset:
    """Memory-mapped window view over the disposable deterministic cache."""

    def __init__(
        self,
        split: str,
        preprocessing_path: Path,
        model_config_path: Path,
        *,
        episode_id: str | None = None,
        maximum_windows: int | None = None,
    ) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("training code may load only train or validation, never test")
        config = load_yaml(preprocessing_path)
        model_config = load_yaml(model_config_path)
        self.cache_root = PROJECT_ROOT / model_config["outputs"]["cache_root"]
        manifest = json.loads((self.cache_root / "cache_manifest.json").read_text(encoding="utf-8"))
        report_path = PROJECT_ROOT / model_config["outputs"]["formal_model_directory"] / "navdiffusion_v0_data_report.json"
        self.preprocessor = NavDiffusionPreprocessor.from_files(preprocessing_path, report_path)
        self.history = int(config["inputs"]["history_frames"])
        self.records: list[tuple[str, int]] = []
        for directory in sorted((self.cache_root / "episodes").iterdir()):
            metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
            if metadata["split"] != split:
                continue
            if episode_id is not None and metadata["episode_id"] != episode_id:
                continue
            self.records.extend((metadata["episode_id"], index) for index in range(metadata["window_count"]))
        if maximum_windows is not None:
            self.records = self.records[:maximum_windows]
        if not self.records:
            raise ValueError(f"no {split} windows selected")
        self._arrays: dict[str, dict[str, np.ndarray]] = {}
        self.cache_manifest = manifest

    def __len__(self) -> int:
        return len(self.records)

    def _episode_arrays(self, episode_id: str) -> dict[str, np.ndarray]:
        if episode_id not in self._arrays:
            directory = self.cache_root / "episodes" / episode_id
            names = (
                "rgb_standardized", "depth_unit", "anchor_indices", "anchor_timestamp_s",
                "goal_normalized", "future_local_m", "future_padding_points",
            )
            self._arrays[episode_id] = {
                name: np.load(directory / f"{name}.npy", mmap_mode="r") for name in names
            }
        return self._arrays[episode_id]

    def __getitem__(self, item: int) -> dict[str, Any]:
        import torch

        episode_id, local_index = self.records[item]
        arrays = self._episode_arrays(episode_id)
        anchor = int(arrays["anchor_indices"][local_index])
        indices = slice(anchor - self.history + 1, anchor + 1)
        rgb = np.asarray(arrays["rgb_standardized"][indices], dtype=np.float32)
        depth_unit = np.asarray(arrays["depth_unit"][indices], dtype=np.float32)
        depth = (depth_unit - self.preprocessor.depth_mean) / self.preprocessor.depth_std
        images = np.concatenate((rgb, depth[:, None]), axis=1)
        trajectory = self.preprocessor.normalize_trajectory(
            np.asarray(arrays["future_local_m"][local_index], dtype=np.float32)
        )
        return {
            "images": torch.from_numpy(images),
            "goal": torch.from_numpy(np.array(
                arrays["goal_normalized"][local_index], dtype=np.float32, copy=True
            )),
            "trajectory": torch.from_numpy(trajectory),
            "episode_id": episode_id,
            "anchor_timestamp_s": float(arrays["anchor_timestamp_s"][local_index]),
            "future_padding_points": int(arrays["future_padding_points"][local_index]),
        }
