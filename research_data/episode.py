"""HDF5 schema helpers shared by the Isaac recorder and host validation."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from .depth import ALIGNMENT_ALGORITHM, ALIGNMENT_VERSION, INVALID_DEPTH_CONVENTION


EPISODE_SCHEMA_VERSION = 2
COLLECTOR_VERSION = "research_forest_collector_v2"


def write_episode(path: Path, payload: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with h5py.File(path, "w") as output:
        output.attrs["schema_version"] = EPISODE_SCHEMA_VERSION
        output.attrs["dataset_version"] = "dataset_v0"
        output.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
        output.create_dataset("expert/global_path_xyz", data=np.asarray(payload["expert_path"], dtype=np.float32))
        state = output.create_group("state")
        for name in ("timestamp_s", "position_xyz", "orientation_xyzw", "linear_velocity_xyz", "angular_velocity_xyz", "command_vw"):
            state.create_dataset(name, data=np.asarray(payload[name], dtype=np.float64 if name == "timestamp_s" else np.float32))
        imu = output.create_group("imu")
        for name in ("timestamp_s", "linear_acceleration_xyz", "angular_velocity_xyz"):
            imu.create_dataset(name, data=np.asarray(payload[f"imu_{name}"], dtype=np.float64 if name == "timestamp_s" else np.float32))
        sensors = output.create_group("sensors")
        sensors.create_dataset("timestamp_s", data=np.asarray(payload["sensor_timestamp_s"], dtype=np.float64))
        sensors.create_dataset("rgb", data=np.asarray(payload["rgb"], dtype=np.uint8), compression="gzip", compression_opts=1)
        sensors.create_dataset(
            "depth_raw_z16", data=np.asarray(payload["depth_raw_z16"], dtype=np.uint16),
            compression="gzip", compression_opts=1, shuffle=True,
        )
        sensors.create_dataset("camera_pose_xyz_xyzw", data=np.asarray(payload["camera_pose"], dtype=np.float32))
        calibration = output.create_group("calibration")
        for name, value in payload["calibration"].items():
            calibration.create_dataset(name, data=np.asarray(value, dtype=np.float64))
        calibration.attrs["alignment_algorithm"] = ALIGNMENT_ALGORITHM
        calibration.attrs["alignment_version"] = ALIGNMENT_VERSION
        calibration.attrs["invalid_depth_convention"] = INVALID_DEPTH_CONVENTION
        output.flush()
        contributions: dict[str, int] = {"rgb": 0, "depth": 0, "state_imu_expert_calibration": 0}

        def account(name: str, item) -> None:
            if not isinstance(item, h5py.Dataset):
                return
            size = int(item.id.get_storage_size())
            if name == "sensors/rgb":
                contributions["rgb"] += size
            elif name == "sensors/depth_raw_z16":
                contributions["depth"] += size
            else:
                contributions["state_imu_expert_calibration"] += size

        output.visititems(account)
    elapsed = time.monotonic() - started
    file_size = path.stat().st_size
    contributions["hdf5_metadata_overhead"] = file_size - sum(contributions.values())
    return {
        "write_time_s": elapsed,
        "file_size_bytes": file_size,
        "disk_write_throughput_bytes_per_s": file_size / elapsed if elapsed > 0.0 else 0.0,
        "compressed_dataset_contributions_bytes": contributions,
    }


def read_metadata(path: Path) -> dict[str, Any]:
    import h5py

    with h5py.File(path, "r") as episode:
        return json.loads(str(episode.attrs["metadata_json"]))
