"""HDF5 schema helpers shared by the Isaac recorder and host validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def write_episode(path: Path, payload: dict[str, Any], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as output:
        output.attrs["schema_version"] = 1
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
        sensors.create_dataset("rgb", data=np.asarray(payload["rgb"], dtype=np.uint8), compression="gzip", compression_opts=4, shuffle=True)
        sensors.create_dataset("depth_raw_m", data=np.asarray(payload["depth_raw_m"], dtype=np.float32), compression="gzip", compression_opts=4, shuffle=True)
        sensors.create_dataset("depth_aligned_to_rgb_m", data=np.asarray(payload["depth_aligned_m"], dtype=np.float32), compression="gzip", compression_opts=4, shuffle=True)
        sensors.create_dataset("camera_pose_xyz_xyzw", data=np.asarray(payload["camera_pose"], dtype=np.float32))
        calibration = output.create_group("calibration")
        for name, value in payload["calibration"].items():
            calibration.create_dataset(name, data=np.asarray(value, dtype=np.float64))


def read_metadata(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as episode:
        return json.loads(str(episode.attrs["metadata_json"]))

