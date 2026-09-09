"""ROS-independent helpers for validating the project image contract."""

from __future__ import annotations

import numpy as np


def stamp_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def image_array(message) -> np.ndarray:
    layouts = {
        "rgb8": (np.dtype("u1"), 3),
        "32FC1": (np.dtype(">f4") if message.is_bigendian else np.dtype("<f4"), 1),
        "16UC1": (np.dtype(">u2") if message.is_bigendian else np.dtype("<u2"), 1),
    }
    if message.encoding not in layouts:
        raise ValueError(f"unsupported encoding {message.encoding!r}")
    dtype, channels = layouts[message.encoding]
    expected_step = message.width * dtype.itemsize * channels
    if message.step != expected_step:
        raise ValueError(f"step {message.step} != tightly-packed step {expected_step}")
    expected_bytes = expected_step * message.height
    if len(message.data) != expected_bytes:
        raise ValueError(f"data bytes {len(message.data)} != {expected_bytes}")
    array = np.frombuffer(message.data, dtype=dtype)
    shape = (message.height, message.width, channels) if channels > 1 else (message.height, message.width)
    return array.reshape(shape)
