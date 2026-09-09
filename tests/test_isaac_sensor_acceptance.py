from types import SimpleNamespace

import numpy as np
import pytest

from tools.sensor_validation import image_array, stamp_seconds


def message_for(array, encoding, step):
    return SimpleNamespace(
        data=array.tobytes(),
        encoding=encoding,
        height=array.shape[0],
        width=array.shape[1],
        is_bigendian=False,
        step=step,
    )


def test_decodes_project_sensor_encodings():
    rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    depth = np.arange(6, dtype=np.float32).reshape(2, 3)
    semantic = np.arange(6, dtype=np.uint16).reshape(2, 3)
    assert np.array_equal(image_array(message_for(rgb, "rgb8", 9)), rgb)
    assert np.array_equal(image_array(message_for(depth, "32FC1", 12)), depth)
    assert np.array_equal(image_array(message_for(semantic, "16UC1", 6)), semantic)


def test_rejects_non_tightly_packed_payload():
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="step"):
        image_array(message_for(rgb, "rgb8", 10))


def test_converts_ros_stamp_to_seconds():
    assert stamp_seconds(SimpleNamespace(sec=7, nanosec=250_000_000)) == 7.25
