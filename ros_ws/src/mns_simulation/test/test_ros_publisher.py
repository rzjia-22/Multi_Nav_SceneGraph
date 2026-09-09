import math

import numpy as np
import pytest

from mns_simulation.ros_publisher import quaternion_xyzw, remap_semantic_ids, time_message


def test_fractional_time_normalization():
    message = time_message(1.9999999996)
    assert message.sec == 2
    assert message.nanosec == 0


def test_isaac_semantic_ids_are_mapped_to_project_contract():
    raw = np.asarray([[0, 10, 11], [12, 13, 99]], dtype=np.uint32)
    labels = remap_semantic_ids(raw, {
        "10": {"class": "ground"},
        "11": {"class": "tree_trunk"},
        "12": {"class": "robot"},
        "13": {"class": "not_in_label_space"},
    })
    assert labels.dtype == np.uint16
    assert labels.tolist() == [[0, 1, 2], [6, 0, 0]]


def test_planar_and_full_orientation_conversion():
    assert quaternion_xyzw(math.pi) == pytest.approx((0.0, 0.0, 1.0, 0.0))
    assert quaternion_xyzw((2.0, 0.0, 0.0, 0.0)) == pytest.approx((0.0, 0.0, 0.0, 1.0))
