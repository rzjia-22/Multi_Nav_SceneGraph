import pytest

from mns_mapping.monitor_node import _GapCounter, _Rate


def test_pipeline_rate_uses_observation_window():
    rate = _Rate()
    for stamp in (1.0, 1.1, 1.2):
        rate.add(stamp)
    assert rate.count == 3
    assert rate.hz == pytest.approx(10.0)


def test_gap_counter_estimates_missing_frames():
    gaps = _GapCounter(10.0)
    for stamp in (1.0, 1.1, 1.2, 1.5, 1.6):
        gaps.add(stamp)
    assert gaps.dropped == 2
