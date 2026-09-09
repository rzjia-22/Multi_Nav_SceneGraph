from mns_multi_robot.partition import Bounds, partition_quadrants


def test_four_way_partition_is_disjoint():
    parts = partition_quadrants(Bounds(-8, 8, -8, 8), gap=1.0)
    assert len(parts) == 4
    assert parts[0].x_max < parts[1].x_min
    assert parts[0].y_max < parts[2].y_min

