from mns_multi_robot.partition import Bounds, partition_quadrants


def test_four_robot_partition_leaves_requested_gap():
    quadrants = partition_quadrants(Bounds(-10, 10, -8, 8), gap=2.0)
    assert len(quadrants) == 4
    assert quadrants[1].x_min - quadrants[0].x_max == 2.0
    assert quadrants[2].y_min - quadrants[0].y_max == 2.0
