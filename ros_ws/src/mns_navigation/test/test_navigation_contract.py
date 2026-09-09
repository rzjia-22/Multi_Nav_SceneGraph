from mns_navigation.coverage import Point2D, plan_zigzag
from mns_navigation.path_follower import PurePursuitFollower


def test_coverage_path_is_consumable_by_velocity_follower():
    route = plan_zigzag((0.0, 3.0, 0.0, 2.0), 1.0)
    command = PurePursuitFollower().command(Point2D(0.0, 0.0), 0.0, route.sampled(0.2))
    assert route.total_length > 3.0
    assert command.linear_x > 0.0
