"""Small ROS conversion helpers."""

from __future__ import annotations

import math

from .coverage import Point2D


def yaw_from_quaternion(quaternion) -> float:
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


def path_message(node, points: tuple[Point2D, ...], frame_id: str):
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Path

    message = Path()
    message.header.stamp = node.get_clock().now().to_msg()
    message.header.frame_id = frame_id
    for point in points:
        pose = PoseStamped()
        pose.header = message.header
        pose.pose.position.x = point.x
        pose.pose.position.y = point.y
        pose.pose.orientation.w = 1.0
        message.poses.append(pose)
    return message


def twist_message(command):
    from geometry_msgs.msg import Twist

    message = Twist()
    message.linear.x = command.linear_x
    message.linear.y = command.linear_y
    message.angular.z = command.angular_z
    return message

