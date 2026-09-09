"""ROS node exposing the stable motion boundary as Twist."""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

from .arbiter import Command, CommandArbiter


class CommandArbiterNode(Node):
    def __init__(self) -> None:
        super().__init__("command_arbiter")
        self.declare_parameter("navigation_timeout", 0.3)
        self.declare_parameter("safety_timeout", 0.2)
        self.declare_parameter("max_linear_speed", 1.0)
        self.declare_parameter("max_angular_speed", 1.5)
        self.arbiter = CommandArbiter(
            navigation_timeout=float(self.get_parameter("navigation_timeout").value),
            safety_timeout=float(self.get_parameter("safety_timeout").value),
            max_linear_speed=float(self.get_parameter("max_linear_speed").value),
            max_angular_speed=float(self.get_parameter("max_angular_speed").value),
        )
        self.publisher = self.create_publisher(Twist, "cmd_vel_safe", 10)
        self.create_subscription(Twist, "cmd_vel/navigation", lambda msg: self._update("navigation", msg), 10)
        self.create_subscription(Twist, "cmd_vel/safety", lambda msg: self._update("safety", msg), 10)
        self.create_timer(0.02, self._publish)

    def _seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _update(self, source: str, message: Twist) -> None:
        self.arbiter.update(
            source,
            Command(message.linear.x, message.linear.y, message.angular.z),
            self._seconds(),
        )

    def _publish(self) -> None:
        _, command = self.arbiter.select(self._seconds())
        message = Twist()
        message.linear.x = command.x
        message.linear.y = command.y
        message.angular.z = command.yaw
        self.publisher.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CommandArbiterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.context.ok():
            node.destroy_node()
        rclpy.try_shutdown()
