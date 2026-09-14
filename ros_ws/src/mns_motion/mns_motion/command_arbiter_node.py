"""ROS node exposing the stable motion boundary as Twist."""

from __future__ import annotations

import json

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Empty, String

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
        self.diagnostic_publisher = self.create_publisher(String, "cmd_vel_safe/diagnostics", 10)
        self.create_subscription(Twist, "cmd_vel/navigation", lambda msg: self._update("navigation", msg), 10)
        self.create_subscription(Twist, "cmd_vel/safety", lambda msg: self._update("safety", msg), 10)
        self.create_subscription(Empty, "mission/reset", self._reset, 10)
        self.create_timer(0.02, self._publish)

    def _reset(self, _message: Empty) -> None:
        self.arbiter.reset()

    def _seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _update(self, source: str, message: Twist) -> None:
        self.arbiter.update(
            source,
            Command(message.linear.x, message.linear.y, message.angular.z),
            self._seconds(),
        )

    def _publish(self) -> None:
        now = self._seconds()
        source, command = self.arbiter.select(now)
        message = Twist()
        message.linear.x = command.x
        message.linear.y = command.y
        message.angular.z = command.yaw
        self.publisher.publish(message)
        diagnostic = String()
        diagnostic.data = json.dumps({
            "source": source,
            "timestamp_s": now,
            "command": [command.x, command.y, command.yaw],
        }, separators=(",", ":"), sort_keys=True)
        self.diagnostic_publisher.publish(diagnostic)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CommandArbiterNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if node.context.ok():
            node.destroy_node()
        rclpy.try_shutdown()
