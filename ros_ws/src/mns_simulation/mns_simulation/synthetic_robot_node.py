"""Deterministic non-GPU integration fixture for ROS/Hydra pipelines."""

from __future__ import annotations

import math
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

from .ros_publisher import StandardRobotPublisher


class SyntheticRobotNode(Node):
    def __init__(self) -> None:
        super().__init__("synthetic_robot")
        self.declare_parameter("robot_id", "robot")
        self.declare_parameter("kind", "go2")
        self.declare_parameter("spawn", [0.0, 0.0, 0.45, 0.0])
        self.declare_parameter("sensor_rate_hz", 10.0)
        self.declare_parameter("publish_clock", False)
        self.robot_id = str(self.get_parameter("robot_id").value)
        self.kind = str(self.get_parameter("kind").value)
        self.x, self.y, self.z, self.yaw = (
            float(value) for value in self.get_parameter("spawn").value
        )
        self.command = (0.0, 0.0, 0.0)
        self.sim_time = 0.0
        self.publisher = StandardRobotPublisher(
            self,
            self.robot_id,
            publish_clock=bool(self.get_parameter("publish_clock").value),
            camera_mount="downward" if self.kind == "uav" else "forward",
        )
        self.create_subscription(Twist, "cmd_vel_safe", self._command, 10)
        self.create_timer(0.02, self._physics)
        self.create_timer(1.0 / float(self.get_parameter("sensor_rate_hz").value), self._sensor)

    def _command(self, message: Twist) -> None:
        self.command = (message.linear.x, message.linear.y, message.angular.z)

    def _physics(self) -> None:
        dt = 0.02
        vx, vy, omega = self.command
        cosine, sine = math.cos(self.yaw), math.sin(self.yaw)
        self.x += (cosine * vx - sine * vy) * dt
        self.y += (sine * vx + cosine * vy) * dt
        self.yaw = math.atan2(math.sin(self.yaw + omega * dt), math.cos(self.yaw + omega * dt))
        self.sim_time += dt
        self.publisher.publish_pose(self.sim_time, (self.x, self.y, self.z), self.yaw, self.command)

    def _sensor(self) -> None:
        height, width = self.publisher.height, self.publisher.width
        columns = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
        rows = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        rgb = np.empty((height, width, 3), dtype=np.uint8)
        rgb[..., 0] = np.clip(60 + columns * 150, 0, 255)
        rgb[..., 1] = np.clip(50 + rows * 150, 0, 255)
        rgb[..., 2] = 80 if self.kind == "go2" else 160
        depth = (3.0 + 1.5 * columns + 0.2 * np.sin(self.sim_time)).repeat(height, axis=0)
        semantic = np.ones((height, width), dtype=np.uint16)
        semantic[:, width // 3 : width // 2] = 2
        semantic[height // 4 : height // 2, 2 * width // 3 : 5 * width // 6] = 4
        self.publisher.publish_images(self.sim_time, rgb, depth, semantic)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SyntheticRobotNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if node.context.ok():
            node.destroy_node()
        rclpy.try_shutdown()
