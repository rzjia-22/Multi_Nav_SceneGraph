"""Depth safety override published separately from navigation commands."""

from __future__ import annotations

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image

from .ros_utils import twist_message
from .safety import DepthSafetyController


class SafetyMonitorNode(Node):
    def __init__(self) -> None:
        super().__init__("safety_monitor")
        self.controller = DepthSafetyController()
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Twist, "cmd_vel/safety", 10)
        self.create_subscription(Image, "camera/depth/image_rect", self._depth, 10)

    def _depth(self, message: Image) -> None:
        depth = np.asarray(self.bridge.imgmsg_to_cv2(message, desired_encoding="32FC1"))
        height, width = depth.shape[:2]
        band = depth[height // 3 : 2 * height // 3]
        thirds = np.array_split(band, 3, axis=1)
        distances = [
            float(np.nanpercentile(section[np.isfinite(section)], 10))
            if np.any(np.isfinite(section)) else float("inf")
            for section in thirds
        ]
        decision = self.controller.decide(*distances)
        if decision.command is not None:
            self.publisher.publish(twist_message(decision.command))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SafetyMonitorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if node.context.ok():
            node.destroy_node()
        rclpy.try_shutdown()
