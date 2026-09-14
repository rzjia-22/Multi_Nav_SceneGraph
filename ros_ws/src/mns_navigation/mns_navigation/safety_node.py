"""Depth safety override published separately from navigation commands."""

from __future__ import annotations

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Empty

from .ros_utils import twist_message
from .safety import DepthSafetyController, depth_sector_distances


class SafetyMonitorNode(Node):
    def __init__(self) -> None:
        super().__init__("safety_monitor")
        self.controller = DepthSafetyController()
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Twist, "cmd_vel/safety", 10)
        self.create_subscription(Image, "camera/depth/image_rect", self._depth, 10)
        self.create_subscription(Empty, "mission/reset", self._reset, 10)

    def _reset(self, _message: Empty) -> None:
        self.controller.reset()

    def _depth(self, message: Image) -> None:
        depth = np.asarray(self.bridge.imgmsg_to_cv2(message, desired_encoding="32FC1"))
        decision = self.controller.decide(*depth_sector_distances(depth))
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
