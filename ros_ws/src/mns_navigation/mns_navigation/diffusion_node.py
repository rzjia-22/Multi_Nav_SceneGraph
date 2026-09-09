"""ROS adapter for RGB-D history, original NavDiffusion, and path following."""

from __future__ import annotations

from pathlib import Path as FilePath

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from mns_interfaces.msg import MissionStatus

from .coverage import Point2D
from .diffusion import DiffusionConfig, DiffusionPlanner, OriginalNavDiffusionPredictor, RGBDHistory
from .path_follower import PurePursuitFollower
from .ros_utils import path_message, twist_message, yaw_from_quaternion


class DiffusionNavigatorNode(Node):
    def __init__(self) -> None:
        super().__init__("diffusion_navigator")
        for name, default in (
            ("robot_id", "robot"),
            ("frame_id", "robot/odom"),
            ("checkpoint", "/workspace/models/navdiffusion.ckpt"),
            ("model_config", "/workspace/config/navigation/diffusion.yaml"),
            ("upstream_source", "/opt/forestnavigation/forest_nav"),
            ("device", "cuda"),
        ):
            self.declare_parameter(name, default)
        self.declare_parameter("history_length", 5)
        self.declare_parameter("planning_rate_hz", 2.0)
        self.bridge = CvBridge()
        self.history = RGBDHistory(int(self.get_parameter("history_length").value))
        config = DiffusionConfig(
            checkpoint=FilePath(str(self.get_parameter("checkpoint").value)),
            model_config=FilePath(str(self.get_parameter("model_config").value)),
            source_path=FilePath(str(self.get_parameter("upstream_source").value)),
            device=str(self.get_parameter("device").value),
        )
        self.planner = DiffusionPlanner(OriginalNavDiffusionPredictor(config), self.history)
        self.follower = PurePursuitFollower()
        self.position: Point2D | None = None
        self.yaw = 0.0
        self.goal: Point2D | None = None
        self.latest_rgb: np.ndarray | None = None
        self.trajectory: tuple[Point2D, ...] = ()
        self.cmd_pub = self.create_publisher(Twist, "cmd_vel/navigation", 10)
        self.path_pub = self.create_publisher(Path, "mission/path", 1)
        self.status_pub = self.create_publisher(MissionStatus, "mission/status", 10)
        self.create_subscription(Image, "camera/color/image_raw", self._rgb, 10)
        self.create_subscription(Image, "camera/depth/image_rect", self._depth, 10)
        self.create_subscription(Odometry, "odom", self._odom, 20)
        goal_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(PoseStamped, "mission/goal", self._goal, goal_qos)
        self.create_timer(1.0 / float(self.get_parameter("planning_rate_hz").value), self._plan)
        self.create_timer(0.05, self._control)
        self.create_timer(1.0, self._status)

    def _rgb(self, message: Image) -> None:
        self.latest_rgb = self.bridge.imgmsg_to_cv2(message, desired_encoding="rgb8")

    def _depth(self, message: Image) -> None:
        if self.latest_rgb is None:
            return
        depth = self.bridge.imgmsg_to_cv2(message, desired_encoding="32FC1")
        self.history.append(self.latest_rgb, depth)
        self.latest_rgb = None

    def _odom(self, message: Odometry) -> None:
        self.position = Point2D(message.pose.pose.position.x, message.pose.pose.position.y)
        self.yaw = yaw_from_quaternion(message.pose.pose.orientation)

    def _goal(self, message: PoseStamped) -> None:
        self.goal = Point2D(message.pose.position.x, message.pose.position.y)

    def _plan(self) -> None:
        if self.position is None or self.goal is None or not self.history.ready:
            return
        self.trajectory = self.planner.trajectory(self.position, self.yaw, self.goal)
        self.path_pub.publish(path_message(self, self.trajectory, str(self.get_parameter("frame_id").value)))

    def _control(self) -> None:
        if self.position is None or len(self.trajectory) < 2:
            return
        self.cmd_pub.publish(twist_message(self.follower.command(self.position, self.yaw, self.trajectory)))

    def _status(self) -> None:
        message = MissionStatus()
        message.header.stamp = self.get_clock().now().to_msg()
        message.robot_id = str(self.get_parameter("robot_id").value)
        message.navigator = "diffusion"
        if self.goal is None:
            message.state = "waiting_for_goal"
        elif not self.history.ready:
            message.state = "waiting_for_history"
        elif len(self.trajectory) < 2:
            message.state = "planning"
        else:
            message.state = "running"
        message.progress = 0.0
        message.detail = f"history={len(self.history)}/{self.history.length}"
        self.status_pub.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DiffusionNavigatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.context.ok():
            node.destroy_node()
        rclpy.try_shutdown()
