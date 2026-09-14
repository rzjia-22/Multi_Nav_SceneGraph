"""ROS adapter for RGB-D history, selectable NavDiffusion, and path following."""

from __future__ import annotations

from pathlib import Path as FilePath
import json
import math
import time

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Empty, String
from mns_interfaces.msg import MissionStatus

from .coverage import Point2D
from .diffusion import DiffusionConfig, DiffusionPlanner, OriginalNavDiffusionPredictor, RGBDHistory
from .navdiffusion_v0.predictor import MNSNavDiffusionConfig, MNSNavDiffusionPredictor
from .path_follower import PurePursuitFollower
from .ros_utils import path_message, twist_message, yaw_from_quaternion


class DiffusionNavigatorNode(Node):
    def __init__(self) -> None:
        super().__init__("diffusion_navigator")
        for name, default in (
            ("robot_id", "robot"),
            ("frame_id", "robot/odom"),
            ("model_backend", "legacy"),
            ("checkpoint", "/workspace/models/navdiffusion.ckpt"),
            ("model_config", "/workspace/config/navigation/diffusion.yaml"),
            ("upstream_source", "/opt/forestnavigation/forest_nav"),
            ("device", "cuda"),
        ):
            self.declare_parameter(name, default)
        self.declare_parameter("history_length", 5)
        self.declare_parameter("planning_rate_hz", 2.0)
        self.declare_parameter("goal_tolerance", 0.35)
        self.declare_parameter("control_waypoints", 8)
        self.bridge = CvBridge()
        self.history = RGBDHistory(int(self.get_parameter("history_length").value))
        backend = str(self.get_parameter("model_backend").value)
        checkpoint = FilePath(str(self.get_parameter("checkpoint").value))
        device = str(self.get_parameter("device").value)
        if backend == "legacy":
            predictor = OriginalNavDiffusionPredictor(DiffusionConfig(
                checkpoint=checkpoint,
                model_config=FilePath(str(self.get_parameter("model_config").value)),
                source_path=FilePath(str(self.get_parameter("upstream_source").value)),
                device=device,
            ))
        elif backend == "mns_v0":
            predictor = MNSNavDiffusionPredictor(MNSNavDiffusionConfig(
                checkpoint=checkpoint,
                device=device,
                output_waypoints=32,
            ))
        else:
            raise ValueError(f"unsupported diffusion model_backend: {backend}")
        self.model_backend = backend
        self.predictor = predictor
        self.planner = DiffusionPlanner(predictor, self.history)
        self.follower = PurePursuitFollower()
        self.position: Point2D | None = None
        self.yaw = 0.0
        self.goal: Point2D | None = None
        self.latest_rgb: np.ndarray | None = None
        self.trajectory: tuple[Point2D, ...] = ()
        self.full_trajectory: tuple[Point2D, ...] = ()
        self.episode_id = ""
        self.planning_error = ""
        self.plan_count = 0
        self.last_planning_ms = 0.0
        self.cmd_pub = self.create_publisher(Twist, "cmd_vel/navigation", 10)
        self.path_pub = self.create_publisher(Path, "mission/path", 1)
        self.full_path_pub = self.create_publisher(Path, "mission/predicted_path_full", 1)
        self.diagnostic_pub = self.create_publisher(String, "mission/planning_diagnostics", 10)
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
        self.create_subscription(Empty, "mission/reset", self._reset, 10)
        self.create_subscription(String, "mission/episode_id", self._episode, 10)
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

    def _episode(self, message: String) -> None:
        self.episode_id = message.data

    def _reset(self, _message: Empty) -> None:
        self.planner.reset()
        self.position = None
        self.goal = None
        self.latest_rgb = None
        self.trajectory = ()
        self.full_trajectory = ()
        self.plan_count = 0
        self.last_planning_ms = 0.0
        self.planning_error = ""

    def _plan(self) -> None:
        if (
            self.position is None
            or self.goal is None
            or not self.history.ready
            or self._goal_reached()
        ):
            return
        start = time.perf_counter()
        try:
            full = self.planner.trajectory(self.position, self.yaw, self.goal)
            control_count = int(self.get_parameter("control_waypoints").value)
            if control_count <= 1 or len(full) < control_count:
                raise ValueError(
                    f"diffusion trajectory has {len(full)} points, needs at least {control_count}"
                )
            self.full_trajectory = full
            self.trajectory = full[:control_count]
            self.last_planning_ms = (time.perf_counter() - start) * 1000.0
            self.plan_count += 1
            self.planning_error = ""
            frame_id = str(self.get_parameter("frame_id").value)
            self.path_pub.publish(path_message(self, self.trajectory, frame_id))
            self.full_path_pub.publish(path_message(self, self.full_trajectory, frame_id))
            local = self.planner.last_local_trajectory
            jumps = np.linalg.norm(np.diff(local, axis=0), axis=1)
            goal_local = self.planner.last_goal_local
            final = local[-1]
            diagnostic = {
                "status": "PASS",
                "episode_id": self.episode_id,
                "timestamp_s": self.get_clock().now().nanoseconds * 1.0e-9,
                "robot_pose_xyyaw": [self.position.x, self.position.y, self.yaw],
                "mission_goal_xy": [self.goal.x, self.goal.y],
                "goal_distance_m": math.hypot(
                    self.goal.x - self.position.x, self.goal.y - self.position.y
                ),
                "history_length": len(self.history),
                "history_ready": self.history.ready,
                "full_local_points": local.tolist(),
                "full_world_points": [[point.x, point.y] for point in self.full_trajectory],
                "control_world_points": [[point.x, point.y] for point in self.trajectory],
                "inference_ms": float(getattr(self.predictor, "last_inference_ms", self.last_planning_ms)),
                "planner_wall_ms": self.last_planning_ms,
                "maximum_consecutive_jump_m": float(jumps.max()) if jumps.size else 0.0,
                "predicted_final_displacement_m": float(np.linalg.norm(final)),
                "goal_progress_dot": float(np.dot(final, goal_local)),
            }
            message = String()
            message.data = json.dumps(diagnostic, separators=(",", ":"), sort_keys=True)
            self.diagnostic_pub.publish(message)
        except Exception as error:
            self.trajectory = ()
            self.full_trajectory = ()
            self.last_planning_ms = (time.perf_counter() - start) * 1000.0
            self.planning_error = f"{type(error).__name__}: {error}"
            message = String()
            message.data = json.dumps({
                "status": "FAIL",
                "episode_id": self.episode_id,
                "timestamp_s": self.get_clock().now().nanoseconds * 1.0e-9,
                "failure_class": "MODEL",
                "error": self.planning_error,
            }, separators=(",", ":"), sort_keys=True)
            self.diagnostic_pub.publish(message)
            self.get_logger().error(self.planning_error)

    def _control(self) -> None:
        if self.position is None:
            return
        if self._goal_reached():
            self.cmd_pub.publish(Twist())
        elif len(self.trajectory) >= 2:
            # A diffusion trajectory is a rolling local horizon, not the
            # mission endpoint. A short stochastic sample must therefore not
            # trigger the follower's local endpoint stop condition.
            command = self.follower.command(
                self.position, self.yaw, self.trajectory, stop_at_goal=False
            )
            self.cmd_pub.publish(twist_message(command))

    def _goal_reached(self) -> bool:
        if self.position is None or self.goal is None:
            return False
        tolerance = float(self.get_parameter("goal_tolerance").value)
        return math.hypot(self.goal.x - self.position.x, self.goal.y - self.position.y) <= tolerance

    def _status(self) -> None:
        message = MissionStatus()
        message.header.stamp = self.get_clock().now().to_msg()
        message.robot_id = str(self.get_parameter("robot_id").value)
        message.navigator = f"diffusion:{self.model_backend}"
        if self.goal is None:
            message.state = "waiting_for_goal"
        elif self._goal_reached():
            message.state = "complete"
        elif not self.history.ready:
            message.state = "waiting_for_history"
        elif self.planning_error:
            message.state = "failed_model"
        elif len(self.trajectory) < 2:
            message.state = "planning"
        else:
            message.state = "running"
        message.progress = 1.0 if message.state == "complete" else 0.0
        message.detail = (
            f"history={len(self.history)}/{self.history.length}"
            f" plans={self.plan_count} planning_ms={self.last_planning_ms:.1f}"
            + (f" error={self.planning_error}" if self.planning_error else "")
        )
        self.status_pub.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DiffusionNavigatorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if node.context.ok():
            node.destroy_node()
        rclpy.try_shutdown()
