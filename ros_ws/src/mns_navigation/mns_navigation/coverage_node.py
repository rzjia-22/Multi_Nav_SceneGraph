"""ROS node for coverage planning, sensor evaluation and path following."""

from __future__ import annotations

import json

from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from mns_interfaces.msg import MissionStatus
from nav_msgs.msg import Odometry, Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener

from .coverage import Point2D, ProgressTracker, plan_connected_coverage, plan_zigzag
from .path_follower import PurePursuitFollower
from .ros_utils import path_message, twist_message, yaw_from_quaternion
from .sensor_coverage import SensorCoverageGrid, plan_residual_coverage


class CoverageNavigatorNode(Node):
    def __init__(self) -> None:
        super().__init__("coverage_navigator")
        self.declare_parameter("robot_id", "robot")
        self.declare_parameter("frame_id", "robot/odom")
        self.declare_parameter("bounds", [-8.0, 8.0, -8.0, 8.0])
        self.declare_parameter("lane_spacing", 1.5)
        self.declare_parameter("planner", "connected")
        self.declare_parameter("obstacles_json", "[]")
        self.declare_parameter("clearance", 0.7)
        self.declare_parameter("lookahead_distance", 0.8)
        self.declare_parameter("max_linear_speed", 0.7)
        self.declare_parameter("max_angular_speed", 1.0)
        self.declare_parameter("sensor_coverage_enabled", False)
        self.declare_parameter("sensor_coverage_resolution", 0.1)
        self.declare_parameter("sensor_coverage_target_ratio", 0.9)
        self.declare_parameter("sensor_coverage_max_range", 3.0)
        self.declare_parameter("sensor_coverage_pixel_stride", 24)
        self.declare_parameter("max_residual_passes", 2)

        bounds = self.get_parameter("bounds").value
        lane_spacing = float(self.get_parameter("lane_spacing").value)
        planner = str(self.get_parameter("planner").value)
        obstacles = tuple(
            Point2D(float(item[0]), float(item[1]))
            for item in json.loads(str(self.get_parameter("obstacles_json").value))
        )
        self.route = (
            plan_connected_coverage(
                bounds,
                lane_spacing,
                obstacles,
                clearance=float(self.get_parameter("clearance").value),
            )
            if planner == "connected"
            else plan_zigzag(bounds, lane_spacing)
        )
        self.samples = self.route.sampled(0.2)
        self.tracker = ProgressTracker(self.route)
        self.follower = PurePursuitFollower(
            lookahead_distance=float(self.get_parameter("lookahead_distance").value),
            max_linear_speed=float(self.get_parameter("max_linear_speed").value),
            max_angular_speed=float(self.get_parameter("max_angular_speed").value),
        )
        self.position: Point2D | None = None
        self.yaw = 0.0
        self.camera_intrinsics: np.ndarray | None = None
        self.residual_passes = 0
        self.residual_exhausted = False
        self.sensor_grid: SensorCoverageGrid | None = None
        if bool(self.get_parameter("sensor_coverage_enabled").value):
            self.sensor_grid = SensorCoverageGrid(
                bounds,
                float(self.get_parameter("sensor_coverage_resolution").value),
                obstacles,
                clearance=float(self.get_parameter("clearance").value),
                start=self.route.waypoints[0],
            )
            self.bridge = CvBridge()
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)

        self.command_pub = self.create_publisher(Twist, "cmd_vel/navigation", 10)
        self.path_pub = self.create_publisher(Path, "mission/path", 1)
        self.status_pub = self.create_publisher(MissionStatus, "mission/status", 10)
        self.create_subscription(Odometry, "odom", self._odom_callback, 20)
        if self.sensor_grid is not None:
            self.create_subscription(
                CameraInfo, "camera/color/camera_info", self._camera_info_callback, 10
            )
            self.create_subscription(
                Image, "camera/depth/image_rect", self._depth_callback, 10
            )
        self.create_timer(0.05, self._control)
        self.create_timer(1.0, self._publish_status)
        self._publish_route()

    def _odom_callback(self, message: Odometry) -> None:
        self.position = Point2D(message.pose.pose.position.x, message.pose.pose.position.y)
        self.yaw = yaw_from_quaternion(message.pose.pose.orientation)

    def _camera_info_callback(self, message: CameraInfo) -> None:
        self.camera_intrinsics = np.asarray(message.k, dtype=np.float64).reshape(3, 3)

    def _depth_callback(self, message: Image) -> None:
        if self.sensor_grid is None or self.camera_intrinsics is None:
            return
        try:
            transform = self.tf_buffer.lookup_transform(
                str(self.get_parameter("frame_id").value),
                message.header.frame_id,
                Time.from_msg(message.header.stamp),
            ).transform
        except TransformException:
            return
        translation = transform.translation
        rotation = transform.rotation
        depth = self.bridge.imgmsg_to_cv2(message, desired_encoding="32FC1")
        self.sensor_grid.observe_depth(
            depth,
            self.camera_intrinsics,
            (translation.x, translation.y, translation.z),
            (rotation.w, rotation.x, rotation.y, rotation.z),
            max_range=float(self.get_parameter("sensor_coverage_max_range").value),
            pixel_stride=int(self.get_parameter("sensor_coverage_pixel_stride").value),
        )

    def _publish_route(self) -> None:
        self.samples = self.route.sampled(0.2)
        self.path_pub.publish(
            path_message(self, self.samples, str(self.get_parameter("frame_id").value))
        )

    def _route_complete(self) -> bool:
        # The follower intentionally stops within goal_tolerance, so requiring
        # an exact 99.9% projection can leave long routes permanently at
        # 99.x% with a zero velocity command.
        return self.route.total_length - self.tracker.distance <= self.follower.goal_tolerance

    def _plan_residual(self) -> bool:
        if self.sensor_grid is None or self.position is None:
            return False
        target = float(self.get_parameter("sensor_coverage_target_ratio").value)
        maximum = int(self.get_parameter("max_residual_passes").value)
        if self.sensor_grid.ratio >= target or self.residual_passes >= maximum:
            return False
        plan = plan_residual_coverage(self.sensor_grid, self.position)
        if plan is None:
            self.residual_exhausted = True
            return False
        self.route = plan.path
        self.tracker = ProgressTracker(self.route)
        self.residual_passes += 1
        self._publish_route()
        return True

    def _control(self) -> None:
        if self.position is None:
            return
        distance = self.tracker.update(self.position.x, self.position.y)
        if self._route_complete() and self._plan_residual():
            distance = 0.0
        local = [self.route.point_at(distance)]
        cursor = distance + 0.2
        while cursor < self.route.total_length:
            local.append(self.route.point_at(cursor))
            cursor += 0.2
        local.append(self.route.point_at(self.route.total_length))
        self.command_pub.publish(
            twist_message(self.follower.command(self.position, self.yaw, local))
        )

    def _publish_status(self) -> None:
        message = MissionStatus()
        message.header.stamp = self.get_clock().now().to_msg()
        message.robot_id = str(self.get_parameter("robot_id").value)
        message.navigator = "coverage"
        sensor_complete = (
            self.sensor_grid is None
            or self.sensor_grid.ratio
            >= float(self.get_parameter("sensor_coverage_target_ratio").value)
            or self.residual_passes >= int(self.get_parameter("max_residual_passes").value)
            or self.residual_exhausted
        )
        route_complete = self._route_complete()
        message.state = "waiting_for_odom" if self.position is None else (
            "complete" if route_complete and sensor_complete else "running"
        )
        message.progress = 1.0 if route_complete else float(self.tracker.fraction)
        sensor_detail = "" if self.sensor_grid is None else (
            f" sensor_coverage={self.sensor_grid.ratio:.3f}"
            f" residual_passes={self.residual_passes}"
        )
        message.detail = f"route_length={self.route.total_length:.2f}m{sensor_detail}"
        self.status_pub.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CoverageNavigatorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if node.context.ok():
            node.destroy_node()
        rclpy.try_shutdown()
