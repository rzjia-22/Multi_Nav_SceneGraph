"""Standard-message publisher shared by synthetic and Isaac runtimes."""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np


PROJECT_SEMANTIC_LABELS = {
    "ground": 1,
    "tree_trunk": 2,
    "foliage": 3,
    "rock": 4,
    "building": 5,
    "robot": 6,
    "other_object": 7,
    # Composite vegetation roots fall back to the existing generic object ID;
    # supported tree assets override their rendered trunk and leaf prims.
    "vegetation": 7,
}


def quaternion_xyzw(orientation_wxyz: float | Sequence[float]) -> tuple[float, float, float, float]:
    """Convert either a planar yaw or Isaac's wxyz quaternion to ROS xyzw."""
    if isinstance(orientation_wxyz, (int, float)):
        yaw = float(orientation_wxyz)
        return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)
    values = tuple(float(value) for value in orientation_wxyz)
    if len(values) != 4:
        raise ValueError("orientation must be a yaw scalar or wxyz quaternion")
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1.0e-9:
        raise ValueError("orientation quaternion cannot be zero")
    w, x, y, z = (value / norm for value in values)
    return x, y, z, w


def remap_semantic_ids(raw_ids: np.ndarray, id_to_labels: dict) -> np.ndarray:
    """Map Isaac instance IDs to the fixed Hydra label-space IDs."""
    raw = np.asarray(raw_ids).squeeze()
    if raw.ndim != 2:
        raise ValueError("semantic annotator output must resolve to a 2-D image")
    mapped = np.zeros(raw.shape, dtype=np.uint16)
    for instance_id, metadata in id_to_labels.items():
        if isinstance(metadata, dict):
            class_name = str(metadata.get("class", metadata.get("semanticLabel", "")))
        else:
            class_name = str(metadata)
        label = PROJECT_SEMANTIC_LABELS.get(class_name, 0)
        if label:
            mapped[raw == int(instance_id)] = label
    return mapped


def time_message(seconds: float):
    from builtin_interfaces.msg import Time

    whole = math.floor(seconds)
    nanos = int(round((seconds - whole) * 1_000_000_000))
    if nanos >= 1_000_000_000:
        whole, nanos = whole + 1, nanos - 1_000_000_000
    return Time(sec=int(whole), nanosec=nanos)


class StandardRobotPublisher:
    def __init__(
        self,
        node,
        robot_id: str,
        *,
        width: int = 160,
        height: int = 120,
        focal_length_px: float = 128.0,
        publish_clock: bool = False,
        camera_mount: str = "forward",
    ) -> None:
        from geometry_msgs.msg import TransformStamped
        from nav_msgs.msg import Odometry
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import CameraInfo, Image
        from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

        self.robot_id = robot_id
        self.width, self.height = width, height
        self.focal_length_px = focal_length_px
        self.camera_mount = camera_mount
        self.Image, self.CameraInfo = Image, CameraInfo
        self.Odometry, self.TransformStamped, self.Clock = Odometry, TransformStamped, Clock
        # Hydra-ROS' image filters and initial CameraInfo lookup request reliable
        # delivery. A best-effort simulator publisher is QoS-incompatible and
        # causes Hydra to wait forever before constructing its sensor model.
        image_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.color_pub = node.create_publisher(Image, "camera/color/image_raw", image_qos)
        self.depth_pub = node.create_publisher(Image, "camera/depth/image_rect", image_qos)
        self.semantic_pub = node.create_publisher(Image, "camera/semantic/image_raw", image_qos)
        self.info_pub = node.create_publisher(CameraInfo, "camera/color/camera_info", image_qos)
        # depth_image_proc derives the CameraInfo sibling name from image_rect.
        # The RGB and depth images are registered, so publish the same calibrated
        # model on the conventional depth topic as well as Hydra's color topic.
        self.depth_info_pub = node.create_publisher(
            CameraInfo, "camera/depth/camera_info", image_qos
        )
        self.odom_pub = node.create_publisher(Odometry, "odom", 20)
        self.clock_pub = node.create_publisher(Clock, "/clock", 20) if publish_clock else None
        self.tf = TransformBroadcaster(node)
        self.static_tf = StaticTransformBroadcaster(node)
        self._publish_static(time_message(0.0))

    @property
    def odom_frame(self) -> str:
        return f"{self.robot_id}/odom"

    @property
    def base_frame(self) -> str:
        return f"{self.robot_id}/base_link"

    @property
    def camera_frame(self) -> str:
        return f"{self.robot_id}/camera_link"

    @property
    def optical_frame(self) -> str:
        return f"{self.robot_id}/camera_optical_frame"

    def _publish_static(self, stamp) -> None:
        mount = self.TransformStamped()
        mount.header.stamp = stamp
        mount.header.frame_id = self.base_frame
        mount.child_frame_id = self.camera_frame
        if self.camera_mount == "downward":
            mount.transform.translation.z = -0.08
        else:
            mount.transform.translation.x = 0.35
            mount.transform.translation.z = 0.25
        mount.transform.rotation.w = 1.0
        optical = self.TransformStamped()
        optical.header.stamp = stamp
        optical.header.frame_id = self.camera_frame
        optical.child_frame_id = self.optical_frame
        if self.camera_mount == "downward":
            optical.transform.rotation.x = 1.0
            optical.transform.rotation.w = 0.0
        else:
            optical.transform.rotation.x = -0.5
            optical.transform.rotation.y = 0.5
            optical.transform.rotation.z = -0.5
            optical.transform.rotation.w = 0.5
        self.static_tf.sendTransform([mount, optical])

    def publish_pose(
        self,
        sim_time: float,
        position: tuple[float, float, float],
        orientation_wxyz: float | Sequence[float],
        velocity: Sequence[float],
    ) -> None:
        stamp = time_message(sim_time)
        odom = self.Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = position
        (
            odom.pose.pose.orientation.x,
            odom.pose.pose.orientation.y,
            odom.pose.pose.orientation.z,
            odom.pose.pose.orientation.w,
        ) = quaternion_xyzw(orientation_wxyz)
        odom.twist.twist.linear.x = velocity[0]
        odom.twist.twist.linear.y = velocity[1]
        if len(velocity) == 3:
            odom.twist.twist.angular.z = velocity[2]
        elif len(velocity) == 6:
            odom.twist.twist.linear.z = velocity[2]
            odom.twist.twist.angular.x = velocity[3]
            odom.twist.twist.angular.y = velocity[4]
            odom.twist.twist.angular.z = velocity[5]
        else:
            raise ValueError("velocity must contain planar vx/vy/wz or full vx/vy/vz/wx/wy/wz")
        self.odom_pub.publish(odom)
        transform = self.TransformStamped()
        transform.header = odom.header
        transform.child_frame_id = self.base_frame
        transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = position
        transform.transform.rotation = odom.pose.pose.orientation
        self.tf.sendTransform(transform)
        if self.clock_pub is not None:
            clock = self.Clock()
            clock.clock = stamp
            self.clock_pub.publish(clock)

    def publish_images(
        self,
        sim_time: float,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        semantics: np.ndarray,
    ) -> None:
        rgb = np.ascontiguousarray(np.asarray(rgb)[..., :3].astype(np.uint8))
        depth = np.ascontiguousarray(np.asarray(depth_m).squeeze().astype(np.float32))
        labels = np.ascontiguousarray(np.asarray(semantics).squeeze().astype(np.uint16))
        if rgb.shape[:2] != depth.shape or depth.shape != labels.shape:
            raise ValueError("RGB, depth, and semantic images must be registered")
        stamp = time_message(sim_time)
        for publisher, array, encoding, step in (
            (self.color_pub, rgb, "rgb8", rgb.shape[1] * 3),
            (self.depth_pub, depth, "32FC1", depth.shape[1] * 4),
            (self.semantic_pub, labels, "16UC1", labels.shape[1] * 2),
        ):
            message = self.Image()
            message.header.stamp = stamp
            message.header.frame_id = self.optical_frame
            message.height, message.width = depth.shape
            message.encoding = encoding
            message.is_bigendian = False
            message.step = step
            message.data = array.tobytes()
            publisher.publish(message)
        info = self.CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = self.optical_frame
        info.height, info.width = depth.shape
        fx = fy = self.focal_length_px
        cx, cy = (info.width - 1.0) / 2.0, (info.height - 1.0) / 2.0
        info.distortion_model = "plumb_bob"
        info.d = [0.0] * 5
        info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.info_pub.publish(info)
        self.depth_info_pub.publish(info)


class ResearchRobotPublisher:
    """ROS contract for the terrain-following DIABLO/D435i research runtime."""

    def __init__(self, node, robot_id: str, rgb_intrinsics: np.ndarray, resolution_wh) -> None:
        from geometry_msgs.msg import TransformStamped
        from nav_msgs.msg import Odometry
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import CameraInfo, Image
        from tf2_ros import TransformBroadcaster

        self.robot_id = robot_id
        self.rgb_intrinsics = np.asarray(rgb_intrinsics, dtype=np.float64)
        self.width, self.height = (int(value) for value in resolution_wh)
        self.Image, self.CameraInfo = Image, CameraInfo
        self.Odometry, self.TransformStamped, self.Clock = Odometry, TransformStamped, Clock
        image_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.color_pub = node.create_publisher(Image, "camera/color/image_raw", image_qos)
        self.depth_pub = node.create_publisher(Image, "camera/depth/image_rect", image_qos)
        self.color_info_pub = node.create_publisher(
            CameraInfo, "camera/color/camera_info", image_qos
        )
        self.depth_info_pub = node.create_publisher(
            CameraInfo, "camera/depth/camera_info", image_qos
        )
        self.odom_pub = node.create_publisher(Odometry, "odom", 20)
        self.clock_pub = node.create_publisher(Clock, "/clock", 100)
        self.tf = TransformBroadcaster(node)

    @property
    def odom_frame(self) -> str:
        return f"{self.robot_id}/odom"

    @property
    def base_frame(self) -> str:
        return f"{self.robot_id}/base_link"

    @property
    def optical_frame(self) -> str:
        return f"{self.robot_id}/camera_optical_frame"

    def publish_clock(self, sim_time: float) -> None:
        message = self.Clock()
        message.clock = time_message(sim_time)
        self.clock_pub.publish(message)

    def publish_pose(
        self,
        sim_time: float,
        position_xyz,
        body_xyzw,
        linear_velocity_body,
        angular_velocity_body,
        body_to_optical_translation,
        body_to_optical_xyzw,
    ) -> None:
        stamp = time_message(sim_time)
        odom = self.Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = (
            float(value) for value in position_xyz
        )
        (
            odom.pose.pose.orientation.x,
            odom.pose.pose.orientation.y,
            odom.pose.pose.orientation.z,
            odom.pose.pose.orientation.w,
        ) = (float(value) for value in body_xyzw)
        (
            odom.twist.twist.linear.x,
            odom.twist.twist.linear.y,
            odom.twist.twist.linear.z,
        ) = (float(value) for value in linear_velocity_body)
        (
            odom.twist.twist.angular.x,
            odom.twist.twist.angular.y,
            odom.twist.twist.angular.z,
        ) = (float(value) for value in angular_velocity_body)
        self.odom_pub.publish(odom)

        base = self.TransformStamped()
        base.header = odom.header
        base.child_frame_id = self.base_frame
        base.transform.translation.x = odom.pose.pose.position.x
        base.transform.translation.y = odom.pose.pose.position.y
        base.transform.translation.z = odom.pose.pose.position.z
        base.transform.rotation = odom.pose.pose.orientation
        camera = self.TransformStamped()
        camera.header.stamp = stamp
        camera.header.frame_id = self.base_frame
        camera.child_frame_id = self.optical_frame
        (
            camera.transform.translation.x,
            camera.transform.translation.y,
            camera.transform.translation.z,
        ) = (float(value) for value in body_to_optical_translation)
        (
            camera.transform.rotation.x,
            camera.transform.rotation.y,
            camera.transform.rotation.z,
            camera.transform.rotation.w,
        ) = (float(value) for value in body_to_optical_xyzw)
        self.tf.sendTransform([base, camera])

    def _camera_info(self, stamp):
        info = self.CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = self.optical_frame
        info.width, info.height = self.width, self.height
        fx, fy = float(self.rgb_intrinsics[0, 0]), float(self.rgb_intrinsics[1, 1])
        cx, cy = float(self.rgb_intrinsics[0, 2]), float(self.rgb_intrinsics[1, 2])
        info.distortion_model = "plumb_bob"
        info.d = [0.0] * 5
        info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return info

    def publish_images(self, sim_time: float, rgb: np.ndarray, registered_depth_m: np.ndarray) -> None:
        rgb = np.ascontiguousarray(np.asarray(rgb)[..., :3].astype(np.uint8))
        depth = np.ascontiguousarray(np.asarray(registered_depth_m).astype(np.float32))
        if rgb.shape != (self.height, self.width, 3):
            raise ValueError(f"Research RGB shape mismatch: {rgb.shape}")
        if depth.shape != (self.height, self.width):
            raise ValueError(f"Research registered depth shape mismatch: {depth.shape}")
        stamp = time_message(sim_time)
        for publisher, array, encoding, step in (
            (self.color_pub, rgb, "rgb8", self.width * 3),
            (self.depth_pub, depth, "32FC1", self.width * 4),
        ):
            message = self.Image()
            message.header.stamp = stamp
            message.header.frame_id = self.optical_frame
            message.height, message.width = self.height, self.width
            message.encoding = encoding
            message.is_bigendian = False
            message.step = step
            message.data = array.tobytes()
            publisher.publish(message)
        info = self._camera_info(stamp)
        self.color_info_pub.publish(info)
        self.depth_info_pub.publish(info)
