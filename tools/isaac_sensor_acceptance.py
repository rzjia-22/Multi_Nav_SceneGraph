#!/usr/bin/env python3
"""Inspect real Isaac ROS 2 sensor messages, not merely topic discovery."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformListener

from sensor_validation import image_array, stamp_seconds


EXPECTED_ENCODINGS = {"rgb": "rgb8", "depth": "32FC1", "semantic": "16UC1"}
ALLOWED_SEMANTIC_IDS = set(range(8))


class SensorObserver(Node):
    def __init__(self, robot_ids: list[str]) -> None:
        super().__init__("isaac_sensor_acceptance")
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        self.robot_ids = robot_ids
        self.counts = {robot_id: defaultdict(int) for robot_id in robot_ids}
        self.stamps = {
            robot_id: {key: [] for key in ("rgb", "depth", "semantic", "info", "odom")}
            for robot_id in robot_ids
        }
        self.samples: dict[str, dict[str, dict]] = {robot_id: {} for robot_id in robot_ids}
        self.errors: dict[str, set[str]] = {robot_id: set() for robot_id in robot_ids}
        self.tf_edges: set[tuple[str, str]] = set()
        self.clock_stamps: list[float] = []
        reliable = QoSProfile(
            depth=20,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        static_qos = QoSProfile(
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        for robot_id in robot_ids:
            prefix = f"/{robot_id}"
            for key, topic in (
                ("rgb", f"{prefix}/camera/color/image_raw"),
                ("depth", f"{prefix}/camera/depth/image_rect"),
                ("semantic", f"{prefix}/camera/semantic/image_raw"),
            ):
                self.create_subscription(
                    Image,
                    topic,
                    lambda message, rid=robot_id, name=key: self._image(rid, name, message),
                    reliable,
                )
            self.create_subscription(
                CameraInfo,
                f"{prefix}/camera/color/camera_info",
                lambda message, rid=robot_id: self._camera_info(rid, message),
                reliable,
            )
            self.create_subscription(
                Odometry,
                f"{prefix}/odom",
                lambda message, rid=robot_id: self._odom(rid, message),
                50,
            )
        self.create_subscription(Clock, "/clock", self._clock_message, 50)
        self.create_subscription(TFMessage, "/tf", self._tf, 200)
        self.create_subscription(TFMessage, "/tf_static", self._tf, static_qos)
        self.tf_buffer = Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)

    def _record_stamp(self, robot_id: str, key: str, stamp) -> None:
        value = stamp_seconds(stamp)
        values = self.stamps[robot_id][key]
        if values and value < values[-1]:
            self.errors[robot_id].add(f"{key} timestamps are not monotonic")
        values.append(value)

    def _image(self, robot_id: str, key: str, message: Image) -> None:
        self.counts[robot_id][key] += 1
        self._record_stamp(robot_id, key, message.header.stamp)
        expected_frame = f"{robot_id}/camera_optical_frame"
        if message.header.frame_id != expected_frame:
            self.errors[robot_id].add(
                f"{key} frame {message.header.frame_id!r} != {expected_frame!r}"
            )
        if message.encoding != EXPECTED_ENCODINGS[key]:
            self.errors[robot_id].add(
                f"{key} encoding {message.encoding!r} != {EXPECTED_ENCODINGS[key]!r}"
            )
            return
        try:
            array = image_array(message)
        except ValueError as error:
            self.errors[robot_id].add(f"{key}: {error}")
            return
        sample = {
            "encoding": message.encoding,
            "frame_id": message.header.frame_id,
            "height": int(message.height),
            "width": int(message.width),
            "stamp_s": stamp_seconds(message.header.stamp),
        }
        if key == "rgb":
            sample.update({"min": int(array.min()), "max": int(array.max()), "mean": float(array.mean())})
            if int(array.max()) == int(array.min()):
                self.errors[robot_id].add("rgb image is constant")
        elif key == "depth":
            finite = array[np.isfinite(array) & (array > 0.0)]
            sample["finite_positive_fraction"] = float(finite.size / array.size)
            if finite.size:
                sample.update({
                    "min_m": float(finite.min()),
                    "median_m": float(np.median(finite)),
                    "max_m": float(finite.max()),
                })
            if finite.size < max(1, int(array.size * 0.01)):
                self.errors[robot_id].add("depth has less than 1% finite positive pixels")
        else:
            unique, counts = np.unique(array.astype(np.uint16), return_counts=True)
            label_counts = {str(int(label)): int(count) for label, count in zip(unique, counts)}
            sample["label_counts"] = label_counts
            unexpected = set(int(value) for value in unique) - ALLOWED_SEMANTIC_IDS
            if unexpected:
                self.errors[robot_id].add(f"semantic has unexpected IDs {sorted(unexpected)}")
            if not any(int(value) > 0 for value in unique):
                self.errors[robot_id].add("semantic image contains only unknown label 0")
        self.samples[robot_id][key] = sample

    def _camera_info(self, robot_id: str, message: CameraInfo) -> None:
        self.counts[robot_id]["info"] += 1
        self._record_stamp(robot_id, "info", message.header.stamp)
        expected_frame = f"{robot_id}/camera_optical_frame"
        if message.header.frame_id != expected_frame:
            self.errors[robot_id].add(f"CameraInfo frame {message.header.frame_id!r} != {expected_frame!r}")
        if message.width <= 0 or message.height <= 0 or len(message.k) != 9:
            self.errors[robot_id].add("CameraInfo dimensions or K matrix are invalid")
        elif message.k[0] <= 0 or message.k[4] <= 0 or not math.isclose(message.k[8], 1.0):
            self.errors[robot_id].add("CameraInfo intrinsics are invalid")
        self.samples[robot_id]["camera_info"] = {
            "frame_id": message.header.frame_id,
            "height": int(message.height),
            "width": int(message.width),
            "fx": float(message.k[0]),
            "fy": float(message.k[4]),
            "cx": float(message.k[2]),
            "cy": float(message.k[5]),
            "stamp_s": stamp_seconds(message.header.stamp),
        }

    def _odom(self, robot_id: str, message: Odometry) -> None:
        self.counts[robot_id]["odom"] += 1
        self._record_stamp(robot_id, "odom", message.header.stamp)
        if message.header.frame_id != f"{robot_id}/odom":
            self.errors[robot_id].add(f"invalid odom frame {message.header.frame_id!r}")
        if message.child_frame_id != f"{robot_id}/base_link":
            self.errors[robot_id].add(f"invalid odom child frame {message.child_frame_id!r}")
        self.samples[robot_id]["odom"] = {
            "frame_id": message.header.frame_id,
            "child_frame_id": message.child_frame_id,
            "position_m": [
                float(message.pose.pose.position.x),
                float(message.pose.pose.position.y),
                float(message.pose.pose.position.z),
            ],
            "stamp_s": stamp_seconds(message.header.stamp),
        }

    def _clock_message(self, message: Clock) -> None:
        value = stamp_seconds(message.clock)
        if self.clock_stamps and value < self.clock_stamps[-1]:
            for errors in self.errors.values():
                errors.add("/clock is not monotonic")
        self.clock_stamps.append(value)

    def _tf(self, message: TFMessage) -> None:
        for transform in message.transforms:
            self.tf_edges.add((transform.header.frame_id, transform.child_frame_id))

    def report(self, elapsed: float, expected_sensor_rate: float, min_wall_rate: float) -> tuple[dict, list[str]]:
        report = {"clock_count": len(self.clock_stamps), "robots": {}}
        failures: list[str] = []
        if len(self.clock_stamps) < 2 or self.clock_stamps[-1] <= self.clock_stamps[0]:
            failures.append("/clock did not advance")
        for robot_id in self.robot_ids:
            robot_failures = sorted(self.errors[robot_id])
            rates = {}
            for key in ("rgb", "depth", "semantic", "info", "odom"):
                count = self.counts[robot_id][key]
                values = self.stamps[robot_id][key]
                wall_rate = count / elapsed
                sim_rate = (
                    (len(values) - 1) / (values[-1] - values[0])
                    if len(values) > 1 and values[-1] > values[0]
                    else 0.0
                )
                rates[key] = {"count": count, "sim_hz": sim_rate, "wall_hz": wall_rate}
                if count < 2:
                    robot_failures.append(f"fewer than two {key} messages")
                if key in ("rgb", "depth", "semantic", "info"):
                    if sim_rate and not 0.8 * expected_sensor_rate <= sim_rate <= 1.2 * expected_sensor_rate:
                        robot_failures.append(f"{key} simulation rate {sim_rate:.2f} Hz is outside tolerance")
                    if wall_rate < min_wall_rate:
                        robot_failures.append(f"{key} wall rate {wall_rate:.2f} Hz is below {min_wall_rate:.2f} Hz")
            expected_edges = {
                (f"{robot_id}/odom", f"{robot_id}/base_link"),
                (f"{robot_id}/base_link", f"{robot_id}/camera_link"),
                (f"{robot_id}/camera_link", f"{robot_id}/camera_optical_frame"),
            }
            missing_edges = sorted(expected_edges - self.tf_edges)
            if missing_edges:
                robot_failures.append(f"missing TF edges {missing_edges}")
            transform_at_image_time = False
            depth_stamps = self.stamps[robot_id]["depth"]
            if depth_stamps:
                try:
                    self.tf_buffer.lookup_transform(
                        f"{robot_id}/odom",
                        f"{robot_id}/camera_optical_frame",
                        Time(seconds=depth_stamps[-1]),
                        timeout=Duration(seconds=0.0),
                    )
                    transform_at_image_time = True
                except Exception as error:  # tf2 exception types vary across ROS distributions
                    robot_failures.append(f"TF unavailable at image stamp: {error}")
            report["robots"][robot_id] = {
                "passed": not robot_failures,
                "failures": robot_failures,
                "rates": rates,
                "samples": self.samples[robot_id],
                "missing_tf_edges": missing_edges,
                "transform_at_image_time": transform_at_image_time,
            }
            failures.extend(f"{robot_id}: {failure}" for failure in robot_failures)
        return report, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robots", nargs="+", required=True)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--expected-sensor-rate", type=float, default=10.0)
    parser.add_argument("--min-wall-rate", type=float, default=1.0)
    args = parser.parse_args()
    if min(args.duration, args.expected_sensor_rate, args.min_wall_rate) <= 0:
        parser.error("duration and rates must be positive")
    rclpy.init()
    observer = SensorObserver(args.robots)
    started = time.monotonic()
    try:
        while rclpy.ok() and time.monotonic() - started < args.duration:
            rclpy.spin_once(observer, timeout_sec=0.1)
        elapsed = time.monotonic() - started
        report, failures = observer.report(elapsed, args.expected_sensor_rate, args.min_wall_rate)
        result = {"elapsed_s": elapsed, "passed": not failures, "failures": failures, **report}
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if not failures else 1
    finally:
        observer.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
