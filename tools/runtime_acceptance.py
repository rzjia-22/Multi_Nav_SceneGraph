#!/usr/bin/env python3
"""Observe a running Phase 1/2 graph and emit a machine-readable verdict."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import time

import rclpy
from hydra_msgs.msg import DsgUpdate
from mns_interfaces.msg import MissionStatus, PipelineStatus
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from tf2_msgs.msg import TFMessage


class RuntimeObserver(Node):
    def __init__(self, robot_ids: list[str], require_mapping: bool) -> None:
        super().__init__("runtime_acceptance")
        self.robot_ids = robot_ids
        self.require_mapping = require_mapping
        self.counts: dict[str, dict[str, int]] = {
            robot_id: defaultdict(int) for robot_id in robot_ids
        }
        self.positions: dict[str, list[tuple[float, float]]] = {
            robot_id: [] for robot_id in robot_ids
        }
        self.pipeline: dict[str, PipelineStatus] = {}
        self.mission: dict[str, MissionStatus] = {}
        self.tf_frames: set[str] = set()
        reliable = QoSProfile(
            depth=20,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        for robot_id in robot_ids:
            prefix = f"/{robot_id}"
            for key, topic, message_type in (
                ("rgb", f"{prefix}/camera/color/image_raw", Image),
                ("depth", f"{prefix}/camera/depth/image_rect", Image),
                ("semantic", f"{prefix}/camera/semantic/image_raw", Image),
                ("camera_info", f"{prefix}/camera/color/camera_info", CameraInfo),
            ):
                self.create_subscription(
                    message_type, topic,
                    lambda _message, rid=robot_id, name=key: self._count(rid, name),
                    reliable,
                )
            self.create_subscription(
                Odometry, f"{prefix}/odom",
                lambda message, rid=robot_id: self._odom(rid, message), 20,
            )
            self.create_subscription(
                MissionStatus, f"{prefix}/mission/status",
                lambda message, rid=robot_id: self.mission.__setitem__(rid, message), 10,
            )
            if require_mapping:
                self.create_subscription(
                    DsgUpdate, f"{prefix}/hydra/backend/dsg",
                    lambda _message, rid=robot_id: self._count(rid, "dsg"), 10,
                )
                self.create_subscription(
                    PipelineStatus, f"{prefix}/hydra/pipeline_status",
                    lambda message, rid=robot_id: self.pipeline.__setitem__(rid, message), 10,
                )
        self.create_subscription(TFMessage, "/tf", self._tf, 100)
        static_qos = QoSProfile(
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(TFMessage, "/tf_static", self._tf, static_qos)

    def _count(self, robot_id: str, key: str) -> None:
        self.counts[robot_id][key] += 1

    def _odom(self, robot_id: str, message: Odometry) -> None:
        self._count(robot_id, "odom")
        self.positions[robot_id].append((message.pose.pose.position.x, message.pose.pose.position.y))

    def _tf(self, message: TFMessage) -> None:
        for transform in message.transforms:
            self.tf_frames.add(transform.header.frame_id)
            self.tf_frames.add(transform.child_frame_id)

    def report(self, elapsed: float) -> tuple[dict, list[str]]:
        report: dict[str, dict] = {}
        failures: list[str] = []
        for robot_id in self.robot_ids:
            counts = dict(self.counts[robot_id])
            start_end = self.positions[robot_id]
            displacement = (
                math.dist(start_end[0], start_end[-1]) if len(start_end) > 1 else 0.0
            )
            pipeline = self.pipeline.get(robot_id)
            mission = self.mission.get(robot_id)
            expected_frames = {
                f"{robot_id}/odom", f"{robot_id}/base_link",
                f"{robot_id}/camera_link", f"{robot_id}/camera_optical_frame",
            }
            missing_frames = sorted(expected_frames - self.tf_frames)
            robot_report = {
                "counts": counts,
                "depth_rate_hz": counts.get("depth", 0) / elapsed,
                "displacement_m": displacement,
                "mission_state": mission.state if mission else None,
                "navigator": mission.navigator if mission else None,
                "missing_tf_frames": missing_frames,
            }
            required = ("rgb", "depth", "semantic", "camera_info", "odom")
            for key in required:
                if counts.get(key, 0) == 0:
                    failures.append(f"{robot_id}: no {key} messages")
            if counts.get("depth", 0) / elapsed < 5.0:
                failures.append(f"{robot_id}: depth rate below 5 Hz")
            if mission is None:
                failures.append(f"{robot_id}: no mission status")
            if missing_frames:
                failures.append(f"{robot_id}: missing TF frames {missing_frames}")
            if self.require_mapping:
                robot_report["dsg_count"] = counts.get("dsg", 0)
                robot_report["mapping_state"] = pipeline.state if pipeline else None
                robot_report["mapping_received"] = int(pipeline.received) if pipeline else 0
                robot_report["mapping_dropped"] = int(pipeline.dropped) if pipeline else 0
                if counts.get("dsg", 0) == 0:
                    failures.append(f"{robot_id}: no backend DSG updates")
                if pipeline is None or pipeline.state != "running":
                    failures.append(f"{robot_id}: mapping pipeline is not running")
            report[robot_id] = robot_report
        return report, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robots", nargs="+", required=True)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--no-mapping", action="store_true")
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("duration must be positive")
    rclpy.init()
    node = RuntimeObserver(args.robots, not args.no_mapping)
    started = time.monotonic()
    try:
        while rclpy.ok() and time.monotonic() - started < args.duration:
            rclpy.spin_once(node, timeout_sec=0.1)
        elapsed = time.monotonic() - started
        report, failures = node.report(elapsed)
        print(json.dumps({
            "elapsed_s": elapsed,
            "passed": not failures,
            "failures": failures,
            "robots": report,
        }, indent=2, sort_keys=True))
        return 0 if not failures else 1
    finally:
        if node.context.ok():
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
