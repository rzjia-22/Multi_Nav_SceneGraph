#!/usr/bin/env python3
"""Exercise the real Isaac Go2 RL backend through its ROS Twist boundary."""

from __future__ import annotations

import argparse
import json
import math
import time

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node


def stamp_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def euler_from_quaternion(quaternion) -> tuple[float, float, float]:
    x, y, z, w = quaternion.x, quaternion.y, quaternion.z, quaternion.w
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return roll, pitch, math.atan2(siny_cosp, cosy_cosp)


class MotionObserver(Node):
    def __init__(self, robot_id: str) -> None:
        super().__init__("go2_motion_acceptance")
        self.publisher = self.create_publisher(Twist, f"/{robot_id}/cmd_vel_safe", 10)
        self.create_subscription(Odometry, f"/{robot_id}/odom", self._odom, 100)
        self.samples: list[dict] = []
        self.start_sim_time: float | None = None

    def _odom(self, message: Odometry) -> None:
        stamp = stamp_seconds(message.header.stamp)
        if self.start_sim_time is None:
            self.start_sim_time = stamp
        roll, pitch, yaw = euler_from_quaternion(message.pose.pose.orientation)
        self.samples.append({
            "t": stamp - self.start_sim_time,
            "x": float(message.pose.pose.position.x),
            "y": float(message.pose.pose.position.y),
            "z": float(message.pose.pose.position.z),
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "vx": float(message.twist.twist.linear.x),
            "vy": float(message.twist.twist.linear.y),
            "wz": float(message.twist.twist.angular.z),
        })

    def send(self, vx: float, wz: float) -> None:
        message = Twist()
        message.linear.x = vx
        message.angular.z = wz
        self.publisher.publish(message)


def nearest(samples: list[dict], target: float) -> dict:
    return min(samples, key=lambda sample: abs(sample["t"] - target))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="go2_1")
    parser.add_argument("--wall-timeout", type=float, default=120.0)
    parser.add_argument("--forward-speed", type=float, default=0.6)
    parser.add_argument("--turn-speed", type=float, default=0.5)
    parser.add_argument("--turn-linear-speed", type=float, default=0.6)
    parser.add_argument(
        "--deadman-duration",
        type=float,
        default=8.0,
        help="seconds to observe after command publication stops at t=12",
    )
    args = parser.parse_args()
    rclpy.init()
    node = MotionObserver(args.robot)
    wall_start = time.monotonic()
    try:
        while rclpy.ok() and time.monotonic() - wall_start < args.wall_timeout:
            rclpy.spin_once(node, timeout_sec=0.02)
            if not node.samples:
                continue
            elapsed = node.samples[-1]["t"]
            if elapsed < 3.0:
                node.send(0.0, 0.0)
            elif elapsed < 7.0:
                node.send(args.forward_speed, 0.0)
            elif elapsed < 9.0:
                # Exercise a moving turn within the actor's observed stable
                # curvature; in-place rotation is not a supported gait.
                node.send(args.turn_linear_speed, args.turn_speed)
            elif elapsed < 11.0:
                node.send(0.0, 0.0)
            elif elapsed < 12.0:
                node.send(args.forward_speed, 0.0)
            elif elapsed >= 12.0 + args.deadman_duration:
                break
            # After 12 s deliberately publish nothing: backend deadman and
            # post-maneuver stability test.
        end_time = 12.0 + args.deadman_duration
        if not node.samples or node.samples[-1]["t"] < end_time:
            print("MNS_GO2_MOTION_RESULT=" + json.dumps({"status": "FAIL", "reason": "timeout"}))
            return 1

        samples = node.samples
        at_3, at_7, at_9, at_11, at_12, at_end = (
            nearest(samples, value) for value in (3.0, 7.0, 9.0, 11.0, 12.0, end_time)
        )
        yaws = np.unwrap([sample["yaw"] for sample in samples])
        for sample, yaw in zip(samples, yaws):
            sample["unwrapped_yaw"] = float(yaw)
        at_7, at_9 = nearest(samples, 7.0), nearest(samples, 9.0)
        heading = at_3["yaw"]
        forward_displacement = (
            (at_7["x"] - at_3["x"]) * math.cos(heading)
            + (at_7["y"] - at_3["y"]) * math.sin(heading)
        )
        yaw_change = abs(at_9["unwrapped_yaw"] - at_7["unwrapped_yaw"])
        final_speed = math.hypot(at_end["vx"], at_end["vy"])
        failures = []
        if min(sample["z"] for sample in samples[200:]) < 0.20:
            failures.append("base height fell below 0.20 m")
        if max(abs(sample["roll"]) for sample in samples[200:]) > 0.70:
            failures.append("absolute roll exceeded 0.70 rad")
        if max(abs(sample["pitch"]) for sample in samples[200:]) > 0.70:
            failures.append("absolute pitch exceeded 0.70 rad")
        if forward_displacement < 0.30:
            failures.append("forward displacement was below 0.30 m")
        if yaw_change < 0.25:
            failures.append("turn response was below 0.25 rad")
        if final_speed > 0.30 or abs(at_end["wz"]) > 0.35:
            failures.append("backend did not settle after command timeout")
        report = {
            "status": "PASS" if not failures else "FAIL",
            "failures": failures,
            "samples": len(samples),
            "sim_duration_s": samples[-1]["t"],
            "base_height_m": {"min": min(sample["z"] for sample in samples[200:]), "final": at_end["z"]},
            "max_abs_roll_rad": max(abs(sample["roll"]) for sample in samples[200:]),
            "max_abs_pitch_rad": max(abs(sample["pitch"]) for sample in samples[200:]),
            "forward_displacement_m": forward_displacement,
            "yaw_change_rad": yaw_change,
            "explicit_stop_speed_mps": math.hypot(at_11["vx"], at_11["vy"]),
            "deadman_final_speed_mps": final_speed,
            "deadman_final_yaw_rate_rps": abs(at_end["wz"]),
            "pulse_displacement_m": math.hypot(at_12["x"] - at_11["x"], at_12["y"] - at_11["y"]),
        }
        print("MNS_GO2_MOTION_RESULT=" + json.dumps(report, sort_keys=True))
        return 0 if not failures else 1
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
