"""Aggregate mission lifecycle; task allocation remains intentionally simple."""

from __future__ import annotations

import json
import math

import rclpy
from geometry_msgs.msg import PoseStamped
from mns_interfaces.msg import MissionStatus
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


class MissionCoordinatorNode(Node):
    def __init__(self) -> None:
        super().__init__("mission_coordinator")
        self.declare_parameter("robot_ids", ["go2_1"])
        self.declare_parameter("goal_frames", ["go2_1/odom"])
        self.declare_parameter("mission_goals", [0.0, 0.0, 0.0])
        self.robot_ids = tuple(str(value) for value in self.get_parameter("robot_ids").value)
        self.goal_frames = tuple(str(value) for value in self.get_parameter("goal_frames").value)
        flat_goals = tuple(float(value) for value in self.get_parameter("mission_goals").value)
        if len(self.goal_frames) != len(self.robot_ids) or len(flat_goals) != 3 * len(self.robot_ids):
            raise ValueError("goal_frames and mission_goals must match robot_ids")
        self.states = {robot_id: "starting" for robot_id in self.robot_ids}
        self.publisher = self.create_publisher(String, "/system/mission_status", 10)
        goal_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._goal_publishers = [
            self.create_publisher(PoseStamped, f"/{robot_id}/mission/goal", goal_qos)
            for robot_id in self.robot_ids
        ]
        self._goals = tuple(
            flat_goals[index : index + 3] for index in range(0, len(flat_goals), 3)
        )
        self._subscriptions = []
        for robot_id in self.robot_ids:
            self._subscriptions.append(
                self.create_subscription(
                    MissionStatus,
                    f"/{robot_id}/mission/status",
                    lambda message, rid=robot_id: self._status(rid, message),
                    10,
                )
            )
        # Mission dispatch must not deadlock when a simulator has not started
        # publishing /clock yet.  Message stamps still use ROS time, while the
        # lifecycle timers deliberately use a monotonic wall clock.
        self._lifecycle_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(1.0, self._publish, clock=self._lifecycle_clock)
        self._goal_timer = self.create_timer(
            2.0, self._publish_goals_once, clock=self._lifecycle_clock
        )

    def _publish_goals_once(self) -> None:
        stamp = self.get_clock().now().to_msg()
        for publisher, frame, (x, y, yaw) in zip(
            self._goal_publishers, self.goal_frames, self._goals
        ):
            message = PoseStamped()
            message.header.stamp = stamp
            message.header.frame_id = frame
            message.pose.position.x = x
            message.pose.position.y = y
            message.pose.orientation.z = math.sin(yaw / 2.0)
            message.pose.orientation.w = math.cos(yaw / 2.0)
            publisher.publish(message)
        self._goal_timer.cancel()

    def _status(self, robot_id: str, message: MissionStatus) -> None:
        self.states[robot_id] = message.state

    def _publish(self) -> None:
        message = String()
        message.data = json.dumps(self.states, sort_keys=True)
        self.publisher.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MissionCoordinatorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if node.context.ok():
            node.destroy_node()
        rclpy.try_shutdown()
