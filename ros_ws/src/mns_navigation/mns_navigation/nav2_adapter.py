"""Forward namespaced mission goals to the standard Nav2 action."""

from __future__ import annotations

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from mns_interfaces.msg import MissionStatus
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy


class Nav2GoalAdapter(Node):
    def __init__(self) -> None:
        super().__init__("nav2_goal_adapter")
        self.declare_parameter("robot_id", "robot")
        self.client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        goal_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(PoseStamped, "mission/goal", self._goal, goal_qos)
        self.status_pub = self.create_publisher(MissionStatus, "mission/status", 10)
        self.pending: PoseStamped | None = None
        self.sent = False
        self.state = "waiting_for_goal"
        self.detail = ""
        self.create_timer(0.2, self._dispatch)
        self.create_timer(1.0, self._status)

    def _goal(self, message: PoseStamped) -> None:
        self.pending = message
        self.sent = False
        self.state = "waiting_for_nav2"

    def _dispatch(self) -> None:
        if self.pending is None or self.sent or not self.client.server_is_ready():
            return
        goal = NavigateToPose.Goal()
        goal.pose = self.pending
        self.sent = True
        future = self.client.send_goal_async(goal)
        future.add_done_callback(self._accepted)

    def _accepted(self, future) -> None:
        handle = future.result()
        if not handle.accepted:
            self.state = "failed"
            self.detail = "Nav2 rejected mission goal"
            return
        self.state = "running"
        handle.get_result_async().add_done_callback(self._result)

    def _result(self, future) -> None:
        status = future.result().status
        self.state = "complete" if status == GoalStatus.STATUS_SUCCEEDED else "failed"
        self.detail = f"Nav2 action status={status}"

    def _status(self) -> None:
        message = MissionStatus()
        message.header.stamp = self.get_clock().now().to_msg()
        message.robot_id = str(self.get_parameter("robot_id").value)
        message.navigator = "nav2"
        message.state = self.state
        message.progress = 1.0 if self.state == "complete" else 0.0
        message.detail = self.detail
        self.status_pub.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Nav2GoalAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.context.ok():
            node.destroy_node()
        rclpy.try_shutdown()
