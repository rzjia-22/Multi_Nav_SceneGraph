"""Observe sensor-to-DSG activity without entering Hydra internals."""

from __future__ import annotations

from collections import deque

import rclpy
from hydra_msgs.msg import DsgUpdate
from mns_interfaces.msg import PipelineStatus
from rclpy.node import Node
from sensor_msgs.msg import Image


class _Rate:
    def __init__(self) -> None:
        self.stamps: deque[float] = deque(maxlen=120)
        self.count = 0

    def add(self, stamp: float) -> None:
        self.stamps.append(stamp)
        self.count += 1

    @property
    def hz(self) -> float:
        if len(self.stamps) < 2 or self.stamps[-1] <= self.stamps[0]:
            return 0.0
        return (len(self.stamps) - 1) / (self.stamps[-1] - self.stamps[0])


class _GapCounter:
    """Estimate missing source frames from monotonically stamped input."""

    def __init__(self, expected_rate_hz: float) -> None:
        self.period = 1.0 / expected_rate_hz if expected_rate_hz > 0.0 else 0.0
        self.previous_stamp: float | None = None
        self.dropped = 0

    def add(self, stamp: float) -> None:
        if self.period and self.previous_stamp is not None and stamp > self.previous_stamp:
            observed_steps = int((stamp - self.previous_stamp) / self.period + 0.5)
            self.dropped += max(0, observed_steps - 1)
        self.previous_stamp = stamp


class MappingMonitorNode(Node):
    def __init__(self) -> None:
        super().__init__("mapping_monitor")
        self.declare_parameter("robot_id", "robot")
        self.declare_parameter("stale_after_s", 3.0)
        self.declare_parameter("expected_input_rate_hz", 10.0)
        self.input_rate = _Rate()
        self.output_rate = _Rate()
        self.input_gaps = _GapCounter(
            float(self.get_parameter("expected_input_rate_hz").value)
        )
        self.last_input_stamp = 0.0
        self.last_output_stamp = 0.0
        self.status_pub = self.create_publisher(PipelineStatus, "hydra/pipeline_status", 10)
        self.create_subscription(Image, "camera/color/image_raw", self._on_input, 20)
        self.create_subscription(DsgUpdate, "hydra/backend/dsg", self._on_output, 10)
        self.create_timer(1.0, self._status)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_input(self, message: Image) -> None:
        now = self._now()
        self.last_input_stamp = now
        self.input_rate.add(now)
        source_stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1.0e-9
        if source_stamp > 0.0:
            self.input_gaps.add(source_stamp)

    def _on_output(self, message: DsgUpdate) -> None:
        now = self._now()
        self.last_output_stamp = now
        self.output_rate.add(now)

    def _status(self) -> None:
        now = self._now()
        stale = float(self.get_parameter("stale_after_s").value)
        state = "waiting_for_sensor"
        if self.last_input_stamp:
            state = "waiting_for_dsg" if not self.last_output_stamp else "running"
        if self.last_input_stamp and now - self.last_input_stamp > stale:
            state = "sensor_stale"
        elif self.last_output_stamp and now - self.last_output_stamp > stale:
            state = "mapping_stale"
        message = PipelineStatus()
        message.header.stamp = self.get_clock().now().to_msg()
        message.robot_id = str(self.get_parameter("robot_id").value)
        message.component = "hydra"
        message.state = state
        message.input_rate_hz = self.input_rate.hz
        message.output_rate_hz = self.output_rate.hz
        message.received = self.input_rate.count
        message.dropped = self.input_gaps.dropped
        message.latency_ms = max(0.0, (self.last_output_stamp - self.last_input_stamp) * 1000.0)
        message.detail = (
            "dropped estimates gaps in RGB source stamps; DSG rate is publication "
            "throughput, not per-frame completion latency"
        )
        self.status_pub.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MappingMonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.context.ok():
            node.destroy_node()
        rclpy.try_shutdown()
