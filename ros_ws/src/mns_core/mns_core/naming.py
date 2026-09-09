"""Deterministic topic and frame names for a robot instance."""

from __future__ import annotations

from dataclasses import dataclass
import re

_VALID_ID = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class RobotNames:
    robot_id: str

    def __post_init__(self) -> None:
        if not _VALID_ID.fullmatch(self.robot_id):
            raise ValueError(
                "robot_id must start with a lowercase letter and contain only "
                "lowercase letters, digits, and underscores"
            )

    @property
    def namespace(self) -> str:
        return f"/{self.robot_id}"

    def topic(self, relative: str) -> str:
        clean = relative.strip("/")
        if not clean:
            raise ValueError("relative topic must not be empty")
        return f"{self.namespace}/{clean}"

    def frame(self, relative: str) -> str:
        clean = relative.strip("/")
        if not clean:
            raise ValueError("relative frame must not be empty")
        return f"{self.robot_id}/{clean}"

    @property
    def odom_frame(self) -> str:
        return self.frame("odom")

    @property
    def base_frame(self) -> str:
        return self.frame("base_link")

    @property
    def camera_link_frame(self) -> str:
        return self.frame("camera_link")

    @property
    def optical_frame(self) -> str:
        return self.frame("camera_optical_frame")

    @property
    def map_frame(self) -> str:
        return self.frame("map")

    @property
    def sensor_topics(self) -> dict[str, str]:
        return {
            "color": self.topic("camera/color/image_raw"),
            "camera_info": self.topic("camera/color/camera_info"),
            "depth": self.topic("camera/depth/image_rect"),
            "semantic": self.topic("camera/semantic/image_raw"),
            "odom": self.topic("odom"),
        }

