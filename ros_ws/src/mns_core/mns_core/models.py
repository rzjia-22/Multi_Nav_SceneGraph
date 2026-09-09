"""Validated configuration models shared by launch and test tooling."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import yaml

from .naming import RobotNames


class RobotKind(str, Enum):
    GO2 = "go2"
    UAV = "uav"


class NavigatorKind(str, Enum):
    COVERAGE = "coverage"
    DIFFUSION = "diffusion"
    NAV2 = "nav2"


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    z: float
    yaw: float

    @classmethod
    def from_value(cls, value: Any) -> "Pose":
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise ValueError("spawn must be [x, y, z, yaw]")
        return cls(*(float(item) for item in value))


@dataclass(frozen=True)
class RobotSpec:
    robot_id: str
    kind: RobotKind
    navigator: NavigatorKind
    spawn: Pose
    mission: str
    mapping_enabled: bool = True
    motion_backend: str = ""
    hydra_robot_id: int = 0

    def __post_init__(self) -> None:
        RobotNames(self.robot_id)
        if self.hydra_robot_id < 0:
            raise ValueError("hydra_robot_id must be non-negative")
        expected = "isaac_rl" if self.kind is RobotKind.GO2 else "kinematic"
        if self.motion_backend and self.motion_backend not in {
            "isaac_rl", "kinematic", "real_unitree"
        }:
            raise ValueError(f"unsupported motion backend: {self.motion_backend}")
        if not self.motion_backend:
            object.__setattr__(self, "motion_backend", expected)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RobotSpec":
        return cls(
            robot_id=str(data["id"]),
            kind=RobotKind(data["kind"]),
            navigator=NavigatorKind(data.get("navigator", "coverage")),
            spawn=Pose.from_value(data["spawn"]),
            mission=str(data.get("mission", "default_coverage")),
            mapping_enabled=bool(data.get("mapping_enabled", True)),
            motion_backend=str(data.get("motion_backend", "")),
            hydra_robot_id=int(data.get("hydra_robot_id", 0)),
        )


@dataclass(frozen=True)
class SystemSpec:
    use_sim_time: bool
    run_root: str
    robots: tuple[RobotSpec, ...]

    def __post_init__(self) -> None:
        ids = [robot.robot_id for robot in self.robots]
        if not ids:
            raise ValueError("at least one robot is required")
        if len(ids) != len(set(ids)):
            raise ValueError("robot ids must be unique")
        hydra_ids = [r.hydra_robot_id for r in self.robots if r.mapping_enabled]
        if len(hydra_ids) != len(set(hydra_ids)):
            raise ValueError("Hydra robot ids must be unique")

    @classmethod
    def load(cls, path: str | Path) -> "SystemSpec":
        with Path(path).open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        if not isinstance(data, Mapping):
            raise ValueError("system config must be a mapping")
        return cls(
            use_sim_time=bool(data.get("use_sim_time", True)),
            run_root=str(data.get("run_root", "runs")),
            robots=tuple(RobotSpec.from_dict(item) for item in data["robots"]),
        )

