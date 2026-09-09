"""Generate every robot instance from one validated roster."""

from __future__ import annotations

import json
from pathlib import Path
import re

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterFile
from nav2_common.launch import RewrittenYaml

from mns_core.models import SystemSpec
from mns_core.naming import RobotNames


def _mission_table(config_path: str) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)["missions"]


def _hydra_node(robot, names: RobotNames, hydra_config: str, label_space: str, log_path: Path) -> Node:
    hydra_ros_share = Path(get_package_share_directory("hydra_ros"))
    arguments = [
        "--config-utilities-file", hydra_config,
        "--config-utilities-file", label_space,
        "--config-utilities-file", str(hydra_ros_share / "config/sinks/mesh_segmenter_sinks.yaml") + "@frontend/objects",
        "--config-utilities-file", str(hydra_ros_share / "config/sinks/active_window_sinks.yaml") + "@active_window",
        "--config-utilities-yaml",
        "{" + ", ".join([
            f"robot_id: {robot.hydra_robot_id}",
            f"odom_frame: '{names.odom_frame}'",
            f"robot_frame: '{names.base_frame}'",
            f"map_frame: '{names.map_frame}'",
            f"log_path: '{log_path}'",
            "enable_lcd: false",
            # The simulator is the authoritative /clock owner.  Let Hydra
            # finalize naturally when that publisher disappears during a
            # coordinated Compose shutdown; ianvs does not treat launch's
            # initial SIGINT as an exit request on Jazzy.
            "exit_after_clock: true",
            "force_shutdown: true",
        ]) + "}",
    ]
    return Node(
        package="hydra_ros",
        executable="hydra_ros_node",
        name="hydra",
        namespace=robot.robot_id,
        output="screen",
        # A real RGB-D run may need more than launch's five-second default to
        # join Hydra worker threads and serialize DSG/mesh/timing outputs.
        sigterm_timeout="45",
        sigkill_timeout="10",
        parameters=[{"use_sim_time": True}],
        arguments=arguments,
        remappings=[
            ("hydra/input/camera/rgb/image_raw", names.sensor_topics["color"]),
            ("hydra/input/camera/rgb/camera_info", names.sensor_topics["camera_info"]),
            ("hydra/input/camera/depth_registered/image_rect", names.sensor_topics["depth"]),
            ("hydra/input/camera/semantic/image_raw", names.sensor_topics["semantic"]),
        ],
    )


def _navigation_nodes(
    robot, names: RobotNames, navigator: str, mission: dict, obstacles_json: str
) -> list:
    common = {"robot_id": robot.robot_id, "frame_id": names.odom_frame, "use_sim_time": True}
    if navigator == "coverage":
        return [Node(
            package="mns_navigation",
            executable="coverage_navigator",
            namespace=robot.robot_id,
            output="screen",
            parameters=[common, {
                "bounds": mission["bounds"],
                "lane_spacing": mission["lane_spacing"],
                "planner": mission["planner"],
                "obstacles_json": obstacles_json,
                "max_linear_speed": 0.5 if robot.kind.value == "uav" else 0.7,
                "sensor_coverage_enabled": robot.kind.value == "go2",
            }],
        )]
    if navigator == "diffusion":
        return [Node(
            package="mns_navigation",
            executable="diffusion_navigator",
            namespace=robot.robot_id,
            output="screen",
            parameters=[common, {"checkpoint": "/workspace/models/navdiffusion.ckpt"}],
        )]
    if navigator == "nav2":
        substitutions = {
            "global_frame": names.odom_frame,
            "local_frame": names.odom_frame,
            "robot_base_frame": names.base_frame,
            # Costmap plugins are nested nodes (for example
            # /go2_1/local_costmap/local_costmap), so a relative sensor topic
            # would incorrectly resolve below the costmap namespace.
            "topic": names.topic("camera/depth/points"),
            "use_sim_time": "true",
        }
        configured_params = ParameterFile(
            RewrittenYaml(
                source_file="/workspace/config/nav2/navigation.yaml",
                root_key=robot.robot_id,
                param_rewrites=substitutions,
                convert_types=True,
            ),
            allow_substs=True,
        )
        return [
            Node(
                package="depth_image_proc",
                executable="point_cloud_xyz_node",
                name="depth_to_points",
                namespace=robot.robot_id,
                output="screen",
                remappings=[
                    ("image_rect", "camera/depth/image_rect"),
                    ("points", "camera/depth/points"),
                ],
            ),
            Node(
                package="nav2_controller",
                executable="controller_server",
                namespace=robot.robot_id,
                output="screen",
                parameters=[configured_params],
                remappings=[("cmd_vel", "cmd_vel/navigation")],
            ),
            Node(
                package="nav2_planner",
                executable="planner_server",
                namespace=robot.robot_id,
                output="screen",
                parameters=[configured_params],
            ),
            Node(
                package="nav2_bt_navigator",
                executable="bt_navigator",
                namespace=robot.robot_id,
                output="screen",
                parameters=[configured_params],
            ),
            Node(
                package="nav2_behaviors",
                executable="behavior_server",
                namespace=robot.robot_id,
                output="screen",
                parameters=[configured_params],
            ),
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_navigation",
                namespace=robot.robot_id,
                parameters=[{"use_sim_time": True, "autostart": True, "node_names": ["controller_server", "planner_server", "behavior_server", "bt_navigator"]}],
            ),
            Node(
                package="mns_navigation",
                executable="nav2_goal_adapter",
                namespace=robot.robot_id,
                parameters=[common],
            ),
        ]
    raise ValueError(f"unsupported navigator: {navigator}")


def _launch(context):
    config_path = LaunchConfiguration("system_config").perform(context)
    override = LaunchConfiguration("navigator").perform(context)
    simulation_mode = LaunchConfiguration("simulation_mode").perform(context)
    mapping = LaunchConfiguration("enable_mapping").perform(context).lower() in {"1", "true", "yes"}
    run_id = LaunchConfiguration("run_id").perform(context)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError("run_id may contain only letters, digits, '.', '_' and '-'")
    run_root = Path(LaunchConfiguration("run_root").perform(context)).resolve()
    run_directory = run_root / run_id
    run_directory.mkdir(parents=True, exist_ok=True)
    spec = SystemSpec.load(config_path)
    missions = _mission_table(LaunchConfiguration("mission_config").perform(context))
    hydra_config = LaunchConfiguration("hydra_config").perform(context)
    label_space = LaunchConfiguration("label_space").perform(context)
    scene_config = LaunchConfiguration("scene_config").perform(context)
    with Path(scene_config).open("r", encoding="utf-8") as stream:
        scene_spec = yaml.safe_load(stream)
    obstacles_json = json.dumps([
        [float(value) for value in item["position"]]
        for item in scene_spec["trees"]
    ])
    actions = []
    goal_frames = []
    mission_goals = []
    for index, robot in enumerate(spec.robots):
        names = RobotNames(robot.robot_id)
        navigator = robot.navigator.value if override == "configured" else override
        mission = missions[robot.mission]
        if "goal" not in mission or len(mission["goal"]) != 3:
            raise ValueError(f"mission '{robot.mission}' requires goal: [x, y, yaw]")
        goal_frames.append(names.odom_frame)
        mission_goals.extend(float(value) for value in mission["goal"])
        if simulation_mode == "synthetic":
            actions.append(Node(
                package="mns_simulation",
                executable="synthetic_robot",
                namespace=robot.robot_id,
                output="screen",
                parameters=[{
                    "robot_id": robot.robot_id,
                    "kind": robot.kind.value,
                    "spawn": [robot.spawn.x, robot.spawn.y, robot.spawn.z, robot.spawn.yaw],
                    "publish_clock": index == 0,
                    "use_sim_time": False,
                }],
            ))
        actions.extend(_navigation_nodes(robot, names, navigator, mission, obstacles_json))
        # The current UAV sensor is nadir-facing mapping RGB-D, not a forward
        # collision sensor.  Feeding it to the planar depth safety controller
        # makes tree canopies look like frontal obstacles and traps the UAV in
        # a turn.  Go2 keeps the reactive safety layer; a future UAV flight
        # backend may provide its own correctly oriented safety source.
        if robot.kind.value == "go2":
            actions.append(Node(
                package="mns_navigation", executable="safety_monitor", namespace=robot.robot_id,
                parameters=[{"use_sim_time": spec.use_sim_time}],
            ))
        actions.append(Node(
            package="mns_motion", executable="command_arbiter", namespace=robot.robot_id,
            parameters=[{"use_sim_time": spec.use_sim_time}],
        ))
        if mapping and robot.mapping_enabled:
            log_path = run_directory / robot.robot_id / "hydra"
            log_path.mkdir(parents=True, exist_ok=True)
            actions.append(_hydra_node(robot, names, hydra_config, label_space, log_path))
            actions.append(Node(
                package="mns_mapping", executable="mapping_monitor", namespace=robot.robot_id,
                parameters=[{"robot_id": robot.robot_id, "use_sim_time": spec.use_sim_time}],
            ))
    actions.append(Node(
        package="mns_multi_robot",
        executable="mission_coordinator",
        parameters=[{
            "robot_ids": [robot.robot_id for robot in spec.robots],
            "goal_frames": goal_frames,
            "mission_goals": mission_goals,
            "use_sim_time": spec.use_sim_time,
        }],
    ))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("system_config"),
        DeclareLaunchArgument("navigator", default_value="configured"),
        DeclareLaunchArgument("simulation_mode", default_value="external"),
        DeclareLaunchArgument("enable_mapping", default_value="true"),
        DeclareLaunchArgument("mission_config", default_value="/workspace/config/missions/coverage.yaml"),
        DeclareLaunchArgument("hydra_config", default_value="/workspace/config/hydra/isaac_input.yaml"),
        DeclareLaunchArgument("label_space", default_value="/workspace/config/hydra/isaac_forest_label_space.yaml"),
        DeclareLaunchArgument("scene_config", default_value="/workspace/config/simulation/forest.yaml"),
        DeclareLaunchArgument("run_root", default_value="/workspace/runs"),
        DeclareLaunchArgument("run_id", default_value="current"),
        OpaqueFunction(function=_launch),
    ])
