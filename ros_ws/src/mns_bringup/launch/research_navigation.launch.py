"""Configurable robotics graph for Research Forest navigation evaluation."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    navigator_package = LaunchConfiguration("navigator_package")
    navigator_executable = LaunchConfiguration("navigator_executable")
    model_backend = LaunchConfiguration("model_backend")
    checkpoint = LaunchConfiguration("checkpoint")
    device = LaunchConfiguration("device")
    robot_id = LaunchConfiguration("robot_id")
    common = {
        "robot_id": robot_id,
        "frame_id": ParameterValue([robot_id, "/odom"], value_type=str),
        "use_sim_time": True,
    }
    return LaunchDescription([
        DeclareLaunchArgument("navigator_package", default_value="mns_navigation"),
        DeclareLaunchArgument("navigator_executable", default_value="diffusion_navigator"),
        DeclareLaunchArgument("model_backend", default_value="mns_v0"),
        DeclareLaunchArgument(
            "checkpoint",
            default_value="/workspace/models/trained/navdiffusion_v0/best.pt",
        ),
        DeclareLaunchArgument("device", default_value="cuda"),
        DeclareLaunchArgument("robot_id", default_value="diablo_1"),
        DeclareLaunchArgument("history_length", default_value="5"),
        DeclareLaunchArgument("planning_rate_hz", default_value="2.0"),
        DeclareLaunchArgument("control_waypoints", default_value="8"),
        DeclareLaunchArgument("goal_tolerance_m", default_value="0.35"),
        Node(
            package=navigator_package,
            executable=navigator_executable,
            name="navigation_model",
            namespace=robot_id,
            output="screen",
            parameters=[common, {
                "model_backend": model_backend,
                "checkpoint": checkpoint,
                "device": device,
                "history_length": ParameterValue(LaunchConfiguration("history_length"), value_type=int),
                "planning_rate_hz": ParameterValue(LaunchConfiguration("planning_rate_hz"), value_type=float),
                "goal_tolerance": ParameterValue(LaunchConfiguration("goal_tolerance_m"), value_type=float),
                "control_waypoints": ParameterValue(LaunchConfiguration("control_waypoints"), value_type=int),
            }],
        ),
        Node(
            package="mns_navigation",
            executable="safety_monitor",
            namespace=robot_id,
            output="screen",
            parameters=[{"use_sim_time": True}],
        ),
        Node(
            package="mns_motion",
            executable="command_arbiter",
            namespace=robot_id,
            output="screen",
            parameters=[{
                "use_sim_time": True,
                "navigation_timeout": 0.30,
                "safety_timeout": 0.20,
                "max_linear_speed": 0.65,
                "max_angular_speed": 1.00,
            }],
        ),
    ])
