"""DIABLO Research Forest NavDiffusion V0 closed-loop robotics graph."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    checkpoint = LaunchConfiguration("checkpoint")
    device = LaunchConfiguration("device")
    common = {
        "robot_id": "diablo_1",
        "frame_id": "diablo_1/odom",
        "use_sim_time": True,
    }
    return LaunchDescription([
        DeclareLaunchArgument(
            "checkpoint",
            default_value="/workspace/models/trained/navdiffusion_v0/best.pt",
        ),
        DeclareLaunchArgument("device", default_value="cuda"),
        Node(
            package="mns_navigation",
            executable="diffusion_navigator",
            name="navdiffusion_v0",
            namespace="diablo_1",
            output="screen",
            parameters=[common, {
                "model_backend": "mns_v0",
                "checkpoint": checkpoint,
                "device": device,
                "history_length": 5,
                "planning_rate_hz": 2.0,
                "goal_tolerance": 0.35,
                "control_waypoints": 8,
            }],
        ),
        Node(
            package="mns_navigation",
            executable="safety_monitor",
            namespace="diablo_1",
            output="screen",
            parameters=[{"use_sim_time": True}],
        ),
        Node(
            package="mns_motion",
            executable="command_arbiter",
            namespace="diablo_1",
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
