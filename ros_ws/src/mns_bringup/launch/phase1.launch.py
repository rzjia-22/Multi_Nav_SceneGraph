from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("system_config", default_value="/workspace/config/robots/phase1.yaml"),
        DeclareLaunchArgument("navigator", default_value="coverage"),
        DeclareLaunchArgument("simulation_mode", default_value="external"),
        DeclareLaunchArgument("enable_mapping", default_value="true"),
        DeclareLaunchArgument("run_id", default_value="current"),
        DeclareLaunchArgument("scene_config", default_value="/workspace/config/simulation/forest.yaml"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([FindPackageShare("mns_bringup"), "launch", "system.launch.py"])
            ),
            launch_arguments={
                "system_config": LaunchConfiguration("system_config"),
                "navigator": LaunchConfiguration("navigator"),
                "simulation_mode": LaunchConfiguration("simulation_mode"),
                "enable_mapping": LaunchConfiguration("enable_mapping"),
                "run_id": LaunchConfiguration("run_id"),
                "scene_config": LaunchConfiguration("scene_config"),
            }.items(),
        ),
    ])
