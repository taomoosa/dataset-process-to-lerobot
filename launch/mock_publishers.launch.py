import os

from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from launch import LaunchDescription


def generate_launch_description() -> LaunchDescription:
    package_name = "dataset_process_to_lerobot"
    default_config = os.path.join(
        get_package_share_directory(package_name), "config", "mock_publishers.yaml"
    )
    config_file = LaunchConfiguration("config_file")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=default_config,
                description="YAML parameter file for all mock publisher nodes",
            ),
            Node(
                package=package_name,
                executable="mock-cameras",
                name="mock_cameras",
                output="screen",
                parameters=[config_file],
            ),
            Node(
                package=package_name,
                executable="mock-robot-actions",
                name="mock_robot_actions",
                output="screen",
                parameters=[config_file],
            ),
            Node(
                package=package_name,
                executable="mock-robot-joint-positions",
                name="mock_robot_joint_positions",
                output="screen",
                parameters=[config_file],
            ),
        ]
    )
