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

    return LaunchDescription(
        [
            DeclareLaunchArgument("config_file", default_value=default_config),
            DeclareLaunchArgument("output_directory", default_value="bags"),
            DeclareLaunchArgument("session_prefix", default_value="teleop"),
            DeclareLaunchArgument("storage_id", default_value="sqlite3"),
            Node(
                package=package_name,
                executable="teleop-bag-recorder",
                name="teleop_bag_recorder",
                output="screen",
                parameters=[
                    LaunchConfiguration("config_file"),
                    {
                        "output_directory": LaunchConfiguration("output_directory"),
                        "session_prefix": LaunchConfiguration("session_prefix"),
                        "storage_id": LaunchConfiguration("storage_id"),
                    },
                ],
            ),
        ]
    )
