import os

from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, Shutdown
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from launch import LaunchDescription


def generate_launch_description() -> LaunchDescription:
    package_name = "dataset_process_to_lerobot"
    default_config = os.path.join(
        get_package_share_directory(package_name), "config", "mock_publishers.yaml"
    )
    config_file = LaunchConfiguration("config_file")
    recorder = Node(
        package=package_name,
        executable="teleop-bag-recorder",
        name="teleop_bag_recorder",
        output="screen",
        parameters=[
            config_file,
            {
                "output_directory": LaunchConfiguration("output_directory"),
                "auto.enabled": ParameterValue(
                    LaunchConfiguration("auto_enabled"), value_type=bool
                ),
                "auto.session_count": ParameterValue(
                    LaunchConfiguration("session_count"), value_type=int
                ),
                "auto.duration_sec": ParameterValue(
                    LaunchConfiguration("duration_sec"), value_type=float
                ),
                "auto.exit_when_done": ParameterValue(
                    LaunchConfiguration("exit_when_done"), value_type=bool
                ),
            },
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("config_file", default_value=default_config),
            DeclareLaunchArgument("output_directory", default_value="bags"),
            DeclareLaunchArgument("auto_enabled", default_value="false"),
            DeclareLaunchArgument("session_count", default_value="1"),
            DeclareLaunchArgument("duration_sec", default_value="5.0"),
            DeclareLaunchArgument("exit_when_done", default_value="false"),
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
            recorder,
            RegisterEventHandler(OnProcessExit(target_action=recorder, on_exit=[Shutdown()])),
        ]
    )
