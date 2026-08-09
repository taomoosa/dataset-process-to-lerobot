from glob import glob

from setuptools import find_packages, setup

PACKAGE_NAME = "dataset_process_to_lerobot"


setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{PACKAGE_NAME}"],
        ),
        (f"share/{PACKAGE_NAME}", ["package.xml", "README.md", "LICENSE"]),
        (f"share/{PACKAGE_NAME}/config", glob("config/*.yaml")),
        (f"share/{PACKAGE_NAME}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="dataset-process-to-lerobot contributors",
    maintainer_email="maintainers@example.com",
    description="Record ROS 2 teleoperation data and convert it to LeRobotDataset V3.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "mock-cameras = dataset_process_to_lerobot.mock_publishers.camera_node:main",
            (
                "mock-robot-actions = "
                "dataset_process_to_lerobot.mock_publishers.robot_action_node:main"
            ),
            (
                "mock-robot-joint-positions = "
                "dataset_process_to_lerobot.mock_publishers.robot_joint_position_node:main"
            ),
            (
                "teleop-bag-recorder = "
                "dataset_process_to_lerobot.recording.teleop_bag_recorder_node:main"
            ),
            (
                "record-teleop-episodes = "
                "dataset_process_to_lerobot.recording.keyboard_controller:main"
            ),
            ("archive-rosbags = dataset_process_to_lerobot.recording.archive_rosbags:main"),
            "rosbag-to-lerobot = dataset_process_to_lerobot.conversion.rosbag_to_lerobot:main",
            "lerobot-video-check = dataset_process_to_lerobot.validation.lerobot_video_check:main",
            (
                "evaluate-lerobot-doctor = "
                "dataset_process_to_lerobot.validation.doctor_evaluator:main"
            ),
            ("evaluate-lerobot-video = dataset_process_to_lerobot.validation.video_evaluator:main"),
            "validate-lerobot-dataset = dataset_process_to_lerobot.validation.pipeline:main",
            (
                "remove-failed-episodes = "
                "dataset_process_to_lerobot.validation.remove_failed_episodes:main"
            ),
            "process-teleop-dataset = dataset_process_to_lerobot.workflow:main",
        ]
    },
)
