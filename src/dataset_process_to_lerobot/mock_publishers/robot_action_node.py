"""Publish mock seven-axis action vectors for one or more robots."""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, MultiArrayDimension

from .common import (
    DEFAULT_JOINT_NAMES,
    join_topic,
    require_positive,
    sampled_sine,
    validate_axis_vector,
    validate_entity_ids,
    validate_joint_names,
)


class MockRobotActionNode(Node):
    def __init__(self) -> None:
        super().__init__("mock_robot_actions")
        self.declare_parameter("robot_ids", ["robot_0", "robot_1"])
        self.declare_parameter("fps", 20.0)
        self.declare_parameter("topic_root", "/mock/robots")
        self.declare_parameter("joint_names", list(DEFAULT_JOINT_NAMES))
        self.declare_parameter("amplitudes", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.04])
        self.declare_parameter("frequency_hz", 0.2)

        self._robot_ids = validate_entity_ids(self.get_parameter("robot_ids").value, "robot_ids")
        fps = require_positive(self.get_parameter("fps").value, "fps")
        self._frequency_hz = require_positive(
            self.get_parameter("frequency_hz").value, "frequency_hz"
        )
        self._joint_names = validate_joint_names(self.get_parameter("joint_names").value)
        self._amplitudes = validate_axis_vector(
            self.get_parameter("amplitudes").value, "amplitudes"
        )
        self._offsets = (0.0,) * len(self._amplitudes)
        topic_root = str(self.get_parameter("topic_root").value)

        self._action_publishers = []
        for robot_id in self._robot_ids:
            topic = join_topic(topic_root, robot_id, "action")
            self._action_publishers.append(self.create_publisher(Float64MultiArray, topic, 10))
            self.get_logger().info(f"Publishing actions for {robot_id} on {topic}")

        self._start_time = self.get_clock().now()
        self._timer = self.create_timer(1.0 / fps, self._publish)

    def _publish(self) -> None:
        elapsed = (self.get_clock().now() - self._start_time).nanoseconds / 1e9
        for index, publisher in enumerate(self._action_publishers):
            message = Float64MultiArray()
            message.layout.dim = [
                MultiArrayDimension(
                    label=",".join(self._joint_names),
                    size=len(self._joint_names),
                    stride=len(self._joint_names),
                )
            ]
            message.data = sampled_sine(
                elapsed,
                index,
                self._amplitudes,
                self._offsets,
                self._frequency_hz,
            )
            publisher.publish(message)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MockRobotActionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
            rclpy.try_shutdown()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
