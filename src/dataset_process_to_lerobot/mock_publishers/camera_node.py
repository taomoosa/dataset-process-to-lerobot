"""Publish mock RGB images for one or more cameras."""

from __future__ import annotations

from typing import Any

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from .camera_streams import RgbStatusStream
from .common import join_topic, require_positive, validate_entity_ids


class MockCameraNode(Node):
    def __init__(self) -> None:
        super().__init__("mock_cameras")
        self.declare_parameter("camera_ids", ["camera_0", "camera_1"])
        self.declare_parameter("fps", 5.0)
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("topic_root", "/mock/cameras")
        self.declare_parameter("frame_id_prefix", "mock")

        self._camera_ids = validate_entity_ids(self.get_parameter("camera_ids").value, "camera_ids")
        fps = require_positive(self.get_parameter("fps").value, "fps")
        width = int(self.get_parameter("width").value)
        height = int(self.get_parameter("height").value)
        if width < 64 or height < 32:
            raise ValueError("parameters 'width' and 'height' must be at least 64x32")
        topic_root = str(self.get_parameter("topic_root").value)
        self._frame_id_prefix = str(self.get_parameter("frame_id_prefix").value).strip("/")
        if not self._frame_id_prefix:
            raise ValueError("parameter 'frame_id_prefix' must not be empty")

        # Future depth streams can implement the same three attributes/method and be
        # added here without changing the publishing loop.
        self._streams = (RgbStatusStream(width=width, height=height),)
        self._stream_publishers: dict[tuple[str, str], Any] = {}
        for camera_id in self._camera_ids:
            for stream in self._streams:
                topic = join_topic(topic_root, camera_id, stream.topic_suffix)
                self._stream_publishers[(camera_id, stream.name)] = self.create_publisher(
                    Image, topic, 10
                )
                self.get_logger().info(f"Publishing {stream.name} for {camera_id} on {topic}")

        self._timer = self.create_timer(1.0 / fps, self._publish)

    def _publish(self) -> None:
        now = self.get_clock().now()
        ros_tick = now.nanoseconds
        stamp = now.to_msg()
        for camera_id in self._camera_ids:
            for stream in self._streams:
                frame_id = f"{self._frame_id_prefix}/{camera_id}_{stream.name}_optical_frame"
                message = stream.build_message(camera_id, ros_tick, stamp, frame_id)
                self._stream_publishers[(camera_id, stream.name)].publish(message)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MockCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
            rclpy.try_shutdown()
        except KeyboardInterrupt:
            # A launch process and the terminal can deliver SIGINT nearly
            # simultaneously. The process is already exiting at this point.
            pass


if __name__ == "__main__":
    main()
