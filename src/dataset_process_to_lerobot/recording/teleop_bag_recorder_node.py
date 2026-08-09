"""Service-controlled rosbag2 recorder for teleoperation episodes."""

from __future__ import annotations

import time
from pathlib import Path

import rclpy
import rosbag2_py
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.serialization import serialize_message
from rosidl_runtime_py.utilities import get_message
from std_srvs.srv import Trigger

from dataset_process_to_lerobot.mock_publishers.common import require_positive

from .recording_utils import (
    discard_session_directory,
    next_session_path,
    validate_session_prefix,
    validate_topics,
)

DEFAULT_TOPICS = (
    "/mock/cameras/camera_0/rgb/image_raw",
    "/mock/cameras/camera_1/rgb/image_raw",
    "/mock/robots/robot_0/action",
    "/mock/robots/robot_1/action",
    "/mock/robots/robot_0/joint_states",
    "/mock/robots/robot_1/joint_states",
)
RECORDING_QOS_DEPTH = 100


class TeleopBagRecorder(Node):
    def __init__(self) -> None:
        super().__init__("teleop_bag_recorder")
        self.declare_parameter("output_directory", "bags")
        self.declare_parameter("session_prefix", "teleop")
        self.declare_parameter("storage_id", "sqlite3")
        self.declare_parameter("topics", list(DEFAULT_TOPICS))
        self.declare_parameter("discard_active_on_shutdown", True)
        self.declare_parameter("auto.enabled", False)
        self.declare_parameter("auto.session_count", 1)
        self.declare_parameter("auto.duration_sec", 5.0)
        self.declare_parameter("auto.pause_sec", 1.0)
        self.declare_parameter("auto.start_delay_sec", 1.0)
        self.declare_parameter("auto.exit_when_done", False)

        self._output_directory = (
            Path(str(self.get_parameter("output_directory").value)).expanduser().resolve()
        )
        self._session_prefix = validate_session_prefix(self.get_parameter("session_prefix").value)
        self._storage_id = str(self.get_parameter("storage_id").value)
        if self._storage_id not in {"sqlite3", "mcap"}:
            raise ValueError("parameter 'storage_id' must be 'sqlite3' or 'mcap'")
        self._topics = validate_topics(self.get_parameter("topics").value)
        self._discard_active_on_shutdown = bool(
            self.get_parameter("discard_active_on_shutdown").value
        )

        self._writer: rosbag2_py.SequentialWriter | None = None
        self._current_session: Path | None = None
        self._last_saved_session: Path | None = None
        self._recording_started_monotonic: float | None = None
        self._message_counts = {topic: 0 for topic in self._topics}
        self._subscriptions_by_topic: dict[str, object] = {}
        self._session_number = 0

        self.create_service(Trigger, "~/start", self._handle_start)
        self.create_service(Trigger, "~/stop", self._handle_stop)
        self.create_service(Trigger, "~/discard", self._handle_discard)
        self.create_service(Trigger, "~/status", self._handle_status)

        self._auto_enabled = bool(self.get_parameter("auto.enabled").value)
        self._auto_session_count = int(self.get_parameter("auto.session_count").value)
        if self._auto_session_count <= 0:
            raise ValueError("parameter 'auto.session_count' must be greater than zero")
        self._auto_duration_sec = require_positive(
            self.get_parameter("auto.duration_sec").value, "auto.duration_sec"
        )
        self._auto_pause_sec = float(self.get_parameter("auto.pause_sec").value)
        self._auto_start_delay_sec = float(self.get_parameter("auto.start_delay_sec").value)
        if self._auto_pause_sec < 0.0 or self._auto_start_delay_sec < 0.0:
            raise ValueError("auto pause and start delay parameters must be non-negative")
        self._auto_exit_when_done = bool(self.get_parameter("auto.exit_when_done").value)
        self._auto_completed = 0
        self._auto_deadline = time.monotonic() + self._auto_start_delay_sec
        self._auto_timer = self.create_timer(0.1, self._advance_auto_mode)

        self.get_logger().info(
            f"Ready to record {len(self._topics)} topics under {self._output_directory}"
        )
        if self._auto_enabled:
            self.get_logger().info(
                f"Automatic mode: {self._auto_session_count} session(s), "
                f"{self._auto_duration_sec:.3f} seconds each"
            )

    @property
    def is_recording(self) -> bool:
        return self._writer is not None

    def _discover_topic_types(self) -> dict[str, str]:
        discovered = dict(self.get_topic_names_and_types())
        result: dict[str, str] = {}
        missing: list[str] = []
        ambiguous: list[str] = []
        for topic in self._topics:
            topic_types = discovered.get(topic, [])
            if not topic_types:
                missing.append(topic)
            elif len(topic_types) != 1:
                ambiguous.append(f"{topic}={topic_types}")
            else:
                result[topic] = topic_types[0]
        if missing:
            raise RuntimeError(f"topics are not currently available: {missing}")
        if ambiguous:
            raise RuntimeError(f"topics have multiple message types: {ambiguous}")
        return result

    def _create_subscriptions(self, topic_types: dict[str, str]) -> None:
        for topic, type_name in topic_types.items():
            if topic in self._subscriptions_by_topic:
                continue
            message_type = get_message(type_name)
            publisher_infos = self.get_publishers_info_by_topic(topic)
            reliability = (
                ReliabilityPolicy.RELIABLE
                if publisher_infos
                and all(
                    info.qos_profile.reliability == ReliabilityPolicy.RELIABLE
                    for info in publisher_infos
                )
                else ReliabilityPolicy.BEST_EFFORT
            )
            qos_profile = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=RECORDING_QOS_DEPTH,
                reliability=reliability,
                durability=DurabilityPolicy.VOLATILE,
            )
            subscription = self.create_subscription(
                message_type,
                topic,
                lambda message, topic_name=topic: self._record_message(topic_name, message),
                qos_profile,
            )
            self._subscriptions_by_topic[topic] = subscription

    def _record_message(self, topic: str, message: object) -> None:
        writer = self._writer
        if writer is None:
            return
        writer.write(topic, serialize_message(message), self.get_clock().now().nanoseconds)
        self._message_counts[topic] += 1

    def _start_recording(self) -> tuple[bool, str]:
        if self.is_recording:
            return False, f"already recording: {self._current_session}"
        writer: rosbag2_py.SequentialWriter | None = None
        session_path: Path | None = None
        try:
            topic_types = self._discover_topic_types()
            self._output_directory.mkdir(parents=True, exist_ok=True)
            self._session_number += 1
            session_path = next_session_path(
                self._output_directory, self._session_prefix, self._session_number
            )
            writer = rosbag2_py.SequentialWriter()
            writer.open(
                rosbag2_py.StorageOptions(uri=str(session_path), storage_id=self._storage_id),
                rosbag2_py.ConverterOptions("cdr", "cdr"),
            )
            for topic, type_name in topic_types.items():
                writer.create_topic(
                    rosbag2_py.TopicMetadata(
                        id=0,
                        name=topic,
                        type=type_name,
                        serialization_format="cdr",
                    )
                )
            self._create_subscriptions(topic_types)
        except Exception as error:
            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    pass
            if session_path is not None and session_path.exists():
                try:
                    discard_session_directory(self._output_directory, session_path)
                except Exception as cleanup_error:
                    self.get_logger().error(
                        f"Could not clean up failed recording {session_path}: {cleanup_error}"
                    )
            self.get_logger().error(f"Could not start recording: {error}")
            return False, str(error)

        self._message_counts = {topic: 0 for topic in self._topics}
        self._current_session = session_path
        self._recording_started_monotonic = time.monotonic()
        self._writer = writer
        self.get_logger().info(f"Recording started: {session_path}")
        return True, f"recording started: {session_path}"

    def _finish_recording(self, save: bool) -> tuple[bool, str]:
        if not self.is_recording or self._current_session is None:
            return False, "no recording is active"

        writer = self._writer
        session_path = self._current_session
        started = self._recording_started_monotonic
        if writer is None:
            return False, "recording state is inconsistent: writer is unavailable"
        self._writer = None
        self._current_session = None
        self._recording_started_monotonic = None
        try:
            writer.close()
            if save:
                self._last_saved_session = session_path
            else:
                discard_session_directory(self._output_directory, session_path)
        except Exception as error:
            self.get_logger().error(f"Could not finish recording {session_path}: {error}")
            return False, f"could not finish recording; data was left at {session_path}: {error}"

        elapsed = time.monotonic() - started if started is not None else 0.0
        total = sum(self._message_counts.values())
        counts = ", ".join(f"{topic}={count}" for topic, count in self._message_counts.items())
        if save:
            self.get_logger().info(
                f"Recording saved: {session_path} ({total} messages, {elapsed:.3f}s)"
            )
            return True, f"saved {session_path}; {total} messages; {counts}"
        self.get_logger().info(f"Recording discarded: {session_path}")
        return True, f"discarded {session_path}; {total} messages were removed"

    def _handle_start(
        self, _request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        response.success, response.message = self._start_recording()
        return response

    def _handle_stop(
        self, _request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        response.success, response.message = self._finish_recording(save=True)
        return response

    def _handle_discard(
        self, _request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        response.success, response.message = self._finish_recording(save=False)
        return response

    def _handle_status(
        self, _request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        response.success = True
        if self.is_recording:
            started = self._recording_started_monotonic
            elapsed = time.monotonic() - started if started is not None else 0.0
            response.message = (
                f"recording {self._current_session}; elapsed={elapsed:.3f}s; "
                f"messages={sum(self._message_counts.values())}"
            )
        else:
            response.message = f"idle; last_saved={self._last_saved_session or 'none'}"
        return response

    def _advance_auto_mode(self) -> None:
        if not self._auto_enabled or time.monotonic() < self._auto_deadline:
            return
        if self.is_recording:
            success, message = self._finish_recording(save=True)
            if not success:
                self.get_logger().error(f"Automatic recording failed to stop: {message}")
                self._auto_enabled = False
                return
            self._auto_completed += 1
            if self._auto_completed >= self._auto_session_count:
                self._auto_enabled = False
                self.get_logger().info(
                    f"Automatic recording complete: {self._auto_completed} session(s)"
                )
                if self._auto_exit_when_done:
                    rclpy.try_shutdown()
                return
            self._auto_deadline = time.monotonic() + self._auto_pause_sec
            return

        success, message = self._start_recording()
        if success:
            self._auto_deadline = time.monotonic() + self._auto_duration_sec
        else:
            self.get_logger().warning(
                f"Automatic recording start delayed: {message}; retrying in 1 second"
            )
            self._auto_deadline = time.monotonic() + 1.0

    def finish_active_session_on_shutdown(self) -> None:
        if not self.is_recording:
            return
        save = not self._discard_active_on_shutdown
        success, message = self._finish_recording(save=save)
        if not success:
            self.get_logger().error(message)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = TeleopBagRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.finish_active_session_on_shutdown()
            node.destroy_node()
            rclpy.try_shutdown()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
