"""Convert mock teleoperation rosbag2 episodes to LeRobotDataset v3.0."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import shutil
import sys
import uuid
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dataset_process_to_lerobot.rosbag_utils import discover_rosbags

from .conversion_validation import (
    ConversionInputValidator,
    ConversionValidationReport,
    InputValidationConfig,
    InputValidationError,
    TimedSample,
    report_to_dict,
    tracked_events,
    write_report,
)

NANOSECONDS_PER_SECOND = 1_000_000_000
EXPECTED_TOPIC_TYPES = {
    "camera": "sensor_msgs/msg/Image",
    "action": "std_msgs/msg/Float64MultiArray",
    "state": "sensor_msgs/msg/JointState",
}
DEFAULT_CAMERAS = (
    ("camera_0", "/mock/cameras/camera_0/rgb/image_raw"),
    ("camera_1", "/mock/cameras/camera_1/rgb/image_raw"),
)
DEFAULT_ACTIONS = (
    ("robot_0", "/mock/robots/robot_0/action"),
    ("robot_1", "/mock/robots/robot_1/action"),
)
DEFAULT_STATES = (
    ("robot_0", "/mock/robots/robot_0/joint_states"),
    ("robot_1", "/mock/robots/robot_1/joint_states"),
)
VALIDATION_SEVERITY_ORDER = {"PASS": 0, "WARN": 1, "FAIL": 2}


@dataclass(frozen=True)
class TopicSet:
    cameras: tuple[tuple[str, str], ...]
    actions: tuple[tuple[str, str], ...]
    states: tuple[tuple[str, str], ...]

    @property
    def required(self) -> tuple[str, ...]:
        return tuple(
            topic for mappings in (self.cameras, self.actions, self.states) for _, topic in mappings
        )


@dataclass(frozen=True)
class BagSchema:
    image_shapes: tuple[tuple[str, tuple[int, int, int]], ...]
    joint_names: tuple[tuple[str, tuple[str, ...]], ...]


def validate_topic_set(topics: TopicSet) -> None:
    """Validate cross-category topic and robot mappings."""
    if not topics.cameras or not topics.actions or not topics.states:
        raise ValueError("at least one camera, action, and state topic is required")
    action_ids = tuple(robot_id for robot_id, _ in topics.actions)
    state_ids = tuple(robot_id for robot_id, _ in topics.states)
    if action_ids != state_ids:
        raise ValueError("action and state robot IDs must match in the same order")
    if any(not topic.startswith("/") for topic in topics.required):
        raise ValueError("all ROS topic names must be absolute")
    if len(set(topics.required)) != len(topics.required):
        raise ValueError("ROS topic names must be unique across all mappings")


def parse_assignments(
    values: Sequence[str] | None,
    defaults: Sequence[tuple[str, str]],
    option_name: str,
) -> tuple[tuple[str, str], ...]:
    """Parse repeatable ID=VALUE options while preserving their order."""
    if not values:
        return tuple(defaults)

    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for value in values:
        key, separator, assigned = value.partition("=")
        key = key.strip()
        assigned = assigned.strip()
        if not separator or not key or not assigned:
            raise ValueError(f"{option_name} must use ID=VALUE: {value!r}")
        if key in seen:
            raise ValueError(f"{option_name} contains duplicate ID {key!r}")
        seen.add(key)
        result.append((key, assigned))
    return tuple(result)


def decode_rgb_image(message: Any) -> np.ndarray:
    """Decode common ROS Image encodings into contiguous HWC RGB uint8."""
    encoding = message.encoding.lower()
    channel_counts = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}
    if encoding not in channel_counts:
        raise ValueError(
            f"unsupported image encoding {message.encoding!r}; "
            f"supported: {', '.join(channel_counts)}"
        )
    height = int(message.height)
    width = int(message.width)
    channels = channel_counts[encoding]
    step = int(message.step)
    if height <= 0 or width <= 0 or step < width * channels:
        raise ValueError(
            f"invalid image layout: width={width}, height={height}, "
            f"step={step}, encoding={encoding}"
        )
    raw = np.frombuffer(message.data, dtype=np.uint8)
    required_size = height * step
    if raw.size < required_size:
        raise ValueError(f"image data has {raw.size} bytes, expected at least {required_size}")
    pixels = raw[:required_size].reshape(height, step)[:, : width * channels]
    pixels = pixels.reshape(height, width, channels)

    rgb: np.ndarray
    if encoding == "rgb8":
        rgb = pixels
    elif encoding == "bgr8":
        rgb = pixels[:, :, ::-1]
    elif encoding == "rgba8":
        rgb = pixels[:, :, :3]
    elif encoding == "bgra8":
        rgb = pixels[:, :, (2, 1, 0)]
    else:
        rgb = np.repeat(pixels, 3, axis=2)
    return np.ascontiguousarray(rgb)


def _source_stamp_ns(message: Any) -> int | None:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    return int(stamp.sec) * NANOSECONDS_PER_SECOND + int(stamp.nanosec)


def _image_fingerprint(image: np.ndarray) -> bytes:
    """Fingerprint decoded pixels, excluding ROS header fields that change per message."""
    digest = hashlib.blake2b(digest_size=16)
    digest.update(image.tobytes())
    return digest.digest()


def resample_events(
    events: Iterable[tuple[int, str, Any]], required_topics: Sequence[str], fps: int
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Zero-order-hold a timestamp-sorted event stream at a fixed integer FPS."""
    if fps <= 0:
        raise ValueError("fps must be greater than zero")
    required = frozenset(required_topics)
    latest: dict[str, Any] = {}
    start_ns: int | None = None
    tick_index = 0
    previous_timestamp: int | None = None

    def tick_ns() -> int:
        assert start_ns is not None
        return start_ns + round(tick_index * NANOSECONDS_PER_SECOND / fps)

    for timestamp, group in itertools.groupby(events, key=lambda event: event[0]):
        if previous_timestamp is not None and timestamp < previous_timestamp:
            raise ValueError("bag messages are not timestamp sorted")
        previous_timestamp = timestamp

        if start_ns is not None:
            while tick_ns() < timestamp:
                yield tick_ns(), latest.copy()
                tick_index += 1

        for _, topic, value in group:
            if topic in required:
                latest[topic] = value

        if start_ns is None and required.issubset(latest):
            start_ns = timestamp

        if start_ns is not None:
            while tick_ns() <= timestamp:
                yield tick_ns(), latest.copy()
                tick_index += 1


def _storage_id(bag_path: Path) -> str:
    import yaml

    metadata_path = bag_path / "metadata.yaml"
    if not metadata_path.is_file():
        raise ValueError(f"rosbag2 metadata not found: {metadata_path}")
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    try:
        storage_id = metadata["rosbag2_bagfile_information"]["storage_identifier"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"storage_identifier is missing from {metadata_path}") from error
    if not isinstance(storage_id, str) or not storage_id:
        raise ValueError(f"invalid storage_identifier in {metadata_path}")
    return storage_id


def _open_reader(bag_path: Path):
    import rosbag2_py

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id=_storage_id(bag_path)),
        rosbag2_py.ConverterOptions(input_serialization_format="", output_serialization_format=""),
    )
    return reader


def _validate_topic_types(reader: Any, topics: TopicSet, bag_path: Path) -> dict[str, str]:
    available = {entry.name: entry.type for entry in reader.get_all_topics_and_types()}
    categories = (
        ("camera", topics.cameras),
        ("action", topics.actions),
        ("state", topics.states),
    )
    for category, mappings in categories:
        expected = EXPECTED_TOPIC_TYPES[category]
        for _, topic in mappings:
            actual = available.get(topic)
            if actual is None:
                raise ValueError(f"required topic {topic!r} is missing from {bag_path}")
            if actual != expected:
                raise ValueError(f"topic {topic!r} has type {actual!r}, expected {expected!r}")
    return available


def _message_types(available: dict[str, str], required_topics: Sequence[str]) -> dict[str, type]:
    from rosidl_runtime_py.utilities import get_message

    return {topic: get_message(available[topic]) for topic in required_topics}


def _deserialize(raw: bytes, message_type: type) -> Any:
    from rclpy.serialization import deserialize_message

    return deserialize_message(raw, message_type)


def inspect_bag(bag_path: Path, topics: TopicSet) -> BagSchema:
    reader = _open_reader(bag_path)
    available = _validate_topic_types(reader, topics, bag_path)
    message_types = _message_types(available, topics.required)
    required = frozenset(topics.required)
    first_messages: dict[str, Any] = {}
    while reader.has_next() and len(first_messages) < len(required):
        topic, raw, _ = reader.read_next()
        if topic in required and topic not in first_messages:
            first_messages[topic] = _deserialize(raw, message_types[topic])
    missing = required.difference(first_messages)
    if missing:
        raise ValueError(
            f"required topics have no messages in {bag_path}: {', '.join(sorted(missing))}"
        )

    image_shapes = []
    for camera_id, topic in topics.cameras:
        image_shapes.append((camera_id, tuple(decode_rgb_image(first_messages[topic]).shape)))

    joint_names = []
    for robot_id, topic in topics.states:
        message = first_messages[topic]
        names = tuple(message.name)
        if len(names) != 7 or len(message.position) != 7:
            raise ValueError(f"state topic {topic!r} must contain exactly 7 named positions")
        joint_names.append((robot_id, names))
    for robot_id, topic in topics.actions:
        if len(first_messages[topic].data) != 7:
            raise ValueError(
                f"action topic {topic!r} for {robot_id!r} must contain exactly 7 values"
            )
    return BagSchema(tuple(image_shapes), tuple(joint_names))


def _decoded_events(
    bag_path: Path, topics: TopicSet, schema: BagSchema
) -> Iterator[tuple[int, str, TimedSample]]:
    reader = _open_reader(bag_path)
    available = _validate_topic_types(reader, topics, bag_path)
    message_types = _message_types(available, topics.required)
    camera_topics = frozenset(topic for _, topic in topics.cameras)
    action_topics = frozenset(topic for _, topic in topics.actions)
    state_topics = frozenset(topic for _, topic in topics.states)
    required = camera_topics | action_topics | state_topics
    expected_shapes = dict(schema.image_shapes)
    expected_names = dict(schema.joint_names)
    camera_ids = {topic: camera_id for camera_id, topic in topics.cameras}
    state_ids = {topic: robot_id for robot_id, topic in topics.states}
    source_indices = {topic: 0 for topic in required}

    while reader.has_next():
        topic, raw, timestamp = reader.read_next()
        if topic not in required:
            continue
        message = _deserialize(raw, message_types[topic])
        if topic in camera_topics:
            value = decode_rgb_image(message)
            if value.shape != expected_shapes[camera_ids[topic]]:
                raise ValueError(f"image shape changed on topic {topic!r}: {value.shape}")
            fingerprint = _image_fingerprint(value)
        elif topic in action_topics:
            value = np.asarray(message.data, dtype=np.float32)
            if value.shape != (7,):
                raise ValueError(f"action topic {topic!r} must contain exactly 7 values")
            fingerprint = None
        else:
            if tuple(message.name) != expected_names[state_ids[topic]]:
                raise ValueError(f"joint names changed on state topic {topic!r}")
            value = np.asarray(message.position, dtype=np.float32)
            if value.shape != (7,):
                raise ValueError(f"state topic {topic!r} must contain exactly 7 positions")
            fingerprint = None
        yield (
            int(timestamp),
            topic,
            TimedSample(
                value=value,
                bag_stamp_ns=int(timestamp),
                source_stamp_ns=_source_stamp_ns(message),
                source_index=source_indices[topic],
                content_fingerprint=fingerprint,
            ),
        )
        source_indices[topic] += 1


def build_features(schema: BagSchema) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    for camera_id, shape in schema.image_shapes:
        features[f"observation.images.{camera_id}"] = {
            "dtype": "video",
            "shape": shape,
            "names": ["height", "width", "channels"],
        }
    axes = [
        f"{robot_id}.{joint_name}" for robot_id, names in schema.joint_names for joint_name in names
    ]
    features["observation.state"] = {
        "dtype": "float32",
        "shape": (len(axes),),
        "names": axes,
    }
    features["action"] = {"dtype": "float32", "shape": (len(axes),), "names": axes}
    return features


def _task_for_bag(bag_path: Path, default_task: str, task_map: dict[str, str]) -> str:
    return task_map.get(str(bag_path), task_map.get(bag_path.name, default_task))


def _should_drop_episode(severity: str, drop_on: str) -> bool:
    return VALIDATION_SEVERITY_ORDER[severity] >= VALIDATION_SEVERITY_ORDER[drop_on.upper()]


def write_conversion_manifest(result: dict[str, Any], path: Path) -> None:
    """Atomically write portable conversion provenance for downstream automation."""
    report = result["validation_report"]
    payload = {
        "version": "1.0",
        "tool": "rosbag-to-lerobot",
        "output": result["output"],
        "repo_id": result["repo_id"],
        "fps": result["fps"],
        "input_validation_policy": result.get("validation_policy", "unknown"),
        "input_validation_drop_on": result.get("validation_drop_on", "unknown"),
        "input_episodes": result["input_episodes"],
        "converted_episodes": result["episodes"],
        "rejected_episodes": len(result["rejected_episodes"]),
        "total_frames": result["total_frames"],
        "episodes": result["episode_results"],
        "input_validation": report_to_dict(report),
    }
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def convert_bags(
    bag_paths: Sequence[Path],
    output_dir: Path,
    repo_id: str,
    fps: int,
    default_task: str,
    task_map: dict[str, str],
    topics: TopicSet,
    robot_type: str,
    video_codec: str,
    validation_policy: str = "warn",
    validation_config: InputValidationConfig | None = None,
    validation_drop_on: str = "fail",
) -> dict[str, Any]:
    if fps <= 0:
        raise ValueError("fps must be greater than zero")
    if not default_task.strip():
        raise ValueError("task must not be empty")
    if validation_policy not in {"off", "warn", "fail", "drop"}:
        raise ValueError("validation_policy must be one of: off, warn, fail, drop")
    if validation_drop_on not in {"warn", "fail"}:
        raise ValueError("validation_drop_on must be one of: warn, fail")
    if validation_config is None:
        validation_config = InputValidationConfig()
    if validation_config.source_gap_min_seconds <= 0:
        raise ValueError("source_gap_min_seconds must be greater than zero")
    if validation_config.source_gap_factor <= 1:
        raise ValueError("source_gap_factor must be greater than one")
    if validation_config.source_drop_factor <= 1:
        raise ValueError("source_drop_factor must be greater than one")
    if validation_config.minimum_camera_source_frames < 2:
        raise ValueError("minimum_camera_source_frames must be at least two")
    if validation_config.duplicate_min_source_frames < 2:
        raise ValueError("duplicate_min_source_frames must be at least two")
    if validation_config.state_motion_threshold < 0 or validation_config.max_findings <= 0:
        raise ValueError(
            "state_motion_threshold must not be negative and max_findings must be positive"
        )
    validate_topic_set(topics)
    resolved_bags = tuple(path.expanduser().resolve() for path in bag_paths)
    if not resolved_bags:
        raise ValueError("at least one rosbag2 directory is required")
    for path in resolved_bags:
        if not path.is_dir():
            raise ValueError(f"rosbag2 directory does not exist: {path}")
    valid_task_keys = {path.name for path in resolved_bags} | {str(path) for path in resolved_bags}
    unknown_task_keys = set(task_map).difference(valid_task_keys)
    if unknown_task_keys:
        raise ValueError(
            f"--task-map does not match an input bag: {', '.join(sorted(unknown_task_keys))}"
        )

    schemas = [inspect_bag(path, topics) for path in resolved_bags]
    schema = schemas[0]
    for path, candidate in zip(resolved_bags[1:], schemas[1:], strict=True):
        if candidate != schema:
            raise ValueError(
                f"camera shapes or robot joint names in {path} do not match the first bag"
            )

    output = output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists; refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.partial-{uuid.uuid4().hex}"

    # These are set before importing LeRobot so all Hugging Face helpers remain offline.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from lerobot.configs import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = None
    episode_frames: list[int] = []
    rejected_episodes: list[dict[str, Any]] = []
    episode_results: list[dict[str, Any]] = []
    validation_report = ConversionValidationReport()
    try:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=build_features(schema),
            root=staging,
            robot_type=robot_type,
            use_videos=True,
            rgb_encoder=RGBEncoderConfig(vcodec=video_codec),
        )
        for episode_index, bag_path in enumerate(resolved_bags):
            task = _task_for_bag(bag_path, default_task, task_map)
            frame_count = 0
            events = _decoded_events(bag_path, topics, schema)
            validator = None
            episode_report = None
            if validation_policy != "off":
                validator = ConversionInputValidator(
                    episode_index=episode_index,
                    bag_path=bag_path,
                    camera_topics=topics.cameras,
                    state_topics=topics.states,
                    config=validation_config,
                )
                events = tracked_events(events, validator)
            for tick_ns, snapshot in resample_events(events, topics.required, fps):
                if validator is not None:
                    validator.observe_output(tick_ns, snapshot)
                frame: dict[str, Any] = {"task": task}
                for camera_id, topic in topics.cameras:
                    frame[f"observation.images.{camera_id}"] = snapshot[topic].value
                frame["observation.state"] = np.concatenate(
                    [snapshot[topic].value for _, topic in topics.states]
                ).astype(np.float32, copy=False)
                frame["action"] = np.concatenate(
                    [snapshot[topic].value for _, topic in topics.actions]
                ).astype(np.float32, copy=False)
                dataset.add_frame(frame)
                frame_count += 1
            if frame_count == 0:
                raise ValueError(f"no synchronized frames could be sampled from {bag_path}")
            if validator is not None:
                episode_report = validator.finish()
                validation_report.episodes.append(episode_report)
                if episode_report.severity != "PASS":
                    if validation_policy == "fail":
                        raise InputValidationError(validation_report)
                    if validation_policy == "drop" and _should_drop_episode(
                        episode_report.severity, validation_drop_on
                    ):
                        dataset.clear_episode_buffer(delete_images=True)
                        rejected_episodes.append(
                            {
                                "input_index": episode_index,
                                "bag_path": str(bag_path),
                                "severity": episode_report.severity,
                            }
                        )
                        episode_results.append(
                            {
                                "input_index": episode_index,
                                "input_bag": str(bag_path),
                                "dataset_episode_index": None,
                                "task": task,
                                "frames": frame_count,
                                "status": "rejected",
                                "validation_severity": episode_report.severity,
                            }
                        )
                        continue
            dataset_episode_index = len(episode_frames)
            dataset.save_episode(parallel_encoding=False)
            episode_frames.append(frame_count)
            episode_results.append(
                {
                    "input_index": episode_index,
                    "input_bag": str(bag_path),
                    "dataset_episode_index": dataset_episode_index,
                    "task": task,
                    "frames": frame_count,
                    "status": "converted",
                    "validation_severity": (
                        episode_report.severity if episode_report is not None else "NOT_RUN"
                    ),
                }
            )
        if not episode_frames:
            raise InputValidationError(validation_report)
        dataset.finalize()
        staging.rename(output)
    except Exception:
        if dataset is not None:
            try:
                dataset.clear_episode_buffer(delete_images=True)
            except Exception:
                pass
        if staging.exists():
            shutil.rmtree(staging)
        raise

    return {
        "output": str(output),
        "repo_id": repo_id,
        "fps": fps,
        "validation_policy": validation_policy,
        "validation_drop_on": validation_drop_on,
        "input_episodes": len(resolved_bags),
        "episodes": len(episode_frames),
        "rejected_episodes": rejected_episodes,
        "episode_results": episode_results,
        "episode_frames": episode_frames,
        "total_frames": sum(episode_frames),
        "validation_report": validation_report,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert one or more rosbag2 directories to a local LeRobotDataset v3.0 dataset."
        )
    )
    parser.add_argument(
        "bags", nargs="*", type=Path, help="explicit rosbag2 directories; each becomes one episode"
    )
    parser.add_argument(
        "--bag-dir",
        action="append",
        type=Path,
        default=[],
        help="directory containing rosbag2 episode directories (repeatable)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="recursively discover metadata.yaml below each --bag-dir",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="new local dataset directory"
    )
    parser.add_argument(
        "--repo-id", default="local/mock_teleop", help="dataset repository ID metadata"
    )
    parser.add_argument("--fps", type=int, default=10, help="dataset and MP4 FPS")
    parser.add_argument("--task", default="operate the mock robots", help="default episode task")
    parser.add_argument(
        "--task-map",
        action="append",
        metavar="BAG=TASK",
        help="override the task by bag basename or absolute path (repeatable)",
    )
    parser.add_argument(
        "--camera-topic", action="append", metavar="ID=TOPIC", help="camera mapping"
    )
    parser.add_argument(
        "--action-topic", action="append", metavar="ID=TOPIC", help="robot action mapping"
    )
    parser.add_argument(
        "--state-topic", action="append", metavar="ID=TOPIC", help="robot state mapping"
    )
    parser.add_argument("--robot-type", default="mock_7axis", help="robot_type dataset metadata")
    parser.add_argument("--video-codec", default="libsvtav1", help="LeRobot RGB video codec")
    parser.add_argument(
        "--input-validation",
        choices=("off", "warn", "fail", "drop"),
        default="warn",
        help=(
            "validate source camera frames; warn keeps findings, fail aborts, and drop skips "
            "episodes at or above --input-validation-drop-on before MP4 encoding"
        ),
    )
    parser.add_argument(
        "--input-validation-report",
        type=Path,
        metavar="PATH",
        help="write input validation as JSON, or Markdown when PATH ends in .md",
    )
    parser.add_argument(
        "--input-validation-drop-on",
        choices=("warn", "fail"),
        default="fail",
        help=(
            "minimum finding severity rejected by --input-validation drop; "
            "the default preserves warning-only episodes"
        ),
    )
    parser.add_argument(
        "--conversion-manifest",
        type=Path,
        metavar="PATH",
        help="write converted/rejected episode provenance as JSON",
    )
    parser.add_argument("--source-gap-min-seconds", type=float, default=1.0)
    parser.add_argument("--source-gap-factor", type=float, default=5.0)
    parser.add_argument("--source-drop-factor", type=float, default=2.5)
    parser.add_argument("--minimum-camera-source-frames", type=int, default=2)
    parser.add_argument("--duplicate-min-source-frames", type=int, default=2)
    parser.add_argument("--state-motion-threshold", type=float, default=1e-4)
    parser.add_argument("--max-input-findings", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        locations = discover_rosbags(args.bags, args.bag_dir, recursive=args.recursive)
        topics = TopicSet(
            cameras=parse_assignments(args.camera_topic, DEFAULT_CAMERAS, "--camera-topic"),
            actions=parse_assignments(args.action_topic, DEFAULT_ACTIONS, "--action-topic"),
            states=parse_assignments(args.state_topic, DEFAULT_STATES, "--state-topic"),
        )
        task_map = dict(parse_assignments(args.task_map, (), "--task-map"))
        validation_config = InputValidationConfig(
            source_gap_min_seconds=args.source_gap_min_seconds,
            source_gap_factor=args.source_gap_factor,
            source_drop_factor=args.source_drop_factor,
            minimum_camera_source_frames=args.minimum_camera_source_frames,
            duplicate_min_source_frames=args.duplicate_min_source_frames,
            state_motion_threshold=args.state_motion_threshold,
            max_findings=args.max_input_findings,
        )
        result = convert_bags(
            bag_paths=[location.path for location in locations],
            output_dir=args.output_dir,
            repo_id=args.repo_id,
            fps=args.fps,
            default_task=args.task,
            task_map=task_map,
            topics=topics,
            robot_type=args.robot_type,
            video_codec=args.video_codec,
            validation_policy=args.input_validation,
            validation_config=validation_config,
            validation_drop_on=args.input_validation_drop_on,
        )
    except Exception as error:
        if isinstance(error, InputValidationError) and args.input_validation_report:
            write_report(error.report, args.input_validation_report)
            print(
                f"wrote input validation report to {args.input_validation_report}", file=sys.stderr
            )
        print(f"conversion failed: {error}", file=sys.stderr)
        return 1

    validation_report = result["validation_report"]
    if args.input_validation_report and args.input_validation != "off":
        write_report(validation_report, args.input_validation_report)
        print(f"wrote input validation report to {args.input_validation_report}", file=sys.stderr)
    if args.conversion_manifest:
        write_conversion_manifest(result, args.conversion_manifest)
        print(f"wrote conversion manifest to {args.conversion_manifest}", file=sys.stderr)

    rejected_count = len(result["rejected_episodes"])
    rejected_summary = f"; dropped {rejected_count} input episode(s)" if rejected_count else ""
    print(
        f"created {result['output']}: {result['episodes']} episodes, "
        f"{result['total_frames']} frames at {result['fps']} FPS; "
        f"input validation {validation_report.overall_severity}{rejected_summary}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
