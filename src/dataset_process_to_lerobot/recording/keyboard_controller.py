"""Keyboard controller for repeated service-controlled teleoperation recordings."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import termios
import tty
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

KEY_ACTIONS = {
    "r": "start",
    "s": "stop",
    "d": "discard",
    "i": "status",
}


def build_recorder_launch_command(
    output_directory: Path,
    config_file: Path | None,
    session_prefix: str,
    storage_id: str,
) -> list[str]:
    command = [
        "ros2",
        "launch",
        "dataset_process_to_lerobot",
        "teleop_recorder.launch.py",
        f"output_directory:={output_directory}",
        f"session_prefix:={session_prefix}",
        f"storage_id:={storage_id}",
    ]
    if config_file is not None:
        command.append(f"config_file:={config_file}")
    return command


@contextmanager
def immediate_keys(stream: TextIO) -> Iterator[None]:
    """Use immediate single-key input on a TTY and restore terminal state afterward."""
    if not stream.isatty():
        yield
        return
    descriptor = stream.fileno()
    previous = termios.tcgetattr(descriptor)
    try:
        tty.setcbreak(descriptor)
        yield
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, previous)


def read_key(stream: TextIO) -> str:
    value = stream.read(1)
    return value.lower() if value else "q"


class RecorderController(Node):
    def __init__(self, service_prefix: str) -> None:
        super().__init__("teleop_recorder_keyboard")
        prefix = "/" + service_prefix.strip("/")
        self._service_clients = {
            action: self.create_client(Trigger, f"{prefix}/{action}")
            for action in ("start", "stop", "discard", "status")
        }

    def wait_until_ready(self, timeout_sec: float) -> bool:
        return self._service_clients["status"].wait_for_service(timeout_sec=timeout_sec)

    def call(self, action: str, timeout_sec: float = 10.0) -> tuple[bool, str]:
        future = self._service_clients[action].call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        if not future.done():
            return False, f"{action} service timed out"
        try:
            response = future.result()
        except Exception as error:
            return False, f"{action} service failed: {error}"
        if response is None:
            return False, f"{action} service returned no response"
        return bool(response.success), str(response.message)


def _stop_child(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
            process.wait()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record multiple teleoperation episodes using single-key controls."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("bags"))
    parser.add_argument("--config-file", type=Path)
    parser.add_argument("--session-prefix", default="teleop")
    parser.add_argument("--storage-id", choices=("sqlite3", "mcap"), default="sqlite3")
    parser.add_argument("--service-prefix", default="/teleop_bag_recorder")
    parser.add_argument("--startup-timeout", type=float, default=20.0)
    parser.add_argument(
        "--connect-only",
        action="store_true",
        help="control an already running recorder instead of launching one",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.startup_timeout <= 0:
        print("--startup-timeout must be greater than zero", file=sys.stderr)
        return 2

    output_directory = args.output_dir.expanduser().resolve()
    config_file = args.config_file.expanduser().resolve() if args.config_file else None
    if config_file is not None and not config_file.is_file():
        print(f"Configuration file does not exist: {config_file}", file=sys.stderr)
        return 2

    child: subprocess.Popen[bytes] | None = None
    if not args.connect_only:
        try:
            child = subprocess.Popen(
                build_recorder_launch_command(
                    output_directory,
                    config_file,
                    args.session_prefix,
                    args.storage_id,
                ),
                start_new_session=True,
            )
        except FileNotFoundError:
            print("Could not find ros2. Source the ROS 2 environment first.", file=sys.stderr)
            return 2

    rclpy.init(args=[])
    controller = RecorderController(args.service_prefix)
    try:
        if not controller.wait_until_ready(args.startup_timeout):
            print("Recorder services did not become ready before the timeout.", file=sys.stderr)
            return 2
        print("Controls: [r] start  [s] save  [d] discard  [i] status  [q] quit")
        with immediate_keys(sys.stdin):
            while True:
                key = read_key(sys.stdin)
                if key == "q":
                    success, message = controller.call("status")
                    if success and message.startswith("recording "):
                        _, discard_message = controller.call("discard")
                        print(f"\n{discard_message}")
                    break
                action = KEY_ACTIONS.get(key)
                if action is None:
                    continue
                success, message = controller.call(action)
                marker = "OK" if success else "ERROR"
                print(f"\n[{marker}] {message}")
        return 0
    except KeyboardInterrupt:
        success, message = controller.call("status")
        if success and message.startswith("recording "):
            controller.call("discard")
        return 130
    finally:
        controller.destroy_node()
        rclpy.try_shutdown()
        if child is not None:
            _stop_child(child)


if __name__ == "__main__":
    raise SystemExit(main())
