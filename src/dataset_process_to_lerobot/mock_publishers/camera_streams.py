"""Image stream generators used by the camera mock node.

Add another stream class with ``name``, ``topic_suffix``, and ``build_message`` to
extend the camera node with depth or other image encodings.
"""

from __future__ import annotations

from dataclasses import dataclass

from builtin_interfaces.msg import Time
from sensor_msgs.msg import Image

_FONT_5X7 = {
    " ": ("00000",) * 7,
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "_": ("00000", "00000", "00000", "00000", "00000", "00000", "11111"),
    ":": ("00000", "00100", "00100", "00000", "00100", "00100", "00000"),
    "?": ("01110", "10001", "00001", "00010", "00100", "00000", "00100"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("01110", "00100", "00100", "00100", "00100", "00100", "01110"),
    "J": ("00001", "00001", "00001", "00001", "10001", "10001", "01110"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
}


def _set_rgb_pixel(
    pixels: bytearray, width: int, height: int, x: int, y: int, color: tuple[int, int, int]
) -> None:
    if x < 0 or y < 0 or x >= width or y >= height:
        return
    offset = (y * width + x) * 3
    pixels[offset : offset + 3] = bytes(color)


def _draw_text(
    pixels: bytearray,
    width: int,
    height: int,
    x: int,
    y: int,
    text: str,
    scale: int,
    color: tuple[int, int, int],
) -> None:
    cursor_x = x
    for character in text.upper():
        glyph = _FONT_5X7.get(character, _FONT_5X7["?"])
        for row, pattern in enumerate(glyph):
            for column, enabled in enumerate(pattern):
                if enabled == "0":
                    continue
                for dy in range(scale):
                    for dx in range(scale):
                        _set_rgb_pixel(
                            pixels,
                            width,
                            height,
                            cursor_x + column * scale + dx,
                            y + row * scale + dy,
                            color,
                        )
        cursor_x += 6 * scale


def render_status_rgb(width: int, height: int, camera_id: str, ros_tick: int) -> bytes:
    if width < 64 or height < 32:
        raise ValueError("camera width and height must be at least 64x32")

    seed = sum((index + 1) * ord(character) for index, character in enumerate(camera_id))
    background = (24 + seed % 48, 32 + (seed * 3) % 48, 48 + (seed * 7) % 48)
    pixels = bytearray(background * (width * height))
    lines = (f"CAMERA: {camera_id}", f"ROS TICK: {ros_tick}")
    longest = max(len(line) for line in lines)
    scale = max(1, min((width - 16) // (longest * 6), (height - 20) // 16))
    line_height = 8 * scale
    total_height = line_height * len(lines)
    origin_y = max(4, (height - total_height) // 2)

    for index, line in enumerate(lines):
        _draw_text(
            pixels,
            width,
            height,
            8,
            origin_y + index * line_height,
            line,
            scale,
            (240, 245, 255),
        )
    return bytes(pixels)


@dataclass(frozen=True)
class RgbStatusStream:
    width: int
    height: int
    name: str = "rgb"
    topic_suffix: str = "rgb/image_raw"

    def build_message(self, camera_id: str, ros_tick: int, stamp: Time, frame_id: str) -> Image:
        message = Image()
        message.header.stamp = stamp
        message.header.frame_id = frame_id
        message.height = self.height
        message.width = self.width
        message.encoding = "rgb8"
        message.is_bigendian = 0
        message.step = self.width * 3
        message.data = render_status_rgb(self.width, self.height, camera_id, ros_tick)
        return message
