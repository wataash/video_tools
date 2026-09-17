#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Wataru Ashihara <wataash0607@gmail.com>
# SPDX-License-Identifier: Apache-2.0
epilog = r"""
video_blur_mask_gui.py -h
video_blur_mask_gui.py paint -h
video_blur_mask_gui.py paint input.mp4
video_blur_mask_gui.py paint --time 12 --sigma 30 input.mp4
video_blur_mask_gui.py encode input.mp4                    # reuse the mask saved by paint
video_blur_mask_gui.py encode --mask x.mask.png -o out.mp4 input.mp4
video_blur_mask_gui.py encode --print-command input.mp4    # print the ffmpeg command only
pytest -v --doctest-modules video_blur_mask_gui.py
"""[1:]

import argparse
import dataclasses
import datetime
import fractions
import json
import logging
import math
import os
import pathlib
import shlex
import string
import subprocess
import sys
import typing as t

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer, QVideoFrame, QVideoSink


try:
    from _colorize import get_colors  # Python 3.13+ (private API)
except ImportError:
    class _NoColors:
        RED = YELLOW = BLUE = WHITE = RESET = ""

    def get_colors(colorize=False, *, file=None):  # type: ignore[misc]
        return _NoColors


class MyFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        c = get_colors(file=sys.stderr)
        color = {
            logging.CRITICAL: c.RED,
            logging.ERROR: c.RED,
            logging.WARNING: c.YELLOW,
            logging.INFO: c.BLUE,
            logging.DEBUG: c.WHITE,
        }[record.levelno]
        func_name = "" if record.funcName == "<module>" else f" {record.funcName}()"
        fmt = f"{color}[%(levelname)1.1s %(asctime)s %(filename)s:%(lineno)d{func_name}] %(message)s{c.RESET}"
        return logging.Formatter(fmt=fmt, datefmt="%T").format(record)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger_handler = logging.StreamHandler()
logger_handler.setFormatter(MyFormatter())
logger.addHandler(logger_handler)


class ArgumentDefaultsRawTextHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter
):
    pass


# -----------------------------------------------------------------------------
# time / frame reference


class TimeOrFrame(t.NamedTuple):
    kind: str  # "time" | "frame" | "last"
    value: fractions.Fraction | int


def parse_time_seconds(value: str) -> fractions.Fraction:
    """
    >>> parse_time_seconds("83.5")
    Fraction(167, 2)
    >>> parse_time_seconds("1:23.5")
    Fraction(167, 2)
    """
    try:
        if ":" not in value:
            seconds = fractions.Fraction(value)
        else:
            fields = value.split(":")
            if len(fields) not in (2, 3):
                raise ValueError
            seconds = fractions.Fraction(0)
            for field in fields:
                if field == "" or field[0] in "+-":  # do not silently accept "1:-30" as 30s
                    raise ValueError
                seconds = seconds * 60 + fractions.Fraction(field)
    except (ValueError, ZeroDivisionError) as exc:
        raise argparse.ArgumentTypeError(f"invalid time: {value!r}") from exc
    if seconds < 0:
        raise argparse.ArgumentTypeError(f"time must be non-negative: {value!r}")
    return seconds


def parse_time_or_frame(value: str) -> TimeOrFrame:
    """
    >>> parse_time_or_frame("last")
    TimeOrFrame(kind='last', value=0)
    >>> parse_time_or_frame("f:120")
    TimeOrFrame(kind='frame', value=120)
    >>> parse_time_or_frame("00:01:23.456").kind
    'time'
    """
    if value == "last":
        return TimeOrFrame(kind="last", value=0)
    for prefix in ("frame:", "f:"):
        if value.startswith(prefix):
            try:
                frame = int(value[len(prefix):])
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"invalid frame: {value!r}") from exc
            if frame < 0:
                raise argparse.ArgumentTypeError(f"frame must be non-negative: {value!r}")
            return TimeOrFrame(kind="frame", value=frame)
    if value.startswith("f") and value[1:].isdigit():
        return TimeOrFrame(kind="frame", value=int(value[1:]))
    return TimeOrFrame(kind="time", value=parse_time_seconds(value))


def resolve_frame_index(value: TimeOrFrame, fps: float, frame_count: int) -> int:
    """
    >>> resolve_frame_index(TimeOrFrame("time", fractions.Fraction(2)), 60.0, 1000)
    120
    >>> resolve_frame_index(TimeOrFrame("last", 0), 60.0, 1000)
    999
    """
    if value.kind == "last":
        return max(0, frame_count - 1)
    if value.kind == "frame":
        frame = int(value.value)
    else:
        frame = round(float(value.value) * fps)
    return min(max(0, frame), max(0, frame_count - 1))


AUTOSAVE_INTERVAL_MS = 3000
RENDER_FPS = 25.0  # decimate playback to this fps; the source can be 60fps
APPROX_BLUR_SIGMA = 6.0  # sigma actually given to GaussianBlur in the approximate preview blur
VIEW_MODES = ("composite", "overlay", "original")
KEY_COLUMNS = ("Frame", "Time (s)", "Shape", "X", "Y", "Scale", "")  # keyframe table
KEY_COLUMN_FRAME, KEY_COLUMN_TIME, KEY_COLUMN_SHAPE, KEY_COLUMN_X, KEY_COLUMN_Y, KEY_COLUMN_SCALE, KEY_COLUMN_DELETE = range(7)


# -----------------------------------------------------------------------------
# blur / ffmpeg


def apply_blur(image: np.ndarray, sigma: float, pixelize: int, approx: bool = False) -> np.ndarray:
    """Reproduce what the ffmpeg side does with `pixelize` and `gblur`.

    With approx=True a large sigma is approximated by downscale -> small sigma -> upscale.
    It looks almost the same and is several times faster, so the playback preview uses it.
    """
    out = image
    if pixelize >= 2:
        h, w = out.shape[:2]
        small = (max(1, int(round(w / pixelize))), max(1, int(round(h / pixelize))))
        out = cv2.resize(out, small, interpolation=cv2.INTER_AREA)
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_NEAREST)
    if sigma <= 0:
        return out
    if approx and sigma > APPROX_BLUR_SIGMA:
        h, w = out.shape[:2]
        ratio = APPROX_BLUR_SIGMA / sigma
        tiny = cv2.resize(out, (max(1, int(w * ratio)), max(1, int(h * ratio))), interpolation=cv2.INTER_AREA)
        tiny = cv2.GaussianBlur(tiny, (0, 0), APPROX_BLUR_SIGMA)
        return cv2.resize(tiny, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(out, (0, 0), sigma)


def blend(base: np.ndarray, blurred: np.ndarray, mask3: np.ndarray) -> np.ndarray:
    """Compute base + (blurred - base) * mask3 cheaply, staying in uint8."""
    diff = cv2.subtract(blurred, base, dtype=cv2.CV_32F)
    cv2.multiply(diff, mask3, dst=diff)
    return cv2.add(diff, base, dtype=cv2.CV_8U)


def mask_bounding_box(mask: np.ndarray, threshold: int = 2) -> tuple[int, int, int, int] | None:
    """
    >>> m = np.zeros((10, 10), dtype=np.uint8); m[3:6, 4:8] = 255
    >>> mask_bounding_box(m)
    (4, 3, 8, 6)
    >>> mask_bounding_box(np.zeros((4, 4), dtype=np.uint8)) is None
    True
    """
    hit = mask > threshold
    rows = np.any(hit, axis=1)
    if not rows.any():
        return None
    cols = np.any(hit, axis=0)
    y0 = int(np.argmax(rows))
    y1 = len(rows) - int(np.argmax(rows[::-1]))
    x0 = int(np.argmax(cols))
    x1 = len(cols) - int(np.argmax(cols[::-1]))
    return x0, y0, x1, y1


@dataclasses.dataclass
class MaskKeyframe:
    """One mask placement, expressed in source-video pixels."""

    frame: int
    center_x: float
    center_y: float
    scale: float = 1.0
    shape_id: str = "A"


@dataclasses.dataclass
class MaskShape:
    """A separately paintable source shape used by one or more keyframes."""

    shape_id: str
    filename: str  # "" for A, whose PNG is the mask path itself
    anchor_x: float
    anchor_y: float

    @property
    def anchor(self) -> tuple[float, float]:
        return self.anchor_x, self.anchor_y


@dataclasses.dataclass
class MaskMotion:
    """Placements of separately painted masks.

    Before the first keyframe the painted A mask is used.  Between adjacent
    keyframes of the same shape centre coordinates and scale are linearly
    interpolated; after the last keyframe its placement is retained.  A pair
    of equal adjacent keys therefore forms a hold interval.  At a keyframe
    whose shape differs from the preceding one, the source shape switches at
    that frame; no misleading pseudo-morph or fade is applied.
    """

    width: int
    height: int
    fps: float
    keyframes: list[MaskKeyframe] = dataclasses.field(default_factory=list)
    shapes: dict[str, MaskShape] = dataclasses.field(default_factory=dict)  # "A" first, always present

    def __post_init__(self) -> None:
        if "A" not in self.shapes:
            self.set_shape("A", "", self.width / 2, self.height / 2)

    def set_shape(self, shape_id: str, filename: str, anchor_x: float, anchor_y: float) -> None:
        self.shapes[shape_id] = MaskShape(shape_id, filename, anchor_x, anchor_y)
        if shape_id == "A":
            self.shapes = {"A": self.shapes["A"], **self.shapes}

    def shape_ids(self) -> list[str]:
        return list(self.shapes)

    def anchor(self, shape_id: str) -> tuple[float, float]:
        return self.shapes[shape_id].anchor

    def shape_at_frame(self, frame: int) -> str:
        """Return the source shape shown at ``frame`` (the change is instant)."""
        previous = next((key for key in reversed(self.keyframes) if key.frame <= frame), None)
        return "A" if previous is None else previous.shape_id

    @property
    def uses_transform(self) -> bool:
        """Whether ffmpeg/the preview need a transformed-mask graph."""
        return len(self.shapes) > 1 or any(
            (key.center_x, key.center_y) != self.anchor(key.shape_id) or key.scale != 1.0
            for key in self.keyframes
        )

    def sort_keyframes(self) -> None:
        self.keyframes.sort(key=lambda key: key.frame)

    def keyframe_at(self, frame: int) -> MaskKeyframe | None:
        return next((key for key in self.keyframes if key.frame == frame), None)

    def transform_at_frame(self, frame: int) -> tuple[float, float, float]:
        """Return the placement at ``frame``, holding the values at both ends."""
        shape_id = self.shape_at_frame(frame)
        x0, y0 = self.anchor(shape_id)
        if not self.keyframes or frame < self.keyframes[0].frame:
            return x0, y0, 1.0
        previous = self.keyframes[0]
        if frame == previous.frame:
            return previous.center_x, previous.center_y, previous.scale
        for following in self.keyframes[1:]:
            if frame <= following.frame:
                if frame == following.frame:
                    return following.center_x, following.center_y, following.scale
                if previous.shape_id != following.shape_id:
                    return previous.center_x, previous.center_y, previous.scale
                amount = (frame - previous.frame) / (following.frame - previous.frame)
                return (
                    previous.center_x + (following.center_x - previous.center_x) * amount,
                    previous.center_y + (following.center_y - previous.center_y) * amount,
                    previous.scale + (following.scale - previous.scale) * amount,
                )
            previous = following
        return previous.center_x, previous.center_y, previous.scale

    def to_dict(self) -> dict[str, t.Any]:
        return {
            "version": 3,
            "video_size": [self.width, self.height],
            "fps": self.fps,
            "anchor": list(self.anchor("A")),
            "shapes": [
                {"id": shape.shape_id, "file": shape.filename, "anchor": [shape.anchor_x, shape.anchor_y]}
                for shape in self.shapes.values() if shape.shape_id != "A"
            ],
            "keyframes": [
                {"frame": key.frame, "center": [key.center_x, key.center_y], "scale": key.scale, "shape": key.shape_id}
                for key in self.keyframes
            ],
        }

    def save(self, path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: pathlib.Path, width: int, height: int, fps: float) -> "MaskMotion":
        motion = cls(width, height, fps)
        if not path.exists():
            return motion
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            version = data.get("version")
            source_w, source_h = map(int, data["video_size"])
            if source_w <= 0 or source_h <= 0:
                raise ValueError("video_size must be positive")
            anchor_x, anchor_y = data.get("anchor", [source_w / 2, source_h / 2])
            motion.set_shape("A", "", float(anchor_x) * width / source_w, float(anchor_y) * height / source_h)
            if version == 1:
                # Version 1 had one A -> B movement.  Two keys preserve it exactly.
                start_frame, end_frame = data["start_frame"], data["end_frame"]
                end_x, end_y = data["end_center"]
                end_scale = float(data["end_scale"])
                if start_frame is not None and end_frame is not None and int(end_frame) > int(start_frame):
                    if int(start_frame) < 0 or end_scale <= 0 or not math.isfinite(end_scale):
                        raise ValueError("version-1 motion requires non-negative frames and a positive scale")
                    base_x, base_y = motion.anchor("A")
                    motion.keyframes = [
                        MaskKeyframe(int(start_frame), base_x, base_y),
                        MaskKeyframe(
                            int(end_frame), float(end_x) * width / source_w,
                            float(end_y) * height / source_h, end_scale,
                        ),
                    ]
            elif version in (2, 3):
                if version == 3:
                    for raw_shape in data.get("shapes", []):
                        shape_id = str(raw_shape["id"])
                        filename = str(raw_shape["file"])
                        anchor_x, anchor_y = raw_shape["anchor"]
                        if not shape_id or shape_id == "A" or pathlib.Path(filename).name != filename:
                            raise ValueError("invalid additional shape")
                        motion.set_shape(
                            shape_id, filename, float(anchor_x) * width / source_w, float(anchor_y) * height / source_h,
                        )
                for raw_key in data["keyframes"]:
                    frame = int(raw_key["frame"])
                    center_x, center_y = raw_key["center"]
                    scale = float(raw_key["scale"])
                    if frame < 0 or scale <= 0 or not math.isfinite(scale):
                        raise ValueError("keyframes require non-negative frames and positive scales")
                    shape_id = str(raw_key.get("shape", "A"))
                    if shape_id not in motion.shapes:
                        raise ValueError("keyframe refers to an unknown shape")
                    motion.keyframes.append(MaskKeyframe(
                        frame, float(center_x) * width / source_w, float(center_y) * height / source_h, scale, shape_id,
                    ))
                motion.sort_keyframes()
                if any(a.frame == b.frame for a, b in zip(motion.keyframes, motion.keyframes[1:])):
                    raise ValueError("duplicate keyframe frames")
            else:
                raise ValueError(f"unsupported version {version!r}")
        except (KeyError, TypeError, ValueError, ZeroDivisionError, json.JSONDecodeError) as exc:
            logger.warning("failed to read motion settings %s: %s", path, exc)
            return cls(width, height, fps)
        return motion


def default_motion_path(mask_path: pathlib.Path) -> pathlib.Path:
    """Store the keyframes next to their PNG without changing the PNG format."""
    return mask_path.with_suffix(".motion.json")


def default_shape_path(mask_path: pathlib.Path, shape_id: str) -> pathlib.Path:
    """Return a sibling PNG path for an additional painted shape."""
    safe_id = "".join(char for char in shape_id if char.isalnum() or char in "-_")
    if not safe_id or safe_id == "A":
        raise ValueError("additional shape needs a non-A identifier")
    return mask_path.with_name(f"{mask_path.stem}.shape-{safe_id}.png")


def motion_shape_paths(motion: MaskMotion, mask_path: pathlib.Path) -> dict[str, pathlib.Path]:
    """Resolve v3 relative shape file names next to the primary A PNG."""
    return {
        shape_id: mask_path if shape_id == "A" else mask_path.parent / shape.filename
        for shape_id, shape in motion.shapes.items()
    }


def transform_mask(
    mask: np.ndarray, anchor_x: float, anchor_y: float, center_x: float, center_y: float, scale: float
) -> np.ndarray:
    """Scale a mask around ``anchor`` and place that anchor at ``center``.

    >>> mask = np.zeros((8, 10), dtype=np.uint8); mask[3:5, 4:6] = 255
    >>> moved = transform_mask(mask, 5, 4, 7, 4, 1)
    >>> mask_bounding_box(moved)
    (6, 3, 8, 5)
    """
    height, width = mask.shape
    matrix = np.array(
        [[scale, 0, center_x - scale * anchor_x], [0, scale, center_y - scale * anchor_y]],
        dtype=np.float32,
    )
    return cv2.warpAffine(
        mask, matrix, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def shape_keyframe_expression(motion: MaskMotion, shape_id: str, field: str) -> str:
    """Build placement expressions for one shape, including its active holds.

    The caller enables the matching overlay only for this shape's intervals.
    Values outside those intervals are harmless and deliberately unspecified.
    """
    base = 1.0 if field == "scale" else motion.anchor(shape_id)[0 if field == "center_x" else 1]
    expression = f"{base:.12g}"
    keys = motion.keyframes
    for index in range(len(keys) - 1, -1, -1):
        key = keys[index]
        end = keys[index + 1].frame / motion.fps if index + 1 < len(keys) else None
        if key.shape_id == shape_id:
            value = getattr(key, field)
            following = keys[index + 1] if index + 1 < len(keys) else None
            if following is not None and following.shape_id == shape_id:
                following_value = getattr(following, field)
                start = key.frame / motion.fps
                segment = f"({value:.12g})+({following_value - value:.12g})*(t-{start:.12g})/{end - start:.12g}"
            else:
                segment = f"{value:.12g}"
        else:
            segment = expression
        if end is not None:
            expression = f"if(lt(t,{end:.12g}),{segment},{expression})"
        else:
            expression = segment
    first_end = keys[0].frame / motion.fps
    if shape_id == "A":
        expression = f"if(lt(t,{first_end:.12g}),{base:.12g},{expression})"
    return expression


def shape_enable_expression(motion: MaskMotion, shape_id: str) -> str:
    """FFmpeg expression selecting the intervals where ``shape_id`` is visible."""
    keys = motion.keyframes
    terms: list[str] = []
    if shape_id == "A" and (not keys or keys[0].frame > 0):
        if keys:
            terms.append(f"lt(t,{keys[0].frame / motion.fps:.12g})")
        else:
            terms.append("1")
    for index, key in enumerate(keys):
        if key.shape_id != shape_id:
            continue
        start = key.frame / motion.fps
        if index + 1 == len(keys):
            terms.append(f"gte(t,{start:.12g})")
        else:
            end = keys[index + 1].frame / motion.fps
            terms.append(f"gte(t,{start:.12g})*lt(t,{end:.12g})")
    return "+".join(terms) or "0"


def build_filter_complex(sigma: float, pixelize: int, motion: MaskMotion | None = None) -> str:
    chain = []
    if pixelize >= 2:
        chain.append(f"pixelize=w={pixelize}:h={pixelize}")
    if sigma > 0:
        chain.append(f"gblur=sigma={sigma:g}:steps=3")
    chain.append("format=yuva420p")
    if motion is None or not motion.uses_transform:
        return (
            "[0:v]split[base][pre];"
            f"[pre]{','.join(chain)}[blurred];"
            "[1:v]format=gray[mask];"
            # The mask is a single still image; framesync repeatlast reuses it for every frame
            "[blurred][mask]alphamerge[blurred_a];"
            "[base][blurred_a]overlay=0:0[out]"
        )

    graph = [
        "[0:v]split=3[base][pre][maskbase]",
        f"[pre]{','.join(chain)}[blurred]",
        # Produce a black canvas from the video so it is always exactly the video size.
        "[maskbase]format=gray,geq=lum='0'[mask0]",
    ]
    mask_label = "mask0"
    for input_index, shape_id in enumerate(motion.shape_ids(), start=1):
        x0, y0 = motion.anchor(shape_id)
        scale = shape_keyframe_expression(motion, shape_id, "scale")
        center_x = shape_keyframe_expression(motion, shape_id, "center_x")
        center_y = shape_keyframe_expression(motion, shape_id, "center_y")
        scaled = f"scaled{input_index}"
        next_mask = f"mask{input_index}"
        enabled = shape_enable_expression(motion, shape_id)
        graph.append(
            f"[{input_index}:v]format=gray,scale=w='trunc(iw*({scale}))':h='trunc(ih*({scale}))':eval=frame[{scaled}]"
        )
        graph.append(
            f"[{mask_label}][{scaled}]overlay=x='{center_x}-({scale})*{x0:.12g}':"
            f"y='{center_y}-({scale})*{y0:.12g}':enable='{enabled}':shortest=1[{next_mask}]"
        )
        mask_label = next_mask
    graph.extend([
        f"[blurred][{mask_label}]alphamerge[blurred_a]",
        "[base][blurred_a]overlay=0:0[out]",
    ])
    return ";".join(graph)


def build_ffmpeg_command(
    input_path: pathlib.Path,
    mask_path: pathlib.Path,
    output_path: pathlib.Path,
    sigma: float,
    pixelize: int,
    crf: int,
    preset: str,
    progress: bool = False,
    motion: MaskMotion | None = None,
    shape_paths: dict[str, pathlib.Path] | None = None,
) -> list[str]:
    command = ["ffmpeg", "-hide_banner", "-y"]
    if progress:
        command += ["-nostats", "-progress", "pipe:1"]
    command += ["-i", str(input_path)]
    shape_paths = shape_paths or {"A": mask_path}
    if motion is not None and motion.uses_transform:
        for shape_id in motion.shape_ids():
            command += ["-loop", "1", "-framerate", f"{motion.fps:.12g}", "-i", str(shape_paths[shape_id])]
    else:
        command += ["-i", str(mask_path)]
    command += [
        "-filter_complex", build_filter_complex(sigma, pixelize, motion),
        "-map", "[out]",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", preset,
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-c:a", "copy",
        str(output_path),
    ]
    return command


def default_mask_path(input_path: pathlib.Path) -> pathlib.Path:
    """Keep the mask in the current directory so the input video's directory stays clean."""
    return pathlib.Path.cwd() / f"{input_path.stem}.mask.png"


def default_output_path(input_path: pathlib.Path) -> pathlib.Path:
    return pathlib.Path.cwd() / f"{input_path.stem}_blur.mp4"


# -----------------------------------------------------------------------------
# mask


class MaskPainter:
    def __init__(self, width: int, height: int, path: pathlib.Path) -> None:
        self.width = width
        self.height = height
        self.path = path
        self.mask = np.zeros((height, width), dtype=np.uint8)
        self.version = 0  # invalidates the preview cache
        self.unsaved = False
        self.undo_stack: list[np.ndarray] = []
        self.redo_stack: list[np.ndarray] = []
        self.undo_limit = 40
        self._stamp_cache: dict[tuple[int, int], np.ndarray] = {}

    def load(self) -> bool:
        if not self.path.exists():
            return False
        loaded = cv2.imread(str(self.path), cv2.IMREAD_GRAYSCALE)
        if loaded is None:
            logger.warning("failed to read mask: %s", self.path)
            return False
        if loaded.shape != (self.height, self.width):
            logger.warning(
                "mask size %sx%s != video %sx%s; resizing",
                loaded.shape[1], loaded.shape[0], self.width, self.height,
            )
            loaded = cv2.resize(loaded, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        self.mask = loaded
        self.version += 1
        return True

    def save(self, snapshot: bool = False) -> pathlib.Path | None:
        """Write to a temporary file and rename it, so a crash cannot corrupt the mask."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp.png")  # cv2.imwrite picks the format from the extension
        if not cv2.imwrite(str(tmp), self.mask):
            raise OSError(f"failed to write mask: {tmp}")
        os.replace(tmp, self.path)
        self.unsaved = False
        snapshot_path = None
        if snapshot:
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            snapshot_path = self.path.with_name(f"{self.path.stem}.{stamp}{self.path.suffix}")
            cv2.imwrite(str(snapshot_path), self.mask)
        return snapshot_path

    def begin_stroke(self) -> None:
        self.undo_stack.append(self.mask.copy())
        del self.undo_stack[: max(0, len(self.undo_stack) - self.undo_limit)]
        self.redo_stack.clear()

    def undo(self) -> bool:
        if not self.undo_stack:
            return False
        self.redo_stack.append(self.mask.copy())
        self.mask = self.undo_stack.pop()
        self.version += 1
        self.unsaved = True
        return True

    def redo(self) -> bool:
        if not self.redo_stack:
            return False
        self.undo_stack.append(self.mask.copy())
        self.mask = self.redo_stack.pop()
        self.version += 1
        self.unsaved = True
        return True

    def clear(self) -> None:
        self.begin_stroke()
        self.mask[:] = 0
        self.version += 1
        self.unsaved = True

    def coverage(self) -> float:
        return float(np.count_nonzero(self.mask > 8)) / (self.width * self.height)

    def _stamp(self, radius: int, hardness: float) -> np.ndarray:
        key = (radius, int(round(hardness * 100)))
        stamp = self._stamp_cache.get(key)
        if stamp is None:
            yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1].astype(np.float32)
            dist = np.sqrt(xx * xx + yy * yy) / max(1.0, float(radius))
            inner = min(0.99, hardness)
            alpha = np.clip((1.0 - dist) / (1.0 - inner), 0.0, 1.0)
            stamp = (alpha * alpha * (3.0 - 2.0 * alpha)).astype(np.float32)
            self._stamp_cache[key] = stamp
        return stamp

    def dab(self, cx: int, cy: int, radius: int, hardness: float, erase: bool) -> None:
        stamp = self._stamp(radius, hardness)
        x0, y0 = cx - radius, cy - radius
        x1, y1 = x0 + stamp.shape[1], y0 + stamp.shape[0]
        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(self.width, x1), min(self.height, y1)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        sub = stamp[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0]
        roi = self.mask[cy0:cy1, cx0:cx1].astype(np.float32)
        if erase:
            roi *= 1.0 - sub
        else:
            roi = np.maximum(roi, sub * 255.0)
        self.mask[cy0:cy1, cx0:cx1] = roi.astype(np.uint8)
        self.version += 1
        self.unsaved = True

    def stroke(
        self, x0: int, y0: int, x1: int, y1: int, radius: int, hardness: float, erase: bool
    ) -> None:
        distance = math.hypot(x1 - x0, y1 - y0)
        steps = max(1, int(distance / max(1.0, radius * 0.25)))
        for i in range(1, steps + 1):
            t = i / steps
            self.dab(
                int(round(x0 + (x1 - x0) * t)),
                int(round(y0 + (y1 - y0) * t)),
                radius, hardness, erase,
            )

# -----------------------------------------------------------------------------
# encode job


class EncodeJob(QtCore.QObject):
    """Run ffmpeg through QProcess and report progress as Qt signals."""

    progress = QtCore.Signal(float)
    done = QtCore.Signal(str, str)  # status ("ok" / "cancelled" / "error"), detail

    def __init__(self, command: list[str], duration: float, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.command = command
        self.duration = duration
        self.cancelled = False
        self.tail: list[str] = []
        self._buffer = ""
        self.process = QtCore.QProcess(self)
        self.process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.errorOccurred.connect(self._on_error)
        self.process.finished.connect(self._on_finished)

    def start(self) -> None:
        self.process.start(self.command[0], self.command[1:])

    def cancel(self) -> None:
        self.cancelled = True
        self.process.terminate()

    def _read_output(self) -> None:
        self._buffer += bytes(self.process.readAllStandardOutput()).decode("utf-8", "replace")
        lines = self._buffer.split("\n")
        self._buffer = lines.pop()
        for line in lines:
            line = line.rstrip("\r")
            if line.startswith("out_time_us=") and self.duration > 0:
                try:
                    seconds = int(line.split("=", 1)[1]) / 1e6
                except ValueError:  # "N/A" until the first frame is written
                    continue
                self.progress.emit(min(1.0, seconds / self.duration))
            elif line and "=" not in line:
                self.tail.append(line)
                del self.tail[: max(0, len(self.tail) - 40)]

    def _on_error(self, error: QtCore.QProcess.ProcessError) -> None:
        if error == QtCore.QProcess.ProcessError.FailedToStart:
            self.done.emit("error", f"failed to start: {shlex.join(self.command)}")

    def _on_finished(self, code: int, _status: QtCore.QProcess.ExitStatus) -> None:
        if self.cancelled:
            self.done.emit("cancelled", "")
        elif code == 0:
            self.done.emit("ok", "")
        else:
            self.done.emit("error", f"ffmpeg exited {code}\n" + "\n".join(self.tail[-12:]))


# -----------------------------------------------------------------------------
# widgets


class SeekSlider(QtWidgets.QSlider):
    """A slider that jumps straight to the clicked position instead of paging.

    It also draws a tick for every keyframe, so the timing of the mask can be
    read from the timeline without opening the table.
    """

    MARKER_COLOR = QtGui.QColor(255, 140, 0, 220)

    def __init__(self, orientation: Qt.Orientation, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(orientation, parent)
        self.markers: list[int] = []
        self.setMinimumHeight(22)

    def set_markers(self, values: t.Iterable[int]) -> None:
        markers = sorted(set(values))
        if markers != self.markers:
            self.markers = markers
            self.update()

    def marker_x(self, value: int) -> int:
        option = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(option)
        style = self.style()
        groove = style.subControlRect(QtWidgets.QStyle.ComplexControl.CC_Slider, option, QtWidgets.QStyle.SubControl.SC_SliderGroove, self)
        handle = style.subControlRect(QtWidgets.QStyle.ComplexControl.CC_Slider, option, QtWidgets.QStyle.SubControl.SC_SliderHandle, self)
        span = max(1, groove.width() - handle.width())
        offset = QtWidgets.QStyle.sliderPositionFromValue(self.minimum(), self.maximum(), value, span, option.upsideDown)
        return groove.x() + handle.width() // 2 + offset

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        if not self.markers:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self.MARKER_COLOR)
        bottom = self.height()
        for value in self.markers:
            x = self.marker_x(value)
            painter.drawPolygon(QtGui.QPolygonF([
                QtCore.QPointF(x, bottom - 7),
                QtCore.QPointF(x - 5, bottom),
                QtCore.QPointF(x + 5, bottom),
            ]))
            painter.drawRect(QtCore.QRectF(x - 0.5, 0, 1.0, bottom - 7))
        painter.end()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._seek_to(event.position().x())
            self.setSliderDown(True)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.isSliderDown():
            self._seek_to(event.position().x())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.isSliderDown():
            self.setSliderDown(False)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _seek_to(self, x: float) -> None:
        span = self.maximum() - self.minimum()
        if span <= 0:
            return
        ratio = min(1.0, max(0.0, x / max(1, self.width())))
        self.setValue(round(self.minimum() + ratio * span))


class VideoCanvas(QtWidgets.QWidget):
    """Shows the composited frame and turns pointer/tablet input into brush strokes.

    Positions are emitted in video pixel coordinates, so the window never deals with
    widget geometry. Tablet events carry pen pressure; mouse events report 1.0.
    """

    pressed = QtCore.Signal(QtCore.QPointF, bool, float)  # position, erase, pressure
    dragged = QtCore.Signal(QtCore.QPointF, float)
    released = QtCore.Signal()
    scrolled = QtCore.Signal(int, bool)  # wheel steps, shift held

    def __init__(self, video_size: QtCore.QSize, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.video_size = video_size
        self.image: QtGui.QImage | None = None
        self.brush_radius = 1.0  # in video pixels, for the cursor ring
        self.erasing = False
        self.cursor_position: QtCore.QPointF | None = None
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.BlankCursor)
        self.setMinimumSize(320, 180)
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(16, 16, 16))
        self.setPalette(palette)

    # -- geometry --------------------------------------------------------

    def fit_scale(self) -> float:
        """Device pixels per video pixel when the frame is fitted to the widget."""
        if self.video_size.isEmpty():
            return 1.0
        ratio = self.devicePixelRatioF()
        return min(
            self.width() * ratio / self.video_size.width(),
            self.height() * ratio / self.video_size.height(),
        )

    def target_rect(self) -> QtCore.QRectF:
        fit = self.fit_scale() / self.devicePixelRatioF()
        width = self.video_size.width() * fit
        height = self.video_size.height() * fit
        return QtCore.QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)

    def to_video(self, position: QtCore.QPointF) -> QtCore.QPointF:
        rect = self.target_rect()
        fit = max(1e-6, self.fit_scale() / self.devicePixelRatioF())
        return QtCore.QPointF((position.x() - rect.x()) / fit, (position.y() - rect.y()) / fit)

    def set_image(self, image: QtGui.QImage) -> None:
        self.image = image
        self.update()

    # -- painting --------------------------------------------------------

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        if self.image is not None:
            painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
            painter.drawImage(self.target_rect(), self.image)
        if self.cursor_position is not None:
            radius = self.brush_radius * self.fit_scale() / self.devicePixelRatioF()
            color = QtGui.QColor(255, 90, 90) if self.erasing else QtGui.QColor(255, 255, 255)
            painter.setPen(QtGui.QPen(color, 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(self.cursor_position, radius, radius)

    # -- input -----------------------------------------------------------

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            self.erasing = event.button() == Qt.MouseButton.RightButton
            self.cursor_position = event.position()
            self.pressed.emit(self.to_video(event.position()), self.erasing, 1.0)
            self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        self.cursor_position = event.position()
        if event.buttons() & (Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton):
            self.dragged.emit(self.to_video(event.position()), 1.0)
        self.update()

    def mouseReleaseEvent(self, _event: QtGui.QMouseEvent) -> None:
        self.erasing = False
        self.released.emit()
        self.update()

    def tabletEvent(self, event: QtGui.QTabletEvent) -> None:
        # Accepting these keeps Qt from synthesizing mouse events, so pressure is not lost.
        event.accept()
        self.cursor_position = event.position()
        eraser = event.pointerType() == QtGui.QPointingDevice.PointerType.Eraser
        pressure = max(0.05, event.pressure())
        if event.type() == QtCore.QEvent.Type.TabletPress:
            self.erasing = eraser or bool(event.buttons() & Qt.MouseButton.RightButton)
            self.pressed.emit(self.to_video(event.position()), self.erasing, pressure)
        elif event.type() == QtCore.QEvent.Type.TabletMove:
            if event.buttons():
                self.dragged.emit(self.to_video(event.position()), pressure)
        elif event.type() == QtCore.QEvent.Type.TabletRelease:
            self.erasing = False
            self.released.emit()
        self.update()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        steps = event.angleDelta().y() // 120
        if steps:
            self.scrolled.emit(steps, bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier))
        event.accept()

    def leaveEvent(self, _event: QtCore.QEvent) -> None:
        self.cursor_position = None
        self.update()


# -----------------------------------------------------------------------------
# main window


class PaintWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        video_path: pathlib.Path,
        mask_path: pathlib.Path,
        output_path: pathlib.Path,
        initial_time: TimeOrFrame,
        sigma: float,
        pixelize: int,
        crf: int,
        preset: str,
    ) -> None:
        super().__init__()
        self.video_path = video_path
        self.mask_path = mask_path
        self.output_path = output_path
        self.preset = preset

        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise SystemExit(f"failed to open video: {video_path}")
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.video_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.video_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.duration = self.frame_count / self.fps if self.frame_count and self.fps else 0.0

        self.painter = MaskPainter(self.video_w, self.video_h, mask_path)
        self.loaded_existing = self.painter.load()
        self.motion_path = default_motion_path(mask_path)
        self.motion = MaskMotion.load(self.motion_path, self.video_w, self.video_h, self.fps)
        self.shape_painters: dict[str, MaskPainter] = {"A": self.painter}
        for shape_id, path in motion_shape_paths(self.motion, mask_path).items():
            if shape_id == "A":
                continue
            painter = MaskPainter(self.video_w, self.video_h, path)
            painter.load()
            self.shape_painters[shape_id] = painter
        self.active_shape_id = "A"
        self.motion_unsaved = False
        self.motion_version = 0

        self.current_frame_index = 0
        self.current_seconds = 0.0
        self.current_bgr: np.ndarray | None = None
        self.view_mode = "composite"
        self.speed = 1.0
        self.painting = False
        self.erasing = False
        self.last_point: tuple[int, int] | None = None
        self.encode_job: EncodeJob | None = None
        self._frame_buffer: np.ndarray | None = None  # keeps the QImage data alive
        self._mask_cache: tuple[t.Any, ...] | None = None
        self._mask3: np.ndarray | None = None
        self._mask_bbox: tuple[int, int, int, int] | None = None
        self._paint_mask_cache: tuple[t.Any, ...] | None = None
        self._paint_mask3: np.ndarray | None = None
        self._coverage_cache: tuple[tuple[str, int] | int, float] = (-1, 0.0)
        self._latest_frame = None

        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        self.video_sink = QVideoSink(self)
        self.player.setVideoSink(self.video_sink)
        self.video_sink.videoFrameChanged.connect(self.on_video_frame)
        self.player.playbackStateChanged.connect(self.on_playback_state_changed)
        self.player.errorOccurred.connect(lambda error, message: self.set_status(f"playback error: {message}"))

        self.setWindowTitle(f"video-blur-mask-gui: {video_path.name}")
        self.resize(1280, 900)
        self._build_ui(sigma, pixelize, crf)
        self._build_shortcuts()

        self.playback_timer = QtCore.QTimer(self)
        self.playback_timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self.playback_timer.timeout.connect(self.play_next_frame)
        self.player.setSource(QtCore.QUrl.fromLocalFile(str(video_path.resolve())))
        self.autosave_timer = QtCore.QTimer(self)
        self.autosave_timer.timeout.connect(self.autosave)
        self.autosave_timer.start(AUTOSAVE_INTERVAL_MS)

        start_frame = resolve_frame_index(initial_time, self.fps, self.frame_count) if self.frame_count else 0
        self.goto_frame(start_frame)
        self.set_status(
            "loaded the existing mask and motion" if self.loaded_existing and self.motion.keyframes
            else "loaded the existing mask" if self.loaded_existing
            else "left drag = blur pen / right drag = eraser / wheel = pen size"
        )

    # -- construction ----------------------------------------------------

    def _build_ui(self, sigma: float, pixelize: int, crf: int) -> None:
        self.canvas = VideoCanvas(QtCore.QSize(self.video_w, self.video_h))
        self.canvas.pressed.connect(self.on_pressed)
        self.canvas.dragged.connect(self.on_dragged)
        self.canvas.released.connect(self.on_released)
        self.canvas.scrolled.connect(self.on_scrolled)

        self.timeline = SeekSlider(Qt.Orientation.Horizontal)
        self.timeline.setRange(0, max(0, self.frame_count - 1))
        self.timeline.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.timeline.valueChanged.connect(self.on_timeline_changed)

        self.play_button = QtWidgets.QPushButton("Play (Space)")
        self.play_button.clicked.connect(self.toggle_playback)
        prev_button = QtWidgets.QPushButton("◀")
        prev_button.setToolTip("previous frame (Left)")
        prev_button.clicked.connect(lambda: self.step_frame(-1))
        next_button = QtWidgets.QPushButton("▶")
        next_button.setToolTip("next frame (Right)")
        next_button.clicked.connect(lambda: self.step_frame(1))

        self.speed_box = QtWidgets.QComboBox()
        for label, value in (("0.25x", 0.25), ("0.5x", 0.5), ("1x", 1.0)):
            self.speed_box.addItem(label, value)
        self.speed_box.setCurrentIndex(2)
        self.speed_box.currentIndexChanged.connect(self.on_speed_changed)
        self.mute_box = QtWidgets.QCheckBox("Mute")
        self.mute_box.toggled.connect(self.audio.setMuted)

        self.view_button = QtWidgets.QPushButton("Composite (v)")
        self.view_button.clicked.connect(self.cycle_view)

        self.pen_slider = self._make_slider(8, max(40, self.video_w // 3), max(4, self.video_w // 22))
        self.pen_slider.valueChanged.connect(self.on_pen_changed)
        self.hardness_slider = self._make_slider(0, 95, 35)
        self.sigma_slider = self._make_slider(0, 80, int(round(sigma)))
        self.sigma_slider.valueChanged.connect(self.render_frame)
        self.pixelize_slider = self._make_slider(0, 80, pixelize)
        self.pixelize_slider.valueChanged.connect(self.render_frame)

        self.shape_box = QtWidgets.QComboBox()
        self.shape_box.currentIndexChanged.connect(self.on_shape_selected)
        self.new_shape_button = QtWidgets.QPushButton("New shape")
        self.new_shape_button.setToolTip("Create a separately paintable shape and select it")
        self.new_shape_button.clicked.connect(self.add_shape)
        self.delete_shape_button = QtWidgets.QPushButton("Delete shape")
        self.delete_shape_button.clicked.connect(self.delete_shape)

        self.motion_add_button = QtWidgets.QPushButton()  # lives in the last table row; rebuilt on refresh
        self.motion_hint_label = QtWidgets.QLabel()
        self.motion_hint_label.setWordWrap(True)
        self.motion_hint_label.setStyleSheet("color: palette(placeholder-text);")
        self.motion_key_table = QtWidgets.QTableWidget(0, len(KEY_COLUMNS))
        self.motion_key_table.setHorizontalHeaderLabels(KEY_COLUMNS)
        self.motion_key_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.motion_key_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.motion_key_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked | QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed)
        self.motion_key_table.setAlternatingRowColors(True)
        self.motion_key_table.setToolTip(
            "Click a row to seek to it; its delete button appears at the right (Delete key also works).\n"
            "X / Y / Scale change in place; double-click Frame or Time to retime the key."
        )
        self.motion_key_table.itemChanged.connect(self.on_motion_table_item_changed)
        self.motion_key_table.itemSelectionChanged.connect(self.on_motion_table_selected)
        self.motion_key_table.verticalHeader().setVisible(False)
        header = self.motion_key_table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(KEY_COLUMN_SHAPE, QtWidgets.QHeaderView.ResizeMode.Interactive)
        header.resizeSection(KEY_COLUMN_SHAPE, 120)
        header.setSectionResizeMode(KEY_COLUMN_DELETE, QtWidgets.QHeaderView.ResizeMode.Stretch)  # keeps the bin beside the values
        self.motion_key_table.setMaximumHeight(220)
        QtGui.QShortcut(
            QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Delete), self.motion_key_table,
            context=Qt.ShortcutContext.WidgetShortcut, activated=lambda: self.delete_motion_keyframe(),
        )

        undo_button = QtWidgets.QPushButton("Undo")
        undo_button.clicked.connect(self.do_undo)
        redo_button = QtWidgets.QPushButton("Redo")
        redo_button.clicked.connect(self.do_redo)
        clear_button = QtWidgets.QPushButton("Clear")
        clear_button.clicked.connect(self.clear_mask)

        save_button = QtWidgets.QPushButton("Save mask (s)")
        save_button.clicked.connect(self.save_mask)
        self.encode_button = QtWidgets.QPushButton("Encode...")
        self.encode_button.clicked.connect(self.start_encode)
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_encode)
        self.crf_box = QtWidgets.QSpinBox()
        self.crf_box.setRange(12, 32)
        self.crf_box.setValue(crf)
        self.crf_box.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)

        row1 = self._make_row(
            self.play_button, prev_button, next_button, None,
            QtWidgets.QLabel("Speed"), self.speed_box, self.mute_box, None,
            QtWidgets.QLabel("View"), self.view_button,
        )
        row1.addStretch(1)
        row2 = self._make_row(
            QtWidgets.QLabel("Pen"), self.pen_slider,
            QtWidgets.QLabel("Hardness"), self.hardness_slider,
            QtWidgets.QLabel("Blur"), self.sigma_slider,
            QtWidgets.QLabel("Pixelize"), self.pixelize_slider, None,
            undo_button, redo_button, clear_button,
        )
        row3 = self._make_row(
            save_button, self.encode_button, self.cancel_button, None,
            QtWidgets.QLabel("crf"), self.crf_box, self.progress,
        )
        self.paint_shape_label = QtWidgets.QLabel("Paint shape")
        shape_row = self._make_row(
            self.paint_shape_label, self.shape_box, self.new_shape_button, self.delete_shape_button,
        )
        shape_row.addStretch(1)
        key_row = self._make_row(QtWidgets.QLabel("Keyframes"), None, self.motion_hint_label)
        key_row.setStretch(key_row.count() - 1, 1)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.timeline)
        layout.addLayout(row1)
        layout.addLayout(row2)
        layout.addLayout(shape_row)
        layout.addLayout(key_row)
        layout.addWidget(self.motion_key_table)
        layout.addLayout(row3)
        self.setCentralWidget(central)

        self.position_label = QtWidgets.QLabel()
        self.position_label.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        self.statusBar().addPermanentWidget(self.position_label)
        mask_label = QtWidgets.QLabel(f"mask: {self.mask_path.name}")
        mask_label.setToolTip(str(self.mask_path))
        self.statusBar().addPermanentWidget(mask_label)
        self.refresh_motion_keyframes()

    def _make_slider(self, minimum: int, maximum: int, value: int) -> QtWidgets.QSlider:
        slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(min(maximum, max(minimum, value)))
        slider.setFixedWidth(120)
        slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # so the arrow keys stay with the window
        return slider

    def _make_row(self, *widgets: QtWidgets.QWidget | None) -> QtWidgets.QHBoxLayout:
        row = QtWidgets.QHBoxLayout()
        for widget in widgets:
            if widget is None:
                row.addSpacing(16)
            else:
                row.addWidget(widget)
        return row

    def _build_shortcuts(self) -> None:
        for keys, handler in (
            ("Space", self.toggle_playback),
            ("Left", lambda: self.step_frame(-1)),
            ("Right", lambda: self.step_frame(1)),
            ("Shift+Left", lambda: self.step_frame(-10)),
            ("Shift+Right", lambda: self.step_frame(10)),
            ("Home", lambda: self.goto_frame(0)),
            ("End", lambda: self.goto_frame(max(0, self.frame_count - 1))),
            ("Ctrl+Z", self.do_undo),
            ("Ctrl+Y", self.do_redo),
            ("Ctrl+Shift+Z", self.do_redo),
            ("V", self.cycle_view),
            ("S", self.save_mask),
            ("[", lambda: self.scale_pen(1 / 1.15)),
            ("]", lambda: self.scale_pen(1.15)),
            ("Q", self.close),
        ):
            QtGui.QShortcut(QtGui.QKeySequence(keys), self, activated=handler)

    # -- frame access ----------------------------------------------------

    def goto_frame(self, frame_index: int) -> None:
        if self.frame_count:
            frame_index = max(0, min(frame_index, self.frame_count - 1))
        else:
            frame_index = max(0, frame_index)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.cap.read()
        if not ok:
            self.set_status(f"failed to read frame {frame_index}")
            return
        self.current_frame_index = frame_index
        timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        self.current_seconds = timestamp if timestamp > 0 else frame_index / self.fps
        self._latest_frame = None
        self.player.setPosition(round(self.current_seconds * 1000))
        self.current_bgr = frame
        self.render_frame()
        self.sync_timeline()
        self.select_keyframe_row(frame_index)

    def sync_timeline(self) -> None:
        self.timeline.blockSignals(True)
        self.timeline.setValue(self.current_frame_index)
        self.timeline.blockSignals(False)
        self.update_labels()

    def step_frame(self, delta: int) -> None:
        self.stop_playback()
        self.goto_frame(self.current_frame_index + delta)

    # -- playback --------------------------------------------------------

    def stop_playback(self) -> None:
        self.player.pause()  # on_playback_state_changed stops the preview timer

    def toggle_playback(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.stop_playback()
            return
        if self.player.mediaStatus() == QMediaPlayer.MediaStatus.EndOfMedia or (self.frame_count and self.current_frame_index >= self.frame_count - 1):
            self.goto_frame(0)
        self.player.play()

    def on_speed_changed(self) -> None:
        self.speed = float(self.speed_box.currentData())
        self.player.setPlaybackRate(self.speed)

    def on_playback_state_changed(self, state) -> None:
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.playback_timer.start(round(1000 / RENDER_FPS))
            self.play_button.setText("Pause (Space)")
        else:
            self.playback_timer.stop()
            self.play_next_frame()
            self.play_button.setText("Play (Space)")

    def on_video_frame(self, frame) -> None:
        if frame.isValid() and self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            # Decode/audio keep their own clock. Never queue expensive previews.
            self._latest_frame = QVideoFrame(frame)

    def play_next_frame(self) -> None:
        frame = self._latest_frame
        if frame is None:
            return
        self._latest_frame = None
        image = frame.toImage()  # Qt applies the video's display rotation.
        if image.isNull():
            return
        image = image.scaled(*self.display_size(), Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
        image = image.convertToFormat(QtGui.QImage.Format.Format_RGB888)
        data = np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), image.bytesPerLine())
        rgb = data[:, :image.width() * 3].reshape(image.height(), image.width(), 3)
        self.current_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self.current_seconds = max(0.0, frame.startTime() / 1_000_000)
        self.current_frame_index = min(max(0, self.frame_count - 1), round(self.current_seconds * self.fps))
        self.render_frame()
        self.sync_timeline()

    def on_timeline_changed(self, value: int) -> None:
        self.stop_playback()
        if value != self.current_frame_index:
            self.goto_frame(value)

    # -- rendering -------------------------------------------------------

    def scaled_mask(self, width: int, height: int) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
        """Return the current-frame mask at display size and its bounding box."""
        shape_id = self.motion.shape_at_frame(self.current_frame_index)
        painter = self.shape_painters[shape_id]
        center_x, center_y, scale = self.motion.transform_at_frame(self.current_frame_index)
        key = (shape_id, painter.version, self.motion_version, width, height, center_x, center_y, scale)
        if self._mask_cache != key or self._mask3 is None:
            small = cv2.resize(painter.mask, (width, height), interpolation=cv2.INTER_LINEAR)
            if self.motion.uses_transform:
                anchor_x, anchor_y = self.motion.anchor(shape_id)
                small = transform_mask(
                    small,
                    anchor_x * width / self.video_w, anchor_y * height / self.video_h,
                    center_x * width / self.video_w, center_y * height / self.video_h, scale,
                )
            normalized = small.astype(np.float32) / 255.0
            self._mask3 = cv2.merge([normalized, normalized, normalized])
            self._mask_bbox = mask_bounding_box(small)
            self._mask_cache = key
        return self._mask3, self._mask_bbox

    def paint_shape_overlay(self, width: int, height: int) -> np.ndarray | None:
        """Return the active source shape when its output placement differs.

        Painting always changes source-mask pixels.  Show that source in cyan
        when the current keyframe uses another shape, or moves/scales this one,
        so a brush stroke cannot appear to do nothing.
        """
        output_shape_id = self.motion.shape_at_frame(self.current_frame_index)
        center_x, center_y, scale = self.motion.transform_at_frame(self.current_frame_index)
        anchor_x, anchor_y = self.motion.anchor(self.active_shape_id)
        if (
            output_shape_id == self.active_shape_id
            and (center_x, center_y, scale) == (anchor_x, anchor_y, 1.0)
        ):
            return None
        painter = self.shape_painters[self.active_shape_id]
        key = (self.active_shape_id, painter.version, width, height)
        if self._paint_mask_cache != key or self._paint_mask3 is None:
            small = cv2.resize(painter.mask, (width, height), interpolation=cv2.INTER_LINEAR)
            normalized = small.astype(np.float32) / 255.0 * 0.55
            self._paint_mask3 = cv2.merge([normalized, normalized, normalized])
            self._paint_mask_cache = key
        return self._paint_mask3

    def display_size(self) -> tuple[int, int]:
        """Device resolution, never upscaled; Qt stretches the result if the widget is larger."""
        scale = min(1.0, self.canvas.fit_scale())
        return max(1, round(self.video_w * scale)), max(1, round(self.video_h * scale))

    def render_frame(self) -> None:
        if self.current_bgr is None:
            return
        width, height = self.display_size()
        scale = width / self.video_w
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        shown = cv2.resize(self.current_bgr, (width, height), interpolation=interpolation)

        mask3, bbox = self.scaled_mask(width, height)
        if self.view_mode != "original" and bbox is not None:
            sigma = self.sigma_slider.value() * scale
            pixelize = int(round(self.pixelize_slider.value() * scale))
            # Blur only the rectangle holding the mask, widened by 3 sigma so the blur is not clipped
            pad = int(3 * sigma) + pixelize + 2
            x0 = max(0, bbox[0] - pad)
            y0 = max(0, bbox[1] - pad)
            x1 = min(width, bbox[2] + pad)
            y1 = min(height, bbox[3] + pad)
            base = shown[y0:y1, x0:x1]
            region = mask3[y0:y1, x0:x1]
            merged = blend(base, apply_blur(base, sigma, pixelize, approx=True), region)
            if self.view_mode == "overlay":
                tint = np.zeros_like(merged)
                tint[:, :, 2] = 255
                merged = blend(merged, tint, region * 0.35)
            shown[y0:y1, x0:x1] = merged

        # The cyan source layer is intentionally independent of keyframe
        # placement: pointer coordinates map directly to its mask pixels.
        paint_mask = self.paint_shape_overlay(width, height)
        if paint_mask is not None:
            cyan = np.empty_like(shown)
            cyan[:, :] = (255, 255, 0)
            shown = blend(shown, cyan, paint_mask)
        self.paint_shape_label.setText(
            "Paint shape (cyan source)" if paint_mask is not None else "Paint shape"
        )

        self._frame_buffer = np.ascontiguousarray(shown)  # QImage does not copy the data
        image = QtGui.QImage(
            self._frame_buffer.data, width, height,
            self._frame_buffer.strides[0], QtGui.QImage.Format.Format_BGR888,
        )
        self.canvas.brush_radius = self.pen_slider.value()
        self.canvas.set_image(image)

    def update_labels(self) -> None:
        seconds = self.current_seconds
        coverage_key = (self.active_shape_id, self.painter.version)
        if self._coverage_cache[0] != coverage_key:
            self._coverage_cache = (coverage_key, self.painter.coverage())
        self.position_label.setText(
            f"{seconds:7.2f}s / {self.duration:.2f}s  frame {self.current_frame_index}/{max(0, self.frame_count - 1)}"
            f"   pen r={self.pen_slider.value()}  shape {self.active_shape_id} {self._coverage_cache[1] * 100:4.1f}%"
            f"{'  ' + str(len(self.motion.keyframes)) + ' keyframes' if self.motion.keyframes else ''}"
            f"{'  *unsaved' if any(p.unsaved for p in self.shape_painters.values()) or self.motion_unsaved else ''}"
        )
        on_key = self.motion.keyframe_at(self.current_frame_index) is not None
        add_text = (
            f"＋  Add key at {seconds:.2f}s (frame {self.current_frame_index}) using shape {self.active_shape_id}"
            if not on_key else f"frame {self.current_frame_index} already has a key; seek elsewhere to add another"
        )
        if self.motion_add_button.text() != add_text:
            self.motion_add_button.setText(add_text)
            self.motion_add_button.setEnabled(not on_key)

    def set_status(self, text: str) -> None:
        self.statusBar().showMessage(text)

    def cycle_view(self) -> None:
        self.view_mode = VIEW_MODES[(VIEW_MODES.index(self.view_mode) + 1) % len(VIEW_MODES)]
        self.view_button.setText(
            {"composite": "Composite (v)", "overlay": "Mask in red (v)", "original": "Original (v)"}[self.view_mode]
        )
        self.render_frame()

    # -- painting --------------------------------------------------------

    def pen_radius(self, pressure: float) -> int:
        # A stylus modulates the width; a mouse always reports 1.0 and gets the full radius.
        return max(2, int(round(self.pen_slider.value() * (0.35 + 0.65 * pressure))))

    def on_pressed(self, position: QtCore.QPointF, erase: bool, pressure: float) -> None:
        self.painting = True
        self.erasing = erase
        self.painter.begin_stroke()
        point = (int(round(position.x())), int(round(position.y())))
        self.painter.dab(*point, self.pen_radius(pressure), self.hardness_slider.value() / 100.0, erase)
        self.last_point = point
        self.render_frame()
        self.update_labels()

    def on_dragged(self, position: QtCore.QPointF, pressure: float) -> None:
        if not self.painting or self.last_point is None:
            return
        point = (int(round(position.x())), int(round(position.y())))
        self.painter.stroke(
            *self.last_point, *point,
            self.pen_radius(pressure), self.hardness_slider.value() / 100.0, self.erasing,
        )
        self.last_point = point
        self.render_frame()
        self.update_labels()

    def on_released(self) -> None:
        self.painting = False
        self.erasing = False
        self.last_point = None

    def on_scrolled(self, steps: int, shift: bool) -> None:
        if shift:
            self.hardness_slider.setValue(self.hardness_slider.value() + 5 * steps)
        else:
            self.scale_pen(1.15**steps)

    def scale_pen(self, factor: float) -> None:
        self.pen_slider.setValue(max(1, int(round(self.pen_slider.value() * factor))))

    def on_pen_changed(self) -> None:
        self.canvas.brush_radius = self.pen_slider.value()
        self.canvas.update()
        self.update_labels()

    def do_undo(self) -> None:
        if self.painter.undo():
            self.render_frame()
            self.update_labels()

    def do_redo(self) -> None:
        if self.painter.redo():
            self.render_frame()
            self.update_labels()

    def clear_mask(self) -> None:
        answer = QtWidgets.QMessageBox.question(self, "Clear", "Clear the whole mask? (undoable)")
        if answer == QtWidgets.QMessageBox.StandardButton.Yes:
            self.painter.clear()
            self.render_frame()
            self.update_labels()

    # -- mask keyframes --------------------------------------------------

    def refresh_shapes(self, selected_shape_id: str | None = None) -> None:
        selected_shape_id = selected_shape_id or self.active_shape_id
        self.shape_box.blockSignals(True)
        self.shape_box.clear()
        for shape_id in self.motion.shape_ids():
            self.shape_box.addItem(f"Shape {shape_id}", shape_id)
        index = self.shape_box.findData(selected_shape_id)
        self.shape_box.setCurrentIndex(max(0, index))
        self.shape_box.blockSignals(False)
        self.delete_shape_button.setEnabled(selected_shape_id != "A")

    def on_shape_selected(self) -> None:
        shape_id = self.shape_box.currentData()
        if shape_id is None or shape_id == self.active_shape_id:
            return
        self.active_shape_id = str(shape_id)
        self.painter = self.shape_painters[self.active_shape_id]
        self._coverage_cache = (-1, 0.0)
        self.render_frame()
        self.update_labels()
        self.set_status(f"editing shape {self.active_shape_id}; keyframe rows choose which shape is used")

    def add_shape(self) -> None:
        shape_id = next((letter for letter in string.ascii_uppercase if letter not in self.shape_painters), None)
        if shape_id is None:
            self.set_status("no free shape letter left")
            return
        filename = default_shape_path(self.mask_path, shape_id).name
        self.motion.set_shape(shape_id, filename, self.video_w / 2, self.video_h / 2)
        self.shape_painters[shape_id] = MaskPainter(self.video_w, self.video_h, self.mask_path.parent / filename)
        self._touch_motion()
        self.refresh_shapes(shape_id)
        self.on_shape_selected()
        self.set_status(
            f"created shape {shape_id}; draw the cyan source mask, then add a keyframe to use it"
        )

    def delete_shape(self) -> None:
        shape_id = self.active_shape_id
        if shape_id == "A":
            return
        if any(key.shape_id == shape_id for key in self.motion.keyframes):
            self.set_status(f"shape {shape_id} is assigned to a keyframe; choose another shape in that row first")
            return
        del self.shape_painters[shape_id]
        del self.motion.shapes[shape_id]
        self.active_shape_id = "A"
        self.painter = self.shape_painters["A"]
        self._touch_motion()
        self.refresh_shapes("A")
        self.render_frame()
        self.update_labels()
        self.set_status(f"removed unused shape {shape_id}")

    def selected_motion_keyframe(self) -> MaskKeyframe | None:
        rows = self.motion_key_table.selectionModel().selectedRows()
        if not rows:
            return None
        item = self.motion_key_table.item(rows[0].row(), KEY_COLUMN_FRAME)
        return None if item is None else self.motion.keyframe_at(int(item.data(Qt.ItemDataRole.UserRole)))

    def refresh_motion_keyframes(self, selected_frame: int | None = None) -> None:
        """Rebuild the table while preserving the selected source frame."""
        if selected_frame is None:
            selected = self.selected_motion_keyframe()
            selected_frame = None if selected is None else selected.frame
        self.refresh_shapes()
        table = self.motion_key_table
        table.blockSignals(True)
        table.clearSpans()
        for row in range(table.rowCount()):
            table.removeCellWidget(row, KEY_COLUMN_FRAME)  # the previous "+" button
        table.setRowCount(len(self.motion.keyframes) + 1)  # the extra row holds the "+" button
        for row, key in enumerate(self.motion.keyframes):
            for column, value in ((KEY_COLUMN_FRAME, str(key.frame)), (KEY_COLUMN_TIME, f"{key.frame / self.fps:.6g}")):
                item = QtWidgets.QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, key.frame)
                item.setToolTip("Double-click to retime this key")
                table.setItem(row, column, item)
            chooser = QtWidgets.QComboBox()
            for shape_id in self.motion.shape_ids():
                chooser.addItem(f"Shape {shape_id}", shape_id)
            chooser.setCurrentIndex(chooser.findData(key.shape_id))
            chooser.setToolTip("Shape shown from this key on")
            chooser.currentIndexChanged.connect(lambda _index, frame=key.frame, box=chooser: self.on_motion_table_shape_changed(frame, box))
            table.setCellWidget(row, KEY_COLUMN_SHAPE, self._as_key_cell(chooser, key.frame))
            for column, field, value, limit in (
                (KEY_COLUMN_X, "center_x", key.center_x, self.video_w),
                (KEY_COLUMN_Y, "center_y", key.center_y, self.video_h),
                (KEY_COLUMN_SCALE, "scale", key.scale, None),
            ):
                box = QtWidgets.QDoubleSpinBox()
                if limit is None:
                    box.setRange(0.05, 8.0)
                    box.setSingleStep(0.05)
                    box.setDecimals(2)
                    box.setSuffix(" x")
                    box.setToolTip("Size relative to the painted shape")
                else:
                    box.setRange(-limit, limit * 2)
                    box.setSingleStep(1.0)
                    box.setDecimals(1)
                    box.setSuffix(" px")
                    box.setToolTip("Centre of the shape at this key")
                box.setValue(value)
                box.valueChanged.connect(lambda number, frame=key.frame, field=field: self.on_motion_table_value_changed(frame, field, number))
                table.setCellWidget(row, column, self._as_key_cell(box, key.frame))
            if key.frame == selected_frame:
                table.selectRow(row)
        add_row = len(self.motion.keyframes)
        placeholder = QtWidgets.QTableWidgetItem()
        placeholder.setFlags(Qt.ItemFlag.NoItemFlags)  # the "+" row is not a key: never selectable
        table.setItem(add_row, KEY_COLUMN_FRAME, placeholder)
        table.setSpan(add_row, KEY_COLUMN_FRAME, 1, len(KEY_COLUMNS))
        self.motion_add_button = QtWidgets.QPushButton()
        self.motion_add_button.setFlat(True)
        self.motion_add_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.motion_add_button.setStyleSheet("text-align: left; padding: 2px 8px;")
        self.motion_add_button.clicked.connect(self.add_motion_keyframe)
        table.setCellWidget(add_row, KEY_COLUMN_FRAME, self.motion_add_button)
        table.blockSignals(False)
        self.update_motion_table_delete_buttons(selected_frame)
        self.timeline.set_markers(key.frame for key in self.motion.keyframes)
        self.motion_hint_label.setText(
            "No keys: the mask stays where it is painted. Seek, then add a key to move, resize, or switch shapes from that time."
            if not self.motion.keyframes else
            "Click a row to seek. X / Y / Scale change in place; double-click Frame or Time to retime. Delete removes the selected key."
        )
        self.update_labels()
        self.render_frame()

    def _as_key_cell(self, widget: QtWidgets.QWidget, frame: int) -> QtWidgets.QWidget:
        """Cell widgets swallow clicks, so select their row on focus and ignore stray wheel input."""
        widget.setProperty("keyframe", frame)
        widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        widget.installEventFilter(self)
        return widget

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        frame = obj.property("keyframe") if isinstance(obj, QtWidgets.QWidget) else None
        if frame is not None:
            if event.type() == QtCore.QEvent.Type.Wheel and not obj.hasFocus():
                return True
            if event.type() == QtCore.QEvent.Type.FocusIn:
                self.select_keyframe_row(int(frame))
        return super().eventFilter(obj, event)

    def select_keyframe_row(self, frame: int) -> None:
        selected = self.selected_motion_keyframe()
        if selected is not None and selected.frame == frame:
            return
        for row, key in enumerate(self.motion.keyframes):
            if key.frame == frame:
                self.motion_key_table.selectRow(row)
                return

    def update_motion_table_delete_buttons(self, selected_frame: int | None = None) -> None:
        """Show the destructive action only beside the selected keyframe row."""
        if selected_frame is None:
            key = self.selected_motion_keyframe()
            selected_frame = None if key is None else key.frame
        for row, key in enumerate(self.motion.keyframes):
            self.motion_key_table.removeCellWidget(row, KEY_COLUMN_DELETE)
            if key.frame != selected_frame:
                continue
            button = QtWidgets.QToolButton()
            button.setAutoRaise(True)
            button.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_TrashIcon))
            button.setIconSize(QtCore.QSize(16, 16))
            button.setToolTip(f"Delete keyframe at frame {key.frame}")
            button.setAccessibleName(f"Delete keyframe at frame {key.frame}")
            button.clicked.connect(lambda _checked=False, target=key: self.delete_motion_keyframe(target))
            cell = QtWidgets.QWidget()
            cell_layout = QtWidgets.QHBoxLayout(cell)
            cell_layout.setContentsMargins(2, 0, 0, 0)
            cell_layout.addWidget(button)
            cell_layout.addStretch(1)
            self.motion_key_table.setCellWidget(row, KEY_COLUMN_DELETE, cell)

    def on_motion_table_selected(self) -> None:
        key = self.selected_motion_keyframe()
        if key is None:
            return
        if key.frame != self.current_frame_index:
            self.stop_playback()
            self.goto_frame(key.frame)
        self.update_motion_table_delete_buttons(key.frame)

    def _touch_motion(self) -> None:
        self.motion_unsaved = True
        self.motion_version += 1

    def on_motion_table_shape_changed(self, frame: int, box: QtWidgets.QComboBox) -> None:
        key = self.motion.keyframe_at(frame)
        if key is None:
            return
        key.shape_id = str(box.currentData())
        self._touch_motion()
        self.refresh_motion_keyframes(key.frame)

    def on_motion_table_value_changed(self, frame: int, field: str, value: float) -> None:
        """X / Y / Scale edits apply live without rebuilding the row being edited."""
        key = self.motion.keyframe_at(frame)
        if key is None:
            return
        setattr(key, field, float(value))
        self._touch_motion()
        self.render_frame()
        self.update_labels()

    def on_motion_table_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        frame = item.data(Qt.ItemDataRole.UserRole)
        if frame is None:
            return
        key = self.motion.keyframe_at(int(frame))
        if key is None:
            return
        try:
            if item.column() == KEY_COLUMN_FRAME:
                value = int(item.text())
            elif item.column() == KEY_COLUMN_TIME:
                value = round(float(item.text()) * self.fps)
            else:
                return
            if value < 0 or (value != key.frame and self.motion.keyframe_at(value) is not None):
                raise ValueError
        except ValueError:
            self.set_status("frame/time must be unique and non-negative")
        else:
            key.frame = value
            self.motion.sort_keyframes()
            self._touch_motion()
        self.refresh_motion_keyframes(key.frame)

    def add_motion_keyframe(self) -> None:
        self.stop_playback()
        bbox = mask_bounding_box(self.painter.mask)
        if bbox is None:
            self.set_status(f"draw shape {self.active_shape_id} before adding a keyframe")
            return
        existing = self.motion.keyframe_at(self.current_frame_index)
        if existing is not None:
            self.refresh_motion_keyframes(existing.frame)
            self.set_status(f"selected existing keyframe at {self.current_seconds:.2f}s")
            return
        if not any(key.shape_id == self.active_shape_id for key in self.motion.keyframes):
            # The first key of a shape anchors it at the centre of what was painted.
            shape = self.motion.shapes[self.active_shape_id]
            shape.anchor_x, shape.anchor_y = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        if self.motion.shape_at_frame(self.current_frame_index) == self.active_shape_id:
            center_x, center_y, scale = self.motion.transform_at_frame(self.current_frame_index)
        else:
            (center_x, center_y), scale = self.motion.anchor(self.active_shape_id), 1.0
        key = MaskKeyframe(self.current_frame_index, center_x, center_y, scale, self.active_shape_id)
        self.motion.keyframes.append(key)
        self.motion.sort_keyframes()
        self._touch_motion()
        self.refresh_motion_keyframes(key.frame)
        self.set_status(
            f"added keyframe at {self.current_seconds:.2f}s; adjust centre and scale in its row, or add the next keyframe"
        )

    def delete_motion_keyframe(self, key: MaskKeyframe | None = None) -> None:
        key = key or self.selected_motion_keyframe()
        if key is None or key not in self.motion.keyframes:
            return
        index = self.motion.keyframes.index(key)
        self.motion.keyframes.remove(key)
        self._touch_motion()
        selected_frame = self.motion.keyframes[min(index, len(self.motion.keyframes) - 1)].frame if self.motion.keyframes else None
        self.refresh_motion_keyframes(selected_frame)
        self.set_status(f"deleted keyframe at frame {key.frame}")

    # -- mask save / encode ----------------------------------------------

    def save_state(self, snapshot: bool = False) -> pathlib.Path | None:
        snapshot_path = None
        for shape_id, painter in self.shape_painters.items():
            saved_snapshot = painter.save(snapshot=snapshot)
            if shape_id == "A":
                snapshot_path = saved_snapshot
        self.motion.save(self.motion_path)
        self.motion_unsaved = False
        return snapshot_path

    def autosave(self) -> None:
        if (not any(painter.unsaved for painter in self.shape_painters.values()) and not self.motion_unsaved) or self.painting:
            return
        try:
            self.save_state()
            self.set_status(f"autosave: {self.mask_path} and {self.motion_path.name}")
        except OSError as exc:
            self.set_status(f"autosave failed: {exc}")
        self.update_labels()

    def save_mask(self) -> None:
        try:
            snapshot = self.save_state(snapshot=True)
        except OSError as exc:
            self.set_status(f"save failed: {exc}")
            return
        self.set_status(f"saved: {self.mask_path} and {self.motion_path.name}  (backup: {snapshot.name if snapshot else '-'})")
        self.update_labels()

    def start_encode(self) -> None:
        if self.encode_job is not None:
            return
        self.stop_playback()
        selected, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "Output file", str(self.output_path), "Video (*.mp4 *.mkv *.mov);;All files (*)"
        )
        if not selected:
            return
        self.output_path = pathlib.Path(selected)
        try:
            self.save_state(snapshot=True)
        except OSError as exc:
            self.set_status(f"save failed: {exc}")
            return
        command = build_ffmpeg_command(
            self.video_path, self.mask_path, self.output_path,
            self.sigma_slider.value(), self.pixelize_slider.value(), self.crf_box.value(), self.preset,
            progress=True, motion=self.motion, shape_paths=motion_shape_paths(self.motion, self.mask_path),
        )
        logger.info("%s", shlex.join(command))
        self.encode_job = EncodeJob(command, self.duration, self)
        self.encode_job.progress.connect(lambda value: self.progress.setValue(int(value * 1000)))
        self.encode_job.done.connect(self.on_encode_done)
        self.progress.setValue(0)
        self.encode_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.set_status(f"encoding: {self.output_path}")
        self.encode_job.start()

    def cancel_encode(self) -> None:
        if self.encode_job is not None:
            self.encode_job.cancel()

    def on_encode_done(self, status: str, detail: str) -> None:
        if status == "ok":
            self.progress.setValue(1000)
            self.set_status(f"done: {self.output_path}")
        elif status == "cancelled":
            self.set_status(f"cancelled; the mask is kept at {self.mask_path}")
        else:
            logger.error("%s", detail)
            self.set_status(f"failed: {detail.splitlines()[0]}  (see the terminal for details)")
        job, self.encode_job = self.encode_job, None
        if job is not None:
            job.deleteLater()
        self.encode_button.setEnabled(True)
        self.cancel_button.setEnabled(False)

    # -- window ----------------------------------------------------------

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.render_frame()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.encode_job is not None:
            answer = QtWidgets.QMessageBox.question(
                self, "Quit", "Encoding is running. Cancel it and quit?"
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.encode_job.cancel()
        if any(painter.unsaved for painter in self.shape_painters.values()) or self.motion_unsaved:
            try:
                self.save_state()
            except OSError as exc:
                logger.error("failed to save mask: %s", exc)
        self.stop_playback()
        self.player.stop()
        self.cap.release()
        event.accept()


# -----------------------------------------------------------------------------
# subcommands


def paint(args: argparse.Namespace) -> int:
    input_path = pathlib.Path(args.input)
    if not input_path.exists():
        print(f"input not found: {input_path}", file=sys.stderr)
        return 1
    mask_path = pathlib.Path(args.mask) if args.mask else default_mask_path(input_path)
    output_path = pathlib.Path(args.output) if args.output else default_output_path(input_path)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    window = PaintWindow(
        video_path=input_path,
        mask_path=mask_path,
        output_path=output_path,
        initial_time=args.time or TimeOrFrame(kind="time", value=0),
        sigma=args.sigma,
        pixelize=args.pixelize,
        crf=args.crf,
        preset=args.preset,
    )
    window.show()
    app.exec()
    print(mask_path)
    return 0


def encode(args: argparse.Namespace) -> int:
    input_path = pathlib.Path(args.input)
    if not input_path.exists():
        print(f"input not found: {input_path}", file=sys.stderr)
        return 1
    mask_path = pathlib.Path(args.mask) if args.mask else default_mask_path(input_path)
    output_path = pathlib.Path(args.output) if args.output else default_output_path(input_path)
    cap = cv2.VideoCapture(str(input_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    motion = MaskMotion.load(default_motion_path(mask_path), width, height, fps)
    shape_paths = motion_shape_paths(motion, mask_path)
    command = build_ffmpeg_command(
        input_path, mask_path, output_path, args.sigma, args.pixelize, args.crf, args.preset,
        motion=motion, shape_paths=shape_paths,
    )
    if args.print_command:
        print(shlex.join(command))
        return 0
    missing = [path for path in shape_paths.values() if not path.exists()]
    if missing:
        print(f"mask not found: {missing[0]}", file=sys.stderr)
        return 1
    logger.info("%s", shlex.join(command))
    return subprocess.run(command).returncode


def add_blur_arguments(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("input", help="input video path")
    subparser.add_argument("--mask", help="mask PNG path; default: ./<input stem>.mask.png")
    subparser.add_argument("-o", "--output", help="output video path; default: ./<input stem>_blur.mp4")
    subparser.add_argument("--sigma", type=float, default=26.0, help="gblur sigma")
    subparser.add_argument("--pixelize", type=int, default=0, help="pixelize block size; 0 to disable")
    subparser.add_argument("--crf", type=int, default=20, help="libx264 crf")
    subparser.add_argument("--preset", default="slow", help="libx264 preset")


def main() -> int:
    parser = argparse.ArgumentParser(formatter_class=ArgumentDefaultsRawTextHelpFormatter, epilog=epilog)
    subparsers = parser.add_subparsers(dest="subcommand_name", required=True)

    subparser = subparsers.add_parser(
        "paint", formatter_class=ArgumentDefaultsRawTextHelpFormatter,
        help="open GUI, paint a blur mask while playing the video, then encode",
    )
    subparser.set_defaults(func=paint)
    add_blur_arguments(subparser)
    subparser.add_argument(
        "--time", type=parse_time_or_frame,
        help="initial seek position as seconds, hh:mm:ss.mmm, frame:N/f:N, or last",
    )

    subparser = subparsers.add_parser(
        "encode", formatter_class=ArgumentDefaultsRawTextHelpFormatter,
        help="encode with an existing mask (GUI crash recovery)",
    )
    subparser.set_defaults(func=encode)
    add_blur_arguments(subparser)
    subparser.add_argument("--print-command", action="store_true", help="print the ffmpeg command and exit")

    parser.add_argument("-q", "--quiet", action="count", default=0,
                        help="decrease verbosity; default: debug, -q: info, -qq: warning, -qqq: error")
    args = parser.parse_args()
    logger.setLevel({0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}.get(args.quiet, logging.ERROR))
    logger.debug("%s", args)
    return args.func(args)


# -----------------------------------------------------------------------------
# tests (pytest)


def test_build_filter_complex():
    graph = build_filter_complex(26, 0)
    assert "gblur=sigma=26" in graph
    # A black mask preserves the input; white exposes the blurred edge.
    # Exercise the connections in FFmpeg without fixing internal graph labels.
    command = ["ffmpeg", "-v", "error", "-filter_complex_threads", "1",
               "-f", "lavfi", "-i", "color=white:s=32x32,drawbox=x=0:y=0:w=16:h=32:color=black:t=fill",
               "-f", "lavfi", "-i", "color=black:s=32x32,drawbox=color=white:t=fill:enable='eq(n,1)'",
               "-filter_complex", graph, "-map", "[out]", "-frames:v", "2",
               "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    logger.debug("%s", shlex.join(command))
    result = subprocess.run(command, check=True, capture_output=True)
    frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(2, 32, 32, 3)
    assert np.all(frames[0, :, :16] == 0)
    assert np.all(frames[0, :, 16:] == 255)
    assert np.all(frames[1, :, 15] > 0)
    assert np.all(frames[1, :, 16] < 255)
    assert "pixelize=w=24:h=24" in build_filter_complex(10, 24)
    assert "pixelize" not in build_filter_complex(10, 0)
    assert "gblur" not in build_filter_complex(0, 24)


def test_build_ffmpeg_command():
    command = build_ffmpeg_command(
        pathlib.Path("a.mp4"), pathlib.Path("m.png"), pathlib.Path("o.mp4"), 26, 0, 20, "slow"
    )
    assert command[-1] == "o.mp4"
    assert command.count("-i") == 2
    assert "-progress" not in command


def test_mask_motion_persistence_and_transform(tmp_path):
    motion = MaskMotion(100, 50, 25, keyframes=[
        MaskKeyframe(25, 50, 25, 1.0),
        MaskKeyframe(75, 80, 15, 1.5),
        MaskKeyframe(100, 80, 15, 1.5),
        MaskKeyframe(125, 20, 35, 0.75),
    ])
    assert motion.transform_at_frame(0) == (50, 25, 1.0)
    assert motion.transform_at_frame(50) == (65, 20, 1.25)
    assert motion.transform_at_frame(90) == (80, 15, 1.5)  # B hold
    assert motion.transform_at_frame(150) == (20, 35, 0.75)
    path = tmp_path / "m.mask.motion.json"
    motion.save(path)
    loaded = MaskMotion.load(path, 200, 100, 25)
    assert loaded.transform_at_frame(50) == (130, 40, 1.25)  # A -> B
    assert loaded.transform_at_frame(90) == (160, 30, 1.5)  # B hold
    assert loaded.to_dict()["version"] == 3
    assert default_motion_path(tmp_path / "m.mask.png") == path


def test_loads_version_1_motion_as_two_keyframes(tmp_path):
    path = tmp_path / "old.motion.json"
    path.write_text(json.dumps({
        "version": 1, "video_size": [100, 50], "fps": 25,
        "start_frame": 25, "end_frame": 75, "anchor": [50, 25],
        "end_center": [80, 15], "end_scale": 1.5,
    }), encoding="utf-8")
    loaded = MaskMotion.load(path, 100, 50, 25)
    assert [(key.frame, key.center_x, key.center_y, key.scale) for key in loaded.keyframes] == [
        (25, 50, 25, 1.0), (75, 80, 15, 1.5),
    ]
    assert loaded.transform_at_frame(50) == (65, 20, 1.25)


def test_multiple_shapes_switch_at_the_keyframe_and_persist(tmp_path):
    motion = MaskMotion(100, 50, 10)
    motion.set_shape("A", "", 20, 25)
    motion.set_shape("B", "m.mask.shape-B.png", 80, 25)
    motion.keyframes = [
        MaskKeyframe(10, 20, 25, 1.0, "A"),
        MaskKeyframe(20, 50, 25, 1.5, "A"),
        MaskKeyframe(30, 80, 25, 1.0, "B"),
    ]
    assert motion.shape_at_frame(29) == "A"
    assert motion.shape_at_frame(30) == "B"
    assert motion.transform_at_frame(15) == (35, 25, 1.25)
    assert motion.transform_at_frame(25) == (50, 25, 1.5)  # A holds before B's instant switch
    assert motion.transform_at_frame(30) == (80, 25, 1.0)
    path = tmp_path / "m.mask.motion.json"
    motion.save(path)
    loaded = MaskMotion.load(path, 200, 100, 10)
    assert loaded.shape_at_frame(29) == "A"
    assert loaded.shape_at_frame(30) == "B"
    assert loaded.anchor("B") == (160, 50)
    assert motion_shape_paths(loaded, tmp_path / "m.mask.png")["B"] == tmp_path / "m.mask.shape-B.png"
    graph = build_filter_complex(3, 0, motion)
    assert "gte(t,3)" in graph and "[2:v]format=gray" in graph


def test_build_motion_filter_and_command():
    motion = MaskMotion(100, 50, 25, keyframes=[
        MaskKeyframe(25, 50, 25), MaskKeyframe(75, 80, 15, 1.5),
        MaskKeyframe(100, 80, 15, 1.5), MaskKeyframe(125, 20, 35, 0.75),
    ])
    graph = build_filter_complex(26, 0, motion)
    assert "split=3" in graph
    assert "scale=w=" in graph
    assert "overlay=x=" in graph
    assert "(t-1)/2" in graph and "(t-3)/1" in graph and "(t-4)/1" in graph
    command = build_ffmpeg_command(
        pathlib.Path("a.mp4"), pathlib.Path("m.png"), pathlib.Path("o.mp4"), 26, 0, 20, "slow", motion=motion
    )
    assert command[command.index("-loop") + 1] == "1"
    assert "-framerate" in command


def test_motion_filter_encodes_a_video(tmp_path):
    input_path = tmp_path / "input.mp4"
    mask_path = tmp_path / "mask.png"
    output_path = tmp_path / "output.mp4"
    source = [
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=32x32:r=10:d=1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(input_path),
    ]
    logger.debug("%s", shlex.join(source))
    subprocess.run(source, check=True, capture_output=True)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[10:22, 4:16] = 255
    assert cv2.imwrite(str(mask_path), mask)
    motion = MaskMotion(32, 32, 10, keyframes=[
        MaskKeyframe(2, 10, 16), MaskKeyframe(4, 24, 16, 1.25),
        MaskKeyframe(6, 24, 16, 1.25), MaskKeyframe(8, 10, 16),
    ])
    motion.set_shape("A", "", 10, 16)
    command = build_ffmpeg_command(input_path, mask_path, output_path, 3, 0, 20, "ultrafast", motion=motion)
    logger.debug("%s", shlex.join(command))
    subprocess.run(command, check=True, capture_output=True)
    cap = cv2.VideoCapture(str(output_path))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 10
    cap.release()


def test_multiple_shape_filter_encodes_a_video(tmp_path):
    input_path = tmp_path / "input.mp4"
    mask_path = tmp_path / "mask.png"
    output_path = tmp_path / "output.mp4"
    source = [
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=32x32:r=10:d=1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(input_path),
    ]
    subprocess.run(source, check=True, capture_output=True)
    mask_a = np.zeros((32, 32), dtype=np.uint8); mask_a[:, :12] = 255
    mask_b = np.zeros((32, 32), dtype=np.uint8); mask_b[:, 20:] = 255
    assert cv2.imwrite(str(mask_path), mask_a)
    shape_b_path = default_shape_path(mask_path, "B")
    assert cv2.imwrite(str(shape_b_path), mask_b)
    motion = MaskMotion(32, 32, 10)
    motion.set_shape("A", "", 6, 16)
    motion.set_shape("B", shape_b_path.name, 26, 16)
    motion.keyframes = [MaskKeyframe(0, 6, 16, 1, "A"), MaskKeyframe(5, 26, 16, 1, "B")]
    command = build_ffmpeg_command(
        input_path, mask_path, output_path, 3, 0, 20, "ultrafast", motion=motion,
        shape_paths=motion_shape_paths(motion, mask_path),
    )
    assert command.count("-i") == 3
    subprocess.run(command, check=True, capture_output=True)
    cap = cv2.VideoCapture(str(output_path))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 10
    cap.release()


def test_default_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert default_mask_path(pathlib.Path("/x/PXL_1.mp4")) == tmp_path / "PXL_1.mask.png"
    assert default_output_path(pathlib.Path("/x/PXL_1.mp4")) == tmp_path / "PXL_1_blur.mp4"


def test_mask_painter(tmp_path):
    painter = MaskPainter(64, 32, tmp_path / "m.mask.png")
    painter.begin_stroke()
    painter.stroke(10, 10, 40, 20, radius=6, hardness=0.3, erase=False)
    assert painter.mask.max() == 255
    assert painter.unsaved
    painted = painter.coverage()
    painter.save(snapshot=True)
    assert not painter.unsaved
    assert (tmp_path / "m.mask.png").exists()
    assert len(list(tmp_path.glob("m.mask.*.png"))) == 1

    painter.begin_stroke()
    painter.stroke(10, 10, 40, 20, radius=20, hardness=0.95, erase=True)
    erased = painter.coverage()
    assert erased < painted
    assert painter.undo()
    assert painter.coverage() == painted
    assert painter.redo()
    assert painter.coverage() == erased

    reloaded = MaskPainter(64, 32, tmp_path / "m.mask.png")
    assert reloaded.load()
    assert reloaded.coverage() == painted


def test_apply_blur():
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[20:, :] = 255
    original = image.copy()
    np.testing.assert_array_equal(apply_blur(image, 0, 0), original)
    assert apply_blur(image, 3.0, 0).shape == image.shape
    assert apply_blur(image, 0, 8).shape == image.shape


if __name__ == "__main__":
    raise SystemExit(main())
