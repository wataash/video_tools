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
import datetime
import fractions
import logging
import math
import os
import pathlib
import shlex
import subprocess
import sys
import typing as t

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt


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


def build_filter_complex(sigma: float, pixelize: int) -> str:
    """
    >>> build_filter_complex(26, 0)
    '[0:v]split[base][pre];[pre]gblur=sigma=26:steps=3,format=yuva420p[blurred];[1:v]format=gray[mask];[blurred][mask]alphamerge[blurred_a];[base][blurred_a]overlay=0:0[out]'
    """
    chain = []
    if pixelize >= 2:
        chain.append(f"pixelize=w={pixelize}:h={pixelize}")
    if sigma > 0:
        chain.append(f"gblur=sigma={sigma:g}:steps=3")
    chain.append("format=yuva420p")
    return (
        "[0:v]split[base][pre];"
        f"[pre]{','.join(chain)}[blurred];"
        "[1:v]format=gray[mask];"
        # The mask is a single still image; framesync repeatlast reuses it for every frame
        "[blurred][mask]alphamerge[blurred_a];"
        "[base][blurred_a]overlay=0:0[out]"
    )


def build_ffmpeg_command(
    input_path: pathlib.Path,
    mask_path: pathlib.Path,
    output_path: pathlib.Path,
    sigma: float,
    pixelize: int,
    crf: int,
    preset: str,
    progress: bool = False,
) -> list[str]:
    command = ["ffmpeg", "-hide_banner", "-y"]
    if progress:
        command += ["-nostats", "-progress", "pipe:1"]
    command += [
        "-i", str(input_path),
        "-i", str(mask_path),
        "-filter_complex", build_filter_complex(sigma, pixelize),
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
    """A slider that jumps straight to the clicked position instead of paging."""

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

        self.current_frame_index = 0
        self.current_bgr: np.ndarray | None = None
        self.view_mode = "composite"
        self.speed = 0.5
        self.painting = False
        self.erasing = False
        self.last_point: tuple[int, int] | None = None
        self.encode_job: EncodeJob | None = None
        self._frame_buffer: np.ndarray | None = None  # keeps the QImage data alive
        self._mask_cache: tuple[int, int, int] | None = None
        self._mask3: np.ndarray | None = None
        self._mask_bbox: tuple[int, int, int, int] | None = None

        self.setWindowTitle(f"video-blur-mask-gui: {video_path.name}")
        self.resize(1280, 900)
        self._build_ui(sigma, pixelize, crf)
        self._build_shortcuts()

        self.playback_timer = QtCore.QTimer(self)
        self.playback_timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self.playback_timer.timeout.connect(self.play_next_frame)
        self.autosave_timer = QtCore.QTimer(self)
        self.autosave_timer.timeout.connect(self.autosave)
        self.autosave_timer.start(AUTOSAVE_INTERVAL_MS)

        start_frame = resolve_frame_index(initial_time, self.fps, self.frame_count) if self.frame_count else 0
        self.goto_frame(start_frame)
        self.set_status(
            "loaded the existing mask" if self.loaded_existing
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
        self.speed_box.setCurrentIndex(1)
        self.speed_box.currentIndexChanged.connect(self.on_speed_changed)

        self.view_button = QtWidgets.QPushButton("Composite (v)")
        self.view_button.clicked.connect(self.cycle_view)

        self.pen_slider = self._make_slider(8, max(40, self.video_w // 3), max(4, self.video_w // 22))
        self.pen_slider.valueChanged.connect(self.on_pen_changed)
        self.hardness_slider = self._make_slider(0, 95, 35)
        self.sigma_slider = self._make_slider(0, 80, int(round(sigma)))
        self.sigma_slider.valueChanged.connect(self.render_frame)
        self.pixelize_slider = self._make_slider(0, 80, pixelize)
        self.pixelize_slider.valueChanged.connect(self.render_frame)

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
            QtWidgets.QLabel("Speed"), self.speed_box, None,
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

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.timeline)
        layout.addLayout(row1)
        layout.addLayout(row2)
        layout.addLayout(row3)
        self.setCentralWidget(central)

        self.position_label = QtWidgets.QLabel()
        self.position_label.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        self.statusBar().addPermanentWidget(self.position_label)
        self.statusBar().addPermanentWidget(QtWidgets.QLabel(f"mask: {self.mask_path}"))

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
        self.current_bgr = frame
        self.render_frame()
        self.sync_timeline()

    def advance_frames(self, step: int) -> bool:
        """For playback: skip with grab() instead of seeking, which is faster than rewinding."""
        for _ in range(step - 1):
            if not self.cap.grab():
                return False
        ok, frame = self.cap.read()
        if not ok:
            return False
        self.current_frame_index += step
        self.current_bgr = frame
        self.render_frame()
        self.sync_timeline()
        return True

    def sync_timeline(self) -> None:
        self.timeline.blockSignals(True)
        self.timeline.setValue(self.current_frame_index)
        self.timeline.blockSignals(False)
        self.update_labels()

    def step_frame(self, delta: int) -> None:
        self.stop_playback()
        self.goto_frame(self.current_frame_index + delta)

    # -- playback --------------------------------------------------------

    def playback_step(self) -> int:
        return max(1, int(round(self.fps * self.speed / RENDER_FPS)))

    def playback_interval_ms(self) -> int:
        return max(1, int(round(1000.0 * self.playback_step() / max(1e-6, self.fps * self.speed))))

    def stop_playback(self) -> None:
        self.playback_timer.stop()
        self.play_button.setText("Play (Space)")

    def toggle_playback(self) -> None:
        if self.playback_timer.isActive():
            self.stop_playback()
            return
        if self.frame_count and self.current_frame_index >= self.frame_count - 1:
            self.goto_frame(0)
        self.play_button.setText("Pause (Space)")
        self.playback_timer.start(self.playback_interval_ms())

    def on_speed_changed(self) -> None:
        self.speed = float(self.speed_box.currentData())
        if self.playback_timer.isActive():
            self.playback_timer.start(self.playback_interval_ms())

    def play_next_frame(self) -> None:
        if not self.advance_frames(self.playback_step()):
            self.stop_playback()
            self.set_status("reached the end")

    def on_timeline_changed(self, value: int) -> None:
        self.stop_playback()
        if value != self.current_frame_index:
            self.goto_frame(value)

    # -- rendering -------------------------------------------------------

    def scaled_mask(self, width: int, height: int) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
        """Return the 3ch float mask at display size and its bounding box (refreshed on every stroke)."""
        key = (self.painter.version, width, height)
        if self._mask_cache != key or self._mask3 is None:
            small = cv2.resize(self.painter.mask, (width, height), interpolation=cv2.INTER_LINEAR)
            normalized = small.astype(np.float32) / 255.0
            self._mask3 = cv2.merge([normalized, normalized, normalized])
            self._mask_bbox = mask_bounding_box(small)
            self._mask_cache = key
        return self._mask3, self._mask_bbox

    def render_frame(self) -> None:
        if self.current_bgr is None:
            return
        # Render at device resolution but never upscale; Qt stretches the result if the widget is larger.
        scale = min(1.0, self.canvas.fit_scale())
        width = max(1, int(round(self.video_w * scale)))
        height = max(1, int(round(self.video_h * scale)))
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

        self._frame_buffer = np.ascontiguousarray(shown)  # QImage does not copy the data
        image = QtGui.QImage(
            self._frame_buffer.data, width, height,
            self._frame_buffer.strides[0], QtGui.QImage.Format.Format_BGR888,
        )
        self.canvas.brush_radius = self.pen_slider.value()
        self.canvas.set_image(image)

    def update_labels(self) -> None:
        seconds = self.current_frame_index / self.fps if self.fps else 0.0
        self.position_label.setText(
            f"{seconds:7.2f}s / {self.duration:.2f}s  frame {self.current_frame_index}/{max(0, self.frame_count - 1)}"
            f"   pen r={self.pen_slider.value()}  mask {self.painter.coverage() * 100:4.1f}%"
            f"{'  *unsaved' if self.painter.unsaved else ''}"
        )

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

    # -- mask save / encode ----------------------------------------------

    def autosave(self) -> None:
        if not self.painter.unsaved or self.painting:
            return
        try:
            self.painter.save()
            self.set_status(f"autosave: {self.mask_path}")
        except OSError as exc:
            self.set_status(f"autosave failed: {exc}")
        self.update_labels()

    def save_mask(self) -> None:
        try:
            snapshot = self.painter.save(snapshot=True)
        except OSError as exc:
            self.set_status(f"save failed: {exc}")
            return
        self.set_status(f"saved: {self.mask_path}  (backup: {snapshot.name if snapshot else '-'})")
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
            self.painter.save(snapshot=True)
        except OSError as exc:
            self.set_status(f"save failed: {exc}")
            return
        command = build_ffmpeg_command(
            self.video_path, self.mask_path, self.output_path,
            self.sigma_slider.value(), self.pixelize_slider.value(), self.crf_box.value(), self.preset,
            progress=True,
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
        if self.painter.unsaved:
            try:
                self.painter.save()
            except OSError as exc:
                logger.error("failed to save mask: %s", exc)
        self.stop_playback()
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
    command = build_ffmpeg_command(
        input_path, mask_path, output_path, args.sigma, args.pixelize, args.crf, args.preset
    )
    if args.print_command:
        print(shlex.join(command))
        return 0
    if not mask_path.exists():
        print(f"mask not found: {mask_path}", file=sys.stderr)
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
    assert apply_blur(image, 0, 0) is image
    assert apply_blur(image, 3.0, 0).shape == image.shape
    assert apply_blur(image, 0, 8).shape == image.shape


if __name__ == "__main__":
    raise SystemExit(main())
