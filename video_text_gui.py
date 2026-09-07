#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Wataru Ashihara <wataash0607@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Place timed text on a video and export it using the same Qt-rendered sprites."""
epilog = """fish / bash:
  python video_text_gui.py edit input.mp4
  python video_text_gui.py encode input.mp4 --settings input.text.json -o output.mp4
  python video_text_gui.py encode input.mp4 --settings input.text.json -n
"""

import argparse
from dataclasses import asdict, dataclass
import json
import logging
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer, QVideoSink

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname).1s %(asctime)s %(filename)s:%(lineno)d] %(message)s", datefmt="%T"))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


class ArgumentDefaultsRawTextHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


def log_command(command):
    # These quoted argv-only commands have identical syntax in fish and bash.
    logger.info("fish / bash: %s", shlex.join([str(x) for x in command]))


@dataclass
class VideoInfo:
    width: int
    height: int
    duration: float


def probe(path):
    command = ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]
    log_command(command)
    data = json.loads(subprocess.check_output(command, text=True))
    stream = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    if stream is None:
        raise ValueError("映像ストリームがありません")
    rotation = next((s["rotation"] for s in stream.get("side_data_list", []) if "rotation" in s), 0)
    w, h = stream["width"], stream["height"]
    if round(rotation) % 180:
        w, h = h, w
    duration = float(stream.get("duration", data["format"].get("duration", 0)))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("動画の長さを取得できません")
    return VideoInfo(w, h, duration)


@dataclass
class Text:
    text: str = "ここにテキスト"
    x: float = 0.5
    y: float = 0.8
    size: int = 64
    font: str = "Noto Sans CJK JP"
    color: str = "#ffffffff"
    outline_color: str = "#ff000000"
    outline: int = 3
    background: str = "#00000000"
    start: float = 0.0
    end: float = 1.0

    def validate(self, duration):
        if not all(isinstance(v, str) for v in (self.text, self.font, self.color, self.outline_color, self.background)):
            raise ValueError("文字・フォント・色の形式が不正です")
        if len(self.text) > 10000 or len(self.font) > 200:
            raise ValueError("テキストが長すぎます")
        for key, lo, hi in (("x", 0, 1), ("y", 0, 1), ("size", 8, 400), ("outline", 0, 30), ("start", 0, duration), ("end", 0, duration + 0.001)):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"{key} は {lo:g} 〜 {hi:g} の範囲で指定してください")
        if not isinstance(self.size, int) or not isinstance(self.outline, int):
            raise ValueError("サイズ・縁取りは整数で指定してください")
        if self.start >= self.end:
            raise ValueError("終了時刻は開始時刻より後にしてください")
        if not all(QtGui.QColor(v).isValid() for v in (self.color, self.outline_color, self.background)):
            raise ValueError("色の指定が不正です")


def load_settings(path, video, info):
    data = json.loads(Path(path).read_text())
    if data.get("version") != 1 or data.get("size") != [info.width, info.height]:
        raise ValueError("設定のバージョンまたは動画サイズが一致しません")
    if Path(data.get("video", "")).resolve() != video.resolve():
        raise ValueError("別の動画の設定です")
    raw = data.get("texts")
    if not isinstance(raw, list) or len(raw) > 100:
        raise ValueError("テキスト一覧が不正です（最大100件）")
    try:
        texts = [Text(**item) for item in raw]
        for item in texts:
            item.validate(info.duration)
    except TypeError as exc:
        raise ValueError("テキスト設定の形式が不正です") from exc
    return texts


def save_settings(path, video, info, texts):
    for item in texts:
        item.validate(info.duration)
    if path.resolve() == video.resolve():
        raise ValueError("元動画には保存できません")
    data = {"version": 1, "video": str(video.resolve()), "size": [info.width, info.height], "texts": [asdict(item) for item in texts]}
    # Atomic replacement prevents incomplete settings after an interrupted write.
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
            temp = Path(f.name)
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        temp.replace(path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def render_text(item):
    """Render once at output resolution; preview and encoder use the same image."""
    font = QtGui.QFont(item.font)
    font.setPixelSize(item.size)
    layouts = []
    for text in item.text.split("\n"):
        layout = QtGui.QTextLayout(text, font)
        layout.beginLayout()
        line = layout.createLine()
        line.setLineWidth(1_000_000)
        layout.endLayout()
        layouts.append((layout, line))
    width = max(line.naturalTextWidth() for _, line in layouts)
    path = QtGui.QPainterPath()
    positioned = []
    height = 0
    for layout, line in layouts:
        origin = QtCore.QPointF((width - line.naturalTextWidth()) / 2, height)
        positioned.append((layout, origin))
        for run in layout.glyphRuns():
            raw = run.rawFont()
            # Asking FreeType for outlines of bitmap emoji can also poison Qt's
            # glyph cache. Never request paths from a color font.
            if any(raw.fontTable(table) for table in ("CBDT", "COLR", "sbix", "SVG ")):
                continue
            for glyph, position in zip(run.glyphIndexes(), run.positions()):
                offset = origin + position
                path.addPath(QtGui.QTransform.fromTranslate(offset.x(), offset.y()).map(raw.pathForGlyph(glyph)))
        height += line.height()
    bounds = path.boundingRect().united(QtCore.QRectF(0, 0, max(1, width), max(1, height)))
    padding = item.outline + (12 if QtGui.QColor(item.background).alpha() else 2)
    bounds.adjust(-padding, -padding, padding, padding)
    w, h = math.ceil(bounds.width()), math.ceil(bounds.height())
    if w * h > 32_000_000:
        raise ValueError("文字画像が大きすぎます。文字数やサイズを減らしてください")
    image = QtGui.QImage(w, h, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(QtGui.QColor(item.background))
    painter = QtGui.QPainter(image)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.translate(-bounds.left(), -bounds.top())
    if item.outline:
        # Draw the outline underneath; a centered stroke would cover thin glyphs.
        pen = QtGui.QPen(QtGui.QColor(item.outline_color), item.outline * 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.strokePath(path, pen)
    # The same shaped layout paints normal text and native color emoji.
    painter.setPen(QtGui.QColor(item.color))
    for layout, origin in positioned:
        layout.draw(painter, origin)
    painter.end()
    return image


def text_rect(item, image, info):
    return QtCore.QRect(round(item.x * info.width - image.width() / 2), round(item.y * info.height - image.height() / 2), image.width(), image.height())


def build_command(video, output, info, texts, sprites, crf=18):
    command = ["ffmpeg", "-hide_banner", "-nostdin", "-n", "-i", str(video)]
    for sprite in sprites:
        command += ["-i", str(sprite)]
    filters = ["[0:v:0]setpts=PTS-STARTPTS,setsar=1,format=rgba[base]"]
    previous = "base"
    for i, item in enumerate(texts):
        # Use exactly the same integer pixel coordinates as the preview.
        rect = text_rect(item, QtGui.QImage(str(sprites[i])), info)
        current = f"v{i}"
        filters.append(f"[{previous}][{i + 1}:v:0]overlay=x={rect.x()}:y={rect.y()}:enable='gte(t,{item.start:.9f})*lt(t,{item.end:.9f})':eof_action=repeat:format=rgb[{current}]")
        previous = current
    filters.append(f"[{previous}]pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p[out]")
    command += ["-filter_complex", ";".join(filters), "-map", "[out]", "-map", "0:a?", "-c:v", "libx264", "-crf", str(crf), "-preset", "medium", "-fps_mode", "passthrough", "-enc_time_base", "filter", "-c:a", "copy", "-map_metadata", "-1", "-metadata:s:v:0", "rotate=0", "-t", f"{info.duration:.9f}", "-movflags", "+faststart", "-progress", "pipe:1", "-nostats", str(output)]
    return command


class Export:
    """Own temporary assets and publish a completed MP4 without overwriting files."""
    def __init__(self, video, output, info, texts, crf=18):
        if output.suffix.lower() != ".mp4":
            raise ValueError("出力には .mp4 を指定してください")
        if output.resolve() == video.resolve() or output.exists():
            raise ValueError("出力先が既に存在します。別の名前を指定してください")
        for item in texts:
            item.validate(info.duration)
        self.output = output
        self.temp = tempfile.TemporaryDirectory(prefix=".video-text-", dir=output.parent)
        self.partial = Path(self.temp.name) / "output.mp4"
        try:
            sprites = []
            for i, item in enumerate(texts):
                path = Path(self.temp.name) / f"text-{i}.png"
                if not render_text(item).save(str(path)):
                    raise OSError("文字画像を保存できません")
                sprites.append(path)
            self.command = build_command(video, self.partial, info, texts, sprites, crf)
        except Exception:
            self.cleanup()
            raise

    def publish(self):
        # Same filesystem: link is atomic and fails if another file now exists.
        os.link(self.partial, self.output)

    def cleanup(self):
        self.temp.cleanup()


class Canvas(QtWidgets.QWidget):
    selected = QtCore.Signal(int)
    moved = QtCore.Signal(float, float)
    drag_started = QtCore.Signal()
    drag_finished = QtCore.Signal()

    def __init__(self, info):
        super().__init__()
        self.info = info
        self.frame = QtGui.QImage()
        self.texts = []
        self.sprites = []
        self.index = -1
        self.seconds = 0.0
        self.drag_offset = None
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(240, 300)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)

    def viewport(self):
        size = QtCore.QSizeF(self.info.width, self.info.height)
        size.scale(QtCore.QSizeF(self.width(), self.height()), Qt.AspectRatioMode.KeepAspectRatio)
        return QtCore.QRectF((self.width() - size.width()) / 2, (self.height() - size.height()) / 2, size.width(), size.height())

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor("#17191d"))
        viewport = self.viewport()
        p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        p.translate(viewport.topLeft())
        p.scale(viewport.width() / self.info.width, viewport.height() / self.info.height)
        p.setClipRect(0, 0, self.info.width, self.info.height)
        if not self.frame.isNull():
            p.drawImage(QtCore.QRect(0, 0, self.info.width, self.info.height), self.frame)
        for i, (item, sprite) in enumerate(zip(self.texts, self.sprites)):
            if item.start <= self.seconds < item.end:
                rect = text_rect(item, sprite, self.info)
                p.drawImage(rect.topLeft(), sprite)
                if i == self.index:
                    pen = QtGui.QPen(QtGui.QColor("#6bd9ff"), 1, Qt.PenStyle.DashLine)
                    pen.setCosmetic(True)
                    p.setPen(pen)
                    p.drawRect(rect)
        p.end()

    def video_point(self, event):
        viewport = self.viewport()
        return QtCore.QPointF((event.position().x() - viewport.x()) * self.info.width / viewport.width(), (event.position().y() - viewport.y()) * self.info.height / viewport.height())

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        point = self.video_point(event)
        for i in reversed(range(len(self.texts))):
            item = self.texts[i]
            if item.start <= self.seconds < item.end and text_rect(item, self.sprites[i], self.info).contains(point.toPoint()):
                self.selected.emit(i)
                self.drag_started.emit()
                self.drag_offset = point - QtCore.QPointF(item.x * self.info.width, item.y * self.info.height)
                return

    def mouseMoveEvent(self, event):
        if self.drag_offset is not None:
            point = self.video_point(event) - self.drag_offset
            self.moved.emit(max(0, min(1, point.x() / self.info.width)), max(0, min(1, point.y() / self.info.height)))

    def mouseReleaseEvent(self, event):
        self.drag_offset = None
        self.drag_finished.emit()


class SeekSlider(QtWidgets.QSlider):
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            option = QtWidgets.QStyleOptionSlider()
            self.initStyleOption(option)
            groove = self.style().subControlRect(QtWidgets.QStyle.ComplexControl.CC_Slider, option, QtWidgets.QStyle.SubControl.SC_SliderGroove, self)
            handle = self.style().subControlRect(QtWidgets.QStyle.ComplexControl.CC_Slider, option, QtWidgets.QStyle.SubControl.SC_SliderHandle, self)
            value = QtWidgets.QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), round(event.position().x() - groove.x() - handle.width() / 2), max(1, groove.width() - handle.width()))
            self.setValue(value)
        super().mousePressEvent(event)


class EditCommand(QtGui.QUndoCommand):
    def __init__(self, window, before, after, label, key):
        super().__init__(label)
        self.window, self.before, self.after = window, before, after
        self.key = key
        self.timestamp = time.monotonic()
        self.first_redo = True

    def id(self):
        return 1 if self.key is not None else -1

    def mergeWith(self, other):
        if self.key != other.key or (self.key[-1] != "drag" and other.timestamp - self.timestamp > 0.8):
            return False
        self.after, self.timestamp = other.after, other.timestamp
        self.setObsolete(self.before[0] == self.after[0])
        return True

    def undo(self):
        self.window.restore(self.before)

    def redo(self):
        # push() calls redo; the live editor has already applied that edit.
        if self.first_redo:
            self.first_redo = False
        else:
            self.window.restore(self.after)


class Window(QtWidgets.QMainWindow):
    def __init__(self, video, info, settings, output):
        super().__init__()
        self.video, self.info, self.settings, self.output = video, info, settings, output
        self.texts = load_settings(settings, video, info) if settings.exists() else [Text(end=info.duration)]
        self.history = QtGui.QUndoStack(self)
        self.history.setUndoLimit(100)
        self.edit_group = 0
        self.loading = False
        self.export = None
        self.process = None
        self.cancelled = False
        self.buffer = ""
        self.errors = ""
        self.setWindowTitle(f"テキスト配置 — {video.name}")
        self.resize(1060, 900)
        self.canvas = Canvas(info)
        self.canvas.texts = self.texts
        self.canvas.selected.connect(self.select)
        self.canvas.moved.connect(self.move_text)
        self.canvas.drag_started.connect(self.break_merge)
        self.canvas.drag_finished.connect(self.break_merge)
        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        self.sink = QVideoSink(self)
        self.player.setVideoSink(self.sink)
        self.sink.videoFrameChanged.connect(self.frame_changed)
        self.player.positionChanged.connect(self.position_changed)
        self.player.playbackStateChanged.connect(self.playback_changed)
        self.player.errorOccurred.connect(lambda error, message: self.show_error(message))
        self.build_ui()
        QtWidgets.QApplication.instance().installEventFilter(self)
        self.refresh()
        self.select(0 if self.texts else -1)
        self.player.setSource(QtCore.QUrl.fromLocalFile(str(video.resolve())))
        self.player.pause()
        self.statusBar().showMessage(f"{info.width} × {info.height} ｜ {info.duration:.2f} 秒 ｜ 文字をドラッグして配置")

    def button(self, label, action):
        button = QtWidgets.QPushButton(label)
        button.clicked.connect(action)
        return button

    @property
    def dirty(self):
        return not self.history.isClean()

    def break_merge(self):
        self.edit_group += 1

    def snapshot(self):
        return ([Text(**asdict(item)) for item in self.texts], self.canvas.index)

    def restore(self, state):
        self.texts[:] = [Text(**asdict(item)) for item in state[0]]
        self.canvas.index = -1
        self.canvas.drag_offset = None
        self.refresh()
        self.select(state[1])

    def record(self, before, label, field=None):
        after = self.snapshot()
        if before[0] != after[0]:
            key = (self.edit_group, after[1], field) if field else None
            self.history.push(EditCommand(self, before, after, label, key))

    def commit(self, before, label, field):
        """Re-render the edited texts and record the edit, or roll back if they cannot be rendered."""
        try:
            self.refresh()
        except ValueError as exc:
            self.restore(before)
            self.show_error(str(exc))
            return False
        self.record(before, label, field)
        return True

    def undo(self):
        self.break_merge()
        self.history.undo()

    def redo(self):
        self.break_merge()
        self.history.redo()

    def eventFilter(self, obj, event):
        # Text/spin-box editors otherwise consume Ctrl+Z with their local history.
        if isinstance(obj, QtWidgets.QWidget) and obj.window() == self and event.type() in (QtCore.QEvent.Type.ShortcutOverride, QtCore.QEvent.Type.KeyPress):
            focus = QtWidgets.QApplication.focusWidget()
            editing = isinstance(focus, (QtWidgets.QPlainTextEdit, QtWidgets.QTextEdit, QtWidgets.QLineEdit, QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox))
            if event.key() == Qt.Key.Key_Space and event.modifiers() == Qt.KeyboardModifier.NoModifier and not editing:
                event.accept()
                if event.type() == QtCore.QEvent.Type.KeyPress and not event.isAutoRepeat():
                    self.toggle_play()
                return True
            undo = event.matches(QtGui.QKeySequence.StandardKey.Undo)
            redo = event.matches(QtGui.QKeySequence.StandardKey.Redo) or (event.key() == Qt.Key.Key_Y and event.modifiers() == Qt.KeyboardModifier.ControlModifier)
            if undo or redo:
                event.accept()
                if event.type() == QtCore.QEvent.Type.KeyPress:
                    self.undo() if undo else self.redo()
                return True
        return super().eventFilter(obj, event)

    def build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        main = QtWidgets.QHBoxLayout()
        layout.addLayout(main, 1)
        main.addWidget(self.canvas, 1)
        panel = QtWidgets.QWidget()
        panel.setMaximumWidth(360)
        side = QtWidgets.QVBoxLayout(panel)
        main.addWidget(panel)
        history_row = QtWidgets.QHBoxLayout()
        self.undo_button = self.button("元に戻す", self.undo)
        self.redo_button = self.button("やり直す", self.redo)
        self.undo_button.setToolTip("Ctrl+Z")
        self.redo_button.setToolTip("Ctrl+Shift+Z / Ctrl+Y")
        for button in (self.undo_button, self.redo_button):
            button.setEnabled(False)
            history_row.addWidget(button)
        self.history.canUndoChanged.connect(self.undo_button.setEnabled)
        self.history.canRedoChanged.connect(self.redo_button.setEnabled)
        side.addLayout(history_row)
        self.list = QtWidgets.QListWidget()
        self.list.setMaximumHeight(130)
        self.list.currentRowChanged.connect(self.select)
        side.addWidget(self.list)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.button("＋ 追加", self.add_text))
        self.duplicate_button = self.button("複製", lambda: self.add_text(duplicate=True))
        self.duplicate_button.setToolTip("選択中の字幕を、装飾・位置・表示時間ごと複製")
        row.addWidget(self.duplicate_button)
        row.addWidget(self.button("削除", self.delete_text))
        side.addLayout(row)
        self.editor = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(self.editor)
        form.setContentsMargins(0, 0, 0, 0)
        self.copy = QtWidgets.QPlainTextEdit()
        self.copy.setUndoRedoEnabled(False)
        self.copy.setMaximumHeight(100)
        self.copy.textChanged.connect(lambda: self.edit_text("text"))
        form.addRow("テキスト", self.copy)
        self.font = QtWidgets.QFontComboBox()
        self.font.currentFontChanged.connect(lambda: self.edit_text("font"))
        form.addRow("フォント", self.font)
        self.fields = {}
        for key, title, lo, hi, decimals in (("size", "サイズ (px)", 8, 400, 0), ("outline", "縁取り (px)", 0, 30, 0), ("x", "横位置 (%)", 0, 100, 2), ("y", "縦位置 (%)", 0, 100, 2), ("start", "開始 (秒)", 0, self.info.duration, 3), ("end", "終了 (秒)", 0, self.info.duration, 3)):
            field = QtWidgets.QDoubleSpinBox()
            field.setDecimals(decimals)
            field.setRange(lo, hi)
            field.setSingleStep(0.1 if key in ("start", "end") else 1)
            field.valueChanged.connect(lambda value, key=key: self.edit_text(key))
            self.fields[key] = field
            form.addRow(title, field)
        self.color_buttons = {}
        for key, title in (("color", "文字色"), ("outline_color", "縁取り色"), ("background", "背景色・透明度")):
            button = self.button(title, lambda checked=False, key=key: self.choose_color(key))
            self.color_buttons[key] = button
            form.addRow(button)
        transparency_row = QtWidgets.QHBoxLayout()
        self.background_transparency = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.background_transparency.setRange(0, 100)
        self.background_transparency.setAccessibleName("背景の透明度")
        self.background_transparency.setToolTip("0%：不透明 ／ 100%：完全透明")
        self.background_transparency.sliderPressed.connect(self.break_merge)
        self.background_transparency.sliderReleased.connect(self.break_merge)
        self.background_transparency.valueChanged.connect(self.change_background_transparency)
        self.transparency_value = QtWidgets.QLabel()
        self.transparency_value.setMinimumWidth(42)
        transparency_row.addWidget(self.background_transparency, 1)
        transparency_row.addWidget(self.transparency_value)
        form.addRow("背景の透明度", transparency_row)
        form.addRow(self.button("左右中央に揃える", lambda: self.fields["x"].setValue(50)))
        side.addWidget(self.editor)
        self.warning = QtWidgets.QLabel()
        self.warning.setWordWrap(True)
        side.addWidget(self.warning)
        side.addStretch()
        side.addWidget(self.button("設定を保存", self.save))
        side.addWidget(self.button("設定を読み込む…", self.load))
        self.export_button = self.button("MP4を書き出す…", self.start_export)
        side.addWidget(self.export_button)
        self.cancel_button = self.button("書き出し中止", self.cancel_export)
        self.cancel_button.setEnabled(False)
        side.addWidget(self.cancel_button)
        controls = QtWidgets.QHBoxLayout()
        self.play_button = self.button("▶ 再生", self.toggle_play)
        self.play_button.setToolTip("Space：再生／一時停止（文字入力中を除く）")
        controls.addWidget(self.play_button)
        self.slider = SeekSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, round(self.info.duration * 1000))
        self.slider.valueChanged.connect(self.seek)
        controls.addWidget(self.slider, 1)
        self.clock = QtWidgets.QLabel()
        controls.addWidget(self.clock)
        mute = QtWidgets.QCheckBox("ミュート")
        mute.toggled.connect(self.audio.setMuted)
        controls.addWidget(mute)
        layout.addLayout(controls)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.hide()
        layout.addWidget(self.progress)
        QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Save, self, activated=self.save)
        self.position_changed(0)

    def refresh(self):
        # Regenerate at edit time, not during each video frame.
        self.canvas.sprites = [render_text(item) for item in self.texts]
        self.list.blockSignals(True)
        selected = self.canvas.index
        self.list.clear()
        for i, item in enumerate(self.texts):
            self.list.addItem(f"T{i + 1}  {item.start:.2f}–{item.end:.2f}s  {item.text.replace(chr(10), ' ')[:24]}")
        self.list.setCurrentRow(selected)
        self.list.blockSignals(False)
        self.canvas.update()
        self.update_warning()

    def select(self, index):
        if index != self.canvas.index:
            self.break_merge()
        self.canvas.index = index
        self.editor.setEnabled(index >= 0)
        self.duplicate_button.setEnabled(index >= 0)
        if 0 <= index < len(self.texts):
            self.loading = True
            item = self.texts[index]
            self.copy.setPlainText(item.text)
            self.font.setCurrentFont(QtGui.QFont(item.font))
            for key, field in self.fields.items():
                field.setValue(getattr(item, key) * (100 if key in ("x", "y") else 1))
            self.list.setCurrentRow(index)
            self.loading = False
            self.update_colors()
        self.canvas.update()
        self.update_warning()

    def edit_text(self, field):
        if self.loading or self.canvas.index < 0:
            return
        item = self.texts[self.canvas.index]
        before = self.snapshot()
        if field == "text":
            item.text = self.copy.toPlainText()
        elif field == "font":
            item.font = self.font.currentFont().family()
        else:
            for key in (("x", "y") if field == "drag" else (field,)):
                value = self.fields[key].value() / (100 if key in ("x", "y") else 1)
                setattr(item, key, int(value) if key in ("size", "outline") else value)
        self.commit(before, "文字を編集", field)

    def move_text(self, x, y):
        self.loading = True
        self.fields["x"].setValue(x * 100)
        self.fields["y"].setValue(y * 100)
        self.loading = False
        self.edit_text("drag")

    def update_colors(self):
        item = self.texts[self.canvas.index]
        for key, button in self.color_buttons.items():
            color = QtGui.QColor(getattr(item, key))
            pixmap = QtGui.QPixmap(20, 20)
            pixmap.fill(color)
            button.setIcon(QtGui.QIcon(pixmap))
            button.setToolTip(color.name(QtGui.QColor.NameFormat.HexArgb))
        transparency = round((255 - QtGui.QColor(item.background).alpha()) * 100 / 255)
        blocker = QtCore.QSignalBlocker(self.background_transparency)
        self.background_transparency.setValue(transparency)
        del blocker
        self.transparency_value.setText(f"{transparency}%")

    def change_background_transparency(self, value):
        if self.loading or self.canvas.index < 0:
            return
        before = self.snapshot()
        item = self.texts[self.canvas.index]
        color = QtGui.QColor(item.background)
        color.setAlpha(round((100 - value) * 255 / 100))
        item.background = color.name(QtGui.QColor.NameFormat.HexArgb)
        field = "drag" if self.background_transparency.isSliderDown() else "background_alpha"
        if self.commit(before, "背景の透明度を変更", field):
            self.update_colors()

    def choose_color(self, key):
        item = self.texts[self.canvas.index]
        before = self.snapshot()
        self.break_merge()
        dialog = QtWidgets.QColorDialog(QtGui.QColor(getattr(item, key)), self)
        dialog.setWindowTitle("色を選択")
        dialog.setOptions(QtWidgets.QColorDialog.ColorDialogOption.ShowAlphaChannel | QtWidgets.QColorDialog.ColorDialogOption.DontUseNativeDialog)

        def preview(color):
            setattr(item, key, color.name(QtGui.QColor.NameFormat.HexArgb))
            self.refresh()
            self.update_colors()

        dialog.currentColorChanged.connect(preview)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.record(before, "色を変更")
        else:
            self.restore(before)
        dialog.deleteLater()

    def update_warning(self):
        messages = []
        for item, sprite in zip(self.texts, self.canvas.sprites):
            try:
                item.validate(self.info.duration)
            except ValueError as exc:
                messages.append(str(exc))
            if not QtCore.QRect(0, 0, self.info.width, self.info.height).contains(text_rect(item, sprite, self.info)):
                messages.append("画面外の文字は切り取られます")
        if self.canvas.index >= 0:
            item = self.texts[self.canvas.index]
            if not item.start <= self.canvas.seconds < item.end:
                messages.append("選択中の文字は現在の再生位置では非表示です")
        self.warning.setText("\n".join(dict.fromkeys(messages)))

    def add_text(self, checked=False, *, duplicate=False):
        if duplicate and self.canvas.index < 0:
            return
        if len(self.texts) >= 100:
            self.show_error("テキストは最大100件です")
            return
        before = self.snapshot()
        item = Text(**asdict(self.texts[self.canvas.index])) if duplicate else Text(end=self.info.duration)
        self.texts.append(item)
        self.refresh()
        self.select(len(self.texts) - 1)
        self.record(before, "字幕を複製" if duplicate else "文字を追加")

    def delete_text(self):
        index = self.canvas.index
        if index >= 0:
            before = self.snapshot()
            self.texts.pop(index)
            self.canvas.index = -1
            self.refresh()
            self.select(min(index, len(self.texts) - 1))
            self.record(before, "文字を削除")

    def frame_changed(self, frame):
        if frame.isValid():
            # QVideoFrame.toImage applies the decoder's display rotation.
            self.canvas.frame = frame.toImage()
            self.canvas.seconds = max(0, frame.startTime() / 1_000_000)
            self.canvas.update()
            self.update_warning()

    def position_changed(self, position):
        self.slider.blockSignals(True)
        self.slider.setValue(position)
        self.slider.blockSignals(False)
        self.clock.setText(f"{position / 1000:05.2f} / {self.info.duration:.2f} 秒")

    def seek(self, value):
        self.player.setPosition(value)

    def playback_changed(self, state):
        self.play_button.setText("⏸ 一時停止" if state == QMediaPlayer.PlaybackState.PlayingState else "▶ 再生")

    def toggle_play(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            if self.player.mediaStatus() == QMediaPlayer.MediaStatus.EndOfMedia:
                self.player.setPosition(0)
            self.player.play()

    def show_error(self, message):
        logger.error(message)
        QtWidgets.QMessageBox.warning(self, "エラー", message)

    def save(self):
        try:
            save_settings(self.settings, self.video, self.info, self.texts)
            self.break_merge()
            self.history.setClean()
            self.statusBar().showMessage(f"設定を保存しました: {self.settings}")
            return True
        except (ValueError, OSError) as exc:
            self.show_error(str(exc))
            return False

    def discard_changes(self):
        if not self.dirty:
            return True
        choice = QtWidgets.QMessageBox.question(self, "未保存の変更", "編集内容を保存しますか？", QtWidgets.QMessageBox.StandardButton.Save | QtWidgets.QMessageBox.StandardButton.Discard | QtWidgets.QMessageBox.StandardButton.Cancel, QtWidgets.QMessageBox.StandardButton.Save)
        return choice == QtWidgets.QMessageBox.StandardButton.Discard or (choice == QtWidgets.QMessageBox.StandardButton.Save and self.save())

    def load(self):
        if not self.discard_changes():
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "設定を読み込む", str(self.settings), "JSON (*.json)")
        if path:
            try:
                texts = load_settings(Path(path), self.video, self.info)
                # Validate renderability before replacing current state.
                for item in texts:
                    render_text(item)
                self.texts[:] = texts
                self.settings = Path(path)
                self.break_merge()
                self.history.clear()
                self.canvas.index = -1
                self.refresh()
                self.select(0 if texts else -1)
            except (OSError, ValueError) as exc:
                self.show_error(str(exc))

    def start_export(self):
        self.player.pause()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "新しいMP4として保存", str(self.output), "MP4 (*.mp4)", options=QtWidgets.QFileDialog.Option.DontConfirmOverwrite)
        if not path:
            return
        try:
            self.export = Export(self.video, Path(path), self.info, self.texts)
        except (ValueError, OSError) as exc:
            self.show_error(str(exc))
            return
        self.process = QtCore.QProcess(self)
        self.process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_progress)
        self.process.finished.connect(self.export_finished)
        self.process.errorOccurred.connect(self.export_error)
        self.cancelled = False
        self.buffer = self.errors = ""
        self.export_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setValue(0)
        self.progress.show()
        log_command(self.export.command)
        self.statusBar().showMessage("書き出し中（開始時点の編集内容を使用）")
        self.process.start(self.export.command[0], self.export.command[1:])

    def read_progress(self):
        self.buffer += bytes(self.process.readAllStandardOutput()).decode("utf-8", "replace")
        lines = self.buffer.split("\n")
        self.buffer = lines.pop()
        for line in lines:
            if line.startswith("out_time_us="):
                try:
                    self.progress.setValue(min(1000, round(int(line.split("=", 1)[1]) / self.info.duration / 1000)))
                except ValueError:
                    pass
            else:
                self.errors = (self.errors + line + "\n")[-6000:]

    def export_error(self, error):
        if error == QtCore.QProcess.ProcessError.FailedToStart:
            self.errors = self.process.errorString()
            self.export_finished(1)

    def export_finished(self, code, *args):
        if self.export is None:
            return
        self.read_progress()
        error = None
        try:
            if self.cancelled:
                self.statusBar().showMessage("書き出しを中止しました")
            elif code == 0:
                self.export.publish()
                self.progress.setValue(1000)
                self.output = self.export.output
                self.statusBar().showMessage(f"書き出しました: {self.output}")
            else:
                error = "書き出しに失敗しました\n" + self.errors + self.buffer
        except OSError as exc:
            error = str(exc)
        finally:
            self.export.cleanup()
            self.export = None
            self.process.deleteLater()
            self.process = None
            self.export_button.setEnabled(True)
            self.cancel_button.setEnabled(False)
        if error:
            self.show_error(error)

    def cancel_export(self):
        if self.process:
            self.cancelled = True
            self.process.kill()

    def closeEvent(self, event):
        if self.export:
            choice = QtWidgets.QMessageBox.question(self, "書き出し中", "書き出しを中止して閉じますか？", QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No, QtWidgets.QMessageBox.StandardButton.No)
            if choice != QtWidgets.QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.cancel_export()
            if self.process:
                self.process.waitForFinished(3000)
        if not self.discard_changes():
            event.ignore()
            return
        self.player.stop()
        event.accept()


def main():
    parser = argparse.ArgumentParser(formatter_class=ArgumentDefaultsRawTextHelpFormatter, epilog=epilog)
    parser.add_argument("-q", "--quiet", action="count", default=0)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("edit", "encode"):
        sub = subparsers.add_parser(action, formatter_class=ArgumentDefaultsRawTextHelpFormatter)
        sub.add_argument("input", type=Path)
        sub.add_argument("--settings", type=Path, help="既定: 作業ディレクトリ/input.text.json")
        sub.add_argument("-o", "--output", type=Path, help="既定: 作業ディレクトリ/input_text.mp4")
        sub.add_argument("-n", "--dry_run", "--dry-run", action="store_true", help="コマンドを表示するだけで実行しない")
    args = parser.parse_args()
    logger.setLevel({0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}.get(args.quiet, logging.ERROR))
    video = args.input.resolve()
    settings = args.settings or Path.cwd() / f"{video.stem}.text.json"
    output = args.output or Path.cwd() / f"{video.stem}_text.mp4"
    if args.dry_run:
        command = [sys.executable, str(Path(__file__).resolve()), args.action, str(video), "--settings", str(settings), "-o", str(output)]
        print("# fish / bash")
        print(shlex.join(command))
        return 0
    try:
        for executable in ("ffprobe", "ffmpeg"):
            if not shutil.which(executable):
                raise ValueError(f"{executable} が見つかりません")
        if not video.is_file():
            raise ValueError(f"動画が見つかりません: {video}")
        info = probe(video)
        if args.action == "encode":
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
        if args.action == "edit":
            window = Window(video, info, settings, output)
            window.show()
            return app.exec()
        texts = load_settings(settings, video, info)
        export = Export(video, output, info, texts)
        try:
            log_command(export.command)
            subprocess.run(export.command, check=True)
            export.publish()
        finally:
            export.cleanup()
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
