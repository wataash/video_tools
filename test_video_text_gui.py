# SPDX-FileCopyrightText: Copyright (c) 2026 Wataru Ashihara <wataash0607@gmail.com>
# SPDX-License-Identifier: Apache-2.0
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from video_text_gui import Export, Text, VideoInfo, load_settings, log_command, probe, render_text, save_settings, text_rect


@pytest.fixture(scope="module", autouse=True)
def app():
    instance = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield instance


def run(command):
    log_command(command)
    return subprocess.run(command, check=True, capture_output=True)


def pixels(image):
    image = image.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
    return np.frombuffer(image.constBits(), dtype=np.uint8).reshape(image.height(), image.width(), 4).copy()


def test_japanese_outline_and_multiline():
    item = Text(text="日本語\nテスト ' : % \\", size=40)
    image = render_text(item)
    rgba = pixels(image)
    assert image.height() > 80
    assert np.count_nonzero(np.all(rgba == [255, 255, 255, 255], axis=2)) > 100
    assert np.count_nonzero(np.all(rgba == [0, 0, 0, 255], axis=2)) > 100
    assert rgba[0, 0, 3] == 0


@pytest.mark.parametrize("text", ["😅", "日本語 😅\n👩‍💻 🇯🇵", "👍🏽"])
def test_color_emoji_renders_with_outline(text):
    image = render_text(Text(text=text, size=64, outline=3))
    rgba = pixels(image).astype(int)
    colored = (rgba[:, :, :3].max(axis=2) - rgba[:, :, :3].min(axis=2) > 50) & (rgba[:, :, 3] > 200)
    assert np.count_nonzero(colored) > 200
    assert rgba[0, 0, 3] == 0


def test_settings_round_trip_and_reject_wrong_video(tmp_path):
    video = tmp_path / "in.mp4"
    video.write_bytes(b"original")
    info = VideoInfo(720, 1280, 26.089)
    texts = [Text(text="日本語\n二行目", end=info.duration), Text(text="2", start=2, end=4, background="#80000000")]
    path = tmp_path / "settings.json"
    save_settings(path, video, info, texts)
    assert load_settings(path, video, info) == texts
    with pytest.raises(ValueError, match="別の動画"):
        load_settings(path, tmp_path / "other.mp4", info)
    with pytest.raises(ValueError):
        save_settings(video, video, info, texts)
    assert video.read_bytes() == b"original"
    bad = json.loads(path.read_text())
    bad["texts"][0]["end"] = float("nan")
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        load_settings(path, video, info)


@pytest.mark.parametrize("changes", [{"end": 0}, {"start": 3, "end": 2}, {"x": float("inf")}, {"color": "no-such-color"}, {"size": 30.5}])
def test_invalid_text(changes):
    with pytest.raises(ValueError):
        Text(**changes).validate(5)


def test_export_refuses_existing_output_and_cleans_failure(tmp_path):
    output = tmp_path / "out.mp4"
    output.write_bytes(b"keep")
    with pytest.raises(ValueError):
        Export(tmp_path / "in.mp4", output, VideoInfo(100, 100, 1), [])
    assert output.read_bytes() == b"keep"
    output.unlink()
    export = Export(tmp_path / "in.mp4", output, VideoInfo(100, 100, 1), [])
    export.partial.write_bytes(b"new")
    output.write_bytes(b"race")
    with pytest.raises(FileExistsError):
        export.publish()
    temp = Path(export.temp.name)
    export.cleanup()
    assert not temp.exists()
    assert output.read_bytes() == b"race"


def test_rotated_video_timing_pixel_placement_and_audio(tmp_path):
    raw = tmp_path / "raw.mp4"
    video = tmp_path / "rotated.mp4"
    output = tmp_path / "out.mp4"
    run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=320x180:r=10:d=2", "-f", "lavfi", "-i", "sine=frequency=440:duration=2.1", "-vf", "settb=1/1000,setpts=PTS+if(gte(N\\,10)\\,30\\,0)", "-fps_mode", "passthrough", "-enc_time_base", "filter", "-c:v", "libx264", "-c:a", "aac", "-shortest", str(raw)])
    run(["ffmpeg", "-v", "error", "-display_rotation", "90", "-i", str(raw), "-c", "copy", str(video)])
    info = probe(video)
    assert (info.width, info.height) == (180, 320)
    texts = [Text(text="日本語😅", size=24, x=0.35, y=0.3, start=0.5, end=1.0), Text(text="二行目\n' : %", size=20, y=0.7, start=1.0, end=1.5)]
    export = Export(video, output, info, texts, crf=12)
    try:
        run(export.command)
        export.publish()
    finally:
        export.cleanup()
    out_info = probe(output)
    assert (out_info.width, out_info.height) == (180, 320)
    decoded = run(["ffmpeg", "-v", "error", "-i", str(output), "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]).stdout
    frames = np.frombuffer(decoded, dtype=np.uint8).reshape(-1, 320, 180, 3)
    assert len(frames) == 20
    timestamps = []
    for path in (video, output):
        data = json.loads(run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames", "-show_entries", "frame=pts_time", "-of", "json", str(path)]).stdout)
        timestamps.append([float(frame["pts_time"]) for frame in data["frames"]])
    assert timestamps[0] == pytest.approx(timestamps[1], abs=0.0001)
    for index in (0, 4, 15, 19):
        assert frames[index].max() < 8
    for index, item in ((5, texts[0]), (9, texts[0]), (10, texts[1]), (14, texts[1])):
        expected = QtGui.QImage(180, 320, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
        expected.fill(QtGui.QColor("black"))
        sprite = render_text(item)
        painter = QtGui.QPainter(expected)
        painter.drawImage(text_rect(item, sprite, info).topLeft(), sprite)
        painter.end()
        expected = pixels(expected)[:, :, :3]
        assert frames[index].max() > 200
        assert np.abs(frames[index].astype(float) - expected).mean() < 2
    for path in (video, output):
        audio = run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-c", "copy", "-f", "md5", "pipe:1"]).stdout
        if path == video:
            original_audio = audio
        else:
            assert audio == original_audio


@pytest.fixture
def editor(tmp_path, app):
    from video_text_gui import Window
    video = tmp_path / "editor.mp4"
    run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=180x320:r=10:d=1", "-c:v", "libx264", str(video)])
    window = Window(video, VideoInfo(180, 320, 1), tmp_path / "editor.json", tmp_path / "out.mp4")
    window.show()
    app.processEvents()
    yield window
    window.discard_changes = lambda: True
    window.close()
    window.deleteLater()
    app.processEvents()


def test_undo_typing_shortcuts_and_redo_branch(editor):
    from PySide6 import QtTest
    w = editor
    original = w.texts[0].text
    w.copy.setFocus()
    w.copy.selectAll()
    QtTest.QTest.keyClicks(w.copy, "abc")
    assert w.texts[0].text == "abc"
    assert w.history.count() == 1
    QtTest.QTest.keyClick(w.copy, QtCore.Qt.Key.Key_Z, QtCore.Qt.KeyboardModifier.ControlModifier)
    assert w.texts[0].text == original
    assert not w.dirty
    QtTest.QTest.keyClick(w.copy, QtCore.Qt.Key.Key_Z, QtCore.Qt.KeyboardModifier.ControlModifier | QtCore.Qt.KeyboardModifier.ShiftModifier)
    assert w.texts[0].text == "abc"
    w.undo()
    w.copy.setPlainText("別の編集")
    assert not w.history.canRedo()
    assert w.undo_button.isEnabled()
    assert not w.redo_button.isEnabled()


def test_undo_drag_add_delete_and_selection(editor):
    w = editor
    original = w.snapshot()
    w.canvas.drag_started.emit()
    w.move_text(0.2, 0.3)
    w.move_text(0.3, 0.4)
    w.canvas.drag_finished.emit()
    assert w.history.count() == 1
    w.canvas.drag_started.emit()
    w.move_text(0.7, 0.6)
    w.canvas.drag_finished.emit()
    assert w.history.count() == 2
    w.undo()
    assert (w.texts[0].x, w.texts[0].y) == (0.3, 0.4)
    w.undo()
    assert w.snapshot() == original
    w.redo()
    w.add_text()
    assert w.canvas.index == 1
    w.copy.setPlainText("追加した文字")
    added = w.snapshot()
    w.delete_text()
    w.undo()
    assert w.snapshot() == added
    assert w.copy.toPlainText() == "追加した文字"
    w.undo()
    w.undo()
    assert len(w.texts) == 1 and w.canvas.index == 0
    w.delete_text()
    assert not w.texts and w.canvas.index == -1
    w.undo()
    assert len(w.texts) == 1 and w.editor.isEnabled()


def test_undo_properties_color_and_saved_state(editor, monkeypatch):
    from PySide6 import QtTest
    w = editor
    w.fields["size"].setValue(70)
    w.fields["size"].setValue(72)
    w.fields["end"].setValue(0.8)
    w.undo()
    assert w.texts[0].end == 1 and w.texts[0].size == 72
    # A focused spin-box must not consume Undo in its internal line editor.
    field = w.fields["size"].lineEdit()
    field.setFocus()
    QtTest.QTest.keyClick(field, QtCore.Qt.Key.Key_Z, QtCore.Qt.KeyboardModifier.ControlModifier)
    assert w.texts[0].size == 64
    QtTest.QTest.keyClick(field, QtCore.Qt.Key.Key_Y, QtCore.Qt.KeyboardModifier.ControlModifier)
    assert w.texts[0].size == 72
    assert w.save() and not w.dirty
    w.fields["size"].setValue(80)
    w.undo()
    assert not w.dirty and w.texts[0].size == 72
    w.undo()
    assert w.dirty
    w.redo()
    assert not w.dirty
    def pick_red(dialog):
        dialog.setCurrentColor(QtGui.QColor("red"))
        return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(QtWidgets.QColorDialog, "exec", pick_red)
    old_color = w.texts[0].color
    w.choose_color("color")
    w.undo()
    assert w.texts[0].color == old_color
    w.redo()
    assert w.texts[0].color == "#ffff0000"


def test_load_clears_history_and_noop_does_not_dirty(editor, monkeypatch):
    w = editor
    w.edit_text("text")
    assert not w.dirty and not w.history.canUndo()
    assert w.save()
    w.copy.setPlainText("changed")
    monkeypatch.setattr(w, "discard_changes", lambda: True)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", lambda *a: (str(w.settings), "JSON"))
    w.load()
    assert not w.dirty
    assert not w.history.canUndo() and not w.history.canRedo()
    assert w.texts[0].text == "ここにテキスト"


@pytest.mark.parametrize("key", ["color", "outline_color", "background"])
@pytest.mark.parametrize("accepted", [True, False])
def test_live_color_preview_commit_or_cancel(editor, monkeypatch, key, accepted):
    w = editor
    w.fields["size"].setValue(70)
    w.undo()
    before = w.snapshot()
    sprite = w.canvas.sprites[0].copy()
    count = w.history.count()

    def pick(dialog):
        for color in ("#ffff0000", "#8000ff00"):
            dialog.setCurrentColor(QtGui.QColor(color))
            assert getattr(w.texts[0], key) == color
            assert w.canvas.sprites[0] != sprite
            assert w.color_buttons[key].toolTip() == color
            assert w.history.count() == count
            assert not w.dirty
        return QtWidgets.QDialog.DialogCode.Accepted if accepted else QtWidgets.QDialog.DialogCode.Rejected

    monkeypatch.setattr(QtWidgets.QColorDialog, "exec", pick)
    w.choose_color(key)
    if accepted:
        assert w.dirty and not w.history.canRedo()
        assert getattr(w.texts[0], key) == "#8000ff00"
        w.undo()
        assert w.snapshot() == before and not w.dirty
        w.redo()
        assert getattr(w.texts[0], key) == "#8000ff00"
    else:
        assert w.snapshot() == before
        assert w.canvas.sprites[0] == sprite
        assert not w.dirty and w.history.canRedo()


def test_background_transparency_slider_live_undo_and_rgb(editor, monkeypatch):
    w = editor
    slider = w.background_transparency
    assert slider.value() == 100
    assert w.transparency_value.text() == "100%"

    def pick(dialog):
        dialog.setCurrentColor(QtGui.QColor("#80336699"))
        assert slider.value() == 50
        return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(QtWidgets.QColorDialog, "exec", pick)
    w.choose_color("background")
    count = w.history.count()
    slider.setSliderDown(True)
    for value in (30, 20, 0):
        slider.setValue(value)
        color = QtGui.QColor(w.texts[0].background)
        assert color.name() == "#336699"
        assert color.alpha() == round((100 - value) * 255 / 100)
        assert w.canvas.sprites[0].pixelColor(0, 0).alpha() == color.alpha()
        assert w.transparency_value.text() == f"{value}%"
    slider.setSliderDown(False)
    assert w.history.count() == count + 1
    w.undo()
    assert w.texts[0].background == "#80336699" and slider.value() == 50
    w.redo()
    assert slider.value() == 0
    slider.setSliderDown(True)
    slider.setValue(100)
    slider.setSliderDown(False)
    assert QtGui.QColor(w.texts[0].background).alpha() == 0
    w.undo()
    assert slider.value() == 0
    w.add_text()
    assert slider.value() == 100
    w.select(0)
    assert slider.value() == 0


def test_duplicate_subtitle_independent_edit_and_undo(editor):
    w = editor
    w.copy.setPlainText('日本語 😅\n複製元')
    w.fields['size'].setValue(48)
    w.fields['x'].setValue(35)
    w.fields['start'].setValue(0.2)
    w.background_transparency.setValue(40)
    original = w.snapshot()
    count = w.history.count()
    w.duplicate_button.click()
    assert len(w.texts) == 2 and w.canvas.index == 1
    assert w.texts[0] == w.texts[1] and w.texts[0] is not w.texts[1]
    assert w.history.count() == count + 1
    w.undo()
    assert w.snapshot() == original
    w.redo()
    assert len(w.texts) == 2 and w.canvas.index == 1
    w.copy.setPlainText('複製先だけ変更')
    assert w.texts[0] == original[0][0]
    assert w.texts[1].text == '複製先だけ変更'
    w.undo()
    w.undo()
    w.delete_text()
    assert not w.duplicate_button.isEnabled()
    w.add_text(duplicate=True)
    assert not w.texts


def test_space_playback_and_text_input(editor, monkeypatch):
    from PySide6 import QtTest
    w = editor
    toggles = []
    monkeypatch.setattr(w, 'toggle_play', lambda: toggles.append(True))
    w.canvas.setFocus()
    QtTest.QTest.keyClick(w.canvas, QtCore.Qt.Key.Key_Space)
    QtTest.QTest.keyClick(w.canvas, QtCore.Qt.Key.Key_Space)
    assert len(toggles) == 2
    repeat = QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, QtCore.Qt.Key.Key_Space, QtCore.Qt.KeyboardModifier.NoModifier, ' ', True)
    QtWidgets.QApplication.sendEvent(w.canvas, repeat)
    assert len(toggles) == 2
    w.copy.setFocus()
    w.copy.selectAll()
    QtTest.QTest.keyClicks(w.copy, 'a b')
    assert w.texts[0].text == 'a b' and len(toggles) == 2
    QtTest.QTest.mouseClick(w.canvas, QtCore.Qt.MouseButton.LeftButton, pos=QtCore.QPoint(5, 5))
    assert w.canvas.hasFocus()
    QtTest.QTest.keyClick(w.canvas, QtCore.Qt.Key.Key_Space)
    assert len(toggles) == 3
    # Space on an action button must play, rather than activate that action.
    w.duplicate_button.setFocus()
    QtTest.QTest.keyClick(w.duplicate_button, QtCore.Qt.Key.Key_Space)
    assert len(toggles) == 4 and len(w.texts) == 1
