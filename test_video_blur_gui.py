# SPDX-FileCopyrightText: Copyright (c) 2026 Wataru Ashihara <wataash0607@gmail.com>
# SPDX-License-Identifier: Apache-2.0
import os
import shlex
import subprocess
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
from PySide6 import QtCore, QtWidgets, QtTest
from PySide6.QtMultimedia import QMediaPlayer
from video_blur_mask_gui import MaskMotion, PaintWindow, TimeOrFrame, default_motion_path, logger


@pytest.fixture(scope='module')
def app():
    instance = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield instance


@pytest.fixture
def blur_window(tmp_path, app, request):
    video = tmp_path / 'video.mp4'
    command = ['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=s=640x360:r=30:d=5']
    if getattr(request, 'param', True):
        command += ['-f', 'lavfi', '-i', 'sine=frequency=440:duration=5', '-c:a', 'aac']
    command += ['-c:v', 'libx264', '-preset', 'ultrafast', str(video)]
    logger.info('fish / bash: %s', shlex.join(command))
    subprocess.run(command, check=True, capture_output=True)
    w = PaintWindow(video, tmp_path / 'mask.png', tmp_path / 'output.mp4', TimeOrFrame('frame', 0), 30, 0, 18, 'medium')
    w.show()
    QtTest.QTest.qWait(250)
    yield w
    w.close()
    w.deleteLater()
    app.processEvents()


def test_blur_realtime_audio_pause_seek_and_speed(blur_window):
    w = blur_window
    assert w.speed == 1.0 and w.speed_box.currentData() == 1.0
    assert w.player.audioOutput() == w.audio and not w.audio.isMuted()
    w.toggle_playback()
    QtTest.QTest.qWait(1150)
    assert w.player.hasAudio()
    assert w.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
    assert 0.8 < w.current_seconds < 1.6
    assert abs(w.current_seconds - w.player.position() / 1000) < 0.2
    w.stop_playback()
    paused = w.player.position()
    QtTest.QTest.qWait(150)
    assert w.player.position() == paused and not w.playback_timer.isActive()
    w.goto_frame(15)
    assert w.current_frame_index == 15
    assert abs(w.player.position() - 500) < 40
    w.step_frame(1)
    assert w.current_frame_index == 16
    w.mute_box.setChecked(True)
    assert w.audio.isMuted()
    w.speed_box.setCurrentIndex(1)
    assert w.player.playbackRate() == 0.5
    w.toggle_playback()
    start = w.player.position()
    QtTest.QTest.qWait(800)
    assert 250 < w.player.position() - start < 600


@pytest.mark.parametrize('blur_window', [False], indirect=True)
def test_slow_preview_skips_frames_without_slowing_media(blur_window, monkeypatch):
    w = blur_window
    render = w.render_frame
    rendered = []
    def slow_render():
        time.sleep(0.075)  # Deliberately slower than the 40 ms preview interval.
        render()
        rendered.append(w.current_seconds)
    monkeypatch.setattr(w, 'render_frame', slow_render)
    start = time.monotonic()
    w.toggle_playback()
    QtTest.QTest.qWait(1200)
    elapsed = time.monotonic() - start
    assert not w.player.hasAudio()
    assert w.player.position() / 1000 > elapsed * 0.7
    assert w.current_seconds > elapsed * 0.6
    assert len(rendered) >= 3
    assert any(b - a > 0.06 for a, b in zip(rendered, rendered[1:]))
    w.stop_playback()
    w.goto_frame(w.frame_count - 1)
    w.toggle_playback()
    QtTest.QTest.qWait(300)
    assert w.current_seconds < 1  # Play at the end restarts from the beginning.


@pytest.mark.parametrize('blur_window', [False], indirect=True)
def test_mask_keyframe_controls_preview_and_persistence(blur_window):
    w = blur_window
    w.painter.begin_stroke()
    w.painter.stroke(80, 120, 140, 150, radius=12, hardness=0.7, erase=False)
    w.goto_frame(30)
    w.add_motion_keyframe()
    anchor_x, anchor_y = w.motion.anchor('A')
    assert [key.frame for key in w.motion.keyframes] == [30]
    w.goto_frame(90)
    w.add_motion_keyframe()
    w.motion_key_table.cellWidget(1, 3).setValue(round(anchor_x + 120))
    w.motion_key_table.cellWidget(1, 4).setValue(round(anchor_y - 30))
    w.motion_key_table.cellWidget(1, 5).setValue(1.5)
    assert [key.frame for key in w.motion.keyframes] == [30, 90]
    assert w.motion.transform_at_frame(60) == (anchor_x + 60, anchor_y - 15, 1.25)
    _mask, bbox = w.scaled_mask(w.video_w, w.video_h)
    assert bbox is not None and bbox[0] > 150
    w.goto_frame(120)
    w.add_motion_keyframe()
    assert w.motion.transform_at_frame(105) == (anchor_x + 120, anchor_y - 30, 1.5)
    w.goto_frame(135)
    w.add_motion_keyframe()
    w.motion_key_table.cellWidget(3, 3).setValue(round(anchor_x - 60))
    assert w.motion.transform_at_frame(127.5) == (anchor_x + 30, anchor_y - 30, 1.5)
    w.motion_key_table.selectRow(2)
    w.delete_motion_keyframe()
    assert [key.frame for key in w.motion.keyframes] == [30, 90, 135]
    w.save_state()
    loaded = MaskMotion.load(default_motion_path(w.mask_path), w.video_w, w.video_h, w.fps)
    assert [key.frame for key in loaded.keyframes] == [30, 90, 135]
    assert loaded.transform_at_frame(60) == (anchor_x + 60, anchor_y - 15, 1.25)


@pytest.mark.parametrize('blur_window', [False], indirect=True)
def test_multiple_painted_shapes_and_keyframe_table(blur_window):
    w = blur_window
    w.painter.begin_stroke()
    w.painter.stroke(80, 120, 140, 150, radius=12, hardness=0.7, erase=False)
    w.goto_frame(30)
    w.add_motion_keyframe()
    w.add_shape()
    assert w.active_shape_id == 'B' and set(w.shape_painters) == {'A', 'B'}
    w.painter.begin_stroke()
    w.painter.stroke(400, 120, 460, 150, radius=12, hardness=0.7, erase=False)
    w.goto_frame(90)
    w.add_motion_keyframe()
    assert [key.shape_id for key in w.motion.keyframes] == ['A', 'B']
    assert w.motion_key_table.rowCount() == 3  # two keys plus the "+" row
    assert w.motion_key_table.item(1, 0).text() == '90'
    w.motion_key_table.selectRow(0)
    assert w.current_frame_index == 30
    time_item = w.motion_key_table.item(1, 1)
    time_item.setText('4')
    assert [key.frame for key in w.motion.keyframes] == [30, 120]
    assert w.motion.shape_at_frame(119) == 'A'
    assert w.motion.shape_at_frame(120) == 'B'
    w.save_state()
    loaded = MaskMotion.load(default_motion_path(w.mask_path), w.video_w, w.video_h, w.fps)
    assert loaded.shape_at_frame(120) == 'B'
    assert (w.mask_path.parent / loaded.shapes['B'].filename).exists()


@pytest.mark.parametrize('blur_window', [False], indirect=True)
def test_new_shape_stroke_is_visible_before_its_first_keyframe(blur_window):
    w = blur_window
    # A is the output shape at frame 0.  B has no keyframe yet, but its source
    # overlay must make the stroke visible at the exact brush coordinates.
    w.add_shape()
    assert w.active_shape_id == 'B'
    assert w.paint_shape_label.text() == 'Paint shape (cyan source)'
    w.painter.begin_stroke()
    w.painter.dab(320, 180, radius=20, hardness=1.0, erase=False)
    w.render_frame()

    assert w.motion.shape_at_frame(w.current_frame_index) == 'A'
    assert w.paint_shape_overlay(w.video_w, w.video_h) is not None
    bgr = w._frame_buffer[180, 320]
    assert bgr[0] > bgr[2] and bgr[1] > bgr[2]  # cyan editing guide

    w.add_motion_keyframe()
    assert w.motion.shape_at_frame(w.current_frame_index) == 'B'
    assert w.paint_shape_overlay(w.video_w, w.video_h) is None

    # Once the key moves B, the output is elsewhere.  Keep the editable
    # source visible at its untransformed brush coordinates.
    w.motion_key_table.cellWidget(0, 3).setValue(400)
    assert w.paint_shape_overlay(w.video_w, w.video_h) is not None
    assert w.paint_shape_label.text() == 'Paint shape (cyan source)'


@pytest.mark.parametrize('blur_window', [False], indirect=True)
def test_keyframe_table_shows_delete_button_only_on_selected_row(blur_window):
    w = blur_window
    w.painter.begin_stroke()
    w.painter.stroke(80, 120, 140, 150, radius=12, hardness=0.7, erase=False)
    for frame in (30, 60, 90):
        w.goto_frame(frame)
        w.add_motion_keyframe()

    w.motion_key_table.selectRow(1)
    assert w.motion_key_table.cellWidget(0, 6) is None
    delete_button = w.motion_key_table.cellWidget(1, 6).findChild(QtWidgets.QToolButton)
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)  # flush deleteLater
    assert w.motion_key_table.findChildren(QtWidgets.QToolButton) == [delete_button]
    assert delete_button.toolTip() == 'Delete keyframe at frame 60'
    QtTest.QTest.mouseClick(delete_button, QtCore.Qt.MouseButton.LeftButton)

    assert [key.frame for key in w.motion.keyframes] == [30, 90]
    assert w.motion_key_table.cellWidget(0, 6) is None
    assert w.motion_key_table.cellWidget(1, 6) is not None
    assert w.selected_motion_keyframe().frame == 90


@pytest.mark.parametrize('blur_window', [False], indirect=True)
def test_keyframes_are_marked_on_the_timeline_and_editable_in_place(blur_window):
    w = blur_window
    assert w.timeline.markers == []
    assert 'No keys' in w.motion_hint_label.text()
    w.painter.begin_stroke()
    w.painter.stroke(80, 120, 140, 150, radius=12, hardness=0.7, erase=False)
    for frame in (30, 90):
        w.goto_frame(frame)
        w.add_motion_keyframe()
    assert w.timeline.markers == [30, 90]
    assert w.motion_key_table.rowCount() == 3  # two keys plus the "+" row
    assert w.motion_key_table.cellWidget(2, 0) is w.motion_add_button
    assert not w.motion_add_button.isEnabled()  # a key already exists here
    w.goto_frame(60)
    assert w.motion_add_button.isEnabled() and 'Add key at 2.00s' in w.motion_add_button.text()
    QtTest.QTest.mouseClick(w.motion_add_button, QtCore.Qt.MouseButton.LeftButton)
    assert w.timeline.markers == [30, 60, 90]
    w.motion_key_table.selectRow(1)
    w.delete_motion_keyframe()
    assert w.timeline.markers == [30, 90]
    assert w.selected_motion_keyframe().frame == 90  # selection stays until another key is reached
    w.goto_frame(30)
    assert w.selected_motion_keyframe().frame == 30
    assert w.motion_key_table.cellWidget(0, 6) is not None

    # Spin boxes edit the model live and never rebuild the row under the cursor.
    x_box = w.motion_key_table.cellWidget(1, 3)
    x_box.setValue(x_box.value() + 7)
    assert w.motion_key_table.cellWidget(1, 3) is x_box
    assert w.motion.keyframes[1].center_x == pytest.approx(w.motion.keyframes[0].center_x + 7)
    x_box.setFocus()
    app = QtWidgets.QApplication.instance()
    app.processEvents()
    assert w.selected_motion_keyframe().frame == 90 and w.current_frame_index == 90

    w.motion_key_table.setFocus()
    QtTest.QTest.keyClick(w.motion_key_table, QtCore.Qt.Key.Key_Delete)
    assert [key.frame for key in w.motion.keyframes] == [30]
    assert w.timeline.markers == [30]
