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
from video_blur_mask_gui import PaintWindow, TimeOrFrame, logger


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
