# video_blur_mask_gui

[video_blur_mask_gui.py](video_blur_mask_gui.py)

A PySide6 (Qt) GUI that plays a video while you paint a blur mask over it with a brush, shows the
blurred result live, and finally burns the mask in with ffmpeg.

It does not track faces. You can keep a mask still, or paint several independent shapes and assign
one shape to each timing keyframe.

## Usage

```sh
./video_blur_mask_gui.py paint input.mp4                       # GUI; mask goes to ./<stem>.mask.png
./video_blur_mask_gui.py paint --time 12 --sigma 30 input.mp4  # initial seek position / blur strength
./video_blur_mask_gui.py paint --pixelize 28 input.mp4         # mosaic on top of the blur
./video_blur_mask_gui.py encode input.mp4                      # encode only, using the saved mask
./video_blur_mask_gui.py encode --print-command input.mp4      # print the ffmpeg command only
```

`--mask` sets the mask path and `-o` the output path; both default to the current directory.
`paint` prints the mask path to stdout when it exits.
`-q` is a top-level option, so it goes before the subcommand.

## Controls

- Left drag: blur pen. Right drag: eraser. Both work while the video is playing, and strokes accumulate.
- A graphics tablet works too: pen pressure scales the stroke width, and the eraser end of the stylus
  erases.
- Wheel: pen size. Shift+wheel: edge hardness. `[` and `]` also change the pen size.
- Space: play/pause. Left/Right: one frame, Shift+Left/Right: ten frames, Home/End: first/last frame.
  Click the timeline to seek. Playback speed is 0.25x / 0.5x / 1x (default: 1x).
  Audio plays at the selected speed, synchronized with the video. "Mute" toggles sound.
- `v`: cycle the view (composite -> mask tinted red -> original). The red view is the easiest way to
  spot places you missed.
- `s`: save the mask (plus a timestamped backup). Ctrl+Z / Ctrl+Y: undo / redo. `q`: quit.
- "Encode..." asks for an output path, then runs ffmpeg in the background with a progress bar.
  "Cancel" terminates it.
- Use "New shape" to create another independent brush mask, select it under "Paint shape", and draw
  it straight away. If the current timing key uses another shape, or moves/scales the selected shape,
  the editable source mask is tinted cyan. This cyan layer follows the brush exactly; it is an editing
  guide and is never encoded. Add a key to assign the painted shape to that time. Every shape is saved
  as a sibling PNG. "Delete shape" only removes an unused shape from the project; it deliberately
  leaves its PNG untouched.
- Seek to a timing point and press the "+ Add key at ..." row at the bottom of the keyframe table. A
  new row lists frame, time, shape, centre, and scale; every key is also drawn as an orange tick on the
  timeline. The Shape, X, Y, and Scale cells are edited in place (spin boxes and a drop-down) and the
  preview follows immediately. Double-click Frame or Time to retime a key; rows are re-sorted at
  once, and duplicate and negative times are rejected.
  Selecting a row seeks to that time, and stepping onto a key's frame selects its row. The selected
  row shows a trash button right after its values; that button or the Delete key removes the key.
  After removal, the next row is selected (or the preceding row at the end). The "+" row is greyed
  out while the current frame already has a key.
- Before the first key, shape A is shown. Adjacent rows using the same shape linearly interpolate
  centre and scale; equal values create a hold. When adjacent rows use different shapes, the source
  shape switches exactly at the later row's time. It does not crossfade or pretend to morph between
  different painted outlines. The preview and encoded video use the same rule.

## Mask persistence

- Autosaved every 3 seconds, and also on quit and when an encode starts.
- Writes go to `<mask>.tmp.png` and are then `os.replace()`d, so a crash cannot leave a corrupt mask.
- `s` and the start of an encode also leave a `<stem>.<YYYYmmdd-HHMMSS>.png` backup.
- An existing mask with the same name is loaded on startup. If an encode crashes, redo it with the
  `encode` subcommand instead of repainting.
- Keyframes and shape assignments are saved alongside the PNG as `<stem>.mask.motion.json`. The
  `encode` subcommand loads that file and every referenced shape PNG automatically, so it produces
  the same motion without opening the GUI. Existing version-1 and version-2 motion files are read
  and are rewritten as version 3 on save.

## Implementation notes

- The filter graph is `[0:v]split` -> blur one branch with `gblur` (and `pixelize` when asked) ->
  `alphamerge` to put the mask in the alpha channel -> `overlay`. For transformed keyframes, ffmpeg
  loops each referenced PNG, selects it for its time interval, scales it and overlays it on a black
  expressions as the preview. For a fixed mask, a single still image is enough: framesync's
  `repeatlast` reuses it for every frame and the output frame count matches the input, so `-loop` is
  not needed.
- `QMediaPlayer` decodes video and audio on its playback clock. `QVideoSink` retains only the latest
  frame; a timer composites previews at up to `RENDER_FPS` (25). Expensive previews skip stale frames
  instead of slowing playback. OpenCV is used for exact frame stepping and paused seeks.
- Frames are reduced to display resolution before conversion to numpy. Mask coverage is cached
  until the mask changes, avoiding a full-resolution scan on every playback frame.
- The frame is composited at device resolution (never upscaled beyond the video's own size) and Qt
  stretches it to the widget, so the preview stays sharp on a HiDPI screen.
- Blurring is applied only inside the mask's bounding box (widened by 3 sigma), and a large sigma is
  approximated by downscale -> small sigma -> upscale (`APPROX_BLUR_SIGMA`). The mean difference from
  the real output is about 2, and it is several times faster than a full-resolution `GaussianBlur`
  per frame.
- The composited numpy array is wrapped in a `QImage` with `Format_BGR888`, so no colour conversion
  or copy happens on the way to the screen. The array has to stay referenced while the image lives.
- ffmpeg runs under `QProcess` with merged channels; progress comes from `-progress pipe:1` parsed in
  `readyReadStandardOutput`, so no worker thread is needed.

```sh
pytest -v --doctest-modules video_blur_mask_gui.py
```
