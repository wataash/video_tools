# video_blur_mask_gui

[video_blur_mask_gui.py](video_blur_mask_gui.py)

A PySide6 (Qt) GUI that plays a video while you paint a blur mask over it with a brush, shows the
blurred result live, and finally burns the mask in with ffmpeg.

It does not track faces. The mask is a single still image, so you play the clip and keep painting over
the area the subjects move through; strokes accumulate into one mask that covers every position.

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
  Click the timeline to seek. Playback speed is 0.25x / 0.5x / 1x.
- `v`: cycle the view (composite -> mask tinted red -> original). The red view is the easiest way to
  spot places you missed.
- `s`: save the mask (plus a timestamped backup). Ctrl+Z / Ctrl+Y: undo / redo. `q`: quit.
- "Encode..." asks for an output path, then runs ffmpeg in the background with a progress bar.
  "Cancel" terminates it.

## Mask persistence

- Autosaved every 3 seconds, and also on quit and when an encode starts.
- Writes go to `<mask>.tmp.png` and are then `os.replace()`d, so a crash cannot leave a corrupt mask.
- `s` and the start of an encode also leave a `<stem>.<YYYYmmdd-HHMMSS>.png` backup.
- An existing mask with the same name is loaded on startup. If an encode crashes, redo it with the
  `encode` subcommand instead of repainting.

## Implementation notes

- The filter graph is `[0:v]split` -> blur one branch with `gblur` (and `pixelize` when asked) ->
  `alphamerge` to put the mask in the alpha channel -> `overlay`.
  A single still image is enough for the mask input: framesync's `repeatlast` reuses it for every
  frame and the output frame count matches the input, so `-loop` is not needed.
- The preview decimates frames with `cap.grab()` down to `RENDER_FPS` (25), which is enough for a
  60fps source at 0.25x-1x playback.
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
