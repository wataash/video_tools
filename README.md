# video_tools

Personal tools for working with video. Each tool is a standalone Python script, paired with a
`<name>.md` that documents it.

| Tool | What it does |
| --- | --- |
| [video_blur_mask_gui](video_blur_mask_gui.md) | Paint a blur mask while the video plays, then burn it in with ffmpeg |
| [video_text_gui](video_text_gui.md) | Place and style timed text with audio playback, then export an MP4 |

## Requirements

- Python 3.12+
- `pip install -r requirements.txt` (numpy / opencv-python / PySide6)
- `ffmpeg` and `ffprobe`

```sh
pytest -v --doctest-modules .
```

## License

Apache-2.0. See [LICENSE](LICENSE).
