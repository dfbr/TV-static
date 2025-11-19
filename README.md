# Analogue TV Static Producer

A fast, interactive Python app that generates classic "analogue TV snow" (static) in real time. It renders an RGB image backed by NumPy and displays it with an OpenGL-accelerated widget (PySide6), supporting dynamic window resizing, FPS control, fullscreen mode, and optional multiÔÇæcolor palettes.

## Demo

<video width="600" autoplay muted loop>
  <source src="[path/to/your/video.mp4](https://github.com/dfbr/TV-static/raw/refs/heads/main/static.mp4)" type="video/mp4">
  Your browser does not support the video tag.
</video>
*Example of the static generator in action*

## Features

- **Live static generator** with Play/Pause and fullscreen mode (F11)
- **OpenGL rendering** for high performance, scales crisply with the window
- **True randomness**: Every pixel is independently random every frame
- **Performance optimized**: 30-60+ FPS at 1920├ù1080 fullscreen
- **FPS meter** and target FPS slider (1ÔÇô240)
- **2ÔÇæcolor mode** with a probability slider (white vs black)
- **MultiÔÇæcolor mode** (up to 12 colors) with perÔÇæcolor weights via palette editor
- **Fullscreen mode** (F11) with auto-hiding cursor and UI
- **Dynamic resizing**: Underlying arrays automatically match the visible GL area
- **Video export** (File ÔåÆ Export VideoÔÇª) with duration, FPS, resolution, aspect ratio, and quality controls
- **Debug mode** (`--debug` flag) for performance timing and extra diagnostics

## Requirements

- Python 3.9+
- Packages:
  - PySide6
  - numpy

You can install them with:

```pwsh
python -m pip install --upgrade pip
python -m pip install PySide6 numpy
```

## How to run

From the repository root:

```pwsh
python .\app.py
```

### CommandÔÇæline options

- `--colors N` / `--colours N` / `-c N`
  - Number of colors in the palette (default 2). If `N > 2`, a contrasting palette is generated and the probability slider is disabled. Maximum 12 colors.
  - Examples:
    ```pwsh
    python .\app.py --colors 2     # black & white, probability slider enabled
    python .\app.py -c 12          # vivid 12ÔÇæcolor palette, weights via editor
    ```
- `--debug`
  - Enables extra debug UI and informational dialogs.

## Controls

- Play/Pause button: starts and stops the live updates
- Probability slider: visible and active in 2ÔÇæcolor mode to select white probability
- Target FPS slider: sets the render/update interval (1ÔÇô240 FPS)
- Palette ÔåÆ Edit ColorsÔÇª: open the palette editor (up to 12 colors) and set perÔÇæcolor weights
- File ÔåÆ Export VideoÔÇª: export a video (MP4) of the animated static; choose duration (default 60s), FPS, resolution/aspect ratio, and quality

## How it works

- NumPy generates a new RGB frame each tick based on the selected palette and weights.
- The frame is uploaded to a GL texture and drawn as a textured quad with nearestÔÇæneighbor filtering for crisp pixels.
- The GL image area emits size changes; the backing NumPy array resizes to match, so you always fill the window.

## Notes and tips

- In multiÔÇæcolor mode (>2 colors), the main probability slider is disabled; use the palette editor to adjust perÔÇæcolor weights.
- The FPS label updates several times per second for responsiveness.
- For maximum performance, keep other applications minimized, and prefer a discrete GPU if available.

## Troubleshooting

- If the window is blank or OpenGL initialization fails, update your graphics drivers and ensure PySide6 supports your platform.
- If PySide6 isnÔÇÖt found, install it with `python -m pip install PySide6` in the same environment you use to run the app.
- If export fails due to missing codecs, install the video dependency:
  ```pwsh
  python -m pip install imageio imageio-ffmpeg
  ```

## License

No explicit license provided. If you plan to distribute or reuse, consider adding a LICENSE file.
