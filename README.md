# Analogue TV Static Producer

A fast, interactive Python app that generates classic “analogue TV snow” (static) in real time. It renders an RGB image backed by NumPy and displays it with an OpenGL-accelerated widget (PySide6), supporting dynamic window resizing, FPS control, and optional multi‑color palettes.

## Features

- Live static generator with Play/Pause
- OpenGL rendering for high FPS, scales crisply with the window
- FPS meter and target FPS slider (1–240)
- 2‑color mode with a probability slider (white vs black)
- Multi‑color mode (up to 12 colors) with per‑color weights via a palette editor (Palette → Edit Colors…)
- Starts with a ~500×500 image area and resizes the underlying arrays to match the visible GL area
- Optional `--debug` flag to show extra UI and debug dialogs/prints
- Export to video (File → Export Video…) with duration, FPS, resolution, aspect ratio, and quality controls; outputs MP4 by default

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

### Command‑line options

- `--colors N` / `--colours N` / `-c N`
  - Number of colors in the palette (default 2). If `N > 2`, a contrasting palette is generated and the probability slider is disabled. Maximum 12 colors.
  - Examples:
    ```pwsh
    python .\app.py --colors 2     # black & white, probability slider enabled
    python .\app.py -c 12          # vivid 12‑color palette, weights via editor
    ```
- `--debug`
  - Enables extra debug UI and informational dialogs.

## Controls

- Play/Pause button: starts and stops the live updates
- Probability slider: visible and active in 2‑color mode to select white probability
- Target FPS slider: sets the render/update interval (1–240 FPS)
- Palette → Edit Colors…: open the palette editor (up to 12 colors) and set per‑color weights
- File → Export Video…: export a video (MP4) of the animated static; choose duration (default 60s), FPS, resolution/aspect ratio, and quality

## How it works

- NumPy generates a new RGB frame each tick based on the selected palette and weights.
- The frame is uploaded to a GL texture and drawn as a textured quad with nearest‑neighbor filtering for crisp pixels.
- The GL image area emits size changes; the backing NumPy array resizes to match, so you always fill the window.

## Notes and tips

- In multi‑color mode (>2 colors), the main probability slider is disabled; use the palette editor to adjust per‑color weights.
- The FPS label updates several times per second for responsiveness.
- For maximum performance, keep other applications minimized, and prefer a discrete GPU if available.

## Troubleshooting

- If the window is blank or OpenGL initialization fails, update your graphics drivers and ensure PySide6 supports your platform.
- If PySide6 isn’t found, install it with `python -m pip install PySide6` in the same environment you use to run the app.
- If export fails due to missing codecs, install the video dependency:
  ```pwsh
  python -m pip install imageio imageio-ffmpeg
  ```

## License

No explicit license provided. If you plan to distribute or reuse, consider adding a LICENSE file.
