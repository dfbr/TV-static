# BW 1024x1024 Viewer

Simple demo app that displays a 1024×1024 black/white image and updates it repeatedly.

Features:
- Start with all-white image
- Play/Pause button to start/stop updates
- Slider to control probability of white per pixel (0.0..1.0)
- FPS shown in window title

Requirements:
- Python 3.9+
- See `requirements.txt` for dependencies (PySide6, numpy)

Run:
```powershell
# (Recommend running inside a venv)
python -m pip install -r requirements.txt
python app.py
```

Notes:
- The app uses a QImage that references the NumPy buffer directly to avoid copies.
- The update interval is controlled by a QTimer set to 30ms (about 33 FPS). You can
  change `self.timer.setInterval(...)` in `app.py` to experiment with higher rates.
