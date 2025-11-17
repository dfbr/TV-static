#!/usr/bin/env python3
"""
Simple black/white 1024x1024 live viewer with play/pause and probability slider.

Usage:
    python app.py

Requires: PySide6, numpy

This displays a 1024x1024 grayscale image (0 or 255) and updates it repeatedly when
playing. A slider sets the probability of a pixel being white (1.0 = all white,
0.0 = all black).
"""
import sys
import time
import numpy as np
import colorsys
# Simple debug flag: pass --debug on the command line to enable debug prints and dialogs.
# We remove the flag from sys.argv so Qt doesn't see it.
DEBUG = False
if "--debug" in sys.argv:
    DEBUG = True
    sys.argv = [a for a in sys.argv if a != "--debug"]
from PySide6.QtCore import Qt, QTimer, QSize, Signal
from PySide6.QtGui import (
    QImage,
    QPixmap,
    QPainter,
    QColor,
    QAction,
)
from PySide6.QtOpenGL import (
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLBuffer,
    QOpenGLVertexArrayObject,
    QOpenGLTexture,
)
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QSizePolicy,
    QDialog,
    QColorDialog,
    QMenuBar,
    QMenu,
    QScrollArea,
    QGridLayout,
    QMessageBox,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget


WIDTH = 1024
HEIGHT = 1024


class GLImageWidget(QOpenGLWidget):
    """OpenGL-backed widget that draws an RGB numpy array using QPainter over the GL surface."""
    # Signal emitted when the widget size changes; provides width, height in dp
    sizeChanged = Signal(int, int)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame = None
        self._program = None
        self._vbo = None
        self._vao = None
        self._tex = None
        self._tex_w = 0
        self._tex_h = 0
        self._pending_upload = False

    def sizeHint(self):
        # Prefer a 500x500 image area initially
        return QSize(500, 500)

    def set_frame(self, arr: np.ndarray):
        # expect uint8 RGB shape (H,W,3)
        self.frame = np.require(arr, dtype=np.uint8, requirements=["C"])
        self._pending_upload = True
        # schedule repaint
        self.update()

    def initializeGL(self):
        # Setup simple textured quad pipeline
        self.makeCurrent()
        self._program = QOpenGLShaderProgram(self.context())
        vsrc = (
            "attribute vec2 aPos;\n"
            "attribute vec2 aUV;\n"
            "varying vec2 vUV;\n"
            "void main(){ vUV = aUV; gl_Position = vec4(aPos, 0.0, 1.0); }\n"
        )
        fsrc = (
            "varying vec2 vUV;\n"
            "uniform sampler2D uTex;\n"
            "void main(){ gl_FragColor = texture2D(uTex, vUV); }\n"
        )
        self._program.addShaderFromSourceCode(QOpenGLShader.Vertex, vsrc)
        self._program.addShaderFromSourceCode(QOpenGLShader.Fragment, fsrc)
        self._program.link()

        # Vertex buffer: triangle strip covering full screen
        verts = np.array([
            -1.0, -1.0,  0.0, 1.0,
             1.0, -1.0,  1.0, 1.0,
            -1.0,  1.0,  0.0, 0.0,
             1.0,  1.0,  1.0, 0.0,
        ], dtype=np.float32)

        self._vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self._vbo.create()
        self._vbo.bind()
        self._vbo.allocate(verts.tobytes(), verts.nbytes)
        self._vbo.release()

        # VAO (optional, simplifies attribute setup)
        self._vao = QOpenGLVertexArrayObject(self)
        self._vao.create()
        self._vao.bind()
        self._vbo.bind()
        pos_loc = self._program.attributeLocation("aPos")
        uv_loc = self._program.attributeLocation("aUV")
        stride = 4 * 4  # 4 floats per vertex * 4 bytes
        self._program.enableAttributeArray(pos_loc)
        self._program.setAttributeBuffer(pos_loc, 0x1406, 0, 2, stride)  # GL_FLOAT = 0x1406
        self._program.enableAttributeArray(uv_loc)
        self._program.setAttributeBuffer(uv_loc, 0x1406, 2 * 4, 2, stride)
        self._vbo.release()
        self._vao.release()

        # Texture setup
        self._tex = QOpenGLTexture(QOpenGLTexture.Target2D)
        self._tex.create()
        self._tex.setMinificationFilter(QOpenGLTexture.Nearest)
        self._tex.setMagnificationFilter(QOpenGLTexture.Nearest)
        self._tex.setWrapMode(QOpenGLTexture.ClampToEdge)

    def paintGL(self):
        self.makeCurrent()

        # Upload new frame if pending
        if self.frame is not None and self._pending_upload:
            h, w = self.frame.shape[:2]
            img = QImage(self.frame.data, w, h, self.frame.strides[0], QImage.Format_RGB888)
            # Recreate texture object every upload to avoid Qt storage/format warnings
            if self._tex is not None:
                self._tex.destroy()
            self._tex = QOpenGLTexture(QOpenGLTexture.Target2D)
            self._tex.create()
            self._tex.setMinificationFilter(QOpenGLTexture.Nearest)
            self._tex.setMagnificationFilter(QOpenGLTexture.Nearest)
            self._tex.setWrapMode(QOpenGLTexture.ClampToEdge)
            self._tex_w, self._tex_h = w, h
            # Upload from QImage
            self._tex.setData(img)
            self._pending_upload = False

        # Clear and draw textured quad
        f = self.context().functions()
        f.glViewport(0, 0, self.width(), self.height())
        f.glClearColor(0.0, 0.0, 0.0, 1.0)
        f.glClear(0x00004000)  # GL_COLOR_BUFFER_BIT

        if self._program and self._tex:
            self._program.bind()
            self._vao.bind()
            self._tex.bind(0)
            self._program.setUniformValue("uTex", 0)
            f.glDrawArrays(0x0005, 0, 4)  # GL_TRIANGLE_STRIP
            self._tex.release()
            self._vao.release()
            self._program.release()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Notify listeners that the drawable image area changed size
        try:
            self.sizeChanged.emit(self.width(), self.height())
        except Exception:
            pass


def generate_contrasting_palette(n: int) -> np.ndarray:
    n = int(max(0, n))
    n = min(12, n)
    if n <= 0:
        return np.zeros((0, 3), dtype=np.uint8)
    if n == 1:
        return np.array([[255, 255, 255]], dtype=np.uint8)
    if n == 2:
        return np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
    cols = []
    for i in range(n):
        h = (i / n) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
        cols.append([int(r * 255), int(g * 255), int(b * 255)])
    return np.array(cols, dtype=np.uint8)


class BWViewer(QWidget):
    def __init__(self, parent=None, start_colors: int = 2):
        super().__init__(parent)

        self.setWindowTitle("BW Viewer")

        # Dynamic image dimensions (start with 500x500 visible image area)
        self.img_w = 500
        self.img_h = 500

        # Backing array: uint8, 0..255. Start all white (255), RGB
        self.frame = np.ones((self.img_h, self.img_w, 3), dtype=np.uint8) * 255

        # RNG
        self.rng = np.random.default_rng()

        # OpenGL-backed image widget
        self.glwidget = GLImageWidget()
        self.glwidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Track GL widget size changes so we can resize our backing arrays
        self.glwidget.sizeChanged.connect(self._on_gl_size_changed)

        # Play / Pause button
        self.btn = QPushButton("Play")
        self.btn.setCheckable(True)
        self.btn.toggled.connect(self._on_play_toggled)

        # Probability slider (0..100 => 0.0..1.0)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(50)
        self.slider.setTickInterval(10)
        self.slider.setToolTip("Probability of white when using 2-color palette")
        self.slider.valueChanged.connect(self._on_prob_changed)

        # Target FPS slider (1..240)
        self.fps_target_label = QLabel("Target FPS: 60")
        self.fps_slider = QSlider(Qt.Horizontal)
        self.fps_slider.setRange(1, 240)
        self.fps_slider.setValue(60)
        self.fps_slider.setTickInterval(15)
        self.fps_slider.setToolTip("Target frames per second")
        self.fps_slider.valueChanged.connect(self._on_fps_changed)

        # FPS label (overlay-like, but placed above controls)
        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setStyleSheet("background-color: rgba(0,0,0,0.5); color: white; padding: 4px;")

        # Menu bar
        menubar = QMenuBar()
        palette_menu = menubar.addMenu("Palette")
        edit_action = QAction("Edit Colors...", self)
        palette_menu.addAction(edit_action)
        edit_action.triggered.connect(self.open_palette_editor)

        # Layout
        top_row = QHBoxLayout()
        top_row.addWidget(self.btn)
        top_row.addWidget(self.slider)
        top_row.addWidget(self.fps_target_label)
        top_row.addWidget(self.fps_slider)
        # Dump palette debug button (only shown in debug mode)
        if DEBUG:
            self.dump_btn = QPushButton("Dump palette")
            self.dump_btn.clicked.connect(self._dump_palette)
            top_row.addWidget(self.dump_btn)
        top_row.addStretch()
        top_row.addWidget(self.fps_label)

        layout = QVBoxLayout(self)
        layout.setMenuBar(menubar)
        layout.addWidget(self.glwidget)
        layout.addLayout(top_row)

        # Timer for updates
        self.timer = QTimer(self)
        # Default interval: ~30ms => ~33 FPS. You can lower for faster updates.
        self.timer.setInterval(30)
        self.timer.timeout.connect(self.update_frame)

        # Probability / palette config
        self.prob = 0.5
        # Palette: list of (r,g,b) uint8; default black & white, or contrasting colors per start_colors
        self.palette = generate_contrasting_palette(start_colors if start_colors else 2)
        # Per-color weights (only used when len(palette)>2)
        self.weights = np.ones(len(self.palette), dtype=np.float32)
        # Disable the single-prob slider when more than 2 colors are used
        self.slider.setEnabled(len(self.palette) <= 2)

        # For FPS measurement
        self._Frames = 0
        self._frames = 0
        self._t0 = time.perf_counter()

        # Keep references (legacy fields, not used by GL path)
        self._qimage = None
        self._pixmap = None

        # Draw the initial white frame
        self.glwidget.set_frame(self.frame)

        # After the widget is shown, ensure the image area starts at 500x500
        # without constraining future resizes.
        QTimer.singleShot(0, self._ensure_initial_image_size)

        # Initialize timer interval from FPS slider default
        self._on_fps_changed(self.fps_slider.value())

        # Menu state
        self.edit_palette_dialog = None

    def _on_play_toggled(self, checked: bool):
        if checked:
            self.btn.setText("Pause")
            self.timer.start()
        else:
            self.btn.setText("Play")
            self.timer.stop()

    def _on_prob_changed(self, value: int):
        self.prob = float(value) / 100.0
    
    def _on_fps_changed(self, value: int):
        # Clamp and update timer interval
        v = max(1, int(value))
        try:
            self.fps_target_label.setText(f"Target FPS: {v}")
        except Exception:
            pass
        interval = max(1, int(1000 / v))
        self.timer.setInterval(interval)

    def update_frame(self):
        # Determine current drawable dimensions
        h = int(max(1, int(self.img_h)))
        width = int(max(1, int(self.img_w)))
        size_px = int(h) * int(width)

        # Update frame based on current palette
        ncol = len(self.palette)
        if ncol == 0:
            # black
            rgb = np.zeros((h, width, 3), dtype=np.uint8)
        elif ncol == 1:
            rgb = np.broadcast_to(self.palette[0], (h, width, 3)).copy()
        elif ncol == 2:
            # use single probability slider (self.prob) to pick palette[1] with prob, else palette[0]
            mask = self.rng.random((h, width)) < self.prob
            rgb = np.empty((h, width, 3), dtype=np.uint8)
            rgb[mask] = self.palette[1]
            rgb[~mask] = self.palette[0]
        else:
            # Use weights to pick a color per pixel (use RNG.choice for clarity)
            wts = np.array(self.weights[:ncol], dtype=np.float64)
            if wts.sum() <= 0:
                wts = np.ones_like(wts)
            probs = (wts / wts.sum()).astype(np.float64)
            # Ensure palette is an array with shape (ncol,3)
            pal = np.require(self.palette, dtype=np.uint8, requirements=["C"]).reshape((-1, 3))
            # Draw indices for each pixel according to probs
            idx = self.rng.choice(len(pal), size=size_px, p=probs)
            cols = pal[idx]
            rgb = cols.reshape((h, width, 3)).astype(np.uint8)

        # store as frame and push to GL widget
        self.frame = rgb
        self.glwidget.set_frame(self.frame)

        # FPS tracking

        # FPS tracking
        self._frames += 1
        now = time.perf_counter()
        elapsed = now - self._t0
        if elapsed >= 0.25:  # update FPS label more frequently for responsiveness
            fps = self._frames / elapsed
            self.fps_label.setText(f"FPS: {fps:.1f}")
            # also update window title occasionally
            self.setWindowTitle(f"BW Viewer — FPS: {fps:.1f}")
            self._frames = 0
            self._t0 = now

    def _update_pixmap(self):
        # For backward compatibility: create an RGB frame and send to GL widget
        arr = np.require(self.frame, dtype=np.uint8, requirements=["C"])
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        self.frame = arr
        self.glwidget.set_frame(self.frame)

    def _set_pixmap_scaled(self):
        """Scale the latest pixmap to fit the label while keeping aspect ratio."""
        if self._pixmap is None:
            return
        target_size = self.label.size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            return
        scaled = self._pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label.setPixmap(scaled)

    def resizeEvent(self, event):
        # Let the base class/layout handle child resizing, GL widget will emit sizeChanged
        super().resizeEvent(event)
        # No direct action here; _on_gl_size_changed will handle backing array resize
        return

    def _on_gl_size_changed(self, w: int, h: int):
        # When the GL image area changes, resize our backing arrays to match
        self._resize_image(w, h)

    def _ensure_initial_image_size(self):
        # Resize the outer window so that the GL image area is ~500x500 at startup
        gw, gh = self.glwidget.width(), self.glwidget.height()
        if gw == 500 and gh == 500:
            return
        extra_w = self.width() - gw
        extra_h = self.height() - gh
        if extra_w < 0 or extra_h < 0:
            return
        self.resize(extra_w + 500, extra_h + 500)

    def _resize_image(self, w: int, h: int):
        if w <= 0 or h <= 0:
            return
        if hasattr(self, "img_w") and hasattr(self, "img_h") and self.img_w == w and self.img_h == h:
            return
        self.img_w, self.img_h = int(w), int(h)
        # Recreate a white RGB frame at the new size
        self.frame = np.ones((self.img_h, self.img_w, 3), dtype=np.uint8) * 255
        self.glwidget.set_frame(self.frame)

    # ---------------- Palette editor ----------------
    def open_palette_editor(self):
        dlg = PaletteDialog(self.palette, self.weights, parent=self)
        if dlg.exec() == QDialog.Accepted:
            palette, weights = dlg.get_result()
            # normalize returned types and lengths
            palette = np.require(palette, dtype=np.uint8, requirements=["C"])  # (n,3)
            weights = np.require(weights, dtype=np.float64, requirements=["C"]).astype(np.float64)
            # ensure lengths match
            if weights.size < palette.shape[0]:
                # pad
                weights = np.pad(weights, (0, palette.shape[0] - weights.size), constant_values=1.0)
            elif weights.size > palette.shape[0]:
                weights = weights[: palette.shape[0]]

            self.palette = palette
            self.weights = weights
            # Show a confirmation to the user so it's obvious what was set
            if DEBUG:
                try:
                    hexcols = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in palette]
                    QMessageBox.information(self, "Palette set", f"Colors: {hexcols}\nWeights: {weights.tolist()}")
                except Exception:
                    pass
                # debug output to help trace palette/weights
                try:
                    print("[DEBUG] New palette:", self.palette.tolist(), "weights:", self.weights.tolist())
                except Exception:
                    print("[DEBUG] New palette set (could not convert to list)")
            # Hide main single probability slider if palette has more than 2 colors
            if len(self.palette) > 2:
                self.slider.setEnabled(False)
            else:
                self.slider.setEnabled(True)

    def _dump_palette(self):
        if not DEBUG:
            return
        try:
            print("[DUMP] palette:", self.palette.tolist(), "weights:", self.weights.tolist())
            QMessageBox.information(self, "Palette Dump", f"Palette: {self.palette.tolist()}\nWeights: {self.weights.tolist()}")
        except Exception as e:
            QMessageBox.information(self, "Palette Dump", f"Error dumping palette: {e}")


class PaletteDialog(QDialog):
    """Dialog to edit up to 12 colors and per-color weights."""
    def __init__(self, palette: np.ndarray, weights: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Colors")
        self.palette = [QColor(int(r), int(g), int(b)) for r, g, b in palette]
        self.weights = list(map(float, weights))

        self.max_colors = 12

        self.layout = QVBoxLayout(self)
        self.grid = QGridLayout()

        self.rows = []  # list of (color_button, weight_slider)

        for i, col in enumerate(self.palette):
            self._add_row(col, int(self.weights[i] if i < len(self.weights) else 100))

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add Color")
        add_btn.clicked.connect(self.on_add)
        btn_row.addWidget(add_btn)
        btn_row.addStretch()
        self.layout.addLayout(self.grid)
        self.layout.addLayout(btn_row)

        # Preview button only in debug mode
        if DEBUG:
            preview_row = QHBoxLayout()
            preview_btn = QPushButton("Preview")
            preview_btn.clicked.connect(self._on_preview)
            preview_row.addWidget(preview_btn)
            preview_row.addStretch()
            self.layout.addLayout(preview_row)

        bottom = QHBoxLayout()
        ok = QPushButton("OK")
        ok.clicked.connect(self.accept)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        bottom.addStretch()
        bottom.addWidget(ok)
        bottom.addWidget(cancel)
        self.layout.addLayout(bottom)

    def _add_row(self, qcolor=None, weight=100):
        if len(self.rows) >= self.max_colors:
            QMessageBox.information(self, "Limit", f"Max {self.max_colors} colors")
            return
        color = qcolor or QColor(255, 255, 255)
        btn = QPushButton()
        btn.setFixedSize(36, 24)
        btn.setStyleSheet(f"background-color: {color.name()}")
        # store the QColor on the button for reliable retrieval later
        btn._color = color

        def on_pick(checked=False, col_btn=btn):
            # clicked signal may pass a boolean 'checked' which we ignore; col_btn is captured
            c = QColorDialog.getColor(parent=self)
            # QColorDialog.getColor may return a QColor or a falsy value on some platforms.
            if isinstance(c, QColor) and c.isValid():
                col_btn.setStyleSheet(f"background-color: {c.name()}")
                col_btn._color = c

        btn.clicked.connect(on_pick)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(weight)

        remove = QPushButton("Remove")

        def on_remove(checked=False, btn_ref=btn):
            # remove row
            for i, (b, s, r) in enumerate(self.rows):
                if b is btn_ref:
                    # remove widgets from grid
                    for w in (b, s, r):
                        w.hide()
                        self.grid.removeWidget(w)
                    self.rows.pop(i)
                    return

        remove.clicked.connect(on_remove)

        row = len(self.rows)
        self.grid.addWidget(btn, row, 0)
        self.grid.addWidget(slider, row, 1)
        self.grid.addWidget(remove, row, 2)
        self.rows.append((btn, slider, remove))

    def on_add(self):
        self._add_row(QColor(255, 255, 255), 100)

    def _on_preview(self):
        # Construct palette and weights from current rows without closing dialog
        cols = []
        weights = []
        for b, s, _ in self.rows:
            if hasattr(b, "_color") and isinstance(b._color, QColor):
                color = b._color
            else:
                ss = b.styleSheet()
                color = QColor(255, 255, 255)
                try:
                    import re

                    m = re.search(r"#[0-9A-Fa-f]{6}", ss)
                    if m:
                        color = QColor(m.group(0))
                except Exception:
                    pass
            cols.append((color.red(), color.green(), color.blue()))
            weights.append(float(s.value()))

        if len(cols) == 0:
            QMessageBox.information(self, "Preview", "No colors defined")
            return

        pal = np.array(cols, dtype=np.uint8)
        w = np.array(weights, dtype=np.float64)
        if w.sum() <= 0:
            w = np.ones_like(w)
        probs = w / w.sum()

        rng = np.random.default_rng()
        idx = rng.choice(len(pal), size=128 * 128, p=probs)
        cols_arr = pal[idx].reshape((128, 128, 3)).astype(np.uint8)
        # Create QImage and show in a small dialog
        qim = QImage(cols_arr.data, 128, 128, cols_arr.strides[0], QImage.Format_RGB888)
        pix = QPixmap.fromImage(qim)
        dlg = QDialog(self)
        dlg.setWindowTitle("Palette Preview")
        l = QVBoxLayout(dlg)
        lbl = QLabel()
        lbl.setPixmap(pix.scaled(256, 256, Qt.KeepAspectRatio))
        l.addWidget(lbl)
        btn = QPushButton("Close")
        btn.clicked.connect(dlg.accept)
        l.addWidget(btn)
        dlg.exec()

    def get_result(self):
        cols = []
        weights = []
        for b, s, _ in self.rows:
            # Prefer the stored QColor object attached to the button
            color = None
            if hasattr(b, "_color") and isinstance(b._color, QColor):
                color = b._color
            else:
                # extract color from stylesheet as a fallback
                ss = b.styleSheet()
                color = QColor(255, 255, 255)
                try:
                    import re

                    m = re.search(r"#[0-9A-Fa-f]{6}", ss)
                    if m:
                        hexcol = m.group(0)
                        color = QColor(hexcol)
                    else:
                        hexcol = ss.split(':')[-1].strip().rstrip(';')
                        color = QColor(hexcol)
                except Exception:
                    pass
            cols.append((color.red(), color.green(), color.blue()))
            weights.append(float(s.value()))
        if len(cols) == 0:
            cols = [(0, 0, 0)]
            weights = [1.0]
        # debug
        if DEBUG:
            try:
                print("[DEBUG] PaletteDialog.get_result -> cols:", cols, "weights:", weights)
            except Exception:
                pass
        return np.array(cols, dtype=np.uint8), np.array(weights, dtype=np.float32)


def _parse_color_count(argv_list):
    # Support: --colors N, --colours N, -c N, or --colors=N / --colours=N
    count = 2
    new_argv = []
    skip_next = False
    for i, tok in enumerate(argv_list):
        if skip_next:
            skip_next = False
            continue
        if tok in ("--colors", "--colours", "-c"):
            if i + 1 < len(argv_list):
                try:
                    count = int(argv_list[i + 1])
                except Exception:
                    pass
                skip_next = True
            continue
        if tok.startswith("--colors=") or tok.startswith("--colours="):
            try:
                count = int(tok.split("=", 1)[1])
            except Exception:
                pass
            continue
        new_argv.append(tok)
    return count, new_argv


def main(argv):
    # Parse custom CLI arg for color count while preserving other args for Qt
    if len(argv) > 0:
        color_count, rest = _parse_color_count(argv[:])
        # Ensure argv[0] (program path) is preserved for Qt
        qt_argv = [argv[0]] + rest[1:] if len(rest) > 0 and rest[0] == argv[0] else [argv[0]] + rest
    else:
        color_count = 2
        qt_argv = argv

    app = QApplication(qt_argv)
    w = BWViewer(start_colors=color_count)
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
