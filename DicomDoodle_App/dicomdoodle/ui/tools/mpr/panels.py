"""
The four MPR views (from the original mpr_gui.py):

* :class:`MPR2DPanel` — one orthogonal slice (matplotlib) with crosshair,
  slice slider / box; click to move the crosshair, wheel to scroll slices;
  a "Detect" button (tumour detection from this view's slice) and the
  dashed outline of the detected tumour's 3D box
* :class:`ThreeDPanel` — the 3D model, rendered by the 3D viewer's VolumeView
  (viewer3d/view.py), with reset-camera and screenshot buttons

Each panel has a header with a maximize / restore button.
"""
from __future__ import annotations


import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QHBoxLayout, QSizePolicy, QSlider, QSpinBox, QVBoxLayout, QWidget

from ....core import mpr
from ...icons import bind_icon
from ...qt_utils import label, make_tool_button

try:
    import vtk  # noqa: F401 -- the 3D view needs it
    HAVE_VTK = True
except Exception:
    HAVE_VTK = False

CROSSHAIR_COLOR = "#00ff66"
DETECTION_BOX_COLOR = "#00e5ff"
DETECT_TIP = ("Detect the tumour starting from the slice shown here: the tumour detector + SAM 2.1 "
              "on this slice and on the matching slices of the other two views give a 3D box, then every slice of "
              "the acquired series inside it is segmented. Click on the tumour first to choose it when other "
              "structures are boxed too; otherwise the most confident box is used.")


class BasePanel(QWidget):
    toggleRequested = pyqtSignal(object)

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.title_label = label(title, "Value")
        self.info_label = label("", "Faint")
        self.toggle_btn = make_tool_button("maximize", "Maximize this view (Esc restores)")
        self.toggle_btn.clicked.connect(lambda: self.toggleRequested.emit(self))
        header = QWidget()
        header.setObjectName("DockBody")
        row = QHBoxLayout(header)
        row.setContentsMargins(10, 4, 4, 4)
        row.addWidget(self.title_label)
        row.addWidget(self.info_label, 1)
        row.addWidget(self.toggle_btn)
        self.header_layout = row             # lets subclasses add header buttons
        self.body = QWidget()
        self.body.setObjectName("MprViewBody")
        self.body.setStyleSheet("#MprViewBody { background-color: black; }")
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(1, 1, 1, 1)
        outer.setSpacing(0)
        outer.addWidget(header)
        outer.addWidget(self.body, 1)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_maximized(self, is_max: bool) -> None:
        bind_icon(self.toggle_btn, "minimize" if is_max else "maximize")
        self.toggle_btn.setToolTip("Restore the 2×2 layout (Esc)" if is_max else "Maximize this view")


class MPR2DPanel(BasePanel):
    positionPicked = pyqtSignal(object, int, int)   # plane, row, col
    sliceChanged = pyqtSignal(object, int)          # plane, index
    detectRequested = pyqtSignal(object)            # plane

    def __init__(self, plane: mpr.Plane, parent=None):
        super().__init__(plane.value.capitalize(), parent)
        self.plane = plane
        self.figure = Figure(facecolor="black")
        self.ax = self.figure.add_axes([0, 0, 1, 1])
        self.ax.set_facecolor("black")
        self.ax.axis("off")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._im = self._hline = self._vline = None
        self._det_rect = None                       # detection cuboid outline, see set_detection_box

        self.detect_btn = make_tool_button("sparkles", DETECT_TIP)
        self.detect_btn.setText("Detect")
        self.detect_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.detect_btn.setIconSize(self.detect_btn.iconSize() * 0.85)
        self.detect_btn.clicked.connect(lambda: self.detectRequested.emit(self.plane))
        self.header_layout.insertWidget(self.header_layout.indexOf(self.toggle_btn), self.detect_btn)

        self.slider = QSlider(Qt.Horizontal)
        self.spin = QSpinBox()
        self.spin.setFixedWidth(78)
        controls = QHBoxLayout()
        controls.setContentsMargins(8, 4, 8, 6)
        controls.addWidget(self.slider, 1)
        controls.addWidget(self.spin)
        holder = QWidget()
        holder.setObjectName("DockBody")
        holder.setLayout(controls)
        self.body_layout.addWidget(self.canvas, 1)
        self.body_layout.addWidget(holder)

        self._updating = False
        self.slider.valueChanged.connect(self._on_slider)
        self.spin.valueChanged.connect(self._on_spin)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

    def set_range(self, n: int, current: int) -> None:
        self._updating = True
        for widget in (self.slider, self.spin):
            widget.setMinimum(0)
            widget.setMaximum(max(n - 1, 0))
            widget.setValue(current)
        self._updating = False

    def _on_slider(self, value: int) -> None:
        if self._updating:
            return
        self._updating = True
        self.spin.setValue(value)
        self._updating = False
        self.sliceChanged.emit(self.plane, value)

    def _on_spin(self, value: int) -> None:
        if self._updating:
            return
        self._updating = True
        self.slider.setValue(value)
        self._updating = False
        self.sliceChanged.emit(self.plane, value)

    def _on_click(self, event) -> None:
        if event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
            self.positionPicked.emit(self.plane, int(round(event.ydata)), int(round(event.xdata)))

    def _on_scroll(self, event) -> None:
        step = 1 if event.button == "up" else -1
        self.slider.setValue(int(np.clip(self.slider.value() + step, self.slider.minimum(), self.slider.maximum())))

    def show_image(self, rgb: np.ndarray, aspect: float, extent_text: str) -> None:
        if self._im is None:
            self._im = self.ax.imshow(rgb, aspect=aspect, interpolation="nearest")
            self._hline = self.ax.axhline(0, color=CROSSHAIR_COLOR, linewidth=0.8, alpha=0.8)
            self._vline = self.ax.axvline(0, color=CROSSHAIR_COLOR, linewidth=0.8, alpha=0.8)
        else:
            self._im.set_data(rgb)
            self._im.set_extent((-0.5, rgb.shape[1] - 0.5, rgb.shape[0] - 0.5, -0.5))
            self.ax.set_aspect(aspect)
        self.info_label.setText(f"  {extent_text}")
        self.canvas.draw_idle()

    def set_crosshair(self, row: float, col: float) -> None:
        if self._hline is not None:
            self._hline.set_ydata([row, row])
            self._vline.set_xdata([col, col])
            self.canvas.draw_idle()

    def set_detect_available(self, available: bool, reason: str = "") -> None:
        self.detect_btn.setEnabled(available)
        self.detect_btn.setToolTip(DETECT_TIP if available else reason)

    def set_detection_box(self, box) -> None:
        """Dashed outline of the detected tumour's 3D box on this slice, as
        (x0, y0, x1, y1) in displayed image coordinates, or None to hide it."""
        if box is None:
            if self._det_rect is not None and self._det_rect.get_visible():
                self._det_rect.set_visible(False)
                self.canvas.draw_idle()
            return
        x0, y0, x1, y1 = box
        if self._det_rect is None:
            self._det_rect = Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=DETECTION_BOX_COLOR,
                                       linewidth=1.0, linestyle="--")
            self.ax.add_patch(self._det_rect)
        else:
            self._det_rect.set_bounds(x0, y0, x1 - x0, y1 - y0)
            self._det_rect.set_visible(True)
        self.canvas.draw_idle()


class ThreeDPanel(BasePanel):
    """The 3D view: the 3D model of the right-hand panel, rendered as in the
    original 3D Model Viewer (viewer3d/view.py) and set up on the "3D view"
    tab. Needs VTK; without it the quadrant says so."""

    def __init__(self, parent=None):
        super().__init__("3D", parent)
        self.view = None
        if not HAVE_VTK:
            note = label("VTK is not installed, so there is no 3D view. Run: pip install vtk", "Faint", wrap=True)
            note.setAlignment(Qt.AlignCenter)
            self.body_layout.addWidget(note, 1)
            return
        from ..viewer3d.view import VolumeView
        self.view = VolumeView()
        self.reset_btn = make_tool_button("refresh", "Reset the camera")
        self.shot_btn = make_tool_button("camera", "Save a picture of the 3D view (PNG)")
        self.reset_btn.clicked.connect(self.view.reset_camera)
        self.shot_btn.clicked.connect(lambda: self.view.save_screenshot())
        for button in (self.reset_btn, self.shot_btn):
            self.header_layout.insertWidget(self.header_layout.indexOf(self.toggle_btn), button)
        self.body_layout.addWidget(self.view, 1)

    def refresh_after_resize(self) -> None:
        if self.view is not None:
            self.view.render()

    def shutdown(self) -> None:
        if self.view is not None:
            self.view.shutdown()
