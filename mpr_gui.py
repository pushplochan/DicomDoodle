"""
mpr_gui.py
----------
PyQt5 GUI for viewing 3D MRI volume(s) loaded from folders of DICOM slices,
as Axial / Coronal / Sagittal / 3D panels in a 2x2 grid.

Features
  - Load Primary Series...: pick a directory of DICOM slices. Whatever
    plane the scan was actually acquired in (axial, coronal or sagittal),
    the DICOM orientation metadata (ImageOrientationPatient) is used to
    auto-detect it and losslessly reformat the volume (pure axis
    permute + flip, no interpolation) so array indexing always gives true
    axial/coronal/sagittal views -- the same normalisation ITK-SNAP does
    internally, so scrolling behaves the same regardless of how the scan
    was acquired.
  - Algorithm combo box: choose the interpolation/resampling strategy
    (see mpr_core.InterpMethod) used to build the displayed MPR volume.
  - Register Second Series...: load another folder (e.g. a different
    sequence/orientation of the same patient) and rigidly align it onto
    the primary series using a multi-resolution Mattes Mutual Information
    registration (see mpr_core.register_rigid_mi for the full rationale).
    Once registered, use the "Show" dropdown to flip between the primary
    and the registered series -- both now share the same voxel grid, so
    the crosshair position stays meaningful across the switch.
  - Click anywhere on the Axial, Coronal or Sagittal image to move the
    crosshair; the other two views jump to that location.
  - Mouse wheel over a 2D view (or its slider/spin-box) scrolls through
    slices, the same as ITK-SNAP.
  - Every panel (including the 3D view) has a small "⤢" button in its
    header: click it to make that panel fill the whole window; click the
    "⤡" button (or press Esc) to go back to the 2x2 layout.
  - Volume Processing...: an optional pipeline (see volume_processing.py)
    of Skull Stripping -> Z-Score Normalization -> 3D Connected Component
    Analysis, applied in that fixed order to whichever of those steps you
    turn on -- any subset works (just Z-score, Z-score then CCA, skull
    strip then Z-score then CCA, etc). Runs on GPU automatically via
    CuPy/cuCIM if installed, otherwise transparently falls back to CPU.
  - 3D Colormap dropdown: recolors the 3D volume render only (does not
    touch the underlying data or the 2D grayscale views).
  - Annotation overlay: if a folder contains a DICOM SEG object (any
    multi-frame file with Modality=SEG, found automatically regardless of
    filename -- so a "..._seg.dcm" convention works, but so would any
    other name), it's loaded and can be overlaid as a colored mask on the
    axial/coronal/sagittal views via the "Annotation overlay" dropdown.
    When you register a second series, if the primary series had an
    annotation it is automatically propagated ("transferred") onto the
    secondary series' own native grid using the same rigid transform
    found during registration (nearest-neighbor only -- label data is
    never blended/interpolated). If the secondary series also has its own
    real annotation, "Compare Annotations..." opens a side-by-side view of
    the secondary's actual annotation vs. the transferred one, on a shared
    slice control, with an overall Dice similarity score.

Run:
    python mpr_gui.py

Dependencies: PyQt5, matplotlib, numpy, SimpleITK, pydicom, vtk (all via
mpr_core / requirements.txt). If `vtk` isn't installed the 3D panel falls
back to a flat max-intensity-projection preview so the rest of the app
still works. volume_processing.py's GPU acceleration (CuPy/cuCIM) is
optional too -- everything works on CPU if they aren't installed.
"""

from __future__ import annotations

import sys
import traceback
from functools import partial
from typing import Optional, Dict, Tuple

import torch  # noqa: F401 - must be imported before PyQt5 on Windows to avoid WinError 1114
import numpy as np

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGridLayout, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QComboBox, QSlider, QSpinBox,
    QFileDialog, QMessageBox, QStackedWidget, QToolButton, QInputDialog,
    QSizePolicy, QFrame, QStatusBar, QProgressDialog, QDialog, QGroupBox,
    QFormLayout, QDoubleSpinBox, QCheckBox,
)

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

import mpr_core as core
from mpr_core import Plane, InterpMethod
import volume_processing as vproc

try:
    import vtk
    from vtkmodules.util import numpy_support as vtk_np
    try:
        from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    except Exception:  # older vtk wheels expose it under vtk.qt instead
        from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    HAVE_VTK = True
except Exception:
    HAVE_VTK = False


# ============================================================================
# App-wide light theme. The image viewports (matplotlib black background)
# are deliberately left alone -- a black viewport is the universal
# radiology-software convention because it gives the correct visual
# context for windowing/contrast on grayscale medical images. Everything
# that is *text/chrome* (toolbar, buttons, labels, dropdowns, status bar)
# gets a light background with clearly legible black text instead.
# ============================================================================

LIGHT_STYLESHEET = """
QMainWindow, QDialog { background-color: #f2f2f2; }
QToolBar { background-color: #f2f2f2; border: none; spacing: 6px; padding: 4px; }
QLabel { color: #000000; }
QPushButton {
    background-color: #ffffff; color: #000000; border: 1px solid #999999;
    border-radius: 4px; padding: 5px 12px;
}
QPushButton:hover { background-color: #e6f0ff; }
QPushButton:pressed { background-color: #cfe4ff; }
QPushButton:disabled { color: #a0a0a0; background-color: #f5f5f5; border-color: #cccccc; }
QComboBox {
    background-color: #ffffff; color: #000000; border: 1px solid #999999;
    border-radius: 4px; padding: 3px 6px; min-height: 22px;
}
QComboBox:disabled { color: #a0a0a0; background-color: #f5f5f5; }
QComboBox QAbstractItemView {
    background-color: #ffffff; color: #000000;
    selection-background-color: #cfe4ff; selection-color: #000000;
}
QSpinBox {
    background-color: #ffffff; color: #000000; border: 1px solid #999999; border-radius: 3px;
}
QStatusBar { background-color: #e8e8e8; color: #000000; }
QStatusBar QLabel { color: #000000; }
QToolTip { color: #000000; background-color: #ffffe0; border: 1px solid #999999; }
QProgressDialog { background-color: #f2f2f2; color: #000000; }
"""

TOGGLE_BTN_STYLE = (
    "QToolButton { color:#000000; background-color:#eeeeee; border:1px solid #999999; "
    "border-radius:3px; padding:2px 8px; font-weight:600; } "
    "QToolButton:hover { background-color:#dddddd; }"
)

# Sliders live inside the black image viewport (not the light toolbar/
# header chrome), so the default OS slider style -- often a subtle gray
# on gray -- is hard to see there. This gives a thick, high-contrast
# groove and a large, brightly colored handle that stands out clearly
# against the black background.
SLIDER_STYLE = """
QSlider::groove:horizontal {
    height: 8px;
    background: #555555;
    border-radius: 4px;
}
QSlider::sub-page:horizontal {
    background: #3ea6ff;
    border-radius: 4px;
}
QSlider::add-page:horizontal {
    background: #555555;
    border-radius: 4px;
}
QSlider::handle:horizontal {
    background: #ffcc00;
    border: 2px solid #ffffff;
    width: 20px;
    height: 20px;
    margin: -7px 0;
    border-radius: 10px;
}
QSlider::handle:horizontal:hover {
    background: #ffe066;
}
QSlider::handle:horizontal:pressed {
    background: #ffffff;
}
"""


# ============================================================================
# Reusable panel header: title + maximize/restore button
# ============================================================================

class BasePanel(QWidget):
    """Common chrome shared by every panel in the 2x2 grid: a light title
    bar with black text and a maximize/restore toggle button. Subclasses
    fill in `self.body` (which stays black -- it holds the image)."""

    toggleRequested = pyqtSignal(object)  # emits self

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(2, 2, 2, 2)
        outer.setSpacing(2)

        header = QHBoxLayout()
        header.setContentsMargins(6, 3, 6, 3)
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("color: #000000; font-weight: 700; font-size: 13px;")
        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color: #333333; font-size: 11px;")
        self.toggle_btn = QToolButton()
        self.toggle_btn.setText("\u2922")  # ⤢
        self.toggle_btn.setToolTip("Maximize this view")
        self.toggle_btn.setStyleSheet(TOGGLE_BTN_STYLE)
        self.toggle_btn.clicked.connect(lambda: self.toggleRequested.emit(self))

        header.addWidget(self.title_label)
        header.addWidget(self.info_label, 1)
        header.addWidget(self.toggle_btn)

        header_frame = QFrame()
        header_frame.setLayout(header)
        header_frame.setStyleSheet("background-color: #dcdcdc; border-bottom: 1px solid #aaaaaa;")

        self.body = QWidget()
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        self.body.setStyleSheet("background-color: black;")

        outer.addWidget(header_frame)
        outer.addWidget(self.body, 1)
        self.setStyleSheet("background-color: #bbbbbb;")  # thin light frame between panels
        # All 4 panels must compete for space identically regardless of
        # their internal content (2D panels have a slider row, the 3D
        # panel doesn't; the VTK widget's size hint differs from
        # matplotlib's) -- otherwise Qt sizes quadrants unevenly based on
        # each widget's own preferred size. Expanding + the grid's equal
        # row/column stretch (set in MainWindow._build_central_widget)
        # together guarantee all 4 quadrants stay exactly equal, including
        # after a maximize/restore cycle.
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_maximized(self, is_max: bool):
        self.toggle_btn.setText("\u2921" if is_max else "\u2922")  # ⤡ / ⤢
        self.toggle_btn.setToolTip("Restore 2x2 layout" if is_max else "Maximize this view")


# ============================================================================
# 2D orthogonal-view panel (axial / coronal / sagittal)
# ============================================================================

class MPR2DPanel(BasePanel):
    """One matplotlib image + slider/spin-box slice control + crosshair."""

    positionPicked = pyqtSignal(object, int, int)   # plane, row, col (displayed coords)
    sliceChanged = pyqtSignal(object, int)           # plane, new index

    def __init__(self, plane: Plane, parent=None):
        super().__init__(plane.value.capitalize(), parent)
        self.plane = plane

        self.figure = Figure(facecolor="black")
        self.ax = self.figure.add_axes([0, 0, 1, 1])  # fill the whole panel
        self.ax.set_facecolor("black")
        self.ax.axis("off")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._im = None
        self._hline = None
        self._vline = None

        controls = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setStyleSheet(SLIDER_STYLE)
        self.slider.setMinimumHeight(24)  # extra room so the bigger handle isn't clipped
        self.spin = QSpinBox()
        self.spin.setFixedWidth(70)
        self.spin.setStyleSheet(
            "QSpinBox { background-color:#ffffff; color:#000000; border:1px solid #888888; "
            "border-radius:3px; padding:1px 3px; }")
        controls.addWidget(self.slider, 1)
        controls.addWidget(self.spin)

        self.body_layout.addWidget(self.canvas, 1)
        self.body_layout.addLayout(controls)

        self.slider.valueChanged.connect(self._on_slider_changed)
        self.spin.valueChanged.connect(self._on_spin_changed)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

        self._updating = False  # re-entrancy guard while syncing slider<->spin

    # -- range / index -----------------------------------------------------

    def set_range(self, n: int, current: int):
        self._updating = True
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(n - 1, 0))
        self.spin.setMinimum(0)
        self.spin.setMaximum(max(n - 1, 0))
        self.slider.setValue(current)
        self.spin.setValue(current)
        self._updating = False

    def _on_slider_changed(self, value: int):
        if self._updating:
            return
        self._updating = True
        self.spin.setValue(value)
        self._updating = False
        self.sliceChanged.emit(self.plane, value)

    def _on_spin_changed(self, value: int):
        if self._updating:
            return
        self._updating = True
        self.slider.setValue(value)
        self._updating = False
        self.sliceChanged.emit(self.plane, value)

    # -- interaction: crosshair click + ITK-SNAP-style wheel scrolling -----

    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        col = int(round(event.xdata))
        row = int(round(event.ydata))
        self.positionPicked.emit(self.plane, row, col)

    def _on_scroll(self, event):
        step = 1 if event.button == "up" else -1
        new_val = int(np.clip(self.slider.value() + step, self.slider.minimum(), self.slider.maximum()))
        self.slider.setValue(new_val)  # triggers _on_slider_changed -> sliceChanged

    # -- rendering ------------------------------------------------------------

    def show_image(self, img_rgb_uint8: np.ndarray, aspect: float, extent_text: str):
        """`img_rgb_uint8` must be an (H, W, 3) uint8 array. The caller
        (MainWindow._refresh_2d_panels) converts grayscale to RGB via
        core.gray_to_rgb and blends in any annotation overlay via
        core.blend_label_overlay before calling this, so this panel never
        needs to know whether an overlay is active. Always rendering RGB
        (rather than switching an AxesImage between grayscale-with-cmap
        and RGB modes depending on whether an overlay is on) avoids any
        risk of matplotlib mis-handling that mode switch on an existing
        AxesImage; with equal R=G=B channels this looks pixel-identical to
        plain grayscale, so there's no visual change when no overlay is
        selected."""
        if self._im is None:
            self._im = self.ax.imshow(img_rgb_uint8, aspect=aspect, interpolation="nearest")
            self._hline = self.ax.axhline(0, color="#00ff66", linewidth=0.8, alpha=0.8)
            self._vline = self.ax.axvline(0, color="#00ff66", linewidth=0.8, alpha=0.8)
        else:
            self._im.set_data(img_rgb_uint8)
            self._im.set_extent((-0.5, img_rgb_uint8.shape[1] - 0.5, img_rgb_uint8.shape[0] - 0.5, -0.5))
            self.ax.set_aspect(aspect)
        self.info_label.setText(extent_text)
        self.canvas.draw_idle()

    def set_crosshair(self, row: float, col: float):
        if self._hline is not None:
            self._hline.set_ydata([row, row])
            self._vline.set_xdata([col, col])
            self.canvas.draw_idle()


# ============================================================================
# 3D panel — real volume rendering via raw VTK.
# ============================================================================

class ThreeDPanel(BasePanel):
    """
    Interactive 3D volume rendering using raw VTK:
      - vtkImageData built directly from the numpy volume (numpy_to_vtk)
        with correct physical voxel spacing.
      - vtkSmartVolumeMapper: VTK auto-selects GPU ray-casting when the
        graphics driver supports it and transparently falls back to a
        software (CPU) ray caster otherwise.
      - vtkColorTransferFunction / vtkPiecewiseFunction give a grayscale,
        semi-transparent look so internal structure is visible; shading is
        enabled for depth cues.
      - vtkInteractorStyleTrackballCamera: left-drag orbit, right-drag /
        scroll to zoom, middle-drag to pan — the standard mouse scheme
        used by ITK-SNAP's and 3D Slicer's 3D views.

    Falls back to a max-intensity-projection preview (pure matplotlib) if
    the `vtk` package is not installed, so the rest of the app still runs.
    """

    def __init__(self, parent=None):
        super().__init__("3D", parent)
        self._mode_vtk = HAVE_VTK
        self.volume_actor: Optional["vtk.vtkVolume"] = None
        self._lo = 0.0
        self._hi = 1.0
        self._colormap_name = vproc.DEFAULT_COLORMAP

        if self._mode_vtk:
            self.vtk_widget = QVTKRenderWindowInteractor(self.body)
            self.body_layout.addWidget(self.vtk_widget, 1)

            self.renderer = vtk.vtkRenderer()
            self.renderer.SetBackground(0.0, 0.0, 0.0)
            self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)

            self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
            self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

            controls = QHBoxLayout()
            reset_btn = QPushButton("Reset Camera")
            reset_btn.clicked.connect(self.reset_camera)
            controls.addWidget(reset_btn)
            self.body_layout.addLayout(controls)

            self.vtk_widget.Initialize()
            self.vtk_widget.Start()
        else:
            self.figure = Figure(facecolor="black")
            self.ax = self.figure.add_axes([0, 0, 1, 1])
            self.ax.set_facecolor("black")
            self.ax.axis("off")
            self.canvas = FigureCanvasQTAgg(self.figure)
            self.body_layout.addWidget(self.canvas, 1)
            note = QLabel("The 'vtk' package is not installed, so this is a flat "
                           "max-intensity-projection preview, not a real 3D render. "
                           "Run: pip install vtk")
            note.setWordWrap(True)
            note.setStyleSheet("color:#bbbbbb; font-size:10px;")
            self.body_layout.addWidget(note)
            self._im = None

    def reset_camera(self):
        if self._mode_vtk:
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()

    def refresh_after_resize(self):
        """Call after this panel is reparented (maximize/restore) so VTK
        picks up the new widget size immediately instead of waiting for
        the next interaction."""
        if self._mode_vtk:
            self.vtk_widget.GetRenderWindow().Render()

    def set_colormap(self, name: str):
        """Recolor the 3D render only -- no data upload, so this is cheap
        and instant even on a large volume."""
        self._colormap_name = name
        if self._mode_vtk and self.volume_actor is not None:
            self._apply_colormap()
            self.vtk_widget.GetRenderWindow().Render()

    def _apply_colormap(self):
        stops = vproc.COLORMAPS.get(self._colormap_name, vproc.COLORMAPS[vproc.DEFAULT_COLORMAP])
        color_func = vtk.vtkColorTransferFunction()
        span = self._hi - self._lo
        for pos, rgb in stops:
            color_func.AddRGBPoint(self._lo + pos * span, *rgb)
        self.volume_actor.GetProperty().SetColor(color_func)

    def set_volume(self, volume: core.Volume):
        # Full resolution, every voxel -- no downsampling. On very large
        # volumes this means more GPU/CPU memory and slower camera
        # interaction (especially without a GPU, where VTK falls back to
        # software ray-casting), but nothing is thrown away or approximated:
        # what you see is an exact volume render of the actual voxel data.
        full = volume.array
        sz, sy, sx = volume.spacing_zyx

        if self._mode_vtk:
            if self.volume_actor is not None:
                self.renderer.RemoveVolume(self.volume_actor)
                self.volume_actor = None

            depth, rows, cols = full.shape  # z, y, x
            vtk_data = vtk_np.numpy_to_vtk(
                full.astype(np.float32).ravel(order="C"), deep=True,
                array_type=vtk.VTK_FLOAT)

            image = vtk.vtkImageData()
            image.SetDimensions(cols, rows, depth)          # x, y, z
            image.SetSpacing(sx, sy, sz)                     # x, y, z
            image.SetOrigin(0.0, 0.0, 0.0)
            image.GetPointData().SetScalars(vtk_data)

            lo, hi = np.percentile(full, [2, 98])
            if hi <= lo:
                lo, hi = float(full.min()), float(full.max() + 1.0)
            self._lo, self._hi = float(lo), float(hi)
            mid = (lo + hi) / 2.0

            opacity_func = vtk.vtkPiecewiseFunction()
            opacity_func.AddPoint(lo, 0.0)
            opacity_func.AddPoint(lo + 0.25 * (hi - lo), 0.03)
            opacity_func.AddPoint(mid, 0.12)
            opacity_func.AddPoint(hi, 0.45)

            vol_prop = vtk.vtkVolumeProperty()
            vol_prop.SetScalarOpacity(opacity_func)
            vol_prop.SetInterpolationTypeToLinear()
            vol_prop.ShadeOn()
            vol_prop.SetAmbient(0.25)
            vol_prop.SetDiffuse(0.7)
            vol_prop.SetSpecular(0.25)
            vol_prop.SetSpecularPower(15.0)

            mapper = vtk.vtkSmartVolumeMapper()  # auto GPU-raycast, falls back to CPU
            mapper.SetInputData(image)

            self.volume_actor = vtk.vtkVolume()
            self.volume_actor.SetMapper(mapper)
            self.volume_actor.SetProperty(vol_prop)
            self._apply_colormap()

            self.renderer.AddVolume(self.volume_actor)
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
        else:
            mip = np.max(full, axis=1)  # a simple coronal-style MIP as a stand-in
            u8 = core.window_to_uint8(mip)
            if self._im is None:
                self._im = self.ax.imshow(np.flipud(u8), cmap="gray", vmin=0, vmax=255)
            else:
                self._im.set_data(np.flipud(u8))
            self.canvas.draw_idle()

        self.info_label.setText(
            f"{volume.shape[2]}x{volume.shape[1]}x{volume.shape[0]} voxels, full resolution (no downsampling)")


# ============================================================================
# Volume processing dialog: Skull Strip -> Z-Score -> 3D Connected
# Components, any subset, always in that order when enabled.
# ============================================================================

class ProcessingDialog(QDialog):
    """
    Lets the user turn each of the three processing steps on/off and set
    their parameters, then hands back a fully-populated
    volume_processing.PipelineConfig when they click Apply. Nothing is
    computed by the dialog itself -- MainWindow runs the (potentially
    slow) pipeline after the dialog closes, with its own progress dialog,
    the same pattern used for registration.
    """

    def __init__(self, config: vproc.PipelineConfig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Volume Processing Pipeline")
        self.setStyleSheet(LIGHT_STYLESHEET)
        self.setMinimumWidth(420)
        self.result_config: Optional[vproc.PipelineConfig] = None

        layout = QVBoxLayout(self)

        gpu_label = QLabel(vproc.gpu_status_text())
        gpu_label.setStyleSheet("font-weight:700; padding:2px 0;")
        layout.addWidget(gpu_label)

        note = QLabel(
            "Steps always apply in this fixed order when turned on: "
            "Skull Strip \u2192 Z-Score \u2192 Connected Components. "
            "Turn on any subset for the combination you want (e.g. Z-Score alone, "
            "Z-Score then CCA, or all three).")
        note.setWordWrap(True)
        note.setStyleSheet("color:#333333; font-size:11px; padding-bottom:6px;")
        layout.addWidget(note)

        # -- 1. Skull strip --------------------------------------------------
        self.ss_group = QGroupBox("1. Skull Strip")
        self.ss_group.setCheckable(True)
        self.ss_group.setChecked(config.skull_strip.enabled)
        ss_layout = QFormLayout(self.ss_group)
        self.ss_method_combo = QComboBox()
        for m in vproc.SkullStripMethod:
            label = m.value
            if m is vproc.SkullStripMethod.HD_BET and not vproc.HAVE_HDBET:
                label += "  [not installed]"
            self.ss_method_combo.addItem(label, userData=m)
        self.ss_method_combo.setCurrentIndex(list(vproc.SkullStripMethod).index(config.skull_strip.method))
        ss_layout.addRow("Method:", self.ss_method_combo)
        self.ss_peel_spin = QDoubleSpinBox()
        self.ss_peel_spin.setRange(0.0, 60.0)
        self.ss_peel_spin.setValue(config.skull_strip.peel_mm)
        self.ss_peel_spin.setSuffix(" mm")
        self.ss_peel_spin.setToolTip(
            "Classical method only: how far inward (in mm) to erode the whole-head mask "
            "to peel away scalp/skull/CSF. Too small leaves skull behind; too large eats "
            "into the brain surface.")
        ss_layout.addRow("Peel depth (classical):", self.ss_peel_spin)
        layout.addWidget(self.ss_group)

        # -- 2. Z-score --------------------------------------------------------
        self.z_group = QGroupBox("2. Z-Score Normalization")
        self.z_group.setCheckable(True)
        self.z_group.setChecked(config.zscore.enabled)
        z_layout = QFormLayout(self.z_group)
        self.z_clip_low = QDoubleSpinBox()
        self.z_clip_low.setRange(0.0, 49.0)
        self.z_clip_low.setValue(config.zscore.clip_low_pct)
        self.z_clip_low.setSuffix(" %ile")
        self.z_clip_high = QDoubleSpinBox()
        self.z_clip_high.setRange(51.0, 100.0)
        self.z_clip_high.setValue(config.zscore.clip_high_pct)
        self.z_clip_high.setSuffix(" %ile")
        z_layout.addRow("Clip low (intensity clipping):", self.z_clip_low)
        z_layout.addRow("Clip high (intensity clipping):", self.z_clip_high)
        layout.addWidget(self.z_group)

        # -- 3. CCA --------------------------------------------------------
        self.cca_group = QGroupBox("3. Connected Component Analysis (3D)")
        self.cca_group.setCheckable(True)
        self.cca_group.setChecked(config.cca.enabled)
        cca_layout = QFormLayout(self.cca_group)
        self.cca_threshold_spin = QDoubleSpinBox()
        self.cca_threshold_spin.setRange(-100000.0, 100000.0)
        self.cca_threshold_spin.setDecimals(3)
        self.cca_threshold_spin.setValue(config.cca.threshold)
        self.cca_threshold_spin.setToolTip(
            "Applied to whatever data reaches this step -- raw intensities if Z-Score is "
            "off, z-score units (roughly -3..+10) if Z-Score is on.")
        self.cca_minsize_spin = QSpinBox()
        self.cca_minsize_spin.setRange(1, 100_000_000)
        self.cca_minsize_spin.setValue(config.cca.min_size_voxels)
        self.cca_conn_combo = QComboBox()
        self.cca_conn_combo.addItem("6-connected (faces only)", 1)
        self.cca_conn_combo.addItem("18-connected", 2)
        self.cca_conn_combo.addItem("26-connected (incl. diagonals)", 3)
        self.cca_conn_combo.setCurrentIndex(config.cca.connectivity - 1)
        cca_layout.addRow("Threshold:", self.cca_threshold_spin)
        cca_layout.addRow("Min component size (voxels):", self.cca_minsize_spin)
        cca_layout.addRow("Connectivity:", self.cca_conn_combo)
        layout.addWidget(self.cca_group)

        btn_row = QHBoxLayout()
        disable_all_btn = QPushButton("Disable All")
        disable_all_btn.clicked.connect(self._on_disable_all)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        apply_btn = QPushButton("Apply")
        apply_btn.setStyleSheet("font-weight:700;")
        apply_btn.clicked.connect(self._on_apply)
        btn_row.addWidget(disable_all_btn)
        btn_row.addStretch()
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(apply_btn)
        layout.addLayout(btn_row)

    def _on_disable_all(self):
        self.ss_group.setChecked(False)
        self.z_group.setChecked(False)
        self.cca_group.setChecked(False)

    def _on_apply(self):
        self.result_config = vproc.PipelineConfig(
            skull_strip=vproc.SkullStripParams(
                enabled=self.ss_group.isChecked(),
                method=self.ss_method_combo.currentData(),
                peel_mm=self.ss_peel_spin.value(),
            ),
            zscore=vproc.ZScoreParams(
                enabled=self.z_group.isChecked(),
                clip_low_pct=self.z_clip_low.value(),
                clip_high_pct=self.z_clip_high.value(),
            ),
            cca=vproc.CCAParams(
                enabled=self.cca_group.isChecked(),
                threshold=self.cca_threshold_spin.value(),
                min_size_voxels=self.cca_minsize_spin.value(),
                connectivity=self.cca_conn_combo.currentData(),
            ),
        )
        self.accept()


class AnnotationCompareDialog(QDialog):
    """
    2x1 side-by-side comparison: the secondary series' own actual DICOM SEG
    annotation (left) vs. the primary series' annotation transferred onto
    the secondary's native grid via the registration transform (right).
    Both panels show the exact same physical slice at all times (one
    shared plane selector + slider drives both), since the point of this
    view is checking how well the transferred annotation lines up with the
    real one -- visually, and via the overall Dice score shown at the top.

    Deliberately a separate, self-contained dialog rather than a mode of
    the main 2x2 grid, so it can't interfere with anything there.
    """

    def __init__(self, secondary_reference_image, actual_seg: Optional[core.Segmentation],
                 transferred_seg: Optional[core.Segmentation], parent=None,
                 initial_plane: str = "Axial"):
        super().__init__(parent)
        self.setWindowTitle("Compare Annotations — Secondary Series")
        self.resize(1100, 650)
        self.setStyleSheet(LIGHT_STYLESHEET)

        self.actual_seg = actual_seg
        self.transferred_seg = transferred_seg

        # Always compare at full native resolution with zero interpolation
        # -- this view exists to validate the annotation itself, so it
        # deliberately doesn't inherit the main window's display Algorithm
        # choice.
        self.volume = core.build_volume(secondary_reference_image, InterpMethod.NONE_NATIVE)
        self.actual_array = (core.resample_label_array_like(actual_seg.label_image, self.volume.sitk_image)
                              if actual_seg is not None else None)
        self.transferred_array = (core.resample_label_array_like(transferred_seg.label_image, self.volume.sitk_image)
                                   if transferred_seg is not None else None)

        layout = QVBoxLayout(self)

        if self.actual_array is not None and self.transferred_array is not None:
            dice = core.dice_coefficient(self.transferred_array, self.actual_array)
            dice_text = f"Overall Dice similarity coefficient (whole volume, transferred vs. actual): {dice:.3f}"
        else:
            dice_text = ("Dice score needs both an actual and a transferred annotation; "
                         "only one is available here.")
        dice_label = QLabel(dice_text)
        dice_label.setStyleSheet("font-weight:700; font-size:13px; padding:6px; color:#000000;")
        layout.addWidget(dice_label)

        # Per-slice IoU: unlike the Dice score above (one static number for
        # the whole volume), this updates every time the slice/plane
        # changes below, so scrubbing through slices shows exactly where
        # the transferred annotation agrees or disagrees with the real one.
        self.slice_metric_label = QLabel("")
        self.slice_metric_label.setStyleSheet(
            "font-weight:700; font-size:13px; padding:0px 6px 6px 6px; color:#0a5c2b;")
        layout.addWidget(self.slice_metric_label)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Plane: "))
        self.plane_combo = QComboBox()
        self.plane_combo.addItem("Axial", userData=Plane.AXIAL)
        self.plane_combo.addItem("Coronal", userData=Plane.CORONAL)
        self.plane_combo.addItem("Sagittal", userData=Plane.SAGITTAL)
        # Open on whichever plane the secondary series was actually
        # acquired in (e.g. a coronal-acquired secondary opens showing
        # Coronal) rather than always Axial -- that's the view you almost
        # always want to check first. Falls back to Axial for "Oblique" /
        # "Unknown" detections, where no single plane is the obvious pick.
        plane_by_name = {"Axial": 0, "Coronal": 1, "Sagittal": 2}
        self.plane_combo.setCurrentIndex(plane_by_name.get(initial_plane, 0))
        self.plane_combo.currentIndexChanged.connect(self._on_plane_changed)
        controls.addWidget(self.plane_combo)

        self.show_filled_mask_check = QCheckBox("Show filled mask")
        self.show_filled_mask_check.setToolTip(
            "Off (default): show only each annotation's boundary, in red. "
            "On: fill the whole region with a translucent color instead.")
        self.show_filled_mask_check.stateChanged.connect(lambda _s: self._refresh())
        controls.addWidget(self.show_filled_mask_check)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setStyleSheet(SLIDER_STYLE)
        self.slider.setMinimumHeight(24)
        self.slider.valueChanged.connect(self._on_slice_changed)
        controls.addWidget(self.slider, 1)
        self.spin = QSpinBox()
        self.spin.setFixedWidth(70)
        self.spin.valueChanged.connect(self.slider.setValue)
        self.slider.valueChanged.connect(self.spin.setValue)
        controls.addWidget(self.spin)
        layout.addLayout(controls)

        panels = QHBoxLayout()
        self._left = self._make_panel(
            "Secondary — Actual Annotation" + ("" if actual_seg is not None else "  (none loaded)"))
        self._right = self._make_panel(
            "Secondary — Transferred from Primary" + ("" if transferred_seg is not None else "  (not available)"))
        panels.addWidget(self._left["container"])
        panels.addWidget(self._right["container"])
        layout.addLayout(panels, 1)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_row = QHBoxLayout()
        close_row.addStretch(1)
        close_row.addWidget(close_btn)
        layout.addLayout(close_row)

        self._on_plane_changed(self.plane_combo.currentIndex())  # explicit: setup above didn't fire the signal

    @staticmethod
    def _make_panel(title: str) -> dict:
        container = QWidget()
        v = QVBoxLayout(container)
        label = QLabel(title)
        label.setStyleSheet("font-weight:600; color:#000000; padding:4px;")
        figure = Figure(facecolor="black")
        ax = figure.add_axes([0, 0, 1, 1])
        ax.set_facecolor("black")
        ax.axis("off")
        canvas = FigureCanvasQTAgg(figure)
        v.addWidget(label)
        v.addWidget(canvas, 1)
        return {"container": container, "ax": ax, "canvas": canvas, "im": None}

    def _on_plane_changed(self, _index: int):
        plane = self.plane_combo.currentData()
        axis = {Plane.AXIAL: 0, Plane.CORONAL: 1, Plane.SAGITTAL: 2}[plane]
        extent = self.volume.shape[axis]
        self.slider.blockSignals(True)
        self.spin.blockSignals(True)
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(extent - 1, 0))
        self.spin.setMinimum(0)
        self.spin.setMaximum(max(extent - 1, 0))
        mid = extent // 2
        self.slider.setValue(mid)
        self.spin.setValue(mid)
        self.slider.blockSignals(False)
        self.spin.blockSignals(False)
        self._refresh()

    def _on_slice_changed(self, _value: int):
        self._refresh()

    def _refresh(self):
        plane = self.plane_combo.currentData()
        idx = self.slider.value()
        gray = core.extract_slice(self.volume, plane, idx)
        rgb_base = core.gray_to_rgb(core.window_to_uint8(gray))
        aspect = core.display_aspect(self.volume, plane)

        actual_slice = (core.extract_label_slice(self.actual_array, plane, idx)
                         if self.actual_array is not None else None)
        transferred_slice = (core.extract_label_slice(self.transferred_array, plane, idx)
                              if self.transferred_array is not None else None)

        if actual_slice is not None and transferred_slice is not None:
            iou = core.iou_coefficient(transferred_slice, actual_slice)
            if np.isnan(iou):
                self.slice_metric_label.setText(
                    f"This slice ({plane.value}, index {idx}): no annotation on either side here — IoU n/a")
            else:
                self.slice_metric_label.setText(
                    f"This slice ({plane.value}, index {idx}): IoU (transferred vs. actual) = {iou:.3f}")
        else:
            self.slice_metric_label.setText(f"This slice ({plane.value}, index {idx}): IoU n/a "
                                             f"(need both an actual and a transferred annotation)")

        show_filled = self.show_filled_mask_check.isChecked()
        for panel, label_slice, seg in [(self._left, actual_slice, self.actual_seg),
                                         (self._right, transferred_slice, self.transferred_seg)]:
            rgb = rgb_base
            if label_slice is not None and seg is not None:
                rgb = (core.blend_label_overlay(rgb_base.copy(), label_slice, seg.segments)
                       if show_filled else core.blend_label_boundary(rgb_base.copy(), label_slice, seg.segments))
            ax = panel["ax"]
            if panel["im"] is None:
                panel["im"] = ax.imshow(rgb, aspect=aspect, interpolation="nearest")
            else:
                panel["im"].set_data(rgb)
                panel["im"].set_extent((-0.5, rgb.shape[1] - 0.5, rgb.shape[0] - 0.5, -0.5))
                ax.set_aspect(aspect)
            panel["canvas"].draw_idle()


# ============================================================================
# Main window
# ============================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MRI MPR Viewer — Axial / Coronal / Sagittal / 3D")
        self.resize(1250, 880)
        self.setStyleSheet(LIGHT_STYLESHEET)

        # Primary series: the reference/fixed volume everything is displayed
        # and registered against.
        self.primary_image = None            # sitk.Image, already auto-oriented
        self.primary_plane = ""              # detected acquisition plane, e.g. "Sagittal"

        # Secondary series: optional, loaded via "Register Second Series...",
        # rigidly aligned onto the primary series' voxel grid.
        self.secondary_image = None
        self.secondary_plane = ""
        self.secondary_folder = ""             # native folder path for the secondary series
        self._primary_folder = ""              # primary folder, kept for fill_contours reload
        self.registration_result: Optional[core.RegistrationResult] = None


        # Annotation (DICOM SEG) support. primary_seg is loaded straight
        # from the primary series' own folder. secondary_native_image is
        # the secondary series BEFORE it was resampled into the primary's
        # grid (self.secondary_image above) -- annotation comparison needs
        # the secondary's own native voxel grid, not the display copy.
        # transferred_seg is the primary's annotation warped onto that
        # native grid via the registration transform.
        self.primary_seg: Optional[core.Segmentation] = None
        self.secondary_native_image = None
        self.secondary_seg_native: Optional[core.Segmentation] = None
        self.transferred_seg: Optional[core.Segmentation] = None
        self.overlay_choice = None            # None, "primary", "secondary_native", or "transferred"
        self._active_label_array: Optional[np.ndarray] = None
        self._active_segments: Dict[int, core.SegmentInfo] = {}

        self.active_source = "primary"       # or "secondary" -- which image is on screen
        self.volume: Optional[core.Volume] = None
        self.frac = [0.5, 0.5, 0.5]          # normalized cursor position (z, y, x) in [0,1]

        # Volume processing (skull strip / z-score / CCA) -- disabled by
        # default, configured via the "Volume Processing..." dialog.
        self.processing_config = vproc.PipelineConfig()
        self.processing_result: Optional[vproc.PipelineResult] = None

        self._build_toolbar()
        self._build_central_widget()
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Load a DICOM folder to begin.")

        self.maximized_panel: Optional[BasePanel] = None
        self.panel_positions: Dict[BasePanel, Tuple[int, int]] = {
            self.axial_panel: (0, 0),
            self.coronal_panel: (0, 1),
            self.sagittal_panel: (1, 0),
            self.three_d_panel: (1, 1),
        }

    # ------------------------------------------------------------------ UI --

    def _build_toolbar(self):
        bar = self.addToolBar("Main")
        bar.setMovable(False)

        load_btn = QPushButton("Load Primary Series...")
        load_btn.clicked.connect(self.on_load_primary)
        bar.addWidget(load_btn)

        bar.addSeparator()
        algo_label = QLabel("  Display algorithm (applies immediately): ")
        algo_label.setToolTip(
            "Controls how the CURRENTLY SHOWN volume is resampled for the axial/coronal/"
            "sagittal/3D views. Changing this instantly rebuilds and redraws every view.")
        bar.addWidget(algo_label)
        self.algo_combo = QComboBox()
        for m in InterpMethod:
            self.algo_combo.addItem(m.value, userData=m)
        self.algo_combo.setCurrentIndex(0)  # default: NONE_NATIVE (zero data loss)
        self.algo_combo.setToolTip(algo_label.toolTip())
        self.algo_combo.currentIndexChanged.connect(self.on_algorithm_changed)
        self.algo_combo.setMinimumWidth(340)
        bar.addWidget(self.algo_combo)

        bar.addSeparator()
        backend_label = QLabel("  Registration engine (applies on next click below): ")
        backend_label.setToolTip(
            "Only affects what happens when you click 'Register Second Series...'. "
            "Changing this dropdown does NOT re-run or change anything by itself, and has "
            "no effect on a registration you already ran. After a registration completes, "
            "the summary line below shows which engine actually produced the result you're "
            "viewing.")
        bar.addWidget(backend_label)
        self.backend_combo = QComboBox()
        for b in core.RegistrationBackend:
            label = b.value
            if b is core.RegistrationBackend.ANTSPY and not core.HAVE_ANTSPY:
                label += "  [not installed]"
            self.backend_combo.addItem(label, userData=b)
        self.backend_combo.setToolTip(backend_label.toolTip())
        self.backend_combo.setMinimumWidth(200)
        bar.addWidget(self.backend_combo)

        self.register_btn = QPushButton("Register Second Series...")
        self.register_btn.setEnabled(False)
        self.register_btn.clicked.connect(self.on_register_second_series)
        bar.addWidget(self.register_btn)

        bar.addWidget(QLabel("  Show: "))
        self.source_combo = QComboBox()
        self.source_combo.addItem("Primary Series")
        self.source_combo.setEnabled(False)
        self.source_combo.setToolTip(
            "Switch between the primary series and a registered second series. Both share "
            "the same voxel grid once registered, so the crosshair stays on the same "
            "physical location when you switch.")
        self.source_combo.currentIndexChanged.connect(self.on_source_changed)
        bar.addWidget(self.source_combo)

        self.reg_info_label = QLabel("")
        self.reg_info_label.setStyleSheet("color:#000000; font-size:11px; padding-left:8px;")
        bar.addWidget(self.reg_info_label)

        # Second toolbar row: volume processing + 3D colormap. Kept
        # separate from the row above since it was already full.
        self.addToolBarBreak()
        bar2 = self.addToolBar("Processing")
        bar2.setMovable(False)

        self.processing_btn = QPushButton("Volume Processing...")
        self.processing_btn.setToolTip(
            "Optional pipeline: Skull Strip -> Z-Score -> 3D Connected Components. "
            "Turn on any subset, in that fixed order.")
        self.processing_btn.clicked.connect(self.on_open_processing_dialog)
        bar2.addWidget(self.processing_btn)

        self.gpu_status_label = QLabel("  " + vproc.gpu_status_text())
        self.gpu_status_label.setStyleSheet("color:#000000; font-size:11px; padding: 0 10px;")
        bar2.addWidget(self.gpu_status_label)

        bar2.addSeparator()
        bar2.addWidget(QLabel("  3D Colormap: "))
        self.colormap_combo = QComboBox()
        for name in vproc.COLORMAPS:
            self.colormap_combo.addItem(name)
        self.colormap_combo.setCurrentText(vproc.DEFAULT_COLORMAP)
        self.colormap_combo.setToolTip(
            "Recolors the 3D volume render only -- does not change the underlying data "
            "or the grayscale 2D views.")
        self.colormap_combo.currentTextChanged.connect(self.on_colormap_changed)
        bar2.addWidget(self.colormap_combo)

        self.processing_info_label = QLabel("")
        self.processing_info_label.setStyleSheet("color:#000000; font-size:11px; padding-left:10px;")
        bar2.addWidget(self.processing_info_label)

        # Third toolbar row: annotation (DICOM SEG) overlay + comparison.
        self.addToolBarBreak()
        bar3 = self.addToolBar("Annotation")
        bar3.setMovable(False)

        annotation_label = QLabel("  Annotation overlay: ")
        annotation_label.setToolTip(
            "Overlays a DICOM SEG annotation on the axial/coronal/sagittal views. Options "
            "depend on what's loaded and which source (Primary/Registered) is shown above.")
        bar3.addWidget(annotation_label)
        self.annotation_combo = QComboBox()
        self.annotation_combo.addItem("None", userData=None)
        self.annotation_combo.setToolTip(annotation_label.toolTip())
        self.annotation_combo.setMinimumWidth(220)
        self.annotation_combo.currentIndexChanged.connect(self.on_annotation_choice_changed)
        bar3.addWidget(self.annotation_combo)

        self.show_filled_mask_check = QCheckBox("Show filled mask")
        self.show_filled_mask_check.setToolTip(
            "Off (default): show only the annotation's boundary, in red -- a clean outline "
            "that doesn't obscure the image underneath. On: fill the whole region with a "
            "translucent color instead.")
        self.show_filled_mask_check.stateChanged.connect(self.on_mask_display_mode_changed)
        bar3.addWidget(self.show_filled_mask_check)

        self.compare_btn = QPushButton("Compare Annotations...")
        self.compare_btn.setEnabled(False)
        self.compare_btn.setToolTip(
            "Side-by-side: the secondary series' own actual annotation vs. the primary's "
            "annotation transferred onto it via registration.")
        self.compare_btn.clicked.connect(self.on_compare_annotations)
        bar3.addWidget(self.compare_btn)

        self.fill_contours_check = QCheckBox("Fill SEG contours")
        self.fill_contours_check.setChecked(True)
        self.fill_contours_check.setToolTip(
            "When ON (default): hollow outline annotations are filled before display/transfer "
            "using binary_fill_holes — this makes them solid in all three views and produces "
            "a solid volume for Dice calculation.\n"
            "When OFF: raw outline pixels only. Use this for 'Boundary'-type annotations "
            "where fill_holes would enclose an anatomically-irrelevant region, inflating the "
            "annotation volume and artificially lowering the Dice score.\n"
            "Changing this re-loads all currently-loaded segmentations automatically.")
        self.fill_contours_check.stateChanged.connect(self.on_fill_contours_changed)
        bar3.addWidget(self.fill_contours_check)

        self.save_transferred_btn = QPushButton("Save Transferred Annotation...")
        self.save_transferred_btn.setEnabled(False)
        self.save_transferred_btn.setToolTip(
            "Save the primary series' annotation (transferred onto the secondary series via "
            "registration) as slice-wise DICOM SEG files.  "
            "Creates one `<slice>_seg.dcm` file per annotated slice to perfectly match "
            "DicomDoodle's native save format.")
        self.save_transferred_btn.clicked.connect(self.on_save_transferred_annotation)
        bar3.addWidget(self.save_transferred_btn)

        self.annotation_info_label = QLabel("")
        self.annotation_info_label.setStyleSheet("color:#000000; font-size:11px; padding-left:10px;")
        bar3.addWidget(self.annotation_info_label)


    def _build_central_widget(self):
        self.info_banner = QLabel("Load a primary series to begin.")
        self.info_banner.setStyleSheet(
            "background-color: #eef6ff; color: #000000; font-weight: 700; "
            "font-size: 12px; padding: 5px 8px; border-bottom: 1px solid #99c2e8;")
        self.info_banner.setWordWrap(True)

        self.axial_panel = MPR2DPanel(Plane.AXIAL)
        self.coronal_panel = MPR2DPanel(Plane.CORONAL)
        self.sagittal_panel = MPR2DPanel(Plane.SAGITTAL)
        self.three_d_panel = ThreeDPanel()

        for p in (self.axial_panel, self.coronal_panel, self.sagittal_panel):
            p.sliceChanged.connect(self.on_slice_changed)
            p.positionPicked.connect(self.on_position_picked)

        for p in (self.axial_panel, self.coronal_panel, self.sagittal_panel, self.three_d_panel):
            p.toggleRequested.connect(self.on_toggle_maximize)

        # Page 0: 2x2 grid
        self.grid_page = QWidget()
        self.grid_layout = QGridLayout(self.grid_page)
        self.grid_layout.setContentsMargins(2, 2, 2, 2)
        self.grid_layout.setSpacing(2)
        # Equal stretch on both rows and both columns: this is what forces
        # all 4 quadrants to always be exactly the same size, independent
        # of each panel's own preferred size hint. These factors live on
        # the layout's row/column indices, not on the widgets, so they are
        # unaffected by later removeWidget()/addWidget() calls during
        # maximize/restore -- restoring a panel to (row, col) always gets
        # it back to an equal-sized quadrant.
        self.grid_layout.setRowStretch(0, 1)
        self.grid_layout.setRowStretch(1, 1)
        self.grid_layout.setColumnStretch(0, 1)
        self.grid_layout.setColumnStretch(1, 1)
        self.grid_layout.addWidget(self.axial_panel, 0, 0)
        self.grid_layout.addWidget(self.coronal_panel, 0, 1)
        self.grid_layout.addWidget(self.sagittal_panel, 1, 0)
        self.grid_layout.addWidget(self.three_d_panel, 1, 1)

        # Page 1: single maximized panel
        self.max_page = QWidget()
        self.max_layout = QVBoxLayout(self.max_page)
        self.max_layout.setContentsMargins(0, 0, 0, 0)

        self.stacked = QStackedWidget()
        self.stacked.addWidget(self.grid_page)
        self.stacked.addWidget(self.max_page)

        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        container_layout.addWidget(self.info_banner)
        container_layout.addWidget(self.stacked, 1)
        self.setCentralWidget(container)

    def _update_info_banner(self):
        """Single, always-visible line answering 'what am I looking at right
        now': which series, which interpolation algorithm is applied to it,
        and -- if it's the registered series -- which registration engine
        actually produced it. This is the one place to check; the toolbar
        controls above it are just how you CHANGE these, not confirmation
        of what's currently active."""
        if self.volume is None:
            self.info_banner.setText("Load a primary series to begin.")
            return

        algo: InterpMethod = self.algo_combo.currentData()
        source_text = "PRIMARY series" if self.active_source == "primary" else "REGISTERED second series"

        if self.active_source == "secondary" and self.registration_result is not None:
            reg = self.registration_result
            metric_text = f", metric {reg.final_metric:.3f}" if np.isfinite(reg.final_metric) else ""
            reg_text = f"  |  aligned onto primary using {reg.backend.value}{metric_text}"
        elif self.active_source == "secondary":
            reg_text = "  |  (no registration recorded for this volume)"
        else:
            reg_text = "  |  registration not applicable to the primary series itself"

        active_steps = self.processing_config.active_steps()
        if active_steps:
            proc_text = f"  |  Processing: {' -> '.join(active_steps)}"
        else:
            proc_text = "  |  Processing: none"

        if self.overlay_choice is None:
            annot_text = "  |  Annotation: none shown"
        else:
            annot_label = self.annotation_combo.currentText()
            annot_text = f"  |  Annotation: {annot_label}"

        self.info_banner.setText(
            f"Showing: {source_text}   |   Display algorithm applied: {algo.value}"
            f"{reg_text}{proc_text}{annot_text}")

    # ------------------------------------------------------------- loading --

    def _pick_and_load_series(self, dialog_title: str):
        """Shared folder-pick + (multi-series-aware) DICOM load used by both
        the primary and the register-second-series flows. Returns
        (image, detected_plane, folder, segmentation) or (None, None, None,
        None) if cancelled or on failure (a message box is already shown on
        failure). `segmentation` is a core.Segmentation if the folder also
        contains a DICOM SEG object (any multi-frame file with
        Modality=SEG, regardless of filename), aligned to `image`'s grid,
        or None if there isn't one -- an annotation is always optional."""
        folder = QFileDialog.getExistingDirectory(self, dialog_title)
        if not folder:
            return None, None, None, None
        try:
            series_ids = core.find_series_ids(folder)
            if len(series_ids) > 1:
                item, ok = QInputDialog.getItem(
                    self, "Multiple series found",
                    "This folder contains more than one DICOM series.\nChoose one:",
                    series_ids, 0, False)
                if not ok:
                    return None, None, None, None
                series_id = item
            elif len(series_ids) == 1:
                series_id = series_ids[0]
            else:
                # Let load_dicom_series attempt its own nested-folder
                # fallback / raise a clear error -- it re-runs discovery,
                # but only in this already-exceptional case.
                series_id = None

            image = core.load_dicom_series(folder, series_id)
            plane = core.get_acquisition_plane(image)

            segmentation = None
            try:
                seg_paths = core.find_seg_files(folder)
                if seg_paths:
                    fill = self.fill_contours_check.isChecked()
                    segmentation = core.load_seg_as_labelmap(seg_paths, image,
                                                              fill_contours=fill)
            except Exception:
                # An unreadable/malformed SEG shouldn't block loading the
                # underlying image series -- the annotation is optional.
                segmentation = None

            return image, plane, folder, segmentation
        except Exception as exc:
            QMessageBox.critical(self, "Failed to load DICOM series",
                                  f"{exc}\n\n{traceback.format_exc(limit=2)}")
            return None, None, None, None

    def on_load_primary(self):
        image, plane, folder, segmentation = self._pick_and_load_series(
            "Select folder containing the PRIMARY DICOM series")
        if image is None:
            return

        self.primary_image = image
        self.primary_plane = plane
        self.primary_seg = segmentation
        self._primary_folder = folder          # stored for fill_contours reload
        self.secondary_image = None
        self.secondary_plane = ""
        self.secondary_folder = ""
        self.secondary_native_image = None
        self.secondary_seg_native = None
        self.transferred_seg = None
        self.registration_result = None
        self.active_source = "primary"
        self.frac = [0.5, 0.5, 0.5]
        self.processing_config = vproc.PipelineConfig()
        self.processing_result = None


        self.source_combo.blockSignals(True)
        self.source_combo.clear()
        self.source_combo.addItem("Primary Series")
        self.source_combo.setEnabled(False)
        self.source_combo.blockSignals(False)
        self.reg_info_label.setText("")
        self.register_btn.setEnabled(True)
        self._refresh_annotation_combo()

        self._rebuild_volume()
        seg_note = ""
        if segmentation is not None:
            seg_note = f"  —  annotation loaded ({len(segmentation.segments)} segment(s))"
        self.statusBar().showMessage(
            f"Loaded {folder}  —  detected acquisition plane: {plane} "
            f"(auto-reformatted to standard axial/coronal/sagittal, ITK-SNAP style){seg_note}")

    # --------------------------------------------------------- registration --

    def on_register_second_series(self):
        if self.primary_image is None:
            QMessageBox.information(self, "Load a primary series first",
                                     "Load the primary DICOM series before registering a second one.")
            return

        moving_image, moving_plane, folder, moving_seg = self._pick_and_load_series(
            "Select folder containing the SECOND DICOM series to register")
        if moving_image is None:
            return

        backend: core.RegistrationBackend = self.backend_combo.currentData()

        progress = QProgressDialog(f"Registering second series onto the primary series "
                                    f"using {backend.value}...",
                                    None, 0, 0, self)  # no cancel button (see mpr_core docstring)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()

        def on_iteration(iteration, metric_value):
            if backend is core.RegistrationBackend.ANTSPY:
                progress.setLabelText(
                    f"Registering with ANTsPy... this can take noticeably longer than "
                    f"SimpleITK; please wait.")
            else:
                progress.setLabelText(
                    f"Registering second series onto the primary series...\n"
                    f"iteration {iteration}, similarity metric {metric_value:.4f}")
            QApplication.processEvents()

        try:
            result = core.register_rigid(self.primary_image, moving_image, backend=backend,
                                          progress_callback=on_iteration)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            progress.close()
            QMessageBox.critical(self, "Registration failed",
                                  f"{exc}\n\n{traceback.format_exc(limit=2)}")
            return

        QApplication.restoreOverrideCursor()
        progress.close()

        self.secondary_image = result.registered_image
        self.secondary_plane = moving_plane
        self.secondary_folder = folder             # remember for "Save Transferred Annotation"
        self.registration_result = result
        self.secondary_native_image = moving_image
        self.secondary_seg_native = moving_seg

        self.transferred_seg = None
        if self.primary_seg is not None:
            try:
                self.transferred_seg = core.transfer_labelmap(
                    self.primary_seg, moving_image, result, source_was_fixed=True)
            except Exception as exc:
                QMessageBox.warning(self, "Annotation transfer failed",
                                     f"Registration succeeded but transferring the primary "
                                     f"series' annotation onto the secondary series failed:\n{exc}")

        self.source_combo.blockSignals(True)
        self.source_combo.clear()
        self.source_combo.addItem("Primary Series")
        self.source_combo.addItem("Registered Second Series")
        self.source_combo.setEnabled(True)
        self.source_combo.setCurrentIndex(1)
        self.source_combo.blockSignals(False)

        tx, ty, tz = result.translation_mm
        rx, ry, rz = result.rotation_deg
        metric_text = f"metric={result.final_metric:.3f}" if np.isfinite(result.final_metric) \
            else "metric=n/a for this backend"
        self.reg_info_label.setText(
            f"[{result.backend.name}] translation=({tx:.2f}, {ty:.2f}, {tz:.2f}) mm   "
            f"rotation=({rx:.2f}, {ry:.2f}, {rz:.2f})°   {metric_text}")

        self.active_source = "secondary"
        self._refresh_annotation_combo()
        self._rebuild_volume()

        seg_bits = []
        if moving_seg is not None:
            seg_bits.append(f"secondary has its own annotation ({len(moving_seg.segments)} segment(s))")
        if self.transferred_seg is not None:
            seg_bits.append("primary's annotation was transferred onto it")
        seg_note = f"  —  {'; '.join(seg_bits)}." if seg_bits else ""
        self.statusBar().showMessage(
            f"Registered {folder} (detected: {moving_plane}) onto the primary series. "
            f"Switch the 'Show' dropdown to compare.{seg_note}")

    def on_source_changed(self, index: int):
        self.active_source = "primary" if index == 0 else "secondary"
        self._refresh_annotation_combo()
        self._rebuild_volume()

    # -------------------------------------------------------- annotation --

    def _refresh_annotation_combo(self):
        """Rebuild the annotation-overlay options to match what's actually
        available for the CURRENTLY shown source. Primary only ever offers
        its own annotation (if any); the registered secondary offers its
        own actual annotation (if any) and/or the primary's annotation
        transferred onto it (if a primary annotation existed at
        registration time) -- either, both, or neither may be present."""
        prior_choice = self.overlay_choice
        self.annotation_combo.blockSignals(True)
        self.annotation_combo.clear()
        self.annotation_combo.addItem("None", userData=None)
        if self.active_source == "primary":
            if self.primary_seg is not None:
                self.annotation_combo.addItem(
                    f"Primary annotation ({len(self.primary_seg.segments)} segment(s))",
                    userData="primary")
        else:
            if self.secondary_seg_native is not None:
                self.annotation_combo.addItem(
                    f"Secondary's own annotation ({len(self.secondary_seg_native.segments)} segment(s))",
                    userData="secondary_native")
            if self.transferred_seg is not None:
                self.annotation_combo.addItem(
                    "Transferred from primary", userData="transferred")

        restore_idx = self.annotation_combo.findData(prior_choice)
        self.annotation_combo.setCurrentIndex(restore_idx if restore_idx >= 0 else 0)
        self.overlay_choice = self.annotation_combo.currentData()
        self.annotation_combo.blockSignals(False)

        self.compare_btn.setEnabled(
            self.secondary_seg_native is not None or self.transferred_seg is not None)
        self.save_transferred_btn.setEnabled(self.transferred_seg is not None)

        self._rebuild_active_label_array()

    def on_annotation_choice_changed(self, _index: int):
        self.overlay_choice = self.annotation_combo.currentData()
        self._rebuild_active_label_array()
        self._refresh_2d_panels()
        self._update_info_banner()

    def on_mask_display_mode_changed(self, _state: int):
        self._refresh_2d_panels()

    def on_fill_contours_changed(self, _state: int):
        """Re-load all currently-loaded DICOM SEG annotations with the new
        fill_contours setting, then rebuild the active overlay and refresh.
        This keeps the loaded segmentations consistent with the checkbox so
        the Dice / IoU numbers in the Compare dialog always reflect the same
        fill choice for both the primary (transferred) and secondary (actual)
        annotations."""
        fill = self.fill_contours_check.isChecked()

        # Re-load primary seg if we have a primary image + folder
        if self.primary_image is not None and hasattr(self, '_primary_folder') and self._primary_folder:
            try:
                seg_paths = core.find_seg_files(self._primary_folder)
                if seg_paths:
                    self.primary_seg = core.load_seg_as_labelmap(
                        seg_paths, self.primary_image, fill_contours=fill)
                else:
                    self.primary_seg = None
            except Exception:
                pass

        # Re-load secondary native seg if available
        if self.secondary_native_image is not None and self.secondary_folder:
            try:
                seg_paths = core.find_seg_files(self.secondary_folder)
                if seg_paths:
                    self.secondary_seg_native = core.load_seg_as_labelmap(
                        seg_paths, self.secondary_native_image, fill_contours=fill)
                else:
                    self.secondary_seg_native = None
            except Exception:
                pass

        # Re-run the annotation transfer if we have registration result
        if self.primary_seg is not None and self.secondary_native_image is not None \
                and self.registration_result is not None:
            try:
                self.transferred_seg = core.transfer_labelmap(
                    self.primary_seg, self.secondary_native_image,
                    self.registration_result, source_was_fixed=True)
            except Exception:
                self.transferred_seg = None

        self._refresh_annotation_combo()
        self._rebuild_active_label_array()
        self._refresh_2d_panels()

    def _rebuild_active_label_array(self):
        """Resample whichever Segmentation is currently selected for
        overlay onto self.volume's exact grid (nearest-neighbor only --
        see resample_label_like), caching the result so every crosshair
        move/slice scroll doesn't repeat the resampling. Re-run this
        whenever self.volume's grid changes (new load, algorithm change,
        processing) or the overlay choice changes."""
        self._active_label_array = None
        self._active_segments = {}
        if self.volume is None or self.overlay_choice is None:
            self.annotation_info_label.setText("")
            return

        segmentation = {
            "primary": self.primary_seg,
            "secondary_native": self.secondary_seg_native,
            "transferred": self.transferred_seg,
        }.get(self.overlay_choice)
        if segmentation is None:
            self.annotation_info_label.setText("")
            return

        try:
            self._active_label_array = core.resample_label_array_like(
                segmentation.label_image, self.volume.sitk_image)
            self._active_segments = segmentation.segments
            self.annotation_info_label.setText(
                "Segments: " + ", ".join(f"{i.label} (#{n})" for n, i in segmentation.segments.items()))
        except Exception as exc:
            self.annotation_info_label.setText(f"Overlay unavailable: {exc}")
            self._active_label_array = None

    def on_compare_annotations(self):
        if self.secondary_native_image is None:
            QMessageBox.information(self, "No secondary series",
                                     "Register a second series first (Register Second Series...).")
            return
        if self.secondary_seg_native is None and self.transferred_seg is None:
            QMessageBox.information(
                self, "Nothing to compare",
                "Neither the secondary series' own annotation nor a primary annotation "
                "transferred onto it is available. Load a secondary series with its own "
                "_seg.dcm, and/or make sure the primary series had an annotation before "
                "you registered the secondary series.")
            return
        dialog = AnnotationCompareDialog(self.secondary_native_image, self.secondary_seg_native,
                                          self.transferred_seg, parent=self,
                                          initial_plane=self.secondary_plane)
        dialog.exec_()

    def on_save_transferred_annotation(self):
        """Save the transferred annotation (primary → secondary) as a DICOM
        SEG file alongside the secondary series' own DICOM slices.

        The user picks the output file path via a standard save-file dialog.
        The default filename is ``transferred_seg.dcm`` inside the secondary
        series' folder, which is auto-detected by both DicomDoodle and this
        MPR viewer the next time that folder is loaded.

        The underlying segmentation is already on the secondary series' own
        native voxel grid (i.e. ``self.transferred_seg.label_image`` has the
        same size/spacing/origin as ``self.secondary_native_image``), so no
        additional resampling is needed before writing.
        """
        if self.transferred_seg is None:
            QMessageBox.information(self, "No transferred annotation",
                                     "Transfer the primary series' annotation first: load a "
                                     "primary series that has a DICOM SEG annotation, then "
                                     "register a second series onto it.")
            return

        if not self.secondary_folder:
            QMessageBox.warning(self, "Secondary folder unknown",
                                 "The secondary series' folder path was not recorded.  "
                                 "Please re-load the secondary series via "
                                 "'Register Second Series...' and try again.")
            return

        # ---- Ask user for output folder ----
        out_folder = QFileDialog.getExistingDirectory(
            self,
            "Select folder to save slice-wise DICOM SEG files",
            self.secondary_folder,
        )
        if not out_folder:
            return  # user cancelled

        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            saved_count = core.save_transferred_seg_slice_wise(
                segmentation=self.transferred_seg,
                source_folder=self.secondary_folder,
                output_folder=out_folder,
            )
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(
                self, "Save failed",
                f"Could not save the transferred annotation:\n\n{exc}\n\n"
                f"{traceback.format_exc(limit=3)}"
            )
            return

        QApplication.restoreOverrideCursor()
        n_segs = len(self.transferred_seg.segments)
        QMessageBox.information(
            self, "Annotation saved",
            f"Transferred annotation saved successfully.\n\n"
            f"  Output folder : {out_folder}\n"
            f"  Slices saved  : {saved_count}\n"
            f"  Segments      : {n_segs}\n\n"
            f"The files will be auto-detected by DicomDoodle and the MPR viewer\n"
            f"the next time the secondary series folder is loaded."
        )
        self.statusBar().showMessage(f"Transferred annotation saved → {saved_count} slices in {out_folder}")

    # -------------------------------------------------------- volume build --

    def _rebuild_volume(self):
        image = self.primary_image if self.active_source == "primary" else self.secondary_image
        if image is None:
            return
        method: InterpMethod = self.algo_combo.currentData()
        try:
            self.volume = core.build_volume(image, method)
        except Exception as exc:
            QMessageBox.critical(self, "Resampling failed", str(exc))
            return

        if self.processing_config.active_steps():
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()
            try:
                self.processing_result = vproc.run_pipeline(
                    self.volume.array, self.volume.spacing_zyx, self.processing_config,
                    sitk_image_for_hdbet=image)
                self.volume.array = self.processing_result.data
                self.processing_info_label.setText(self.processing_result.summary_text())
            except Exception as exc:
                QMessageBox.critical(self, "Volume processing failed", str(exc))
                self.processing_result = None
                self.processing_info_label.setText("Processing failed -- see error dialog.")
            finally:
                QApplication.restoreOverrideCursor()
        else:
            self.processing_result = None
            self.processing_info_label.setText("")

        z, y, x = self.volume.shape
        sz, sy, sx = self.volume.spacing_zyx
        plane = self.primary_plane if self.active_source == "primary" else self.secondary_plane
        source_label = "Primary" if self.active_source == "primary" else "Registered second series"
        self.statusBar().showMessage(
            f"{source_label} (detected input: {plane})   "
            f"Volume: {x}x{y}x{z} voxels   spacing (x,y,z)=({sx:.3f},{sy:.3f},{sz:.3f}) mm   "
            f"algorithm: {method.value}")

        self.axial_panel.set_range(z, self._idx(0))
        self.coronal_panel.set_range(y, self._idx(1))
        self.sagittal_panel.set_range(x, self._idx(2))
        self.three_d_panel.set_volume(self.volume)
        self._rebuild_active_label_array()
        self._refresh_2d_panels()
        self._update_info_banner()

    def on_algorithm_changed(self, _index: int):
        self._rebuild_volume()

    def on_colormap_changed(self, name: str):
        self.three_d_panel.set_colormap(name)

    def on_open_processing_dialog(self):
        if self.volume is None:
            QMessageBox.information(self, "Load a series first",
                                     "Load a primary series before configuring processing.")
            return
        dlg = ProcessingDialog(self.processing_config, self)
        if dlg.exec_() == QDialog.Accepted and dlg.result_config is not None:
            self.processing_config = dlg.result_config
            self._rebuild_volume()

    # --------------------------------------------------------- cursor logic --

    def _idx(self, axis: int) -> int:
        """Convert normalized fraction -> absolute voxel index for the given
        axis (0=z/axial, 1=y/coronal, 2=x/sagittal) against the CURRENT
        volume's shape. Keeping the cursor as a fraction means it survives
        switching algorithms or sources, since the primary and a
        registered secondary volume always share the same grid."""
        if self.volume is None:
            return 0
        extent = self.volume.shape[axis]
        return int(round(self.frac[axis] * (extent - 1)))

    def on_slice_changed(self, plane: Plane, value: int):
        if self.volume is None:
            return
        axis = {Plane.AXIAL: 0, Plane.CORONAL: 1, Plane.SAGITTAL: 2}[plane]
        extent = self.volume.shape[axis]
        self.frac[axis] = value / max(extent - 1, 1)
        self._refresh_2d_panels()

    def on_position_picked(self, plane: Plane, row: int, col: int):
        """User clicked inside one 2D view: move the crosshair, i.e. update
        the *other two* axes' fractions (see mpr_core.extract_slice for the
        flip convention this inverts)."""
        if self.volume is None:
            return
        z, y, x = self.volume.shape

        if plane is Plane.AXIAL:
            y_idx = np.clip((y - 1) - row, 0, y - 1)
            x_idx = np.clip(col, 0, x - 1)
            self.frac[1] = y_idx / max(y - 1, 1)
            self.frac[2] = x_idx / max(x - 1, 1)
        elif plane is Plane.CORONAL:
            z_idx = np.clip((z - 1) - row, 0, z - 1)
            x_idx = np.clip(col, 0, x - 1)
            self.frac[0] = z_idx / max(z - 1, 1)
            self.frac[2] = x_idx / max(x - 1, 1)
        else:  # SAGITTAL
            z_idx = np.clip((z - 1) - row, 0, z - 1)
            y_idx = np.clip(col, 0, y - 1)
            self.frac[0] = z_idx / max(z - 1, 1)
            self.frac[1] = y_idx / max(y - 1, 1)

        self.axial_panel.set_range(z, self._idx(0))
        self.coronal_panel.set_range(y, self._idx(1))
        self.sagittal_panel.set_range(x, self._idx(2))
        self._refresh_2d_panels()

    def _refresh_2d_panels(self):
        if self.volume is None:
            return
        z, y, x = self.volume.shape
        cz, cy, cx = self._idx(0), self._idx(1), self._idx(2)

        ax_img = core.extract_slice(self.volume, Plane.AXIAL, cz)
        cor_img = core.extract_slice(self.volume, Plane.CORONAL, cy)
        sag_img = core.extract_slice(self.volume, Plane.SAGITTAL, cx)

        ax_rgb = core.gray_to_rgb(core.window_to_uint8(ax_img))
        cor_rgb = core.gray_to_rgb(core.window_to_uint8(cor_img))
        sag_rgb = core.gray_to_rgb(core.window_to_uint8(sag_img))

        if self._active_label_array is not None:
            if self.show_filled_mask_check.isChecked():
                blend = lambda rgb, lbl: core.blend_label_overlay(rgb, lbl, self._active_segments)
            else:
                blend = lambda rgb, lbl: core.blend_label_boundary(rgb, lbl, self._active_segments)
            ax_rgb = blend(ax_rgb, core.extract_label_slice(self._active_label_array, Plane.AXIAL, cz))
            cor_rgb = blend(cor_rgb, core.extract_label_slice(self._active_label_array, Plane.CORONAL, cy))
            sag_rgb = blend(sag_rgb, core.extract_label_slice(self._active_label_array, Plane.SAGITTAL, cx))

        self.axial_panel.show_image(
            ax_rgb, core.display_aspect(self.volume, Plane.AXIAL), f"slice {cz + 1}/{z}")
        self.coronal_panel.show_image(
            cor_rgb, core.display_aspect(self.volume, Plane.CORONAL), f"slice {cy + 1}/{y}")
        self.sagittal_panel.show_image(
            sag_rgb, core.display_aspect(self.volume, Plane.SAGITTAL), f"slice {cx + 1}/{x}")

        # crosshairs: each panel shows where the OTHER two current indices project
        self.axial_panel.set_crosshair(row=(y - 1) - cy, col=cx)
        self.coronal_panel.set_crosshair(row=(z - 1) - cz, col=cx)
        self.sagittal_panel.set_crosshair(row=(z - 1) - cz, col=cy)

    # ------------------------------------------------------------ maximize --

    def on_toggle_maximize(self, panel: BasePanel):
        if self.maximized_panel is panel:
            self.max_layout.removeWidget(panel)
            row, col = self.panel_positions[panel]
            self.grid_layout.addWidget(panel, row, col)
            self.stacked.setCurrentIndex(0)
            panel.set_maximized(False)
            self.maximized_panel = None
        elif self.maximized_panel is None:
            self.grid_layout.removeWidget(panel)
            self.max_layout.addWidget(panel)
            self.stacked.setCurrentIndex(1)
            panel.set_maximized(True)
            self.maximized_panel = panel
        # else: another panel is already maximized -> ignore until restored

        # VTK's render window doesn't always repaint immediately after being
        # reparented between layouts; nudge it so the 3D view doesn't look
        # frozen/blank right after a maximize or restore.
        if isinstance(panel, ThreeDPanel):
            panel.refresh_after_resize()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self.maximized_panel is not None:
            self.on_toggle_maximize(self.maximized_panel)
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        # Explicitly finalize the VTK render window; otherwise VTK can
        # segfault on interpreter exit on some platforms.
        if self.three_d_panel._mode_vtk:
            try:
                self.three_d_panel.vtk_widget.GetRenderWindow().Finalize()
                self.three_d_panel.interactor.TerminateApp()
            except Exception:
                pass
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
