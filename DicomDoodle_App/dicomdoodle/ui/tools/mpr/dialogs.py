"""
MPR dialog (from the original mpr_gui.py):

* :class:`AnnotationCompareDialog` — the secondary series' own annotation
  next to the primary's annotation transferred onto it, on one shared
  slice control, with the whole-volume Dice and per-slice IoU
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QCheckBox, QComboBox, QDialog, QHBoxLayout, QSlider, QSpinBox, QVBoxLayout, QWidget

from ....core import mpr
from ...qt_utils import label, make_button


class AnnotationCompareDialog(QDialog):
    def __init__(self, secondary_image, actual: Optional[mpr.Segmentation],
                 transferred: Optional[mpr.Segmentation], parent=None, initial_plane: str = "Axial"):
        super().__init__(parent)
        self.setWindowTitle("Compare Annotations — Secondary Series")
        self.resize(1150, 700)
        self.actual, self.transferred = actual, transferred
        # always compared at native resolution, independent of the display algorithm
        self.volume = mpr.build_volume(secondary_image, mpr.InterpMethod.NONE_NATIVE)
        self.actual_array = (mpr.resample_label_array_like(actual.label_image, self.volume.sitk_image)
                             if actual is not None else None)
        self.transferred_array = (mpr.resample_label_array_like(transferred.label_image, self.volume.sitk_image)
                                  if transferred is not None else None)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        if self.actual_array is not None and self.transferred_array is not None:
            dice = mpr.dice_coefficient(self.transferred_array, self.actual_array)
            dice_text = f"Overall Dice (whole volume, transferred vs. actual): {dice:.3f}"
        else:
            dice_text = "Dice needs both an actual and a transferred annotation; only one is available."
        layout.addWidget(label(dice_text, "Heading", wrap=True))
        self.slice_metric = label("", "Value")
        layout.addWidget(self.slice_metric)

        controls = QHBoxLayout()
        controls.addWidget(label("Plane", "Muted"))
        self.plane_combo = QComboBox()
        for name, plane in (("Axial", mpr.Plane.AXIAL), ("Coronal", mpr.Plane.CORONAL),
                            ("Sagittal", mpr.Plane.SAGITTAL)):
            self.plane_combo.addItem(name, plane)
        self.plane_combo.setCurrentIndex({"Axial": 0, "Coronal": 1, "Sagittal": 2}.get(initial_plane, 0))
        self.filled = QCheckBox("Show filled mask")
        self.slider = QSlider(Qt.Horizontal)
        self.spin = QSpinBox()
        self.spin.setFixedWidth(78)
        controls.addWidget(self.plane_combo)
        controls.addWidget(self.filled)
        controls.addWidget(self.slider, 1)
        controls.addWidget(self.spin)
        layout.addLayout(controls)

        panels = QHBoxLayout()
        self._left = self._make_panel("Secondary — actual annotation" + ("" if actual else "  (none loaded)"))
        self._right = self._make_panel("Secondary — transferred from primary" + ("" if transferred else "  (n/a)"))
        panels.addWidget(self._left["container"])
        panels.addWidget(self._right["container"])
        layout.addLayout(panels, 1)
        close = make_button("Close", role="primary")
        close.clicked.connect(self.accept)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(close)
        layout.addLayout(row)

        self.plane_combo.currentIndexChanged.connect(self._plane_changed)
        self.filled.stateChanged.connect(lambda _s: self._refresh())
        self.slider.valueChanged.connect(self.spin.setValue)
        self.spin.valueChanged.connect(self.slider.setValue)
        self.slider.valueChanged.connect(lambda _v: self._refresh())
        self._plane_changed()

    @staticmethod
    def _make_panel(title: str) -> dict:
        container = QWidget()
        v = QVBoxLayout(container)
        v.setContentsMargins(0, 0, 0, 0)
        figure = Figure(facecolor="black")
        ax = figure.add_axes([0, 0, 1, 1])
        ax.set_facecolor("black")
        ax.axis("off")
        canvas = FigureCanvasQTAgg(figure)
        v.addWidget(label(title, "Value"))
        v.addWidget(canvas, 1)
        return {"container": container, "ax": ax, "canvas": canvas, "im": None}

    def _plane_changed(self, *_args) -> None:
        plane = self.plane_combo.currentData()
        extent = self.volume.shape[{mpr.Plane.AXIAL: 0, mpr.Plane.CORONAL: 1, mpr.Plane.SAGITTAL: 2}[plane]]
        for widget in (self.slider, self.spin):
            widget.blockSignals(True)
            widget.setRange(0, max(extent - 1, 0))
            widget.setValue(extent // 2)
            widget.blockSignals(False)
        self._refresh()

    def _refresh(self) -> None:
        plane, idx = self.plane_combo.currentData(), self.slider.value()
        base = mpr.gray_to_rgb(mpr.window_to_uint8(mpr.extract_slice(self.volume, plane, idx)))
        aspect = mpr.display_aspect(self.volume, plane)
        actual = mpr.extract_label_slice(self.actual_array, plane, idx) if self.actual_array is not None else None
        moved = (mpr.extract_label_slice(self.transferred_array, plane, idx)
                 if self.transferred_array is not None else None)
        if actual is not None and moved is not None:
            iou = mpr.iou_coefficient(moved, actual)
            self.slice_metric.setText(f"This slice ({plane.value}, index {idx}): "
                                      + ("no annotation on either side — IoU n/a" if np.isnan(iou)
                                         else f"IoU (transferred vs. actual) = {iou:.3f}"))
        else:
            self.slice_metric.setText(f"This slice ({plane.value}, index {idx}): IoU n/a")
        for panel, label_slice, seg in ((self._left, actual, self.actual), (self._right, moved, self.transferred)):
            rgb = base
            if label_slice is not None and seg is not None:
                rgb = (mpr.blend_label_overlay(base.copy(), label_slice, seg.segments) if self.filled.isChecked()
                       else mpr.blend_label_boundary(base.copy(), label_slice, seg.segments, aspect=aspect))
            if panel["im"] is None:
                panel["im"] = panel["ax"].imshow(rgb, aspect=aspect, interpolation="nearest")
            else:
                panel["im"].set_data(rgb)
                panel["im"].set_extent((-0.5, rgb.shape[1] - 0.5, rgb.shape[0] - 0.5, -0.5))
                panel["ax"].set_aspect(aspect)
            panel["canvas"].draw_idle()
