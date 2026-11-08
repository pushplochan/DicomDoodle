"""
"Plot Results": the current slice shown twice, side by side —
left with the segmented regions blended in (closed hand-drawn outlines
filled), right with only the outlines exactly as they are saved to the
annotation file. Both views zoom independently; either can be exported
as a PNG.
"""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QDialog, QFileDialog, QHBoxLayout, QLabel, QSlider, QSpinBox,
                             QVBoxLayout)

from ...core import mask_ops
from ..qt_utils import label, make_button, swatch_icon
from ..widgets.cards import Card
from ..widgets.image_canvas import ImageCanvas


class ResultsDialog(QDialog):
    def __init__(self, image: np.ndarray, label_map: np.ndarray, outline_map: np.ndarray, lut: np.ndarray,
                 legend: Sequence[tuple], title: str, default_name: str = "slice", parent=None):
        """``label_map``: the filled regions; ``outline_map``: the outlines as saved."""
        super().__init__(parent)
        self.setWindowTitle(f"Segmentation Results — {title}")
        self.resize(1200, 700)
        self._image, self._labels, self._outline, self._lut = image, label_map, outline_map, lut
        self._default_name = default_name

        self.filled_view = ImageCanvas()
        self.boundary_view = ImageCanvas()
        for view in (self.filled_view, self.boundary_view):
            view.zoom_mode = True
            view.setMinimumSize(380, 380)

        self.opacity = QSlider(Qt.Horizontal)
        self.opacity.setRange(5, 100)
        self.opacity.setValue(35)
        self.opacity.setFixedWidth(160)
        self.thickness = QSpinBox()
        self.thickness.setRange(1, 6)
        self.thickness.setValue(2)
        self.thickness.setSuffix(" px")

        controls = QHBoxLayout()
        controls.addWidget(label("Fill opacity", "Muted"))
        controls.addWidget(self.opacity)
        controls.addSpacing(16)
        controls.addWidget(label("Boundary width", "Muted"))
        controls.addWidget(self.thickness)
        controls.addSpacing(16)
        for name, rgb in legend:
            chip = QLabel()
            chip.setPixmap(swatch_icon(rgb, 12).pixmap(12, 12))
            controls.addWidget(chip)
            controls.addWidget(label(name))
            controls.addSpacing(8)
        controls.addStretch(1)

        left, right = Card("Image with segmented region"), Card("Image with boundary (as saved)")
        left.add(self.filled_view, 1)
        right.add(self.boundary_view, 1)
        save_left = make_button("Save image…", "download")
        save_right = make_button("Save image…", "download")
        left.add_header_widget(save_left)
        right.add_header_widget(save_right)

        views = QHBoxLayout()
        views.setSpacing(12)
        views.addWidget(left, 1)
        views.addWidget(right, 1)

        close_btn = make_button("Close", role="primary")
        bottom = QHBoxLayout()
        bottom.addWidget(label("Ctrl/mouse wheel zooms · middle-drag pans · double middle-click fits", "Faint"))
        bottom.addStretch(1)
        bottom.addWidget(close_btn)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(12)
        root.addLayout(controls)
        root.addLayout(views, 1)
        root.addLayout(bottom)

        self.opacity.valueChanged.connect(self._render)
        self.thickness.valueChanged.connect(self._render)
        save_left.clicked.connect(lambda: self._save(self._filled, "segmented"))
        save_right.clicked.connect(lambda: self._save(self._boundary, "boundary"))
        close_btn.clicked.connect(self.accept)
        self._render()

    def _render(self) -> None:
        self._filled = mask_ops.blend_filled(self._image, self._labels, self._lut, self.opacity.value() / 100)
        self._boundary = mask_ops.overlay_labels(self._image, mask_ops.thicken(self._outline, self.thickness.value()),
                                                 self._lut)
        self.filled_view.set_image(self._filled)
        self.boundary_view.set_image(self._boundary)

    def _save(self, rgb: np.ndarray, suffix: str) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save image", f"{self._default_name}_{suffix}.png",
                                              "PNG image (*.png)")
        if path:
            if not path.lower().endswith(".png"):
                path += ".png"
            cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
