"""Series and current-slice details shown in the right-hand panel."""
from __future__ import annotations

import os
from typing import Dict, Optional, Sequence, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFormLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ...core.dicom_series import DicomSeries
from ..qt_utils import label, set_badge
from .cards import Card


def _value_label() -> QLabel:
    widget = label("—", "Value")
    widget.setTextInteractionFlags(Qt.TextSelectableByMouse)
    widget.setWordWrap(True)
    return widget


class _Form(QWidget):
    def __init__(self, keys: Sequence[str], parent=None):
        super().__init__(parent)
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(5)
        form.setLabelAlignment(Qt.AlignLeft)
        self.values: Dict[str, QLabel] = {}
        for key in keys:
            self.values[key] = _value_label()
            form.addRow(label(key, "Muted"), self.values[key])

    def set(self, key: str, text: str) -> None:
        self.values[key].setText(text or "—")


class InfoPanel(QWidget):
    SERIES_KEYS = ("Patient", "Patient ID", "Study date", "Modality", "Series", "Plane",
                   "Matrix", "Pixel spacing", "Slice thickness", "Slices")
    SLICE_KEYS = ("File", "Instance", "Labels")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.series_form = _Form(self.SERIES_KEYS)
        self.slice_form = _Form(self.SLICE_KEYS)
        self.status_badge = QLabel()
        self.status_badge.hide()

        series_card = Card("Series")
        series_card.add(self.series_form)
        slice_card = Card("Current slice")
        badge_row = QHBoxLayout()
        badge_row.addWidget(self.status_badge)
        badge_row.addStretch(1)
        slice_card.add_layout(badge_row)
        slice_card.add(self.slice_form)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(series_card)
        layout.addWidget(slice_card)

    def show_series(self, series: Optional[DicomSeries]) -> None:
        if series is None:
            for key in self.SERIES_KEYS:
                self.series_form.set(key, "")
            return
        info = series.info
        self.series_form.set("Patient", info.patient_name.replace("^", " ").strip())
        self.series_form.set("Patient ID", info.patient_id)
        date = info.study_date
        self.series_form.set("Study date", f"{date[6:8]}-{date[4:6]}-{date[0:4]}" if len(date) == 8 else date)
        self.series_form.set("Modality", " · ".join(x for x in (info.modality, info.manufacturer) if x))
        self.series_form.set("Series", info.series_description or series.name)
        self.series_form.set("Plane", info.plane)
        self.series_form.set("Matrix", f"{info.columns} × {info.rows}" if info.rows else "")
        self.series_form.set("Pixel spacing", f"{info.pixel_spacing[0]:.3f} × {info.pixel_spacing[1]:.3f} mm"
                             if info.pixel_spacing else "")
        self.series_form.set("Slice thickness", f"{info.slice_thickness:g} mm" if info.slice_thickness else "")
        self.series_form.set("Slices", str(len(series)))

    def show_slice(self, path: str = "", instance: Optional[int] = None,
                   labels_present: Sequence[str] = (), status: Tuple[Optional[str], str] = (None, "")) -> None:
        self.slice_form.set("File", os.path.basename(path))
        self.slice_form.set("Instance", "" if instance is None else str(instance))
        self.slice_form.set("Labels", ", ".join(labels_present) if labels_present else "none")
        set_badge(self.status_badge, *status)
