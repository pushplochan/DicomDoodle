"""
The right-hand panel of the Multiplanar & 3D Viewer (from the 3D Model
Viewer) — everything about the series of the 3D model in one place:

    SeriesCard   the axial / coronal / sagittal folders and "Build 3D model"
    VolumeCard   lesion volume of each series and their average (formerly the
                 separate Lesion Volume Calculator), with Copy and Export CSV
    MetricsCard  shape and texture of the 3D lesion, "Show lesion surface in 3D"
    SidePanel    the three cards, top to bottom
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGridLayout,
                             QHBoxLayout, QLineEdit, QVBoxLayout, QWidget)

from ....core.dicom_series import SegFileFilter
from ....core.tumor_metrics import Metric, format_metric
from ....core.volume_calculator import ORIENTATIONS, OrientationVolume, average_volume_cm3
from ...guide import help_button
from ...qt_utils import label, make_button, make_tool_button
from ...widgets.cards import Card

# metric keys of core.tumor_metrics -> names in the panel
METRIC_NAMES = {"Volume (cc)": "Volume (3D model)", "Surface Area (mm²)": "Surface area",
                "Sphericity": "Sphericity", "Texture Contrast": "Texture contrast",
                "Texture Homogeneity": "Texture homogeneity"}


class SeriesCard(Card):
    """Which series to load: one folder per orientation, the annotation files, Build."""

    buildRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Series", parent)
        self.add_header_widget(help_button("viewer3d"))
        self.add(label("Same patient · one folder per orientation", "Faint", wrap=True))
        self.edits: Dict[str, QLineEdit] = {}
        grid = QGridLayout()
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(6)
        for row, name in enumerate(ORIENTATIONS):
            edit = QLineEdit()
            edit.setPlaceholderText("Not selected")
            edit.textChanged.connect(self._update)
            browse = make_tool_button("folder", f"Choose the {name.lower()} series folder")
            clear = make_tool_button("x", f"Remove the {name.lower()} series")
            browse.clicked.connect(lambda _c=False, n=name: self._browse(n))
            clear.clicked.connect(lambda _c=False, e=edit: e.clear())
            grid.addWidget(label(name, "Muted"), row, 0)
            grid.addWidget(edit, row, 1)
            grid.addWidget(browse, row, 2)
            grid.addWidget(clear, row, 3)
            self.edits[name] = edit
        grid.setColumnStretch(1, 1)
        self.add_layout(grid)

        self.filter_combo = QComboBox()
        for option in (SegFileFilter.ALL, SegFileFilter.NATIVE, SegFileFilter.SEGMENTATION):
            self.filter_combo.addItem(option.value, option)
        self.filter_combo.setToolTip("Which annotation files show and measure the lesion")
        self.add(label("Annotation files", "Muted"))
        self.add(self.filter_combo)
        self.build_btn = make_button("Build 3D model", "cube", role="primary",
                                     tooltip="Measure the lesion volume and build the 3D model")
        self.build_btn.clicked.connect(self.buildRequested.emit)
        self.add(self.build_btn)
        self.info = label("", "Faint", wrap=True)
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.info.hide()
        self.add(self.info)
        self._update()

    def set_folder(self, name: str, folder: str) -> None:
        edit = self.edits[name]
        edit.setText(folder)
        edit.setCursorPosition(len(folder))       # show the end of the path: the series name
        edit.setToolTip(folder)

    def paths(self) -> Dict[str, str]:
        return {name: edit.text().strip() for name, edit in self.edits.items()}

    def seg_filter(self) -> SegFileFilter:
        return self.filter_combo.currentData()

    def show_info(self, text: str) -> None:
        """Reference grid, voxel spacing and warnings of the last build."""
        self.info.setText(text)
        self.info.setVisible(bool(text))

    def _browse(self, name: str) -> None:
        current = self.edits[name].text().strip() or next((p for p in self.paths().values() if p), "")
        start = os.path.dirname(current) if current else os.path.expanduser("~")
        folder = QFileDialog.getExistingDirectory(self.window(), f"Select the {name} DICOM Folder", start)
        if folder:
            self.set_folder(name, folder)

    def _update(self, *_args) -> None:
        self.build_btn.setEnabled(any(self.paths().values()))


class VolumeCard(Card):
    """Lesion volume of every loaded series and their average."""

    def __init__(self, parent=None):
        super().__init__("Lesion volume", parent)
        self.add_header_widget(help_button("volume"))
        self.results: List[OrientationVolume] = []
        self.total = label("—", "Title")
        self.total.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.caption = label("Build the 3D model to measure the lesion.", "Muted", wrap=True)
        self.rows = QVBoxLayout()
        self.rows.setSpacing(8)
        self.notes = label("", "Faint", wrap=True)
        self.notes.hide()
        self.copy_btn = make_button("Copy", "copy", tooltip="Copy the table, to paste into a spreadsheet")
        self.export_btn = make_button("Export CSV…", "download")
        self.copy_btn.clicked.connect(self.copy)
        self.export_btn.clicked.connect(self.export_csv)
        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        buttons.addWidget(self.copy_btn, 1)
        buttons.addWidget(self.export_btn, 1)
        self.add(self.total)
        self.add(self.caption)
        self.add_layout(self.rows)
        self.add(self.notes)
        self.add_layout(buttons)
        self._set_buttons()

    def show_results(self, results: List[OrientationVolume]) -> None:
        self.results = results
        while self.rows.count():
            item = self.rows.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        counted = [r for r in results if r.counted]
        if counted:
            self.total.setText(f"{average_volume_cm3(results):.2f} cm³")
            self.caption.setText("Lesion volume from the annotated slices" if len(counted) == 1 else
                                 f"Average of {len(counted)} annotated series")
        else:
            self.total.setText("—")
            self.caption.setText("No lesion annotation was found in the loaded series.")
        for result in results:
            self.rows.addWidget(self._row(result))
        notes = [f"{r.orientation}: {w}" for r in results for w in r.warnings]
        self.notes.setText("\n".join(notes))
        self.notes.setVisible(bool(notes))
        self._set_buttons()

    @staticmethod
    def _row(r: OrientationVolume) -> QWidget:
        if r.error:
            value, detail = "not measured", r.error
        elif not r.annotated_slices:
            value, detail = "—", f"No annotated slices among {r.slices}"
        else:
            value = f"{r.volume_cm3:.2f} cm³"
            detail = (f"{r.annotated_slices} of {r.slices} slices annotated\nVoxel {r.pixel_spacing[0]:.3g} × "
                      f"{r.pixel_spacing[1]:.3g} × {r.slice_thickness:.3g} mm  ({r.voxel_mm3:.3g} mm³)")
        row = QWidget()
        row.setToolTip(r.folder if r.error else f"{r.folder}\n{r.voxels:,} lesion voxels")
        layout = QVBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        head = QHBoxLayout()
        head.addWidget(label(r.orientation, "Value"))
        head.addStretch(1)
        head.addWidget(label(value, "Value" if r.counted else "Muted"))
        layout.addLayout(head)
        layout.addWidget(label(detail, "Faint", wrap=True))
        return row

    def _set_buttons(self) -> None:
        measured = any(not r.error for r in self.results)
        self.copy_btn.setEnabled(measured)
        self.export_btn.setEnabled(measured)

    def _table(self) -> List[list]:
        rows = [[r.orientation, r.slices, r.annotated_slices, r.pixel_spacing[0], r.pixel_spacing[1],
                 r.slice_thickness, round(r.voxel_mm3, 6), r.voxels, round(r.volume_cm3, 4)]
                for r in self.results if not r.error]
        rows.append(["Average", "", "", "", "", "", "", "", round(average_volume_cm3(self.results), 4)])
        return rows

    def copy(self) -> None:
        if self.results:
            header = "Orientation\tSlices\tAnnotated\tSpacing X\tSpacing Y\tThickness\tVoxel mm3\tVoxels\tVolume cm3"
            QApplication.clipboard().setText("\n".join([header] + ["\t".join(map(str, r)) for r in self._table()]))

    def export_csv(self) -> None:
        if not self.results:
            return
        path, _ = QFileDialog.getSaveFileName(self.window(), "Export lesion volumes", "lesion_volumes.csv",
                                              "CSV (*.csv)")
        if path:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["orientation", "slices", "annotated_slices", "spacing_x_mm", "spacing_y_mm",
                                 "slice_thickness_mm", "voxel_mm3", "lesion_voxels", "volume_cm3"])
                writer.writerows(self._table())


class MetricsCard(Card):
    """Shape and texture metrics of the 3D lesion."""

    def __init__(self, parent=None):
        super().__init__("3D lesion metrics", parent)
        self.metrics: Optional[Dict[str, Metric]] = None
        self.form = QFormLayout()
        self.form.setVerticalSpacing(5)
        self.add_layout(self.form)
        self.hint = label("Build the 3D model to measure the lesion's shape.", "Faint", wrap=True)
        self.add(self.hint)
        self.show_surface = QCheckBox("Show lesion surface in 3D")
        self.show_surface.setEnabled(False)
        self.add(self.show_surface)

    def show_metrics(self, metrics: Optional[Dict[str, Metric]], note: str) -> None:
        self.metrics = metrics
        while self.form.rowCount():
            self.form.removeRow(0)
        for key, value in (metrics or {}).items():
            value_label = label(format_metric(key, value), "Value")
            value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.form.addRow(label(METRIC_NAMES.get(key, key), "Muted"), value_label)
        self.hint.setText(note)
        self.show_surface.setEnabled(metrics is not None)


class SidePanel(QWidget):
    """Series, lesion volume and 3D lesion metrics, top to bottom."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.series = SeriesCard()
        self.volume = VolumeCard()
        self.metrics = MetricsCard()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        for card in (self.series, self.volume, self.metrics):
            layout.addWidget(card)
        layout.addStretch(1)
