"""
Annotated DICOM Viewer — review a series together with its saved
annotation files (original "Annotated Dicom Viewer").

* every slice with an annotation shows it in its label colors
* "Show Mask" fills each closed region with a distinct color (original)
* Zoom mode: the wheel zooms instead of changing slice (original); Ctrl +
  wheel always zooms, middle-drag pans
* the segment category / type of each file are shown, and "Add Ontology"
  writes standard coded terms (DICOM's code dictionary, else BioPortal)
  into the annotation files
* choose which annotation files to read when a folder holds several kinds
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from PyQt5.QtCore import QSize, Qt, QTimer
from PyQt5.QtWidgets import (QAction, QComboBox, QDockWidget, QFileDialog, QHeaderView, QLabel, QMainWindow,
                             QSizePolicy, QSlider, QTableWidget, QTableWidgetItem, QToolBar, QVBoxLayout,
                             QWidget)

from ...core import mask_ops
from ...core.dicom_series import DicomSeries, SegFileFilter, annotation_files_in
from ...core.labels import EMPTY_SEGMENT_NAMES, LabelRegistry
from ...core.seg_io import SegmentRecord, read_segmentation
from ...settings import DEFAULTS
from ...workers import run_task
from ..dialogs.progress_dialog import TaskProgressDialog
from ..guide import help_action
from ..icons import bind_icon
from ..label_manager import LabelManager
from ..qt_utils import label, show_error, show_info, swatch_icon
from ..widgets.cards import Card, panel_body
from ..widgets.image_canvas import ImageCanvas
from ..widgets.slice_navigator import SliceNavigator

log = logging.getLogger("dicomdoodle")


@dataclass
class SliceAnnotations:
    label_map: Optional[np.ndarray] = None
    segments: List[SegmentRecord] = field(default_factory=list)
    files: List[str] = field(default_factory=list)


def load_annotations(ctx, series: DicomSeries, seg_filter: SegFileFilter,
                     registry: LabelRegistry) -> Dict[int, SliceAnnotations]:
    """Read every slice's annotation file(s) (background job)."""
    result: Dict[int, SliceAnnotations] = {}
    for i in range(len(series)):
        ctx(i, len(series), f"Reading annotations {i + 1}/{len(series)}…")
        entry = SliceAnnotations()
        for path in seg_filter.candidates_for(series.file(i)):
            if not os.path.isfile(path):
                continue
            try:
                data = read_segmentation(path, registry, series.image(i).shape)
            except Exception as exc:
                log.warning("Could not read %s: %s", os.path.basename(path), exc)
                continue
            if entry.label_map is None:
                entry.label_map = data.label_map.copy()
            else:
                empty = entry.label_map == 0
                entry.label_map[empty] = data.label_map[empty]
            entry.segments += data.segments
            entry.files.append(path)
        if entry.files:
            result[i] = entry
    return result


def load_viewer_data(ctx, folder: str, seg_filter: SegFileFilter, registry: LabelRegistry):
    series = DicomSeries.load(folder, ctx)
    return series, load_annotations(ctx, series, seg_filter, registry)


class OverlayViewerWindow(QMainWindow):
    def __init__(self, labels: LabelManager, settings, initial_folder: str = ""):
        super().__init__()
        self.labels, self.settings = labels, settings
        self.series: Optional[DicomSeries] = None
        self.annotations: Dict[int, SliceAnnotations] = {}
        self.index = 0
        self.setWindowTitle("Annotated DICOM Viewer")
        self.resize(1320, 880)

        self.canvas = ImageCanvas()
        self.canvas.placeholder = "Open a folder that contains DICOM slices and their _seg.dcm files"
        self.navigator = SliceNavigator()
        center = QWidget()
        layout = QVBoxLayout(center)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.canvas, 1)
        nav_holder = QWidget()
        nav_holder.setObjectName("DockBody")
        nav_layout = QVBoxLayout(nav_holder)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.addWidget(self.navigator)
        layout.addWidget(nav_holder)
        self.setCentralWidget(center)

        self._build_toolbar()
        self._build_side_panel()
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(120)
        self._redraw_timer.timeout.connect(self.render_overlay)

        self.navigator.valueChanged.connect(self.show_slice)
        self.canvas.sliceScrolled.connect(lambda step: self.show_slice(self.index + step))
        self.canvas.zoomChanged.connect(lambda _z: self._redraw_timer.start())
        self.labels.labelsChanged.connect(self.render_overlay)
        if initial_folder:
            QTimer.singleShot(0, lambda: self.open_folder(initial_folder))

    # ---------------------------------------------------------------- layout
    def _build_toolbar(self) -> None:
        bar = QToolBar("Viewer", self)
        bar.setMovable(False)
        bar.setIconSize(QSize(20, 20))
        bar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(bar)

        def add(text, icon_name, slot, checkable=False, tip=""):
            action = QAction(text, self)
            bind_icon(action, icon_name)
            action.setCheckable(checkable)
            action.setToolTip(tip or text)
            action.triggered.connect(slot)
            bar.addAction(action)
            return action

        add("Load DICOM Folder", "folder-open", lambda: self.open_folder_dialog())
        bar.addSeparator()
        self.zoom_action = add("Zoom", "zoom-in", self._toggle_zoom_mode, checkable=True,
                               tip="Zoom mode: the mouse wheel zooms instead of changing slice")
        add("Fit", "fit", lambda: self.canvas.fit_to_window())
        bar.addSeparator()
        self.mask_action = add("Show Mask", "droplet", lambda _c: self.render_overlay(), checkable=True,
                               tip="Fill every annotated region with its own color")
        bar.addWidget(QLabel("  Opacity "))
        self.opacity = QSlider(Qt.Horizontal)
        self.opacity.setRange(10, 100)
        self.opacity.setValue(100)
        self.opacity.setFixedWidth(120)
        self.opacity.valueChanged.connect(lambda _v: self.render_overlay())
        bar.addWidget(self.opacity)
        bar.addSeparator()
        bar.addWidget(QLabel("  Annotation files "))
        self.filter_combo = QComboBox()
        for option in (SegFileFilter.NATIVE, SegFileFilter.SEGMENTATION, SegFileFilter.ALL):
            self.filter_combo.addItem(option.value, option)
        self.filter_combo.currentIndexChanged.connect(lambda _i: self.reload_annotations())
        bar.addWidget(self.filter_combo)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        bar.addWidget(spacer)
        self.ontology_action = add("Add Ontology", "share", lambda: self.open_ontology_search(),
                                   tip="Search DICOM codes (and BioPortal) and write segment category / type "
                                       "into the files")
        bar.addAction(help_action(self, "viewer"))

    def _build_side_panel(self) -> None:
        body = panel_body()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        self.file_label = label("—", "Value", wrap=True)
        file_card = Card("Annotation file")
        file_card.add(self.file_label)
        self.segments_table = QTableWidget(0, 3)
        self.segments_table.setHorizontalHeaderLabels(["Label", "Category", "Type"])
        self.segments_table.verticalHeader().hide()
        self.segments_table.setEditTriggers(QTableWidget.NoEditTriggers)
        header = self.segments_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.segments_table.setWordWrap(True)
        self.segments_table.setMinimumHeight(160)
        segments_card = Card("Segments on this slice")
        segments_card.add(self.segments_table)
        self.summary_label = label("", "Muted", wrap=True)
        summary_card = Card("Series")
        summary_card.add(self.summary_label)
        for card in (file_card, segments_card, summary_card):
            layout.addWidget(card)
        layout.addStretch(1)
        dock = QDockWidget("Annotation", self)
        dock.setObjectName("OverlayInfo")
        dock.setWidget(body)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetClosable)
        dock.setMinimumWidth(320)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    # --------------------------------------------------------------- loading
    @property
    def seg_filter(self) -> SegFileFilter:
        return self.filter_combo.currentData()

    def open_folder_dialog(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select DICOM Folder",
                                                  self.series.folder if self.series else
                                                  self.settings.get("last_dir") or os.path.expanduser("~"))
        if folder:
            self.open_folder(folder)

    def open_folder(self, folder: str) -> None:
        dialog = TaskProgressDialog("Loading annotated series", self)
        state = {}

        def loaded(result) -> None:
            state["data"] = result
            dialog.accept()

        def failed(message: str, details: str) -> None:
            dialog.accept()
            show_error(self, "Could not load the folder", message, details)

        task = run_task(load_viewer_data, folder, self.seg_filter, self.labels.registry,
                        on_result=loaded, on_error=failed, on_progress=dialog.set_progress,
                        on_cancelled=dialog.accept)
        dialog.cancelRequested.connect(task.cancel)
        dialog.exec_()
        if "data" in state:
            self.series, self.annotations = state["data"]
            self.labels.notify_transient_added()
            self.navigator.set_count(len(self.series))
            self.index = -1
            self.canvas.fit_to_window()
            self.show_slice(0)
            self._update_overview()

    def reload_annotations(self) -> None:
        if self.series is None:
            return
        dialog = TaskProgressDialog("Reading annotations", self)
        state = {}
        task = run_task(load_annotations, self.series, self.seg_filter, self.labels.registry,
                        on_result=lambda r: (state.update(data=r), dialog.accept()),
                        on_error=lambda m, d: (dialog.accept(), show_error(self, "Annotations", m, d)),
                        on_progress=dialog.set_progress, on_cancelled=dialog.accept)
        dialog.cancelRequested.connect(task.cancel)
        dialog.exec_()
        if "data" in state:
            self.annotations = state["data"]
            self.labels.notify_transient_added()
            self._update_overview()
            self.render_overlay()
            self._update_info()

    def _update_overview(self) -> None:
        annotated = {i for i, a in self.annotations.items() if a.label_map is not None and a.label_map.any()}
        blank = set(self.annotations) - annotated
        self.navigator.set_markers("blank", blank, "text_muted", "Blank mask", "dot")
        self.navigator.set_markers("seg", annotated, "success", "Annotated")
        missing = len(self.series) - len(self.annotations)
        self.summary_label.setText(f"{len(self.series)} slices · {len(annotated)} annotated · "
                                   f"{len(blank)} blank masks"
                                   + (f" · {missing} without an annotation file" if missing else ""))
        self.setWindowTitle(f"Annotated DICOM Viewer — {self.series.name}")

    # --------------------------------------------------------------- display
    def show_slice(self, index: int) -> None:
        if self.series is None:
            return
        index = max(0, min(len(self.series) - 1, int(index)))
        if index == self.index:
            return
        self.index = index
        self.navigator.set_value(index, emit=False)
        self.canvas.set_image(self.series.image(index))
        self.render_overlay()
        self._update_info()

    def render_overlay(self) -> None:
        if self.series is None:
            return
        entry = self.annotations.get(self.index)
        if entry is None or entry.label_map is None or not entry.label_map.any():
            self.canvas.set_overlay(None)
            return
        alpha = self.opacity.value() / 100
        if self.mask_action.isChecked():
            colors = mask_ops.region_fill(entry.label_map)
            rgba = np.zeros(colors.shape[:2] + (4,), dtype=np.uint8)
            filled = colors.any(axis=-1)
            rgba[filled, :3] = colors[filled]
            rgba[filled, 3] = int(255 * alpha)
        else:
            grow = max(1, int(round(1.6 / max(self.canvas.zoom, 1e-3))))
            rgba = mask_ops.rgba_overlay(mask_ops.thicken(entry.label_map, grow),
                                         self.labels.registry.color_lut(), alpha)
        self.canvas.set_overlay(rgba)

    def _update_info(self) -> None:
        entry = self.annotations.get(self.index)
        lines = [(f"Slice {self.index + 1} / {len(self.series)}", True)]
        if entry is not None:
            first = entry.segments[0] if entry.segments else None
            lines.append((f"Segment Category: {first.category if first else 'None'}", False))
            lines.append((f"Segment Type: {first.type if first else 'None'}", False))
        badge = [("Annotated", "success")] if entry and entry.label_map is not None and entry.label_map.any() \
            else ([("Blank mask", "text_muted")] if entry else [])
        self.canvas.set_hud(top_left=lines, top_right=badge)
        self.file_label.setText("\n".join(os.path.basename(f) for f in entry.files) if entry else "No annotation file")
        segments = entry.segments if entry else []
        self.segments_table.setRowCount(len(segments))
        for row, seg in enumerate(segments):
            item = QTableWidgetItem(seg.label)
            value = self.labels.registry.value_for_segment(seg.label, create=False)
            if value:
                item.setIcon(swatch_icon(self.labels.registry.color_for(value)))
            self.segments_table.setItem(row, 0, item)
            self.segments_table.setItem(row, 1, QTableWidgetItem(seg.category))
            self.segments_table.setItem(row, 2, QTableWidgetItem(seg.type))

    def _toggle_zoom_mode(self, checked: bool) -> None:
        self.canvas.zoom_mode = checked

    # -------------------------------------------------------------- ontology
    def open_ontology_search(self) -> None:
        if self.series is None:
            show_info(self, "Add Ontology", "Please load a DICOM folder first.")
            return
        files = annotation_files_in(self.series.folder, self.seg_filter)
        if not files:
            show_info(self, "Add Ontology", "No annotation files were found in the loaded folder.")
            return
        from .ontology_search import OntologySearchWindow
        api_key = self.settings.get("bioportal_api_key") or DEFAULTS["bioportal_api_key"]   # empty: the built-in key
        names = sorted({seg.label for entry in self.annotations.values() for seg in entry.segments
                        if seg.label.strip().casefold() not in EMPTY_SEGMENT_NAMES})
        self._ontology = OntologySearchWindow(api_key, self.series.folder, files, names)
        self._ontology.applied.connect(lambda _n: self.reload_annotations())
        self._ontology.show()
