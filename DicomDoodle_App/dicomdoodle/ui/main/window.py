"""
MainWindow — the slice-by-slice annotator (the original "DICOM Viewer and
Annotator" window).

Layout:
    toolbar ─────────────────────────────────────────────────────────
                            tool strip: Box · Pen · Eraser · Pan
    [Labels & Display]      image canvas (HUD, SAM box, drawing)      [Batch | Series]
                            slice navigator (timeline with file markers and the auto-detect range)
    status bar: cursor value · zoom · AI model status

This module owns the shared state (loaded series, current slice) and the
display. The actual work is delegated to controllers:
``annotation.AnnotationController``, ``batch.BatchController``,
``drawing.DrawingController`` and ``tools.ToolLauncher``.
"""
from __future__ import annotations

import base64
import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QByteArray, Qt, QTimer
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDockWidget, QFileDialog, QFormLayout, QLabel,
                             QMainWindow, QScrollArea, QShortcut, QSlider, QSpinBox, QStackedWidget,
                             QVBoxLayout, QWidget)

from ... import APP_NAME
from ...core import mask_ops
from ...core.ai_models import ModelManager
from ...core.annotation import AnnotationSession, SliceRange
from ...core.seg_io import read_segmentation
from ...settings import Settings
from ...workers import run_task
from ...workers.jobs import load_series_job
from ..dialogs.progress_dialog import TaskProgressDialog
from ..guide import close_guide, configure as configure_guide, help_button
from ..icons import bind_icon
from ..label_manager import LabelManager
from ..qt_utils import ask_choice, label, make_button, show_error
from ..theme import theme_manager
from ..widgets.annotation_canvas import AnnotationCanvas
from ..widgets.batch_panel import AUTO_RANGE_COLOR, BatchPanel
from ..widgets.busy_overlay import BusyOverlay
from ..widgets.cards import Card, panel_body
from ..widgets.info_panel import InfoPanel
from ..widgets.label_panel import LabelPanel
from ..widgets.log_panel import LogPanel, QtLogHandler
from ..widgets.slice_navigator import SliceNavigator
from ..widgets.toast import ToastManager
from ..widgets.tool_strip import ToolStrip
from .actions import MainActions
from .annotation import AnnotationController
from .batch import BatchController
from .drawing import DrawingController
from .tools import ToolLauncher
from .welcome import WelcomePage

log = logging.getLogger("dicomdoodle")


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings, labels: LabelManager, models: ModelManager,
                 log_handler: QtLogHandler):
        super().__init__()
        self.settings, self.labels, self.models = settings, labels, models
        self.session: Optional[AnnotationSession] = None
        self.current = 0
        self.busy = False
        self.contrast, self.brightness = 1.0, 0.0
        self._saved_cache: Dict[str, Tuple[float, np.ndarray]] = {}
        self._detection: Optional[tuple] = None   # (index, box, text) of the last YOLO box

        self.setWindowTitle(APP_NAME)
        self.setAcceptDrops(True)
        self.setDockOptions(QMainWindow.AnimatedDocks | QMainWindow.AllowTabbedDocks)
        self.resize(1480, 920)

        self._build_central()
        self._build_docks(log_handler)
        self.annotation = AnnotationController(self)
        self.batch = BatchController(self)
        self.drawing = DrawingController(self)
        self.tools = ToolLauncher(self)
        configure_guide(settings, self.tools.guide_action)
        self.actions = MainActions(self)
        self._build_status_bar()
        self._populate_panels_menu()
        self.toasts = ToastManager(self)

        self._overlay_timer = QTimer(self)
        self._overlay_timer.setSingleShot(True)
        self._overlay_timer.setInterval(120)
        self._overlay_timer.timeout.connect(self.render_saved_overlay)

        self._connect_signals()
        self._restore_layout()
        self._sync_view_actions()
        self.actions.rebuild_recent(settings.get("recent_folders", []))
        self.welcome.set_recent(settings.get("recent_folders", []))
        self.refresh_series_overview()
        self.update_ui_state()

    # ================================================================ layout
    def _build_central(self) -> None:
        self.canvas = AnnotationCanvas()
        self.canvas.placeholder = "Open a DICOM folder to start"
        self.busy_overlay = BusyOverlay(self.canvas)
        self.navigator = SliceNavigator()
        nav_bar = QWidget()
        nav_bar.setObjectName("DockBody")
        nav_layout = QVBoxLayout(nav_bar)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.addWidget(self.navigator)

        self.tool_strip = ToolStrip()        # filled with the tool actions by MainActions

        viewer = QWidget()
        viewer_layout = QVBoxLayout(viewer)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.setSpacing(0)
        viewer_layout.addWidget(self.tool_strip)
        viewer_layout.addWidget(self.canvas, 1)
        viewer_layout.addWidget(nav_bar)

        self.welcome = WelcomePage()
        self.stack = QStackedWidget()
        self.stack.addWidget(self._scroll(self.welcome))    # scrolls instead of making small screens too small
        self.stack.addWidget(viewer)
        self.setCentralWidget(self.stack)

    def _scroll(self, content: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QScrollArea.NoFrame)
        area.setWidget(content)
        return area

    def _dock(self, title: str, object_name: str, content: QWidget, area) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(object_name)
        dock.setWidget(content)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetClosable
                         | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(area, dock)
        return dock

    def _build_docks(self, log_handler: QtLogHandler) -> None:
        # Left: labels + display ------------------------------------------------
        left = panel_body()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)
        self.label_panel = LabelPanel(self.labels, int(self.settings.get("active_label")))
        labels_card = Card("Labels")
        labels_card.add_header_widget(help_button("labels"))
        labels_card.add(self.label_panel)
        left_layout.addWidget(labels_card)
        left_layout.addWidget(self._build_display_card())
        left_layout.addStretch(1)
        self.left_dock = self._dock("Labels & Display", "LabelsDock", self._scroll(left), Qt.LeftDockWidgetArea)
        self.left_dock.setMinimumWidth(280)

        # Right: batch + info (tabbed) ------------------------------------------
        batch_body = panel_body()
        batch_layout = QVBoxLayout(batch_body)
        batch_layout.setContentsMargins(10, 10, 10, 10)
        self.batch_panel = BatchPanel()
        batch_layout.addWidget(self.batch_panel)
        batch_layout.addStretch(1)
        self.batch_dock = self._dock("Batch Processing", "BatchDock", self._scroll(batch_body), Qt.RightDockWidgetArea)

        info_body = panel_body()
        info_layout = QVBoxLayout(info_body)
        info_layout.setContentsMargins(10, 10, 10, 10)
        self.info_panel = InfoPanel()
        info_layout.addWidget(self.info_panel)
        info_layout.addStretch(1)
        self.info_dock = self._dock("Series Info", "InfoDock", self._scroll(info_body), Qt.RightDockWidgetArea)
        self.tabifyDockWidget(self.batch_dock, self.info_dock)
        self.batch_dock.raise_()
        self.batch_dock.setMinimumWidth(300)

        # Bottom: activity log ------------------------------------------------
        self.log_panel = LogPanel(log_handler)
        self.log_dock = self._dock("Activity Log", "LogDock", self.log_panel, Qt.BottomDockWidgetArea)
        self.log_dock.hide()
        self._default_state = None

    def _build_display_card(self) -> Card:
        card = Card("Display")
        card.add_header_widget(help_button("viewing"))
        self.style_combo = QComboBox()
        self.style_combo.addItem("Boundary outline", "boundary")
        self.style_combo.addItem("Filled regions", "filled")
        self.line_width_spin = QSpinBox()
        self.line_width_spin.setRange(1, 6)
        self.line_width_spin.setSuffix(" px")
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(5, 100)
        self.show_saved_check = QCheckBox("Show saved annotation files")
        self.show_saved_check.setToolTip("Faint layer showing the _seg.dcm already on disk for the slice")
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(10, 300)
        self.contrast_slider.setValue(100)
        reset = make_button("Reset image", "reset", role="ghost",
                            tooltip="Back to the original brightness and contrast")
        reset.clicked.connect(lambda: (self.brightness_slider.setValue(0), self.contrast_slider.setValue(100)))

        form = QFormLayout()
        form.setVerticalSpacing(8)
        form.setHorizontalSpacing(10)
        form.addRow(label("Style", "Muted"), self.style_combo)
        form.addRow(label("Outline", "Muted"), self.line_width_spin)
        form.addRow(label("Fill", "Muted"), self.opacity_slider)
        form.addRow(label("Brightness", "Muted"), self.brightness_slider)
        form.addRow(label("Contrast", "Muted"), self.contrast_slider)
        card.add_layout(form)
        card.add(self.show_saved_check)
        card.add(reset)
        return card

    def _build_status_bar(self) -> None:
        bar = self.statusBar()
        self.cursor_label = QLabel("")
        self.zoom_label = QLabel("")
        self.ai_label = QLabel("")
        self.ai_label.setToolTip("AI model status")
        for widget in (self.cursor_label, self.zoom_label, self.ai_label):
            bar.addPermanentWidget(widget)
        self.set_ai_status("idle", "AI models load on first use")

    def _populate_panels_menu(self) -> None:
        for dock in (self.left_dock, self.batch_dock, self.info_dock, self.log_dock):
            self.actions.panels_menu.addAction(dock.toggleViewAction())

    def _connect_signals(self) -> None:
        c = self.canvas
        c.sliceScrolled.connect(lambda step: self.go_to_slice(self.current + step))
        c.cursorMoved.connect(self._show_cursor_value)
        c.zoomChanged.connect(self._zoom_changed)
        c.detectRequested.connect(self.annotation.segment_box)
        c.boxChanged.connect(lambda _has: self.update_ui_state())
        self.navigator.valueChanged.connect(self.go_to_slice)
        self.welcome.openRequested.connect(self.open_folder_dialog)
        self.welcome.recentRequested.connect(self.open_folder)
        self.welcome.toolRequested.connect(self.tools.open_by_key)
        self.welcome.guideRequested.connect(self.tools.help)

        self.label_panel.activeChanged.connect(self._active_label_changed)
        self.labels.labelsChanged.connect(self._labels_changed)
        self.batch_panel.rangeAction.connect(self.batch.range_action)
        self.batch_panel.findRangeRequested.connect(self.batch.find_tumour_range)
        self.batch_panel.runAutoRequested.connect(self.batch.run_auto_detect)
        self.batch_panel.autoBlankToggled.connect(self.batch.set_auto_blank)

        s = self.settings
        self.style_combo.setCurrentIndex(max(0, self.style_combo.findData(s.get("overlay_mode"))))
        self.line_width_spin.setValue(int(s.get("line_width")))
        self.opacity_slider.setValue(int(float(s.get("fill_opacity")) * 100))
        self.show_saved_check.setChecked(bool(s.get("show_saved_overlay")))
        self.style_combo.currentIndexChanged.connect(lambda _i: self.set_overlay_mode(self.style_combo.currentData()))
        self.line_width_spin.valueChanged.connect(lambda v: self._display_setting("line_width", v))
        self.opacity_slider.valueChanged.connect(lambda v: self._display_setting("fill_opacity", v / 100))
        self.show_saved_check.toggled.connect(self.set_show_saved)
        self.brightness_slider.valueChanged.connect(self._image_adjusted)
        self.contrast_slider.valueChanged.connect(self._image_adjusted)

        theme_manager().themeChanged.connect(self._sync_view_actions)
        self._labels_changed()

        # 1-9 select a label (ignored while typing in text fields)
        for n in range(1, 10):
            QShortcut(QKeySequence(str(n)), self, activated=lambda p=n: self.label_panel.select_position(p))
        for key, step in ((Qt.Key_Left, -1), (Qt.Key_Right, 1), (Qt.Key_Up, -1), (Qt.Key_Down, 1),
                          (Qt.Key_PageUp, -10), (Qt.Key_PageDown, 10)):
            shortcut = QShortcut(QKeySequence(key), self.canvas, activated=lambda s=step: self.go_to_slice(self.current + s))
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        QShortcut(QKeySequence(Qt.Key_Home), self.canvas, activated=lambda: self.go_to_slice(0)).setContext(
            Qt.WidgetWithChildrenShortcut)
        QShortcut(QKeySequence(Qt.Key_End), self.canvas, activated=lambda: self.go_to_slice(10 ** 6)).setContext(
            Qt.WidgetWithChildrenShortcut)

    # ============================================================ loading
    def open_folder_dialog(self) -> None:
        start = self.settings.get("last_dir") or os.path.expanduser("~")
        folder = QFileDialog.getExistingDirectory(self, "Select DICOM Folder", start)
        if folder:
            self.open_folder(folder)

    def open_folder(self, folder: str, confirm: bool = True) -> None:
        if self.busy or (confirm and not self.confirm_discard_changes("open another folder")):
            return
        dialog = TaskProgressDialog("Loading DICOM series", self)
        dialog.message.setText(folder)
        state = {}

        def loaded(session: AnnotationSession) -> None:
            state["session"] = session
            dialog.accept()

        def failed(message: str, details: str) -> None:
            dialog.accept()
            show_error(self, "Could not load the folder", message, details)

        task = run_task(load_series_job, folder, on_result=loaded, on_error=failed,
                        on_progress=dialog.set_progress, on_cancelled=dialog.accept)
        dialog.cancelRequested.connect(task.cancel)
        dialog.exec_()
        if "session" in state:
            self._set_session(state["session"])

    def _set_session(self, session: AnnotationSession) -> None:
        self.session = session
        series = session.series
        self._saved_cache.clear()
        self.batch.forget_series()
        self.settings.add_recent_folder(series.folder)
        self.actions.rebuild_recent(self.settings.get("recent_folders", []))
        self.welcome.set_recent(self.settings.get("recent_folders", []))
        self.navigator.set_count(len(series))
        self.stack.setCurrentIndex(1)
        self.info_panel.show_series(series)
        self.current = -1
        self.go_to_slice(0)
        self.canvas.fit_to_window()
        self.canvas.setFocus()
        blank = len(session.blank_files)
        note = ""
        if session.files:
            note = f" · {len(session.files) - blank} annotated" + (f", {blank} blank" if blank else "")
        log.info("Loaded %s: %d slices%s", series.folder, len(series), note)
        self.toasts.show(f"Loaded {len(series)} slices from “{series.name}”{note}", "success")
        if series.skipped:
            log.warning("Skipped %d file(s): %s", len(series.skipped), "; ".join(series.skipped[:5]))
            self.toasts.show(f"{len(series.skipped)} file(s) could not be used — see View ▸ Panels ▸ Activity Log",
                             "warning", 6000)
        self.refresh_series_overview()

    def close_series(self) -> None:
        if self.session is None or self.busy or not self.confirm_discard_changes("close the series"):
            return
        self.session = None
        self.batch.forget_series()
        self.canvas.remove_box()
        self.canvas.show_detection(None)
        self.canvas.set_contours({}, {})
        self.canvas.set_strokes(None)
        self.canvas.set_overlay(None)
        self.canvas.clear_image()
        self.navigator.set_count(0)
        self.info_panel.show_series(None)
        self.stack.setCurrentIndex(0)
        self.setWindowTitle(APP_NAME)
        self.refresh_series_overview()
        self.update_ui_state()

    def confirm_discard_changes(self, doing: str) -> bool:
        """Ask what to do with unsaved annotations before they would be lost."""
        if self.session is None:
            return True
        dirty = [i for i in self.session.dirty_slices() if not self.session.annotation(i).is_empty]
        if not dirty:
            return True
        choice = ask_choice(self, "Unsaved annotations",
                            f"{len(dirty)} slice(s) have annotations that are not saved yet.",
                            ["Save all", "Discard"],
                            informative=f"Save them before you {doing}?")
        if choice is None:
            return False
        if choice == 0:
            return self.annotation.save_all_modified(blocking=True)
        return True

    # ============================================================ display
    def go_to_slice(self, index: int) -> None:
        if self.session is None or self.busy or self.canvas.is_drawing:
            if self.session is not None:
                self.navigator.set_value(self.current, emit=False)
            return
        index = max(0, min(len(self.session.series) - 1, int(index)))
        if index == self.current:
            return
        self.current = index
        self.navigator.set_value(index, emit=False)
        self.canvas.remove_box()
        self.canvas.show_detection(None)
        self.render_image()
        self.render_annotation()
        self.render_saved_overlay()
        self.refresh_slice_status()
        self.update_ui_state()

    def render_image(self) -> None:
        if self.session is None:
            return
        image = self.session.series.image(self.current)
        self.canvas.set_image(mask_ops.adjust_brightness_contrast(image, self.contrast, self.brightness))

    def render_annotation(self) -> None:
        """The working annotation of the current slice: AI regions as vector outlines
        (or filled), hand-drawn outlines as the pixels that will be saved."""
        if self.session is None:
            return
        ann = self.session.annotation(self.current, create=False)
        filled = self.settings.get("overlay_mode") == "filled"
        fill = float(self.settings.get("fill_opacity")) if filled else 0.0
        self.canvas.set_stroke_style(self.labels.registry.color_lut(), fill)
        self.canvas.set_strokes(ann.strokes if ann is not None else None)
        if ann is None or not ann.label_map.any():
            self.canvas.set_contours({}, {})
            return
        contours = mask_ops.label_contours(ann.label_map)
        colors = {v: self.labels.registry.color_for(v) for v in contours}
        self.canvas.set_contours(contours, colors, float(self.settings.get("line_width")), fill)

    def saved_label_map(self, index: int) -> Optional[np.ndarray]:
        """The annotation file on disk for a slice, decoded (cached by file time)."""
        path = self.session.series.seg_path(index)
        if not os.path.isfile(path):
            return None
        mtime = os.path.getmtime(path)
        cached = self._saved_cache.get(path)
        if cached and cached[0] == mtime:
            return cached[1]
        before = len(self.labels.registry.labels())
        try:
            data = read_segmentation(path, self.labels.registry, self.session.series.image(index).shape)
        except Exception as exc:
            log.warning("Could not read %s: %s", os.path.basename(path), exc)
            return None
        if len(self.labels.registry.labels()) != before:
            self.labels.notify_transient_added()
        self._saved_cache[path] = (mtime, data.label_map)
        return data.label_map

    def render_saved_overlay(self) -> None:
        """Faint layer with the slice's annotation file — hidden while that file is being edited."""
        if (self.session is None or not self.settings.get("show_saved_overlay")
                or self.session.is_loaded(self.current)):
            self.canvas.set_overlay(None)
            return
        saved = self.saved_label_map(self.current)
        if saved is None or not saved.any():
            self.canvas.set_overlay(None)
            return
        thickness = max(1, int(round(1.6 / max(self.canvas.zoom, 1e-3))))
        rgba = mask_ops.rgba_overlay(mask_ops.thicken(saved, thickness), self.labels.registry.color_lut(), 0.55)
        self.canvas.set_overlay(rgba)

    def refresh_slice_status(self) -> None:
        """HUD, info panel and window title for the current slice."""
        if self.session is None:
            return
        series, i = self.session.series, self.current
        ann = self.session.annotation(i, create=False)
        blank = self.session.is_blank(i)
        if ann is not None and ann.dirty and not ann.is_empty:
            status = ("warning", "Unsaved changes")
        elif self.session.is_saved(i):
            status = ("success", "Saved · blank mask" if blank else "Saved")
        elif blank:
            status = ("neutral", "Blank mask")
        elif i in self.session.files:
            status = ("info", "Annotation file on disk")
        else:
            status = (None, "")
        token = {"warning": "warning", "success": "success", "info": "accent",
                 "neutral": "text_muted"}.get(status[0])
        active = self.label_panel.active_label()
        instance = series.instance_numbers[i]
        tool = self.canvas.tool
        if tool == "pen":
            chip = (f"Pen {self.canvas.pen_size} px  ·  {active.display_name}", active.color)
        elif tool == "eraser":
            chip = (f"Eraser {self.canvas.eraser_size} px", None)
        else:
            chip = (f"Label: {active.display_name}", active.color)
        self.canvas.set_hud(
            top_left=[(f"Slice {i + 1} / {len(series)}", True),
                      (f"Instance {instance if instance is not None else '—'}  ·  {os.path.basename(series.file(i))}", False)],
            top_right=[(status[1], token)] if token else [],
            bottom_left=chip,
        )
        present = []
        if ann is not None:
            present = [self.labels.registry.get(v).display_name if self.labels.registry.get(v) else f"Label {v}"
                       for v in ann.label_values()]
        self.info_panel.show_slice(series.file(i), instance, present, status)
        self.setWindowTitle(f"{series.name} — Slice {i + 1}/{len(series)} — {APP_NAME}")

    def refresh_series_overview(self) -> None:
        """Timeline markers/bands, the batch panel and the automatic blank-mask switch."""
        s = self.session
        auto_blank = bool(self.settings.get("auto_blank_masks"))
        self.actions.auto_blank.setChecked(auto_blank)
        if s is None:
            for name in ("blank", "disk", "saved", "dirty"):
                self.navigator.set_markers(name, [], "text_faint", "")
            self.navigator.set_bands([])
            self.batch_panel.show_auto_range(SliceRange(), 0, None)
            self.batch_panel.show_blank_masks(auto_blank, None)
            return
        unsaved = {i for i in s.dirty_slices() if not s.annotation(i).is_empty}
        outlined = s.files - s.blank_files
        self.navigator.set_markers("blank", s.blank_files - unsaved, "text_muted", "Blank mask (no lesion)", "dot")
        self.navigator.set_markers("disk", outlined - s.saved_this_session, "text_faint", "Annotation file on disk")
        self.navigator.set_markers("saved", s.saved_this_session - s.blank_files, "success", "Saved this session")
        self.navigator.set_markers("dirty", unsaved, "warning", "Unsaved changes")
        rng, bands = s.auto_range, []
        if rng.start is not None or rng.end is not None:
            start = rng.start if rng.start is not None else rng.end
            end = rng.end if rng.end is not None else rng.start
            bands.append((start, end, AUTO_RANGE_COLOR, "Auto-detect range"))
        self.navigator.set_bands(bands)
        self.batch_panel.show_auto_range(rng, len(s.auto_targets()), self.label_panel.active_label().display_name)
        self.batch_panel.show_blank_masks(auto_blank, (len(outlined), len(s.blank_files),
                                                       len(s.series) - len(s.files)))

    def annotation_changed(self) -> None:
        """Call after the current slice's label map changed."""
        self.render_annotation()
        self.refresh_slice_status()
        self.refresh_series_overview()
        self.update_ui_state()

    def mark_saved(self, index: int, blank: bool = False) -> None:
        self.session.mark_saved(index, blank)
        self._saved_cache.pop(self.session.series.seg_path(index), None)

    # ============================================================ state
    def update_ui_state(self) -> None:
        has = self.session is not None
        idle = not self.busy
        for action in self.actions.series_actions:
            action.setEnabled(has and idle)
        self.actions.open.setEnabled(idle)
        if has:
            ann = self.session.annotation(self.current, create=False)
            self.actions.undo.setEnabled(idle and ann is not None and ann.can_undo)
            self.actions.redo.setEnabled(idle and ann is not None and ann.can_redo)
            self.actions.detect_box.setEnabled(idle and self.canvas.box_coords() is not None)
            self.actions.save_all.setEnabled(idle and bool(self.session.dirty_slices()))
        self.batch_panel.setEnabled(has and idle)
        self.navigator.setEnabled(has and idle)
        self.canvas.editing_enabled = has and idle
        self.drawing.refresh()

    def set_busy(self, busy: bool, message: str = "", detail: str = "") -> None:
        self.busy = busy
        if busy:
            self.navigator.stop_playback()
            self.busy_overlay.start(message, detail)
        else:
            self.busy_overlay.stop()
        self.update_ui_state()

    def set_ai_status(self, state: str, text: str) -> None:
        colors = {"ready": "success", "loading": "warning", "error": "danger", "idle": "text_faint"}
        color = getattr(theme_manager().theme, colors.get(state, "text_faint"))
        self.ai_label.setText(f'<span style="color:{color}">●</span>&nbsp;{text}')

    # ============================================================ view settings
    def set_overlay_mode(self, mode: str) -> None:
        self.settings.set("overlay_mode", mode)
        self._sync_view_actions()
        self.render_annotation()

    def set_show_saved(self, show: bool) -> None:
        self.settings.set("show_saved_overlay", bool(show))
        self._sync_view_actions()
        self.render_saved_overlay()

    def _display_setting(self, key: str, value) -> None:
        self.settings.set(key, value)
        self.render_annotation()

    def _image_adjusted(self, *_args) -> None:
        self.brightness = float(self.brightness_slider.value())
        self.contrast = self.contrast_slider.value() / 100.0
        self.render_image()

    def toggle_theme(self) -> None:
        theme = theme_manager().toggle()
        self.settings.set("theme", theme.name)
        self.render_saved_overlay()

    def _sync_view_actions(self, *_args) -> None:
        mode = self.settings.get("overlay_mode")
        self.actions.style_boundary.setChecked(mode == "boundary")
        self.actions.style_filled.setChecked(mode == "filled")
        self.style_combo.blockSignals(True)
        self.style_combo.setCurrentIndex(max(0, self.style_combo.findData(mode)))
        self.style_combo.blockSignals(False)
        show = bool(self.settings.get("show_saved_overlay"))
        self.actions.show_saved.setChecked(show)
        self.show_saved_check.blockSignals(True)
        self.show_saved_check.setChecked(show)
        self.show_saved_check.blockSignals(False)
        dark = theme_manager().is_dark
        self.actions.theme.setText("Switch to Light Theme" if dark else "Switch to Dark Theme")
        bind_icon(self.actions.theme, "sun" if dark else "moon")
        self.actions.theme.setToolTip("Switch to light theme (Ctrl+T)" if dark else "Switch to dark theme (Ctrl+T)")

    def apply_settings(self) -> None:
        """Re-read preferences after the Preferences dialog was saved."""
        s = self.settings
        theme_manager().apply(s.get("theme"))
        self.models.configure(models_dir=s.get("models_dir"), device=s.get("device"))
        for widget, value in ((self.line_width_spin, int(s.get("line_width"))),
                              (self.opacity_slider, int(float(s.get("fill_opacity")) * 100))):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        self._sync_view_actions()
        self.render_annotation()
        self.render_saved_overlay()
        self.refresh_series_overview()

    def reset_layout(self) -> None:
        for dock, area in ((self.left_dock, Qt.LeftDockWidgetArea), (self.batch_dock, Qt.RightDockWidgetArea),
                           (self.info_dock, Qt.RightDockWidgetArea), (self.log_dock, Qt.BottomDockWidgetArea)):
            dock.setFloating(False)
            self.addDockWidget(area, dock)
            dock.show()
        self.tabifyDockWidget(self.batch_dock, self.info_dock)
        self.batch_dock.raise_()
        self.log_dock.hide()
        self._default_dock_widths()

    def _default_dock_widths(self) -> None:
        """Side panels just wide enough for their content, so the image gets the rest."""
        self.resizeDocks([self.left_dock, self.batch_dock], [280, 300], Qt.Horizontal)

    # ============================================================ callbacks
    def _active_label_changed(self, value: int) -> None:
        chosen = int(value) != self.canvas.pen_value     # (the label list also re-announces it after changes)
        self.settings.set("active_label", int(value))
        self.canvas.pen_value = int(value)
        self.canvas.update_cursor()                       # the pen's dot takes the label's color
        if chosen and self.canvas.tool == "eraser":       # choosing a color returns to the pen (original)
            self.drawing.set_tool("pen")
        self.actions.rebuild_labels_menu(self.labels.labels(), value)
        self.refresh_slice_status()
        self.refresh_series_overview()

    def _labels_changed(self) -> None:
        self.actions.rebuild_labels_menu(self.labels.labels(), self.label_panel.active_value)
        self.render_annotation()
        self.render_saved_overlay()
        self.refresh_slice_status()

    def _show_cursor_value(self, x: int, y: int) -> None:
        if self.session is None or x < 0:
            self.cursor_label.setText("")
            return
        try:
            value = self.session.series.raw_pixels(self.current)[y, x]
            self.cursor_label.setText(f"x {x} · y {y} · value {value}")
        except Exception:
            self.cursor_label.setText(f"x {x} · y {y}")

    def _zoom_changed(self, zoom: float) -> None:
        self.zoom_label.setText(f"Zoom {zoom * 100:.0f}%")
        if self.session is not None and self.settings.get("show_saved_overlay"):
            self._overlay_timer.start()

    # ============================================================ window
    def _restore_layout(self) -> None:
        geometry = self.settings.get("window_geometry")
        state = self.settings.get("window_state")
        try:
            if geometry:
                self.restoreGeometry(QByteArray(base64.b64decode(geometry)))
            if state and self.restoreState(QByteArray(base64.b64decode(state))):
                return
        except Exception:
            pass
        self._default_dock_widths()

    def closeEvent(self, event) -> None:
        if self.busy:
            event.ignore()
            self.toasts.show("Please wait until the current job finishes.", "warning")
            return
        if not self.confirm_discard_changes("exit"):
            event.ignore()
            return
        self.settings.update(
            window_geometry=base64.b64encode(bytes(self.saveGeometry())).decode("ascii"),
            window_state=base64.b64encode(bytes(self.saveState())).decode("ascii"),
        )
        self.tools.close_all()
        close_guide()
        event.accept()

    def dragEnterEvent(self, event) -> None:
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        if any(u.isLocalFile() and os.path.isdir(u.toLocalFile()) for u in urls):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        for url in event.mimeData().urls():
            if url.isLocalFile() and os.path.isdir(url.toLocalFile()):
                self.open_folder(url.toLocalFile())
                break
