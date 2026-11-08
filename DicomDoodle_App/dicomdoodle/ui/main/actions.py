"""
Every action of the main window — menus, toolbar buttons and keyboard
shortcuts — defined in one place. Each action only calls a method of the
window or one of its controllers, so behaviour lives elsewhere.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Callable, List, Optional, Sequence, Union

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QAction, QActionGroup, QSizePolicy, QToolBar, QWidget

from ..icons import bind_icon
from ..qt_utils import swatch_icon
from ..widgets.annotation_canvas import TOOLS

if TYPE_CHECKING:
    from .window import MainWindow

Shortcut = Union[None, str, Sequence[str]]


class MainActions:
    def __init__(self, window: "MainWindow"):
        self.w = w = window
        an, batch, drawing, tools = w.annotation, w.batch, w.drawing, w.tools
        act = self._action

        # File
        self.open = act("Open DICOM Folder…", "folder-open", "Ctrl+O", w.open_folder_dialog,
                        "Load a folder of DICOM slices")
        self.save = act("Save Slice Annotation", "save", "Ctrl+S", an.save_current,
                        "Save this slice's annotation as <slice>_seg.dcm")
        self.save_all = act("Save All Modified Slices", "save-all", "Ctrl+Shift+S", an.save_all_modified,
                            "Save every slice with unsaved changes")
        self.close_series = act("Close Series", "x", None, w.close_series)
        self.load_card = act("Load Model Card…", "download", None, w.ai.choose_card,
                             "Install the DicomDoodle model card: the AI models, a separate download")
        self.preferences = act("Preferences…", "settings", "Ctrl+,", tools.preferences)
        self.quit = act("Exit", None, "Ctrl+Q", w.close)

        # Edit
        self.undo = act("Undo", "undo", "Ctrl+Z", an.undo, "Undo the last change on this slice", low=True)
        self.redo = act("Redo", "redo", ["Ctrl+Y", "Ctrl+Shift+Z"], an.redo, "Redo", low=True)
        self.reset = act("Reset Slice", "reset", "Ctrl+R", an.reset_slice,
                         "Clear this slice's annotation and reload the image (Refresh)", low=True)
        self.remove_box = act("Remove Box", "x", None, w.canvas.remove_box)

        # View
        self.zoom_in = act("Zoom In", "zoom-in", ["Ctrl+=", "Ctrl++"], w.canvas.zoom_in, low=True)
        self.zoom_out = act("Zoom Out", "zoom-out", "Ctrl+-", w.canvas.zoom_out, low=True)
        self.fit = act("Fit to Window", "fit", "Ctrl+0", w.canvas.fit_to_window, low=True)
        self.actual_size = act("Actual Size (100%)", None, None, w.canvas.zoom_actual)
        self.style_group = QActionGroup(w)
        self.style_boundary = act("Boundary Outline", None, None, lambda: w.set_overlay_mode("boundary"), checkable=True)
        self.style_filled = act("Filled Regions", None, None, lambda: w.set_overlay_mode("filled"), checkable=True)
        for a in (self.style_boundary, self.style_filled):
            self.style_group.addAction(a)
        self.show_saved = act("Show Saved Annotation Files", "eye", None, w.set_show_saved, checkable=True,
                              tip="Faint layer with the _seg.dcm already on disk for this slice")
        self.theme = act("Switch Theme", "moon", "Ctrl+T", w.toggle_theme,
                         tip="Switch between dark and light theme", low=True)
        self.reset_layout = act("Reset Panel Layout", None, None, w.reset_layout)

        # Annotate
        self.auto_detect = act("Auto Detect", "sparkles", "Ctrl+D", an.auto_detect,
                               "YOLO finds the lesion, SAM outlines it — current slice")
        self.detect_box = act("Detect in Box", "box-detect", "Ctrl+Return", an.segment_box,
                              "Segment the box you drew with SAM (Enter, or right-click the box)")
        self.plot = act("Plot Results…", "image", "Ctrl+P", an.plot_results,
                        "Show this slice with the segmented region and its boundary")
        self.add_label = act("Add Label…", "plus", None, w.label_panel.add_label,
                             "Create a new label with any color")

        # Tools of the left mouse button (the tool strip above the image)
        keys = {"box": "B", "pen": ["P", "Ctrl+M"], "eraser": "E", "pan": "H"}
        tips = {"box": "Drag a box around the lesion, then press Enter: SAM outlines it",
                "pen": "Draw an outline by hand in the active label's color",
                "eraser": "Rub out outlines — AI outlines become lines you can erase",
                "pan": "Drag to move the image (also: middle mouse button, or Space + drag)"}
        self.tool_group = QActionGroup(w)
        self.tool_actions = {}
        for key, (name, icon_name) in TOOLS.items():
            self.tool_actions[key] = act(f"{name} Tool", icon_name, keys[key],
                                         lambda _c=False, k=key: drawing.set_tool(k), tips[key], checkable=True)
            self.tool_actions[key].setIconText(name)
            self.tool_group.addAction(self.tool_actions[key])
        self.tool_actions["box"].setChecked(True)
        self.edit_saved = act("Edit Saved Outline", "download", "Ctrl+L", drawing.edit_saved_outline,
                              "Load this slice's annotation file as outlines to correct with the pen and eraser")
        self.size_down = act("Smaller Pen / Eraser", None, "[", lambda: drawing.change_size(-1))
        self.size_up = act("Bigger Pen / Eraser", None, "]", lambda: drawing.change_size(1))

        # Batch
        self.auto_start = act("Set Auto-Detect Range Start", None, None, lambda: batch.range_action("start"))
        self.auto_end = act("Set Auto-Detect Range End", None, None, lambda: batch.range_action("end"))
        self.find_range = act("Find Tumour Range", "search", None, batch.find_tumour_range,
                              "From this slice: the slices the tumour spans become the auto-detect range")
        self.clear_auto = act("Clear Auto-Detect Range", "x", None, lambda: batch.range_action("clear"))
        self.run_auto = act("Run Auto-Detect && Save", "sparkles", None, batch.run_auto_detect)
        self.auto_blank = act("Save Blank Masks Automatically", None, None, batch.set_auto_blank, checkable=True,
                              tip="When you save, every slice without an annotation file gets a blank mask")

        # Tools
        self.overlay_viewer = act("Annotated DICOM Viewer", "layers", None, tools.overlay_viewer,
                                  "Review a series with its saved annotations", low=True)
        self.mpr = act("Multiplanar && 3D Viewer", "grid", None, tools.mpr_viewer,
                       "Axial / coronal / sagittal views and the 3D model, lesion volume, registration, "
                       "skull stripping", low=True)
        self.converter = act("Multi-frame to Single-frame Converter", "scissors", None,
                             tools.multiframe_converter, "Split a multi-frame DICOM into slices", low=True)

        # Help
        self.help = act("User Guide", "book", "F1", tools.help, "Illustrated, searchable guide to every feature")
        self.quick_start = act("Quick Start", "zap", None, lambda: tools.help("quick-start"),
                               "Your first annotation in five minutes, on one of your series")
        self.shortcuts = act("Keyboard Shortcuts", "keyboard", None, tools.shortcuts)
        self.citations = act("References && Citations", "bookmark", None, tools.citations,
                             "How to cite DicomDoodle and the methods it builds on")
        self.about = act("About DicomDoodle", "info", None, tools.about)

        # shorter captions for the toolbar buttons
        for action, caption in ((self.open, "Open"), (self.save, "Save"), (self.detect_box, "Detect Box"),
                                (self.plot, "Results"), (self.edit_saved, "Edit saved outline")):
            action.setIconText(caption)

        self.series_actions = [self.save, self.save_all, self.close_series, self.undo, self.redo, self.reset,
                               self.zoom_in, self.zoom_out, self.fit, self.actual_size, self.auto_detect,
                               self.detect_box, self.plot, *self.tool_actions.values(), self.size_down, self.size_up,
                               self.auto_start, self.auto_end, self.find_range, self.clear_auto, self.run_auto]
        self._build_menus()
        self._build_toolbar()
        w.tool_strip.populate(list(self.tool_actions.values()), self.edit_saved)

    # ------------------------------------------------------------- helpers
    def _action(self, text: str, icon_name: Optional[str], shortcut: Shortcut, slot: Callable,
                tip: str = "", checkable: bool = False, low: bool = False) -> QAction:
        action = QAction(text, self.w)
        if icon_name:
            bind_icon(action, icon_name)
        keys: List[str] = [] if shortcut is None else ([shortcut] if isinstance(shortcut, str) else list(shortcut))
        if keys:
            action.setShortcuts([QKeySequence(k) for k in keys])
        clean = text.replace("&&", "&").rstrip("…")
        action.setToolTip(f"{clean}  ({keys[0]})" if keys else clean)
        action.setStatusTip(tip or clean)
        if tip:
            action.setToolTip(f"<b>{clean}</b>" + (f"  <span>({keys[0]})</span>" if keys else "") + f"<br>{tip}")
        action.setCheckable(checkable)
        if low:
            action.setPriority(QAction.LowPriority)
        if checkable:
            action.triggered.connect(slot)
        else:
            action.triggered.connect(lambda _checked=False: slot())
        self.w.addAction(action)
        return action

    # --------------------------------------------------------------- menus
    def _build_menus(self) -> None:
        bar = self.w.menuBar()
        file_menu = bar.addMenu("&File")
        file_menu.addAction(self.open)
        self.recent_menu = file_menu.addMenu("Open Recent")
        bind_icon(self.recent_menu, "clock")
        file_menu.addSeparator()
        for a in (self.save, self.save_all):
            file_menu.addAction(a)
        file_menu.addSeparator()
        file_menu.addAction(self.close_series)
        file_menu.addSeparator()
        file_menu.addAction(self.load_card)
        file_menu.addAction(self.preferences)
        file_menu.addAction(self.quit)

        edit = bar.addMenu("&Edit")
        for a in (self.undo, self.redo):
            edit.addAction(a)
        edit.addSeparator()
        edit.addAction(self.reset)
        edit.addAction(self.remove_box)

        view = bar.addMenu("&View")
        for a in (self.zoom_in, self.zoom_out, self.fit, self.actual_size):
            view.addAction(a)
        view.addSeparator()
        style = view.addMenu("Annotation Style")
        style.addAction(self.style_boundary)
        style.addAction(self.style_filled)
        view.addAction(self.show_saved)
        view.addSeparator()
        self.panels_menu = view.addMenu("Panels")
        view.addAction(self.reset_layout)
        view.addSeparator()
        view.addAction(self.theme)

        annotate = bar.addMenu("&Annotate")
        for a in (self.auto_detect, self.detect_box):
            annotate.addAction(a)
        annotate.addSeparator()
        for a in (*self.tool_actions.values(), self.edit_saved):
            annotate.addAction(a)
        annotate.addSeparator()
        annotate.addAction(self.plot)
        annotate.addSeparator()
        self.labels_menu = annotate.addMenu("Active Label")
        bind_icon(self.labels_menu, "tag")
        annotate.addAction(self.add_label)

        batch = bar.addMenu("&Batch")
        for a in (self.auto_start, self.auto_end, self.find_range, self.clear_auto, self.run_auto):
            batch.addAction(a)
        batch.addSeparator()
        batch.addAction(self.auto_blank)

        tools = bar.addMenu("&Tools")
        for a in (self.overlay_viewer, self.mpr, self.converter):
            tools.addAction(a)

        help_menu = bar.addMenu("&Help")
        for a in (self.help, self.quick_start, self.shortcuts):
            help_menu.addAction(a)
        help_menu.addSeparator()
        for a in (self.citations, self.about):
            help_menu.addAction(a)

    def rebuild_recent(self, folders: List[str]) -> None:
        self.recent_menu.clear()
        existing = [f for f in folders if os.path.isdir(f)]
        for folder in existing:
            action = self.recent_menu.addAction(folder)
            action.triggered.connect(lambda _c=False, f=folder: self.w.open_folder(f))
        self.recent_menu.setEnabled(bool(existing))

    def rebuild_labels_menu(self, labels, active_value: int) -> None:
        self.labels_menu.clear()
        group = QActionGroup(self.labels_menu)
        for position, lab in enumerate(labels, start=1):
            text = f"{lab.display_name}" + (f"\t{position}" if position <= 9 else "")
            action = self.labels_menu.addAction(swatch_icon(lab.color), text)
            action.setCheckable(True)
            action.setChecked(lab.value == active_value)
            action.triggered.connect(lambda _c=False, v=lab.value: self.w.label_panel.set_active(v))
            group.addAction(action)

    # ------------------------------------------------------------- toolbar
    def _build_toolbar(self) -> None:
        bar = QToolBar("Main toolbar", self.w)
        bar.setObjectName("MainToolbar")
        bar.setMovable(False)
        bar.setIconSize(QSize(20, 20))
        bar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.w.addToolBar(Qt.TopToolBarArea, bar)
        for a in (self.open, self.save):
            bar.addAction(a)
        bar.addSeparator()
        for a in (self.auto_detect, self.detect_box, self.plot):
            bar.addAction(a)
        bar.addSeparator()
        for a in (self.undo, self.redo, self.reset):
            bar.addAction(a)
        bar.addSeparator()
        for a in (self.zoom_out, self.fit, self.zoom_in):
            bar.addAction(a)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        bar.addWidget(spacer)
        for a in (self.overlay_viewer, self.mpr, self.converter):
            bar.addAction(a)
        bar.addSeparator()
        bar.addAction(self.theme)
        self.toolbar = bar
