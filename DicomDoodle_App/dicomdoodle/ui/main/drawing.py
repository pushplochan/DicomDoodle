"""
Drawing by hand in the main window (it replaces the separate Manual Draw
window of earlier versions):

* the tool of the left mouse button — Box (SAM), Pen, Eraser or Pan
* the pen / eraser size ([ and ] keys), remembered between sessions
* "Edit saved outline" — loads the slice's annotation file as outlines to
  correct (the original's "Load Annotation")

The canvas paints strokes straight into the slice's stroke layer
(core.annotation.SliceAnnotation.strokes), so undo / redo, saving, Save All,
batches and the automatic blank masks treat them like any other annotation.
They are saved exactly as drawn; small gaps are closed when saving.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

import numpy as np
from PyQt5.QtCore import QObject

from ...core.annotation import SliceAnnotation
from ...core.seg_io import read_segmentation
from ..qt_utils import ask_yes_no, show_error, show_info

if TYPE_CHECKING:
    from .window import MainWindow

log = logging.getLogger("dicomdoodle")

HINTS = {
    "box": "Drag a box around the lesion, then press Enter — SAM outlines it",
    "pen": "Drag to draw the outline in the active label's color  ·  [ ] size  ·  Ctrl+Z undo",
    "eraser": "Drag to rub out outlines  ·  [ ] size  ·  Ctrl+Z undo",
    "pan": "Drag to move the image  ·  Ctrl + mouse wheel zooms",
}


class DrawingController(QObject):
    def __init__(self, window: "MainWindow"):
        super().__init__(window)
        self.w = window
        c = window.canvas
        c.pen_size = int(window.settings.get("pen_size"))
        c.eraser_size = int(window.settings.get("eraser_size"))
        c.pen_value = window.label_panel.active_value
        c.annotation_for_drawing = self._annotation_for_drawing
        c.strokeStarted.connect(self._stroke_changed)
        c.strokeFinished.connect(self._stroke_changed)
        c.toolRequested.connect(self.set_tool)
        window.tool_strip.sizeChanged.connect(self._size_changed)

    @property
    def tool(self) -> str:
        return self.w.canvas.tool

    def _annotation_for_drawing(self) -> Optional[SliceAnnotation]:
        w = self.w
        return w.session.annotation(w.current) if w.session is not None and not w.busy else None

    def _stroke_changed(self) -> None:
        self.w.annotation_changed()

    # ----------------------------------------------------------------- tools
    def set_tool(self, tool: str) -> None:
        w = self.w
        changed = tool != w.canvas.tool
        w.canvas.set_tool(tool)
        w.actions.tool_actions[tool].setChecked(True)
        self.refresh()
        w.refresh_slice_status()
        if changed and w.session is not None:   # the hint, also when the tool strip is too narrow for it
            w.statusBar().showMessage(f"{w.actions.tool_actions[tool].iconText()}: {HINTS[tool]}", 6000)

    def change_size(self, step: int) -> None:
        """[ and ]: a smaller or bigger pen / eraser."""
        if self.tool in ("pen", "eraser"):
            spin = self.w.tool_strip.size_spin
            spin.setValue(spin.value() + step)

    def _size_changed(self, value: int) -> None:
        c = self.w.canvas
        if c.tool == "eraser":
            c.eraser_size = value
            self.w.settings.set("eraser_size", value)
        elif c.tool == "pen":
            c.pen_size = value
            self.w.settings.set("pen_size", value)
        c.update_cursor()
        self.w.refresh_slice_status()

    def refresh(self) -> None:
        """Sync the tool strip with the tool and the current slice."""
        w, c = self.w, self.w.canvas
        w.tool_strip.show_size(c.tool in ("pen", "eraser"), c.eraser_size if c.tool == "eraser" else c.pen_size)
        hint = HINTS[c.tool]
        can_edit = self.can_edit_saved()
        if can_edit and c.tool in ("pen", "eraser"):
            ann = w.session.annotation(w.current, create=False)
            if ann is None or (ann.is_empty and not ann.can_undo):
                hint = "This slice has a saved outline — Edit saved outline (Ctrl+L) loads it to correct it"
        w.tool_strip.set_hint(hint)
        w.actions.edit_saved.setEnabled(can_edit)

    # --------------------------------------------------- edit saved outline
    def can_edit_saved(self) -> bool:
        """The current slice has an annotation file with an outline (not a blank mask)."""
        w = self.w
        s = w.session
        return s is not None and not w.busy and w.current in s.files and not s.is_blank(w.current)

    def edit_saved_outline(self) -> None:
        """Load the slice's annotation file as outlines to correct with the pen and eraser."""
        w = self.w
        if w.session is None or w.busy:
            return
        s, index = w.session, w.current
        path = s.series.seg_path(index)
        name = os.path.basename(path)
        if not s.has_file_on_disk(index):
            show_info(w, "Edit saved outline", f"Slice {index + 1} has no annotation file yet.")
            return
        if s.is_blank(index):
            show_info(w, "Edit saved outline", f"The annotation file of slice {index + 1} is a blank mask: "
                                               "there is no outline to edit. Draw one with the pen.")
            return
        ann = s.annotation(index)
        if ann.dirty and not ann.is_empty and not ask_yes_no(
                w, "Edit saved outline", f"Slice {index + 1} has unsaved changes.\n\n"
                                         f"Replace them with the outline saved in {name}?", yes_text="Replace"):
            return
        try:
            data = read_segmentation(path, w.labels.registry, s.series.image(index).shape)
        except Exception as exc:
            show_error(w, "Could not read the annotation file", str(exc))
            return
        w.labels.notify_transient_added()
        ann.replace(np.zeros_like(ann.label_map), mark_dirty=False, strokes=data.label_map)
        s.mark_loaded(index)
        log.info("Slice %d: %s loaded for editing", index + 1, name)
        if self.tool in ("box", "pan"):
            self.set_tool("pen")
        w.render_saved_overlay()
        w.annotation_changed()
        w.toasts.show(f"Loaded {name} — correct it with the pen and eraser, then save (Ctrl+S).", "info", 4500)
