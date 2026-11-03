"""
Single-slice annotation work for the main window:

* Auto Detect      — YOLO (-> YOLOE) box + SAM mask on the current slice
* Detect in Box    — SAM inside the box the user drew
* Undo / Redo / Reset slice (AI regions and hand-drawn outlines alike)
* Save (current slice, or every modified slice) as <slice>_seg.dcm — each
  save also writes the automatic blank masks (see batch.BatchController)
* Plot Results

Drawing by hand (pen, eraser, Edit saved outline) is in drawing.py.

AI inference and file writing run in background threads; the window shows
a busy overlay meanwhile and stays responsive.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import numpy as np
from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import QCheckBox, QMessageBox

from ...workers import run_task
from ...workers.jobs import BatchItem, SaveResult, SegmentResult, auto_detect_job, save_many_job, \
    save_slice_job, segment_box_job
from ..dialogs.progress_dialog import TaskProgressDialog
from ..qt_utils import ask_yes_no, show_error, show_info

if TYPE_CHECKING:
    from .window import MainWindow

log = logging.getLogger("dicomdoodle")


class AnnotationController(QObject):
    def __init__(self, window: "MainWindow"):
        super().__init__(window)
        self.w = window

    # --------------------------------------------------------------- helpers
    def _ready(self) -> bool:
        return self.w.session is not None and not self.w.busy and not self.w.canvas.is_drawing

    def _ai_error(self, message: str, details: str) -> None:
        self.w.set_ai_status("error", "AI error")
        show_error(self.w, "AI model error", message, details)

    def _ai_started(self) -> None:
        if not self.w.models.sam_loaded:
            self.w.set_ai_status("loading", "Loading AI models…")

    def _ai_finished(self) -> None:
        self.w.set_busy(False)
        self.w.ai.refresh()

    # ---------------------------------------------------------- auto detect
    def auto_detect(self) -> None:
        if not self._ready() or not self.w.ai.explain("detect"):
            return
        w = self.w
        index, label_value = w.current, w.label_panel.active_value
        image = w.session.series.image(index)
        w.set_busy(True, "Auto Detect", "YOLO is looking for a lesion, then SAM outlines it…")
        self._ai_started()
        run_task(auto_detect_job, w.models, image, index,
                 on_progress=lambda _d, _t, msg: w.busy_overlay.set_message(msg),
                 on_result=lambda r: self._apply_result(r, label_value, from_box=False),
                 on_error=self._ai_error, on_finished=self._ai_finished)

    # ---------------------------------------------------------- box detect
    def segment_box(self) -> None:
        if not self._ready() or not self.w.ai.explain("segment"):
            return
        w = self.w
        box = w.canvas.box_coords()
        if box is None:
            w.toasts.show("Draw a box around the lesion first (left-drag on the image).", "info")
            return
        index, label_value = w.current, w.label_panel.active_value
        w.set_busy(True, "Segmenting with SAM", "Finding the precise outline inside your box…")
        self._ai_started()
        run_task(segment_box_job, w.models, w.session.series.image(index), box, index,
                 on_result=lambda r: self._apply_result(r, label_value, from_box=True),
                 on_error=self._ai_error, on_finished=self._ai_finished)

    def _apply_result(self, result: SegmentResult, label_value: int, from_box: bool) -> None:
        w = self.w
        if w.session is None:
            return
        slice_no = result.index + 1
        if result.mask is None or not np.any(result.mask):
            what = "SAM found nothing inside the box" if from_box else "No detection found for this slice"
            w.toasts.show(f"{what} (slice {slice_no}).", "warning")
            log.info("%s (slice %d)", what, slice_no)
            return
        ann = w.session.annotation(result.index)
        ann.paint(result.mask, label_value)
        label = w.labels.get(label_value)
        name = label.display_name if label else f"Label {label_value}"
        if from_box:
            w.canvas.remove_box()
        elif result.detection is not None and result.index == w.current:
            w.canvas.show_detection(result.detection.box, result.detection.source)
        source = "box" if from_box else (result.detection.source if result.detection else "AI")
        log.info("Slice %d: %s segmented as “%s” (%d px, %.1f s)", slice_no, source, name,
                 int(result.mask.sum()), result.seconds)
        w.toasts.show(f"Slice {slice_no}: region added as “{name}” — press Ctrl+S to save.", "success")
        w.annotation_changed()

    # ------------------------------------------------------- undo / reset
    def undo(self) -> None:
        if self._ready():
            ann = self.w.session.annotation(self.w.current, create=False)
            if ann is not None and ann.undo():
                self.w.annotation_changed()

    def redo(self) -> None:
        if self._ready():
            ann = self.w.session.annotation(self.w.current, create=False)
            if ann is not None and ann.redo():
                self.w.annotation_changed()

    def reset_slice(self) -> None:
        """'Refresh': reload the slice image and clear its annotation (undoable)."""
        if not self._ready():
            return
        w = self.w
        try:
            w.session.series.reload_image(w.current)
        except Exception as exc:
            log.warning("Could not reload slice %d: %s", w.current + 1, exc)
        ann = w.session.annotation(w.current, create=False)
        if ann is not None:
            ann.clear()
        w.session.loaded_files.discard(w.current)     # the saved file is no longer being edited
        w.canvas.remove_box()
        w.canvas.show_detection(None)
        w.render_image()
        w.render_saved_overlay()
        w.annotation_changed()
        log.info("Slice %d reset", w.current + 1)

    # ------------------------------------------------------------------ save
    def _confirm_overwrite(self, index: int) -> bool:
        """Ask before replacing an outline saved in an earlier session (blank masks, and
        files loaded with Edit saved outline, never ask)."""
        w = self.w
        s = w.session
        if (not w.settings.get("confirm_overwrite") or s.is_saved(index) or s.is_blank(index)
                or s.is_loaded(index) or not s.has_file_on_disk(index)):
            return True
        box = QMessageBox(QMessageBox.Question, "Replace annotation file?",
                          f"Slice {index + 1} already has an annotation file from an earlier session:\n"
                          f"{os.path.basename(w.session.series.seg_path(index))}",
                          QMessageBox.NoButton, w)
        box.setInformativeText("Replace it with the current annotation?")
        replace = box.addButton("Replace", QMessageBox.AcceptRole)
        box.addButton("Cancel", QMessageBox.RejectRole)
        dont_ask = QCheckBox("Don't ask again")
        box.setCheckBox(dont_ask)
        box.exec_()
        if box.clickedButton() is replace and dont_ask.isChecked():
            w.settings.set("confirm_overwrite", False)
        return box.clickedButton() is replace

    def save_current(self) -> None:
        if not self._ready():
            return
        w = self.w
        index = w.current
        ann = w.session.annotation(index)
        if ann.is_empty and not ask_yes_no(w, "Save blank mask?",
                                           f"Slice {index + 1} has no annotation.\n\nSave an empty "
                                           "annotation file (no lesion on this slice)?",
                                           yes_text="Save blank mask"):
            return
        if not self._confirm_overwrite(index):
            return
        series = w.session.series
        w.set_busy(True, "Saving annotation", os.path.basename(series.seg_path(index)))
        run_task(save_slice_job, ann.label_map.copy(), series.file(index), series.seg_path(index), index,
                 w.labels.snapshot(), strokes=ann.strokes.copy(), blanks=w.batch.blank_items(exclude=[index]),
                 on_progress=lambda _d, _t, msg: w.busy_overlay.set_message(msg),
                 on_result=self._saved, on_error=lambda m, d: show_error(w, "Save failed", m, d),
                 on_finished=lambda: w.set_busy(False))

    def _saved(self, result: SaveResult) -> None:
        w = self.w
        w.mark_saved(result.index, result.blank)
        blanks = w.batch.apply_blank_masks(result.blanks)
        log.info("Saved %s", result.path)
        note = f" · blank masks for {blanks} slice(s) without annotation" if blanks else ""
        w.toasts.show(f"Saved {os.path.basename(result.path)}{note}", "success")
        failed = result.blanks.errors
        for outcome in failed:
            log.warning("Blank mask for slice %d not saved: %s", outcome.index + 1, outcome.message)
        if failed:
            w.toasts.show(f"{len(failed)} blank mask(s) could not be saved — see View ▸ Panels ▸ Activity Log",
                          "warning", 6000)
        w.render_saved_overlay()
        w.refresh_slice_status()
        w.refresh_series_overview()
        w.update_ui_state()

    def save_all_modified(self, blocking: bool = False) -> bool:
        """Save every slice with unsaved changes (and the automatic blank masks).
        Returns True if all were saved."""
        w = self.w
        if w.session is None or w.busy:
            return False
        indices = w.session.dirty_slices()
        blanks = w.batch.blank_items(exclude=indices)
        if not indices and not blanks:
            if not blocking:
                show_info(w, "Nothing to save", "All annotations are already saved.")
            return True
        series = w.session.series
        annotations = {i: w.session.annotation(i) for i in indices}
        items = [BatchItem(i, series.file(i), series.seg_path(i), label_map=a.label_map.copy(),
                           strokes=a.strokes.copy()) for i, a in annotations.items()]
        dialog = TaskProgressDialog(f"Saving {len(items)} slice(s)" if items else "Saving blank masks", w)
        outcome = {}

        def done(report) -> None:
            for o in report.saved:
                w.mark_saved(o.index, o.status == "blank")
            blank_count = w.batch.apply_blank_masks(report)
            outcome["ok"] = not report.errors and not report.cancelled
            lines = [f"✓ {len(report.saved)} slice(s) saved"] if items else []
            if blank_count:
                lines.append(f"✓ {blank_count} blank mask(s) for slices without annotation")
            lines += [f"✗ Slice {o.index + 1}: {o.message}" for o in report.errors[:8]]
            dialog.finish("Saved" if outcome["ok"] else "Finished with problems", lines,
                          "success" if outcome["ok"] else "warning")
            w.render_saved_overlay()
            w.refresh_slice_status()
            w.refresh_series_overview()

        def failed(message: str, details: str) -> None:
            dialog.finish("Saving failed", [message], "error")

        task = run_task(save_many_job, items, w.labels.snapshot(), blanks, on_result=done, on_error=failed,
                        on_progress=dialog.set_progress)
        dialog.cancelRequested.connect(task.cancel)
        dialog.exec_()
        w.update_ui_state()
        return outcome.get("ok", False)

    # ------------------------------------------------------------------ plot
    def plot_results(self) -> None:
        if not self._ready():
            return
        w = self.w
        ann = w.session.annotation(w.current, create=False)
        if ann is None or ann.is_empty:
            show_info(w, "Nothing to plot yet", "This slice has no annotation. Run Auto Detect, "
                                                "draw a box and Detect, or draw with the pen first.")
            return
        from ..dialogs.results_dialog import ResultsDialog
        registry = w.labels.registry
        legend = [(registry.get(v).display_name if registry.get(v) else f"Label {v}", registry.color_for(v))
                  for v in ann.label_values()]
        series = w.session.series
        name = os.path.splitext(os.path.basename(series.file(w.current)))[0]
        ResultsDialog(series.image(w.current), ann.filled_map(), ann.outline_map(), registry.color_lut(), legend,
                      f"Slice {w.current + 1}", name, w).exec_()
