"""
Batch processing for the main window:

* the Auto Detect Range + "Run auto-detect & save" (original "Save Auto
  Detected Masks")
* "Find tumour range": sets that range from one slice of the tumour, with
  the tumour detector on this slice and on the two other views through it
  (core/tumour_range.py)
* automatic blank masks: every save also writes a blank mask for each slice
  that has no annotation file yet. This replaces the original's two blank
  ranges and its "Save Blank Masks" button — the user annotates only the
  slices with a lesion.

As in the original, slices already saved during this session are skipped.
New: the user is asked before replacing annotation files from earlier
sessions or slices with unsaved work (blank masks are replaced without
asking), jobs run in the background with a Cancel button, and finished
slices show their results immediately.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Iterable, List, Optional

import numpy as np
from PyQt5.QtCore import QObject

from ...core.tumour_range import TumourRange, TumourRangeFinder
from ...workers import run_task
from ...workers.jobs import BatchItem, BatchReport, auto_detect_batch_job, find_tumour_range_job
from ..dialogs.progress_dialog import TaskProgressDialog
from ..qt_utils import ask_choice, show_error, show_info

if TYPE_CHECKING:
    from .window import MainWindow

log = logging.getLogger("dicomdoodle")

RANGE_NAME = "Auto Detect Range"


class BatchController(QObject):
    def __init__(self, window: "MainWindow"):
        super().__init__(window)
        self.w = window
        self._finder: Optional[TumourRangeFinder] = None   # keeps the series' 3D volume between presses
        self.found: Optional[TumourRange] = None            # the last "Find tumour range" result

    # --------------------------------------------------------------- range
    def range_action(self, action: str) -> None:
        """Set the start or end of the auto-detect range to the current slice, or clear it."""
        w = self.w
        if w.session is None or w.busy:
            return
        rng = w.session.auto_range
        if action == "start":
            rng.start = w.current
            log.info("%s start set to slice %d", RANGE_NAME, w.current + 1)
        elif action == "end":
            rng.end = w.current
            log.info("%s end set to slice %d", RANGE_NAME, w.current + 1)
        else:
            rng.clear()
            self.show_found(None)
            log.info("%s cleared", RANGE_NAME)
        if rng.is_complete:
            w.toasts.show(f"{RANGE_NAME}: slices {rng.describe()}", "info", 2500)
        w.refresh_series_overview()

    # ---------------------------------------------------------- tumour range
    def find_tumour_range(self) -> None:
        """'Find tumour range': the slices the tumour spans become the range,
        found from the current slice. A box drawn on the slice tells which
        structure is meant."""
        w = self.w
        if w.session is None or w.busy or w.canvas.is_drawing or not w.ai.explain("range"):
            return
        if self._finder is None or self._finder.series is not w.session.series:
            self._finder = TumourRangeFinder(w.models, w.session.series)
        w.set_busy(True, "Finding the tumour range",
                   "The tumour detector looks at this slice, then at the two other views through the tumour…")
        run_task(find_tumour_range_job, self._finder, w.current, w.canvas.box_coords(),
                 on_progress=lambda _d, _t, msg: w.busy_overlay.set_message(msg),
                 on_result=self._range_found,
                 on_error=lambda message, details: show_error(w, "Could not find the tumour range", message, details),
                 on_finished=lambda: w.set_busy(False))

    def _range_found(self, result) -> None:
        w = self.w
        if w.session is None or self._finder is None or self._finder.series is not w.session.series:
            return
        if not isinstance(result, TumourRange):          # NoTumourFound
            log.info("Find tumour range: %s", result)
            w.toasts.show(f"{result} Go to a slice where the tumour is large and clear, or draw a box "
                          "around it first.", "warning", 7000)
            return
        rng = w.session.auto_range
        rng.start, rng.end = result.first, result.last
        log.info("%s (%.1f s on %s); this slice: %s; %s", result.summary_text(), result.seconds, result.device,
                 result.seed_how, "; ".join(f"{v.plane}: {v.how}" for v in result.views))
        self.show_found(result)
        if result.seed == w.current and w.canvas.box_coords() is None:
            w.canvas.show_detection(result.seed_box, "Tumour")
        if result.estimated:
            w.toasts.show(f"{RANGE_NAME}: slices {rng.describe()}, estimated — the detector found no box on the "
                          "other views. Check the range.", "warning", 7000)
        else:
            w.toasts.show(f"{RANGE_NAME}: slices {rng.describe()} — check it, then run auto-detect & save",
                          "success", 5000)
        w.refresh_series_overview()

    def show_found(self, result: Optional[TumourRange]) -> None:
        self.found = result
        self.w.batch_panel.show_found_range(result)

    def forget_series(self) -> None:
        """A series was closed: drop its 3D volume and the last result."""
        self._finder = None
        self.show_found(None)

    # ------------------------------------------------- automatic blank masks
    def blank_items(self, exclude: Iterable[int] = ()) -> List[BatchItem]:
        """The blank masks to write with a save: one for every slice that has no
        annotation file and no outline yet (``exclude``: the slices being saved)."""
        w = self.w
        if w.session is None or not w.settings.get("auto_blank_masks"):
            return []
        series = w.session.series
        return [BatchItem(i, series.file(i), series.seg_path(i), shape=series.image(i).shape)
                for i in w.session.blank_mask_targets(exclude)]

    def apply_blank_masks(self, report: BatchReport) -> int:
        """Record the automatic blank masks a save has written; returns how many."""
        written = report.auto_blanks
        for outcome in written:
            self.w.session.mark_auto_blank(outcome.index)
        if written:
            log.info("Saved blank masks for %d slice(s) without annotation", len(written))
        return len(written)

    def set_auto_blank(self, enabled: bool) -> None:
        self.w.settings.set("auto_blank_masks", bool(enabled))
        log.info("Automatic blank masks %s", "on" if enabled else "off")
        self.w.refresh_series_overview()

    # ------------------------------------------------------- target checks
    def _choose_targets(self, indices: List[int], what: str) -> Optional[List[int]]:
        """Apply the skip rules and ask about files that would be replaced."""
        s = self.w.session
        saved, existing, fresh = s.split_existing(indices)
        with_work = [i for i in fresh + existing
                     if s.annotation(i, create=False) is not None and not s.annotation(i).is_empty
                     and s.annotation(i).dirty]
        risky = sorted(set(existing) | set(with_work))
        targets = fresh + existing
        if risky:
            notes = []
            if existing:
                notes.append(f"• {len(existing)} slice(s) already have an annotation file from an earlier session")
            if with_work:
                notes.append(f"• {len(with_work)} slice(s) have unsaved annotations in this window")
            choice = ask_choice(self.w, what, f"{what} for {len(targets)} slice(s)?",
                                ["Replace them", "Skip those slices"], informative="\n".join(notes))
            if choice is None:
                return None
            if choice == 1:
                targets = [i for i in targets if i not in risky]
        if saved:
            log.info("%s: skipping %d slice(s) already saved this session", what, len(saved))
        if not targets:
            show_info(self.w, what, "There is nothing left to process — every slice in the range was skipped.")
            return None
        return sorted(targets)

    def _items(self, indices: List[int], with_images: bool) -> List[BatchItem]:
        series = self.w.session.series
        return [BatchItem(i, series.file(i), series.seg_path(i), series.image(i) if with_images else None)
                for i in indices]

    # ---------------------------------------------------------- auto detect
    def run_auto_detect(self) -> None:
        w = self.w
        if w.session is None or w.busy or not w.ai.explain("detect"):
            return
        indices = w.session.auto_targets()
        if not indices:
            show_info(w, "Auto-detect range", "Set the start and end of the Auto Detect Range first.")
            return
        targets = self._choose_targets(indices, "Auto-detect and save")
        if targets is None:
            return
        label_value = w.label_panel.active_value
        w.set_ai_status("loading", "AI working…")
        self._run("Auto-detecting and saving", auto_detect_batch_job,
                  [w.models, self._items(targets, True), label_value, w.labels.snapshot(),
                   self.blank_items(exclude=targets)],
                  skipped=len(indices) - len(targets), label_value=label_value)

    # ---------------------------------------------------------------- runner
    def _run(self, title: str, job: Callable, args: list, skipped: int, label_value: Optional[int] = None) -> None:
        w = self.w
        dialog = TaskProgressDialog(title, w)
        w.busy = True
        w.update_ui_state()

        def done(report: BatchReport) -> None:
            self._apply_report(report, label_value)
            dialog.finish(*self._summary(report, skipped))

        def failed(message: str, details: str) -> None:
            dialog.finish("Batch failed", [message], "error")

        task = run_task(job, *args, on_result=done, on_error=failed, on_progress=dialog.set_progress)
        dialog.cancelRequested.connect(task.cancel)
        dialog.exec_()
        w.busy = False
        w.ai.refresh()
        w.render_annotation()
        w.render_saved_overlay()
        w.refresh_slice_status()
        w.refresh_series_overview()
        w.update_ui_state()

    def _apply_report(self, report: BatchReport, label_value: Optional[int]) -> None:
        s = self.w.session
        for outcome in report.saved:
            ann = s.annotation(outcome.index)
            if outcome.label_map is not None:
                ann.replace(outcome.label_map, mark_dirty=False)
            elif not ann.is_empty:
                ann.replace(np.zeros_like(ann.label_map), mark_dirty=False)
            self.w.mark_saved(outcome.index, blank=outcome.label_map is None)
        self.apply_blank_masks(report)
        log.info("Batch finished: %d saved, %d failed%s", len(report.saved), len(report.errors),
                 " (cancelled)" if report.cancelled else "")

    @staticmethod
    def _summary(report: BatchReport, skipped: int):
        lines = [f"✓ {len(report.saved)} slice(s) saved"]
        if report.count("annotated"):
            lines.append(f"&nbsp;&nbsp;• {report.count('annotated')} with a detected lesion")
        if report.count("no-detection"):
            lines.append(f"&nbsp;&nbsp;• {report.count('no-detection')} blank (nothing detected)")
        if skipped:
            lines.append(f"• {skipped} slice(s) skipped")
        if report.auto_blanks:
            lines.append(f"✓ {len(report.auto_blanks)} blank mask(s) for slices without annotation")
        for outcome in report.errors[:6]:
            lines.append(f"✗ Slice {outcome.index + 1}: {outcome.message}")
        if report.cancelled:
            return "Stopped", ["Cancelled before all slices were processed."] + lines, "warning"
        if report.errors:
            return "Finished with problems", lines, "warning"
        return "Finished", lines, "success"
