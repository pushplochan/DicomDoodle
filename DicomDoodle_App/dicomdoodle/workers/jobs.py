"""
The long-running jobs of the main annotator, written as plain functions
that ``workers.tasks.run_task`` executes in a background thread.

Each job only *computes and writes files*; it never touches widgets or the
session in use. Results are returned to the GUI thread, which updates the
state. Labels are passed as a ``{value: SegLabel}`` snapshot so the user can
keep editing labels while a job runs.

Each job tells the SEG writer how its segments were made (DICOM Segment
Algorithm Type): Auto Detect batches are AUTOMATIC, SAM regions the user
asked for and reviewed SEMIAUTOMATIC, pen outlines MANUAL.

Every job that saves annotations can also be handed ``blanks``: the slices
that still have no annotation file. Each gets an automatic blank mask after
the job's own work, so the user only annotates slices with a lesion.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.ai_models import Detection, ModelManager
from ..core.annotation import AnnotationSession
from ..core.dicom_series import DicomSeries
from ..core.mask_ops import boundary_map, outline_map
from ..core.progress import OperationCancelled
from ..core.labels import SegLabel
from ..core.seg_io import AI_ASSISTED, AUTO_DETECTED, HAND_DRAWN, write_segmentation
from .tasks import TaskContext

if TYPE_CHECKING:
    from ..core.tumour_range import TumourRangeFinder


# --------------------------------------------------------------------------- #
# Batch items, reports and automatic blank masks
# --------------------------------------------------------------------------- #

@dataclass
class BatchItem:
    index: int
    dicom_path: str
    output_path: str
    image: Optional[np.ndarray] = None
    label_map: Optional[np.ndarray] = None     # "save all modified": the slice's regions ...
    strokes: Optional[np.ndarray] = None       # ... and its hand-drawn outlines
    shape: Tuple[int, int] = (0, 0)            # size of an automatic blank mask


@dataclass
class BatchOutcome:
    index: int
    output_path: str
    status: str                       # "annotated" | "blank" | "no-detection" | "auto-blank" | "error"
    label_map: Optional[np.ndarray] = None
    message: str = ""


@dataclass
class BatchReport:
    outcomes: List[BatchOutcome] = field(default_factory=list)
    cancelled: bool = False
    seconds: float = 0.0

    def count(self, status: str) -> int:
        return sum(1 for o in self.outcomes if o.status == status)

    @property
    def saved(self) -> List[BatchOutcome]:
        """The slices the job was asked to save (automatic blank masks are listed apart)."""
        return [o for o in self.outcomes if o.status not in ("error", "auto-blank")]

    @property
    def auto_blanks(self) -> List[BatchOutcome]:
        return [o for o in self.outcomes if o.status == "auto-blank"]

    @property
    def errors(self) -> List[BatchOutcome]:
        return [o for o in self.outcomes if o.status == "error"]


def _step(ctx: TaskContext, done: int, total: int, message: str) -> bool:
    """Report progress; False once the user has pressed Cancel."""
    try:
        ctx(done, total, message)
        return True
    except OperationCancelled:
        return False


def save_blank_masks(ctx: TaskContext, items: Sequence[BatchItem], labels: Dict[int, SegLabel],
                     report: BatchReport) -> None:
    """Automatic blank masks: an empty SEG for every item whose slice still has no annotation file
    (MANUAL: the user annotated the lesion slices, so the others hold no lesion)."""
    for k, item in enumerate(items):
        if not _step(ctx, k, len(items), f"Blank masks for slices without annotation  ({k + 1}/{len(items)})"):
            report.cancelled = True
            return
        if os.path.exists(item.output_path):      # saved in the meantime: never replace it with a blank
            continue
        try:
            write_segmentation(np.zeros(item.shape, dtype=np.uint8), item.dicom_path, item.output_path,
                               item.index, labels=labels, method=HAND_DRAWN)
            report.outcomes.append(BatchOutcome(item.index, item.output_path, "auto-blank"))
        except Exception as exc:
            report.outcomes.append(BatchOutcome(item.index, item.output_path, "error", message=str(exc)))


# --------------------------------------------------------------------------- #
# Single-slice jobs
# --------------------------------------------------------------------------- #

def load_series_job(ctx: TaskContext, folder: str) -> AnnotationSession:
    """Load the slices, then find which already have annotation files (and which are blank)."""
    session = AnnotationSession(DicomSeries.load(folder, ctx))
    session.scan_files(ctx)
    return session


@dataclass
class SegmentResult:
    index: int                      # slice the job ran on
    mask: Optional[np.ndarray]      # SAM mask, None if nothing was detected
    detection: Optional[Detection] = None
    seconds: float = 0.0


def auto_detect_job(ctx: TaskContext, models: ModelManager, image: np.ndarray, index: int) -> SegmentResult:
    """YOLO (-> YOLOE) box, then SAM inside it — the 'Auto Detect' button."""
    start = time.time()
    ctx(0, 2, "Looking for a lesion (YOLO)…")
    detection = models.detect_box(image)
    if detection is None:
        return SegmentResult(index, None, None, time.time() - start)
    ctx(1, 2, f"Segmenting the {detection.source} box with SAM…")
    mask = models.segment_box(image, detection.box)
    return SegmentResult(index, mask, detection, time.time() - start)


def find_tumour_range_job(ctx: TaskContext, finder: "TumourRangeFinder", index: int,
                          hint: Optional[Sequence[float]] = None):
    """'Find tumour range' from slice ``index`` (core/tumour_range.py). Returns
    the TumourRange, or the NoTumourFound — an answer for the user, not a failure."""
    from ..core.mpr.tumour_detect import NoTumourFound
    try:
        return finder.find(index, hint, ctx)
    except NoTumourFound as exc:
        return exc


def segment_box_job(ctx: TaskContext, models: ModelManager, image: np.ndarray,
                    box: Sequence[float], index: int) -> SegmentResult:
    """SAM inside a box drawn by the user — 'Detect' on a box."""
    start = time.time()
    ctx(0, 1, "Segmenting the box with SAM…")
    mask = models.segment_box(image, box)
    return SegmentResult(index, mask, None, time.time() - start)


@dataclass
class SaveResult:
    index: int
    path: str
    blank: bool = False                                        # saved as a blank mask
    blanks: BatchReport = field(default_factory=BatchReport)   # automatic blank masks written with it


def save_slice_job(ctx: TaskContext, label_map: np.ndarray, dicom_path: str, output_path: str,
                   index: int, labels: Dict[int, SegLabel], strokes: Optional[np.ndarray] = None,
                   blanks: Sequence[BatchItem] = ()) -> SaveResult:
    """Write one slice's annotation — the outline of each filled region plus the
    hand-drawn outlines (``strokes``) as drawn — then the automatic blank masks."""
    ctx(0, 1, "Saving annotation…")
    gt = outline_map(label_map, strokes)
    write_segmentation(gt, dicom_path, output_path, index, labels=labels,
                       method=AI_ASSISTED if np.any(gt) else HAND_DRAWN,
                       manual_values=hand_drawn_values(label_map, strokes))
    result = SaveResult(index, output_path, blank=not np.any(gt))
    save_blank_masks(ctx, blanks, labels, result.blanks)
    return result


def hand_drawn_values(label_map: Optional[np.ndarray], strokes: Optional[np.ndarray]) -> List[int]:
    """Labels drawn only with the pen (no AI region of that label on the slice)."""
    if strokes is None:
        return []
    regions = set(np.unique(label_map)) if label_map is not None else set()
    return [int(v) for v in np.unique(strokes) if v != 0 and v not in regions]


# --------------------------------------------------------------------------- #
# Batch jobs
# --------------------------------------------------------------------------- #

def auto_detect_batch_job(ctx: TaskContext, models: ModelManager, items: List[BatchItem],
                          label_value: int, labels: Dict[int, SegLabel],
                          blanks: Sequence[BatchItem] = ()) -> BatchReport:
    """'Run auto-detect & save': YOLO -> SAM on every slice, blank if nothing found."""
    report, start = BatchReport(), time.time()
    for k, item in enumerate(items):
        if not _step(ctx, k, len(items), f"Slice {item.index + 1}: detecting  ({k + 1}/{len(items)})"):
            report.cancelled = True
            break
        try:
            detection = models.detect_box(item.image)
            label_map = np.zeros(item.image.shape[:2], dtype=np.uint8)
            if detection is not None:
                _step(ctx, k, len(items), f"Slice {item.index + 1}: segmenting  ({k + 1}/{len(items)})")
                label_map[models.segment_box(item.image, detection.box)] = label_value
            status = "annotated" if label_map.any() else "no-detection"
            write_segmentation(boundary_map(label_map), item.dicom_path, item.output_path,
                               item.index, labels=labels, method=AUTO_DETECTED)
            report.outcomes.append(BatchOutcome(item.index, item.output_path, status,
                                                label_map if label_map.any() else None))
        except Exception as exc:
            report.outcomes.append(BatchOutcome(item.index, item.output_path, "error", message=str(exc)))
    if not report.cancelled:
        save_blank_masks(ctx, blanks, labels, report)
    report.seconds = time.time() - start
    return report


def save_many_job(ctx: TaskContext, items: List[BatchItem], labels: Dict[int, SegLabel],
                  blanks: Sequence[BatchItem] = ()) -> BatchReport:
    """'Save All Modified Slices': write each item's regions (as outlines) and hand-drawn outlines."""
    report, start = BatchReport(), time.time()
    for k, item in enumerate(items):
        if not _step(ctx, k, len(items), f"Saving slice {item.index + 1}  ({k + 1}/{len(items)})"):
            report.cancelled = True
            break
        try:
            gt = outline_map(item.label_map, item.strokes)
            write_segmentation(gt, item.dicom_path, item.output_path, item.index, labels=labels,
                               method=AI_ASSISTED if np.any(gt) else HAND_DRAWN,
                               manual_values=hand_drawn_values(item.label_map, item.strokes))
            status = "annotated" if gt.any() else "blank"
            report.outcomes.append(BatchOutcome(item.index, item.output_path, status))
        except Exception as exc:
            report.outcomes.append(BatchOutcome(item.index, item.output_path, "error", message=str(exc)))
    if not report.cancelled:
        save_blank_masks(ctx, blanks, labels, report)
    report.seconds = time.time() - start
    return report
