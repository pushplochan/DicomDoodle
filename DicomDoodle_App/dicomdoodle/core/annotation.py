"""
In-memory annotation state for the main annotator window.

* :class:`SliceAnnotation` — the annotation of one slice in two layers
  (filled AI regions, and outlines drawn by hand with the pen), with
  multi-level undo / redo and a "modified since last save" flag.
* :class:`SliceRange` — a start/end pair picked by the user (the auto-detect range).
* :class:`AnnotationSession` — everything for one loaded series: per-slice
  annotations, which slices have annotation files (and which of those are
  blank masks), which were saved this session or loaded for editing, and
  the auto-detect range.

Unlike the original tool, annotations are kept per slice, so scrolling away
and back no longer loses unsaved work. Blank masks are automatic: every save
also gives each slice without an annotation file a blank mask (see
:meth:`AnnotationSession.blank_mask_targets`), so the user only annotates
the slices that show a lesion.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from . import mask_ops
from .dicom_series import DicomSeries
from .progress import Progress, ensure_progress
from .seg_io import is_blank_segmentation

MAX_UNDO_STEPS = 30


class SliceAnnotation:
    """Annotation of one slice plus its edit history, in two layers:

    * ``label_map`` — filled regions (AI results), saved as their outlines;
    * ``strokes``   — outlines drawn by hand with the pen, saved exactly as drawn.

    Both use label values (0 = background); :meth:`outline_map` combines them
    into what is written to the annotation file.
    """

    def __init__(self, shape):
        self.label_map = np.zeros(shape, dtype=np.uint8)
        self.strokes = np.zeros(shape, dtype=np.uint8)
        self._undo: List[Tuple[np.ndarray, np.ndarray]] = []
        self._redo: List[Tuple[np.ndarray, np.ndarray]] = []
        self.dirty = False            # changed since it was last saved / loaded

    # ---------------------------------------------------------------- state
    @property
    def is_empty(self) -> bool:
        return not np.any(self.label_map) and not np.any(self.strokes)

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def label_values(self) -> List[int]:
        """Label values present in either layer, ascending."""
        return sorted(set(mask_ops.label_values(self.label_map)) | set(mask_ops.label_values(self.strokes)))

    def outline_map(self) -> np.ndarray:
        """The slice as it is saved: region outlines, with the hand-drawn outlines on top."""
        return mask_ops.outline_map(self.label_map, self.strokes)

    def filled_map(self) -> np.ndarray:
        """The regions plus the inside of closed hand-drawn outlines (for plots)."""
        out = mask_ops.fill_outlines(self.strokes)
        regions = self.label_map > 0
        out[regions] = self.label_map[regions]
        return out

    # -------------------------------------------------------------- editing
    def _push_undo(self) -> None:
        self._undo.append((self.label_map.copy(), self.strokes.copy()))
        if len(self._undo) > MAX_UNDO_STEPS:
            self._undo.pop(0)
        self._redo.clear()

    def begin_edit(self) -> None:
        """Snapshot for undo before an in-place edit (e.g. a pen stroke on ``strokes``)."""
        self._push_undo()
        self.dirty = True

    def paint(self, mask: np.ndarray, value: int) -> None:
        """Set every pixel of ``mask`` to ``value`` (later regions overwrite earlier ones)."""
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self.label_map.shape or not mask.any():
            return
        self._push_undo()
        self.label_map[mask] = value
        self.dirty = True

    def replace(self, label_map: np.ndarray, mark_dirty: bool = True,
                strokes: Optional[np.ndarray] = None) -> None:
        """New content: ``label_map`` regions and ``strokes`` outlines (default: none)."""
        self._push_undo()
        self.label_map = np.asarray(label_map, dtype=np.uint8).copy()
        self.strokes = (np.zeros_like(self.label_map) if strokes is None
                        else np.asarray(strokes, dtype=np.uint8).copy())
        self.dirty = mark_dirty

    def clear(self) -> None:
        if self.is_empty:
            return
        self._push_undo()
        self.label_map[:] = 0
        self.strokes[:] = 0
        self.dirty = True

    def regions_to_outlines(self) -> bool:
        """Turn the filled regions into their outlines on the hand-drawn layer, so the
        eraser removes exactly the lines that would be saved. The saved result does
        not change. Belongs to the current edit (call after :meth:`begin_edit`);
        returns True if there were regions."""
        if not np.any(self.label_map):
            return False
        outline = mask_ops.boundary_map(self.label_map)
        free = (outline > 0) & (self.strokes == 0)
        self.strokes[free] = outline[free]
        self.label_map[:] = 0
        return True

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append((self.label_map, self.strokes))
        self.label_map, self.strokes = self._undo.pop()
        self.dirty = True
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append((self.label_map, self.strokes))
        self.label_map, self.strokes = self._redo.pop()
        self.dirty = True
        return True


@dataclass
class SliceRange:
    """Start / end slice indices (0-based, either order) chosen by the user."""

    start: Optional[int] = None
    end: Optional[int] = None

    @property
    def is_complete(self) -> bool:
        return self.start is not None and self.end is not None

    def indices(self) -> List[int]:
        if not self.is_complete:
            return []
        lo, hi = sorted((self.start, self.end))
        return list(range(lo, hi + 1))

    def clear(self) -> None:
        self.start = self.end = None

    def describe(self) -> str:
        """Human text with 1-based slice numbers, e.g. '12 – 20 (9 slices)'."""
        fmt = lambda v: "—" if v is None else str(v + 1)
        text = f"{fmt(self.start)} – {fmt(self.end)}"
        if self.is_complete:
            n = len(self.indices())
            text += f"  ({n} slice{'s' if n != 1 else ''})"
        return text


class AnnotationSession:
    """Annotation state for one loaded DICOM series."""

    def __init__(self, series: DicomSeries):
        self.series = series
        self.annotations: Dict[int, SliceAnnotation] = {}
        self.saved_this_session: Set[int] = set()
        self.files: Set[int] = set()            # slices with an annotation file on disk
        self.blank_files: Set[int] = set()      # ... of which the file is a blank mask
        self.loaded_files: Set[int] = set()     # slices whose file was loaded for editing (Edit saved outline)
        self.auto_range = SliceRange()

    def scan_files(self, progress: Optional[Progress] = None) -> None:
        """Find the slices that already have an annotation file, and which of those are blank masks."""
        progress = ensure_progress(progress)
        self.files.clear()
        self.blank_files.clear()
        for i in range(len(self.series)):
            progress(i, len(self.series), "Checking annotation files…")
            if not self.has_file_on_disk(i):
                continue
            self.files.add(i)
            try:
                if is_blank_segmentation(self.series.seg_path(i)):
                    self.blank_files.add(i)
            except Exception:   # unreadable: counts as an annotation, so it is never replaced without asking
                pass

    # ------------------------------------------------------------ per slice
    def annotation(self, index: int, create: bool = True) -> Optional[SliceAnnotation]:
        ann = self.annotations.get(index)
        if ann is None and create:
            ann = SliceAnnotation(self.series.image(index).shape)
            self.annotations[index] = ann
        return ann

    def mark_saved(self, index: int, blank: bool = False) -> None:
        """The user saved this slice (``blank``: as a blank mask)."""
        self.saved_this_session.add(index)
        self.files.add(index)
        if blank:
            self.blank_files.add(index)
        else:
            self.blank_files.discard(index)
        ann = self.annotations.get(index)
        if ann is not None:
            ann.dirty = False

    def mark_auto_blank(self, index: int) -> None:
        """An automatic blank mask was written. It does not count as saved by the user."""
        self.files.add(index)
        self.blank_files.add(index)

    def mark_loaded(self, index: int) -> None:
        """The slice's annotation file was loaded into the window to be corrected."""
        self.loaded_files.add(index)

    def is_loaded(self, index: int) -> bool:
        return index in self.loaded_files

    def is_saved(self, index: int) -> bool:
        return index in self.saved_this_session

    def is_blank(self, index: int) -> bool:
        """True if the slice's annotation file is a blank mask."""
        return index in self.blank_files

    def has_file_on_disk(self, index: int) -> bool:
        return os.path.isfile(self.series.seg_path(index))

    def dirty_slices(self) -> List[int]:
        return sorted(i for i, a in self.annotations.items() if a.dirty)

    def annotated_slices(self) -> List[int]:
        return sorted(i for i, a in self.annotations.items() if not a.is_empty)

    # ---------------------------------------------------------- blank masks
    def blank_mask_targets(self, exclude: Iterable[int] = ()) -> List[int]:
        """Slices that get an automatic blank mask with the next save: no annotation
        file yet and no outline in this window (``exclude``: the slices being saved)."""
        skip = set(exclude)
        return [i for i in range(len(self.series))
                if i not in skip and not self.has_file_on_disk(i)
                and (i not in self.annotations or self.annotations[i].is_empty)]

    # ---------------------------------------------------------------- batch
    def auto_targets(self) -> List[int]:
        return [i for i in self.auto_range.indices() if i < len(self.series)]

    def split_existing(self, indices: Iterable[int]):
        """Split batch targets into (saved this session, file from earlier, free).

        The original tool skipped slices saved during the session; files from
        earlier sessions are reported so the user can decide. Blank masks
        count as free: replacing one never loses an outline.
        """
        saved, existing, fresh = [], [], []
        for i in indices:
            if self.is_saved(i):
                saved.append(i)
            elif self.has_file_on_disk(i) and not self.is_blank(i):
                existing.append(i)
            else:
                fresh.append(i)
        return saved, existing, fresh
