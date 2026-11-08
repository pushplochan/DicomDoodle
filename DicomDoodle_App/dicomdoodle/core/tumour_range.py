"""
"Find tumour range" in the main window's Batch Processing panel: the first
and last slice of a tumour, found in a few seconds from one slice that
shows it, as the range for "Run auto-detect & save".

The series' 3D volume comes from the Multiplanar view's loader
(core/mpr/loading.py) and the detection from its tumour detector
(TumourDetector.find_range in core/mpr/tumour_detect.py): the detector boxes
the tumour on the slice, the two other views (coronal and sagittal for an
axial series) are cut from the volume through the centre of that box, and
the detector's box around the same place on each gives the tumour's extent
along the slice direction. The range covers both extents. Only the
detector runs, not SAM; the slices of the range are outlined later by the
batch.

Slice numbers and boxes are converted between the volume and the main
window's slices through each slice file's position and orientation
(ImagePositionPatient, ImageOrientationPatient), so the order in which the
main window lists the slices does not matter.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pydicom
import SimpleITK as sitk

from .ai_models import ModelManager
from .dicom_series import DicomSeries
from .mpr.loading import find_series, get_acquisition_plane, load_dicom_series
from .mpr.tumour_detect import AXIS_PLANE, PLANE_AXES, NoTumourFound, SeedView, TumourDetector, acquisition_axis
from .mpr.volume import InterpMethod, Volume, window_to_uint8
from .progress import Progress, ensure_progress

Box = Tuple[float, float, float, float]   # (x0, y0, x1, y1) in pixels, 0 = left / top edge of the first pixel

PREVIEW_SIZE = 360      # longest side of a preview image, pixels


@dataclass
class RangeView:
    """One of the two other views through the tumour, as the panel previews it."""

    plane: str                          # "coronal", "sagittal" or "axial"
    image: np.ndarray                   # uint8, superior (or anterior) up, square pixels
    box: Box                            # the detector's box on it, in image pixels
    along_rows: bool                    # the slice direction runs down the image (else across it)
    band: Tuple[float, float]           # the range along that direction, in image pixels
    slices: Optional[Tuple[int, int]]   # the box's first and last slice of the main window (0-based)
    how: str
    estimated: bool = False             # no detector box here: the box was guessed from the seed's size


@dataclass
class TumourRange:
    first: int                  # first and last slice of the main window (0-based, first <= last)
    last: int
    seed: int                   # the slice it was found from
    seed_box: Box               # the detector's box on that slice, in image pixels
    seed_how: str
    views: List[RangeView]
    estimated: bool             # no detector box on the other views: guessed from the tumour's size here
    device: str
    seconds: float

    @property
    def count(self) -> int:
        return self.last - self.first + 1

    def summary_text(self) -> str:
        """e.g. 'Tumour range: slices 114-151 (38 slices) from slice 133 · coronal 114-150, sagittal 115-151'."""
        views = ", ".join(f"{v.plane}: no box" if v.estimated or not v.slices
                          else f"{v.plane} {v.slices[0] + 1}–{v.slices[1] + 1}" for v in self.views)
        source = "estimated from the tumour's size on slice" if self.estimated else "from slice"
        return (f"Tumour range: slices {self.first + 1}–{self.last + 1} ({self.count} slices) {source} "
                f"{self.seed + 1}" + (f" · {views}" if views else ""))


@dataclass
class _Geometry:
    """Where one slice file's pixels are in the patient (DICOM image plane module)."""

    origin: np.ndarray      # ImagePositionPatient: centre of the first pixel, mm
    u: np.ndarray           # direction of increasing column index
    v: np.ndarray           # direction of increasing row index
    du: float               # spacing between columns, mm
    dv: float               # spacing between rows, mm

    @classmethod
    def read(cls, path: str) -> Optional["_Geometry"]:
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
            iop = [float(x) for x in ds.ImageOrientationPatient]
            dv, du = (float(x) for x in ds.PixelSpacing)
            return cls(np.array([float(x) for x in ds.ImagePositionPatient]), np.array(iop[:3]),
                       np.array(iop[3:6]), du, dv)
        except Exception:
            return None


class _Stack:
    """The 3D volume a main-window slice belongs to, and the conversions
    between its array coordinates (z, y, x) and the main window's slices."""

    def __init__(self, image: sitk.Image, files: List[str], series: DicomSeries):
        sx, sy, sz = image.GetSpacing()
        # a view of the image's pixels, not a copy (the detector only reads them)
        self.volume = Volume(sitk.GetArrayViewFromImage(image), (sz, sy, sx), image.GetOrigin(),
                             image.GetDirection(), False, InterpMethod.NONE_NATIVE, image)
        self.image = image
        self.acquired = get_acquisition_plane(image)
        self.axis = acquisition_axis(self.volume, self.acquired)
        self.plane = AXIS_PLANE[self.axis]
        self.files = {_norm(f) for f in files}
        self.geometry: Dict[int, _Geometry] = {}
        self.slice_of: Dict[int, int] = {}          # main-window slice -> slice of the volume
        for j, path in enumerate(series.files):
            g = _Geometry.read(path) if _norm(path) in self.files else None
            if g is None:
                continue
            z = self._to_volume(g, 0.0, 0.0)[self.axis]
            if abs(z - round(z)) < 0.25 and 0 <= round(z) < self.volume.shape[self.axis]:
                self.geometry[j], self.slice_of[j] = g, int(round(z))

    def _to_volume(self, g: _Geometry, col: float, row: float) -> np.ndarray:
        """Array coordinates (z, y, x) of the pixel centre (col, row) of a slice."""
        point = g.origin + col * g.du * g.u + row * g.dv * g.v
        x, y, z = self.image.TransformPhysicalPointToContinuousIndex(point.tolist())
        return np.array([z, y, x])

    def _from_volume(self, g: _Geometry, zyx) -> Tuple[float, float]:
        """(col, row) pixel centre on a slice of the array point (z, y, x)."""
        index_xyz = [float(zyx[2]), float(zyx[1]), float(zyx[0])]
        d = np.array(self.image.TransformContinuousIndexToPhysicalPoint(index_xyz)) - g.origin
        return float(d @ g.u) / g.du, float(d @ g.v) / g.dv

    def box_to_volume(self, j: int, box: Box) -> Box:
        """A box on main-window slice j -> the same box on the volume's slice (its rows and columns)."""
        _, r, c = PLANE_AXES[self.plane]
        corners = [self._to_volume(self.geometry[j], x - 0.5, y - 0.5) for x in box[0::2] for y in box[1::2]]
        cols, rows = [p[c] + 0.5 for p in corners], [p[r] + 0.5 for p in corners]
        return min(cols), min(rows), max(cols), max(rows)

    def box_from_volume(self, j: int, box: Box) -> Box:
        """The inverse of box_to_volume."""
        a, r, c = PLANE_AXES[self.plane]
        corners = []
        for col in box[0::2]:
            for row in box[1::2]:
                zyx = np.zeros(3)
                zyx[a], zyx[r], zyx[c] = self.slice_of[j], row - 0.5, col - 0.5
                x, y = self._from_volume(self.geometry[j], zyx)
                corners.append((x + 0.5, y + 0.5))
        xs, ys = zip(*corners)
        return min(xs), min(ys), max(xs), max(ys)

    def main_slices(self, lo: int, hi: int) -> Optional[Tuple[int, int]]:
        """First and last main-window slice among the volume's slices [lo, hi)."""
        inside = [j for j, z in self.slice_of.items() if lo <= z < hi]
        return (min(inside), max(inside)) if inside else None


def _norm(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


class TumourRangeFinder:
    """Finds tumour ranges in one loaded series; keeps its 3D volume for the
    next time (a few hundred MB for a large series)."""

    def __init__(self, models: ModelManager, series: DicomSeries):
        self.models, self.series = models, series
        self._detector: Optional[TumourDetector] = None     # made by find(): it imports torch
        self._stack: Optional[_Stack] = None

    def find(self, index: int, hint: Optional[Box] = None, progress: Optional[Progress] = None) -> TumourRange:
        """The tumour range from main-window slice ``index``.

        hint: a box drawn around the tumour on that slice, in image pixels;
            only detector boxes centred inside it are used (none: the box itself).
        Raises NoTumourFound when the detector proposes nothing on the slice.
        """
        t0 = time.time()
        step = ensure_progress(progress)
        stack = self._stack_for(index, step)
        found_hint = None if hint is None else stack.box_to_volume(index, hint)
        if self._detector is None:
            self._detector = TumourDetector(self.models, tune_threads=False)   # leave torch's settings alone
        try:
            found = self._detector.find_range(stack.volume, stack.plane, stack.slice_of[index], stack.acquired,
                                              hint=found_hint, progress=progress)
        except NoTumourFound:
            raise NoTumourFound(f"The tumour detector found nothing on slice {index + 1}.") from None
        seed = found.seeds[0]
        first, last = stack.main_slices(*found.slices) or (index, index)
        views = [self._view(stack, sv, found.extent(sv), found.slices) for sv in found.seeds[1:]]
        return TumourRange(first, last, index, stack.box_from_volume(index, seed.box), seed.how, views,
                           found.estimated, found.device, time.time() - t0)

    def _stack_for(self, index: int, step: Progress) -> _Stack:
        path = _norm(self.series.file(index))
        stack = self._stack
        if stack is None or path not in stack.files:
            step(0, 0, "Building the 3D volume of the series (only the first time)…")
            info = next((s for s in find_series(self.series.folder)
                         if not s.multi_frame and path in {_norm(f) for f in s.files}), None)
            if info is None:
                raise RuntimeError(f"Slice {index + 1} is not part of a stack of parallel slices (a localiser, "
                                   "for example), so there are no other views to look at. Use a slice of the "
                                   "main series.")
            self._stack = stack = None     # the old volume can go before the new one is read
            stack = _Stack(load_dicom_series(self.series.folder, info.key), info.files, self.series)
            self._stack = stack
        if index not in stack.slice_of:
            raise RuntimeError(f"The position of slice {index + 1} could not be read from its file "
                               "(ImagePositionPatient / ImageOrientationPatient).")
        return stack

    @staticmethod
    def _view(stack: _Stack, sv: SeedView, extent, slices) -> RangeView:
        """The slice of ``sv`` as shown in the multiplanar view (rows flipped:
        superior / anterior up), with square pixels, at most PREVIEW_SIZE."""
        a, r, c = PLANE_AXES[sv.plane]
        image = np.ascontiguousarray(np.flipud(window_to_uint8(np.take(stack.volume.array, sv.index, axis=a))))
        h, w = image.shape
        sp = stack.volume.spacing_zyx
        fx, fy = sp[c] / min(sp[r], sp[c]), sp[r] / min(sp[r], sp[c])
        scale = min(1.0, PREVIEW_SIZE / max(w * fx, h * fy))
        fx, fy = fx * scale, fy * scale
        size = (max(1, round(w * fx)), max(1, round(h * fy)))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        x0, y0, x1, y1 = sv.box
        box = (x0 * fx, (h - y1) * fy, x1 * fx, (h - y0) * fy)
        lo, hi = slices
        along_rows = r == stack.axis
        band = ((h - hi) * fy, (h - lo) * fy) if along_rows else (lo * fx, hi * fx)
        return RangeView(sv.plane.value, image, box, along_rows, band,
                         stack.main_slices(*extent) if extent else None, sv.how, sv.estimated)
