"""
Tumour detection behind the "Detect" button on each 2D view of the
Multiplanar & 3D Viewer (from zero_shot_tumor/tumour_detect.py). The tumour
detector — the model card's lesion detector, YOLO26x-seg trained on BraTS 2021
slices with one class "tumor" (before it: best_flair.pt, YOLO11x-seg trained on
BraTS 2021 FLAIR slices) — proposes boxes, the three orthogonal views agree on
a 3D cuboid around the tumour, and SAM 2.1 turns the boxes inside that cuboid
into a mask on every slice of the acquired series.

The detector sees each slice the way best_flair.pt's training slices were
made (_Slice.det): min-max of the slice itself to uint8, transposed so that
the patient's left-right axis runs down the rows (the BraTS axial slices were
saved from nibabel arrays without reorientation). Its training data were
skull-stripped FLAIR; T1+C series and heads with skull and scalp are outside
what it was trained on. SAM 2.1 sees the slice with one window for the
whole volume (_Slice.model).

For one press of Detect on slice ``index`` of view ``plane``:

1. Seed slice. The detector proposes boxes at low confidence (CONF). If the
   user's last click was on this view, only boxes under the crosshair are
   kept (no box there: SAM is prompted with the click itself). Among boxes
   of at most AREA_MAX of the slice, the most confident is taken as the
   tumour and refined into a mask by SAM.
2. Corresponding slices. The other two views are cut through the seed
   mask's centre. On each, the box must contain the seed slice and line up
   with the seed mask along the axis both views share; the best (line-up x
   confidence) is refined by SAM. A view with no such box falls back to a
   box built from the seed mask's size.
3. Cuboid. Every array axis is seen by two of the three views; the union of
   their mask extents is the 3D search box, padded by MARGIN_MM in-plane.
   Its extent along the acquisition axis is the tumour's slice range.
4. Range. The detector runs on every acquired slice in that range, from the
   seed outwards in both directions. Of the boxes mostly inside the cuboid
   (INSIDE_MIN), the one that best overlaps the tumour on the neighbouring
   slice goes to SAM (no such box: the neighbour's box is the prompt). The
   mask is filled, must overlap the neighbour's (FOLLOW_MIN), and is clipped
   to the cross-section expected on that slice from the two orthogonal seed
   masks plus CLIP_FRAC (stops SAM spreading into neighbouring structures).
   A 3-slice majority along the acquisition axis then removes single-slice
   spikes and gaps (slices up to SMOOTH_MAX_MM thick).

On data_AIIMS (15 cases, 45 T1+C series, Detect on the slice with the
largest tumour area, with a click on the tumour) the median 3D Dice against
the expert outlines was 0.92 — measured with the YOLOE-pf detector this
pipeline used before best_flair.pt replaced it (zero_shot_tumor/README.md).

find_range() runs steps 1-3 with detector boxes only, for "Find tumour
range" in the main window (core/tumour_range.py): the seed box stands in
for the seed mask, so the other two views are cut through the box's centre,
their boxes are not refined by SAM, and the cuboid's extent along the
acquisition axis is the tumour's slice range — three detector calls.

Everything works in the Volume's (z, y, x) index space, before the display
flip of extract_slice; DetectionResult.display_box converts back for
drawing. Reformatted views with anisotropic pixels are resized to square
pixels before the models see them. The models come from the app's
ModelManager (loaded once, shared with the main window).
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from ..ai_models import ModelManager
from ..progress import Progress, ensure_progress
from .segmentation import Segmentation, SegmentInfo
from .volume import Plane, Volume

CONF = 0.01            # low on purpose: the tumour only has to be somewhere among the proposals
AREA_MAX = 0.10        # boxes over 10 % of the slice are head/brain-sized anatomy
MIN_AREA_MM2 = 10.0    # smaller boxes / masks are vessels or noise
LINE_IOU_MIN = 0.3     # corresponding view: 1D overlap with the seed mask along the shared axis
INSIDE_MIN = 0.5       # "under the cuboid": at least this fraction of the box lies inside it
MATCH_IOU_MIN = 0.3    # range slice: overlap of the detector box with the tumour's box on the neighbouring slice
FOLLOW_MIN = 0.5       # range slice: mask overlap with the neighbouring slice's, as a fraction of the smaller
SMOOTH_MAX_MM = 2.0    # 3-slice majority only for slices up to this thick (a thicker window blurs real shape)
MARGIN_MM = 3.0        # cuboid padding on every side
CLIP_FRAC = 0.25       # range slice: SAM may exceed the expected cross-section by this fraction of its size
                       # (at least MARGIN_MM): the two central chords underestimate irregular tumours off-centre
CPU_THREADS = min(32, os.cpu_count() or 1)   # importing ultralytics caps torch at 8 threads

DETECT_NAME = "Detected tumour"
DETECT_COLOR = (255, 225, 25)   # overlay colour, distinct from the red of a loaded annotation

# (slice axis, row axis, column axis) of each view, in (z, y, x) array order.
# np.take(array, i, axis=slice_axis) keeps the other two axes in this order.
PLANE_AXES = {Plane.AXIAL: (0, 1, 2), Plane.CORONAL: (1, 0, 2), Plane.SAGITTAL: (2, 0, 1)}
AXIS_PLANE = {axes[0]: plane for plane, axes in PLANE_AXES.items()}

Box = Tuple[float, float, float, float]    # (x0, y0, x1, y1) = (col0, row0, col1, row1), half-open
Interval = Tuple[int, int]                 # [lo, hi) voxel indices


class NoTumourFound(RuntimeError):
    """The detector proposed no usable box on the seed slice."""


# --------------------------------------------------------------------------- #
# Small geometry helpers (boxes and intervals in array coordinates)
# --------------------------------------------------------------------------- #

def acquisition_axis(volume: Volume, acquisition_plane: str) -> int:
    """Array axis the series was acquired along; 'Oblique' / 'Unknown' fall
    back to the coarsest axis (the slice direction of a 2D acquisition)."""
    named = {"Axial": 0, "Coronal": 1, "Sagittal": 2}
    if acquisition_plane in named:
        return named[acquisition_plane]
    return int(np.argmax(volume.spacing_zyx))


def _iou(a: Box, b: Box) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def _iou_1d(a, b) -> float:
    inter = min(a[1], b[1]) - max(a[0], b[0])
    return inter / (max(a[1], b[1]) - min(a[0], b[0])) if inter > 0 else 0.0


def _inside(box: Box, rect: Box) -> float:
    """Fraction of ``box`` lying inside ``rect``."""
    w = min(box[2], rect[2]) - max(box[0], rect[0])
    h = min(box[3], rect[3]) - max(box[1], rect[1])
    area = (box[2] - box[0]) * (box[3] - box[1])
    return max(w, 0.0) * max(h, 0.0) / area if area > 0 else 0.0


def _extent(mask2d: np.ndarray, dim: int) -> Optional[Interval]:
    """[lo, hi) of the mask's rows (dim 0) or columns (dim 1)."""
    nz = np.flatnonzero(mask2d.any(axis=1 - dim))
    return (int(nz[0]), int(nz[-1]) + 1) if nz.size else None


def _box_of(mask2d: np.ndarray) -> Optional[Box]:
    rows, cols = _extent(mask2d, 0), _extent(mask2d, 1)
    return None if rows is None else (cols[0], rows[0], cols[1], rows[1])


def _clip_box(box: Box, shape) -> Tuple[int, int, int, int]:
    h, w = shape
    return (int(np.clip(math.floor(box[0]), 0, w)), int(np.clip(math.floor(box[1]), 0, h)),
            int(np.clip(math.ceil(box[2]), 0, w)), int(np.clip(math.ceil(box[3]), 0, h)))


def _box_mask(shape, box: Box) -> np.ndarray:
    m = np.zeros(shape, bool)
    x0, y0, x1, y1 = _clip_box(box, shape)
    m[y0:y1, x0:x1] = True
    return m


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    return labels == 1 + int(np.argmax(sizes))


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #

@dataclass
class SeedView:
    """The tumour on one slice of one view (steps 1-2)."""

    plane: Plane
    index: int
    box: Box               # prompt given to SAM, array coordinates of that slice
    how: str               # how the box was chosen, for the status line
    mask: np.ndarray       # 2D SAM mask (or the box itself if SAM's mask was rejected)
    estimated: bool = False   # no detector box lined up: the box was built from the seed's size


@dataclass
class DetectionResult:
    mask: np.ndarray                              # (Z, Y, X) bool, on the Volume's grid
    cuboid: Tuple[Interval, Interval, Interval]   # [lo, hi) along z, y, x
    shape: Tuple[int, int, int]                   # grid the cuboid indices refer to
    acquisition_plane: Plane
    seeds: List[SeedView]
    slice_sources: Dict[int, str]                 # acquired slice -> "detector", "neighbour" or "majority"
    volume_ml: float
    device: str
    seconds: float

    @property
    def centre(self) -> Tuple[int, int, int]:
        """(z, y, x) centre of mass of the mask (cuboid centre if empty)."""
        if self.mask.any():
            c = ndimage.center_of_mass(self.mask)
        else:
            c = [(lo + hi - 1) / 2 for lo, hi in self.cuboid]
        return tuple(int(round(v)) for v in c)

    def to_segmentation(self, reference_image: sitk.Image) -> Segmentation:
        """The mask as a Segmentation on ``reference_image``'s grid (the grid
        detect() ran on), for the viewer's annotation overlay."""
        label = sitk.GetImageFromArray(self.mask.astype(np.uint8))
        label.CopyInformation(reference_image)
        return Segmentation(label, {1: SegmentInfo(1, DETECT_NAME, DETECT_COLOR)})

    def display_box(self, plane: Plane, index: int, shape) -> Optional[Box]:
        """Cuboid outline on slice ``index`` of ``plane`` in the display
        coordinates of extract_slice (rows flipped) for a volume of ``shape``,
        or None if that slice misses the cuboid. ``shape`` may differ from the
        grid detect() ran on (display algorithm changed since); the cuboid is
        rescaled by field-of-view fraction, like the viewer's crosshair."""
        a, r, c = PLANE_AXES[plane]
        scaled = [(lo * n / m, hi * n / m) for (lo, hi), n, m in zip(self.cuboid, shape, self.shape)]
        if not scaled[a][0] <= index + 0.5 < scaled[a][1]:
            return None
        (r0, r1), (c0, c1) = scaled[r], scaled[c]
        return (c0 - 0.5, shape[r] - r1 - 0.5, c1 - 0.5, shape[r] - r0 - 0.5)

    def summary_text(self) -> str:
        seed = self.seeds[0]
        seed_text = f"seed: {seed.plane.value} slice {seed.index + 1}, {seed.how}"
        if not self.slice_sources:
            return f"Detection found no tumour slices inside the cuboid ({seed_text})."
        zs = sorted(self.slice_sources)
        n_det = sum(v == "detector" for v in self.slice_sources.values())
        return (f"Detected tumour: {self.volume_ml:.1f} mL on {len(zs)} {self.acquisition_plane.value} slices "
                f"{zs[0] + 1}-{zs[-1] + 1} ({n_det} detector boxes, {len(zs) - n_det} from the neighbouring "
                f"slice); {seed_text}; {self.seconds:.0f} s on {self.device}")

    def details_text(self) -> str:
        """One line per view used to build the cuboid, plus the cuboid itself."""
        lines = [f"{s.plane.value} slice {s.index + 1}: {s.how}" for s in self.seeds]
        (z0, z1), (y0, y1), (x0, x1) = self.cuboid
        lines.append(f"cuboid (1-based, inclusive): z {z0 + 1}-{z1}, y {y0 + 1}-{y1}, x {x0 + 1}-{x1}")
        return "\n".join(lines)


@dataclass
class RangeResult:
    """Steps 1-3 with detector boxes only (TumourDetector.find_range)."""

    slices: Interval                              # [lo, hi) along the acquisition axis
    cuboid: Tuple[Interval, Interval, Interval]   # [lo, hi) along z, y, x
    acquisition_plane: Plane
    seeds: List[SeedView]                         # each mask is its box
    estimated: bool                               # no detector box on the other views: from the seed's size
    device: str
    seconds: float

    def extent(self, seed: SeedView) -> Optional[Interval]:
        """[lo, hi) of one view's box along the acquisition axis (None for a
        view that does not show that axis: a slice of the acquired plane)."""
        _, r, c = PLANE_AXES[seed.plane]
        k = PLANE_AXES[self.acquisition_plane][0]
        return _extent(seed.mask, 0) if r == k else _extent(seed.mask, 1) if c == k else None


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #

@dataclass
class _Slice:
    gray: np.ndarray      # uint8 (rows, cols) at array resolution, volume-wide window
    model: np.ndarray     # uint8 RGB resized to square pixels — what SAM sees
    det: np.ndarray       # uint8 RGB, square pixels, slice min-max, maybe transposed — what the detector sees
    transposed: bool      # det is model's grid transposed (x axis down the rows)
    fx: float             # model pixels per array column
    fy: float             # model pixels per array row
    px_mm2: float         # area of one array pixel


class TumourDetector:
    """Runs detect() and find_range() with the app's shared models (lesion detector + SAM 2.1).

    tune_threads: on a CPU, let torch use up to CPU_THREADS threads (a
        process-wide setting, for detect()'s many SAM calls); find_range()'s
        few detector calls do not need it.
    """

    def __init__(self, models: ModelManager, tune_threads: bool = True):
        self.models = models
        self.device = models.device_name()
        if tune_threads and models.runs_on_cpu():
            try:
                import torch
                torch.set_num_threads(CPU_THREADS)
            except Exception:
                pass
        self._lo, self._hi = 0.0, 1.0

    # -- model calls on one slice ---------------------------------------------

    def _slice(self, volume: Volume, plane: Plane, index: int) -> _Slice:
        a, r, c = PLANE_AXES[plane]
        img = np.take(volume.array, index, axis=a).astype(np.float32)
        gray = (np.clip((img - self._lo) / (self._hi - self._lo), 0, 1) * 255).astype(np.uint8)
        sp = volume.spacing_zyx
        px = min(sp[r], sp[c])
        h, w = gray.shape
        size = (max(1, round(w * sp[c] / px)), max(1, round(h * sp[r] / px)))
        model = gray if size == (w, h) else cv2.resize(gray, size, interpolation=cv2.INTER_LINEAR)
        # the detector's training slices: min-max of the slice itself, the x (left-right) axis down
        # the rows — views with x as the column axis (axial, coronal) are transposed
        det = ((img - img.min()) / (np.ptp(img) + 1e-8) * 255).astype(np.uint8)
        det = det if size == (w, h) else cv2.resize(det, size, interpolation=cv2.INTER_LINEAR)
        transposed = c == 2
        if transposed:
            det = np.ascontiguousarray(det.T)
        return _Slice(gray, np.repeat(model[..., None], 3, axis=-1), np.repeat(det[..., None], 3, axis=-1),
                      transposed, size[0] / w, size[1] / h, sp[r] * sp[c])

    def _boxes(self, s: _Slice) -> Tuple[np.ndarray, np.ndarray]:
        """Detector boxes, (N, 4) in array coordinates, and their confidences, size-filtered."""
        res = self.models.predict("yolo", s.det, conf=CONF, iou=0.7, max_det=300)[0]
        if res.boxes is None or not len(res.boxes):
            return np.zeros((0, 4)), np.zeros(0)
        b = res.boxes.xyxy.cpu().numpy()
        if s.transposed:
            b = b[:, [1, 0, 3, 2]]
        b = b / [s.fx, s.fy, s.fx, s.fy]
        conf = res.boxes.conf.cpu().numpy()
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        keep = (area <= AREA_MAX * s.gray.size) & (area * s.px_mm2 >= MIN_AREA_MM2)
        return b[keep], conf[keep]

    def _segment(self, s: _Slice, box: Optional[Box] = None, point=None) -> np.ndarray:
        """SAM mask for a box (or one positive point, (col, row)), back at
        array resolution, largest piece only."""
        if box is not None:
            x0, y0, x1, y1 = _clip_box(box, s.gray.shape)
            prompt = dict(bboxes=[[x0 * s.fx, y0 * s.fy, x1 * s.fx, y1 * s.fy]])
        else:
            prompt = dict(points=[[point[0] * s.fx, point[1] * s.fy]], labels=[1])
        res = self.models.predict("sam", s.model, **prompt)[0]
        h, w = s.gray.shape
        if res.masks is None or not len(res.masks.data):
            return np.zeros((h, w), bool)
        m = res.masks.data[0].cpu().numpy().astype(np.uint8)
        return _largest_component(cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool))

    # -- pipeline -----------------------------------------------------------------

    def detect(self, volume: Volume, plane: Plane, index: int, acquisition_plane: str,
               pick: Optional[Tuple[int, int]] = None, progress: Optional[Progress] = None) -> DetectionResult:
        """Steps 1-4 (module docstring) from slice ``index`` of ``plane``.

        acquisition_plane: get_acquisition_plane() of the series.
        pick: (row, col) of the crosshair on this slice, in array coordinates,
            when the user clicked on the tumour in this view; None to use the
            most confident box.
        progress: reports before every slice; raises OperationCancelled once
            Cancel was pressed.
        """
        t0 = time.time()
        step = ensure_progress(progress)
        self._set_window(volume)
        k = acquisition_axis(volume, acquisition_plane)

        step(0, 3, f"Detector + SAM on {plane.value} slice {index + 1} (seed)…")
        seeds = [self._seed_view(volume, plane, index, pick)]
        for n, other in enumerate(p for p in PLANE_AXES if p is not plane):
            step(1 + n, 3, f"Detector + SAM on the corresponding {other.value} slice…")
            seeds.append(self._cross_view(volume, seeds[0], other))
        cuboid = self._cuboid(seeds, volume.spacing_zyx, volume.shape, k)
        mask, sources = self._range(volume, seeds, cuboid, k, step)

        return DetectionResult(
            mask=mask, cuboid=cuboid, shape=tuple(volume.shape), acquisition_plane=AXIS_PLANE[k],
            seeds=seeds, slice_sources=sources,
            volume_ml=float(mask.sum() * np.prod(volume.spacing_zyx) / 1000.0),
            device=self.device, seconds=time.time() - t0)

    def find_range(self, volume: Volume, plane: Plane, index: int, acquisition_plane: str,
                   hint: Optional[Box] = None, progress: Optional[Progress] = None) -> RangeResult:
        """Steps 1-3 with detector boxes only (no SAM): the tumour's slice
        range along the acquisition axis, from slice ``index`` of ``plane``.

        hint: a box the user drew around the tumour on that slice (array
            coordinates): the most confident detector box centred inside it
            is the seed, or the hint itself where there is none.
        """
        t0 = time.time()
        step = ensure_progress(progress)
        self._set_window(volume)
        k = acquisition_axis(volume, acquisition_plane)

        step(0, 3, f"Detector on {plane.value} slice {index + 1}…")
        seeds = [self._seed_box(volume, plane, index, hint)]
        for n, other in enumerate(p for p in PLANE_AXES if p is not plane):
            step(1 + n, 3, f"Detector on the {other.value} slice through the tumour…")
            seeds.append(self._cross_view(volume, seeds[0], other, refine=False))
        # a box the detector found beats one built from the seed's size
        detected = [sv for sv in seeds[1:] if not sv.estimated]
        cuboid = self._cuboid(seeds[:1] + (detected or seeds[1:]), volume.spacing_zyx, volume.shape, k)
        return RangeResult(cuboid[k], cuboid, AXIS_PLANE[k], seeds, not detected, self.device, time.time() - t0)

    def _set_window(self, volume: Volume) -> None:
        """One window for the whole volume (what SAM sees, _Slice.model)."""
        self._lo, self._hi = (float(v) for v in np.percentile(volume.array[::4, ::4, ::4], [0.5, 99.5]))
        if self._hi <= self._lo:
            self._hi = self._lo + 1.0

    @staticmethod
    def _no_box(plane: Plane, index: int) -> NoTumourFound:
        return NoTumourFound(f"The detector proposed no box between {MIN_AREA_MM2:.0f} mm² and "
                             f"{AREA_MAX:.0%} of the slice on {plane.value} slice {index + 1}.")

    def _seed_box(self, volume, plane, index, hint: Optional[Box]) -> SeedView:
        """Step 1 for find_range: the detector's box is the tumour (no SAM)."""
        s = self._slice(volume, plane, index)
        boxes, conf = self._boxes(s)
        how = "most confident box"
        if hint is not None:
            cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
            inside = (hint[0] <= cx) & (cx <= hint[2]) & (hint[1] <= cy) & (cy <= hint[3])
            if not inside.any():
                hint = tuple(float(v) for v in hint)
                return SeedView(plane, index, hint, "the drawn box (no detector box inside it)",
                                _box_mask(s.gray.shape, hint))
            boxes, conf, how = boxes[inside], conf[inside], "most confident box inside the drawn box"
        if not len(boxes):
            raise self._no_box(plane, index)
        best = int(np.argmax(conf))
        box = tuple(float(v) for v in boxes[best])
        return SeedView(plane, index, box, how + f" ({conf[best]:.2f})", _box_mask(s.gray.shape, box))

    def _seed_view(self, volume, plane, index, pick) -> SeedView:
        """Step 1: the tumour on the slice the user pressed Detect on."""
        s = self._slice(volume, plane, index)
        boxes, conf = self._boxes(s)
        how = "most confident box"
        if pick is not None:
            row, col = pick[0] + 0.5, pick[1] + 0.5
            under = (boxes[:, 0] <= col) & (col <= boxes[:, 2]) & (boxes[:, 1] <= row) & (row <= boxes[:, 3])
            if under.any():
                boxes, conf, how = boxes[under], conf[under], "most confident box under the crosshair"
            else:
                m = self._segment(s, point=(col, row))
                if m.sum() * s.px_mm2 >= MIN_AREA_MM2:
                    return SeedView(plane, index, _box_of(m), "SAM at the crosshair (no box there)", m)
                how = "most confident box (nothing found at the crosshair)"
        if not len(boxes):
            raise self._no_box(plane, index)
        best = int(np.argmax(conf))
        box, how = tuple(boxes[best]), how + f" ({conf[best]:.2f})"
        mask = self._segment(s, box)
        if mask.sum() * s.px_mm2 < MIN_AREA_MM2:
            mask, how = _box_mask(mask.shape, box), how + "; SAM mask empty, box used"
        return SeedView(plane, index, box, how, mask)

    def _cross_view(self, volume, seed: SeedView, plane: Plane, refine: bool = True) -> SeedView:
        """Step 2: the tumour on the slice of ``plane`` through the seed
        (``refine`` False: the chosen box is the result, no SAM)."""
        a0, r0, c0 = PLANE_AXES[seed.plane]
        a, r, c = PLANE_AXES[plane]
        # cut through the seed mask's centre of mass (its widest chord if that line misses a
        # C-shaped mask); `shared` is the seed axis this view also shows
        profile = seed.mask.sum(axis=1) if a == r0 else seed.mask.sum(axis=0)
        index = int(round(np.average(np.arange(profile.size), weights=profile)))
        if not profile[index]:
            index = int(np.argmax(profile))
        shared = c0 if a == r0 else r0
        known = _extent(seed.mask, 1 if shared == c0 else 0)
        dim = {r: 0, c: 1}                         # this view's array axis -> 2D dim

        def along(box: Box, axis: int):
            return (box[1], box[3]) if dim[axis] == 0 else (box[0], box[2])

        s = self._slice(volume, plane, index)
        best, best_score, best_conf = None, 0.0, 0.0
        for b, cf in zip(*self._boxes(s)):
            lo, hi = along(b, a0)
            line = _iou_1d(along(b, shared), known)
            if lo <= seed.index + 0.5 <= hi and line >= LINE_IOU_MIN:
                score = line * cf
                if score > best_score:
                    best, best_score, best_conf = tuple(b), score, cf
        how, estimated = f"box lined up with the seed ({best_conf:.2f})", False
        if best is None:
            # as wide as the seed along the shared axis, as deep as the seed's mean size along a0
            sp = volume.spacing_zyx
            size_mm = np.mean([(e[1] - e[0]) * sp[ax] for ax, e in
                               ((r0, _extent(seed.mask, 0)), (c0, _extent(seed.mask, 1)))])
            half = 0.5 * size_mm / sp[a0]
            span = {shared: known, a0: (seed.index + 0.5 - half, seed.index + 0.5 + half)}
            best = (span[c][0], span[r][0], span[c][1], span[r][1])
            how, estimated = "no box lined up, box from the seed's size", True
        if not refine:
            best = tuple(float(v) for v in best)
            return SeedView(plane, index, best, how, _box_mask(s.gray.shape, best), estimated)
        mask = self._segment(s, best)
        on_seed = _extent(mask, dim[a0])
        if (mask.sum() * s.px_mm2 < MIN_AREA_MM2 or _iou_1d(_extent(mask, dim[shared]), known) < LINE_IOU_MIN
                or not on_seed[0] <= seed.index < on_seed[1]):
            mask, how = _box_mask(mask.shape, best), how + "; SAM mask did not match, box used"
        return SeedView(plane, index, best, how, mask, estimated)

    @staticmethod
    def _cuboid(seeds: List[SeedView], spacing, shape, k: int) -> Tuple[Interval, Interval, Interval]:
        """Step 3: union of the two mask extents seen along each axis, padded
        in-plane. Along the acquisition axis ``k`` it is exactly the slice
        range the other views see (padding there only added slices next to
        the tumour, where SAM latched onto neighbouring structures)."""
        ext = {0: [], 1: [], 2: []}
        for sv in seeds:
            _, r, c = PLANE_AXES[sv.plane]
            ext[r].append(_extent(sv.mask, 0))
            ext[c].append(_extent(sv.mask, 1))
        out = []
        for ax in range(3):
            pad = 0 if ax == k else int(math.ceil(MARGIN_MM / spacing[ax]))
            lo = min(e[0] for e in ext[ax] if e) - pad
            hi = max(e[1] for e in ext[ax] if e) + pad
            out.append((max(lo, 0), min(hi, shape[ax])))
        return tuple(out)

    def _range(self, volume, seeds: List[SeedView], cuboid, k: int, step: Progress):
        """Step 4: detector + SAM on every acquired slice in the cuboid's
        range, outwards from the acquired-plane seed."""
        plane = AXIS_PLANE[k]
        _, r, c = PLANE_AXES[plane]
        rect = (cuboid[c][0], cuboid[r][0], cuboid[c][1], cuboid[r][1])
        sp = volume.spacing_zyx
        pad_c, pad_r = (math.ceil(MARGIN_MM / sp[ax]) for ax in (c, r))

        # the two seed views that are not slices of the acquired plane both
        # contain axis k; each gives the tumour's chord along one in-plane axis
        chords = {}
        for sv in seeds:
            a, rr, cc = PLANE_AXES[sv.plane]
            if a != k:
                chords[cc if rr == k else rr] = (sv.mask, rr == k)

        def chord(axis: int, i: int) -> Optional[Interval]:
            m, k_is_row = chords[axis]
            nz = np.flatnonzero(m[i, :] if k_is_row else m[:, i])
            return (int(nz[0]), int(nz[-1]) + 1) if nz.size else None

        def allowed(i: int) -> Box:
            """Where SAM's mask may lie on slice i: the expected cross-section
            plus CLIP_FRAC of its size (at least MARGIN_MM), inside the cuboid;
            the whole cuboid where neither seed mask reaches slice i."""
            er, ec = chord(r, i), chord(c, i)
            if er is None and ec is None:
                return rect
            er, ec = er or cuboid[r], ec or cuboid[c]
            pc = max(pad_c, CLIP_FRAC * (ec[1] - ec[0]))
            pr = max(pad_r, CLIP_FRAC * (er[1] - er[0]))
            return (max(ec[0] - pc, rect[0]), max(er[0] - pr, rect[1]),
                    min(ec[1] + pc, rect[2]), min(er[1] + pr, rect[3]))

        def segment(i: int, prev: np.ndarray):
            """Mask on slice i following the tumour mask ``prev`` of its neighbour."""
            s = self._slice(volume, plane, i)
            prev = prev & _box_mask(s.gray.shape, rect)
            here = _box_of(prev)
            if here is None:
                return None, None
            under = [tuple(b) for b in self._boxes(s)[0] if _inside(b, rect) >= INSIDE_MIN]
            best = max(under, key=lambda b: _iou(b, here), default=None)
            prompts = [(best, "detector")] if best is not None and _iou(best, here) >= MATCH_IOU_MIN else []
            prompts.append((here, "neighbour"))
            limit = _box_mask(s.gray.shape, allowed(i))
            for box, src in prompts:
                m = ndimage.binary_fill_holes(self._segment(s, box)) & limit
                if (m.sum() * s.px_mm2 >= MIN_AREA_MM2
                        and (m & prev).sum() >= FOLLOW_MIN * min(m.sum(), prev.sum())):
                    return m, src
            return None, None

        lo, hi = cuboid[k]
        start = next(sv for sv in seeds if sv.plane is plane)
        i0 = min(max(start.index, lo), hi - 1)
        out = np.zeros(volume.shape, bool)
        sources: Dict[int, str] = {}
        done, first = 0, start.mask
        for direction in (1, -1):
            prev = first
            i = i0 if direction > 0 else i0 - 1
            while lo <= i < hi:
                step(3 + done, 3 + (hi - lo), f"Detector + SAM on {plane.value} slice {i + 1} ({done + 1}/{hi - lo})")
                done += 1
                m, src = segment(i, prev)
                if m is not None:
                    idx = [slice(None)] * 3
                    idx[k] = i
                    out[tuple(idx)] = m
                    sources[i] = src
                    prev = m
                    if i == i0:
                        first = m
                i += direction

        if sp[k] <= SMOOTH_MAX_MM:
            size = [1, 1, 1]
            size[k] = 3
            out = ndimage.median_filter(out.astype(np.uint8), size=size, mode="constant") > 0
            present = np.flatnonzero(out.any(axis=tuple(ax for ax in range(3) if ax != k)))
            sources = {int(i): sources.get(int(i), "majority") for i in present}
        return out, sources
