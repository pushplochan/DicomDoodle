"""
DICOM SEG annotations for MPR (from the original mpr_core.py): stacking SEG
frames into a 3D label volume aligned with a series, resampling/transferring
labels between registered series, and saving a transferred annotation back
as DicomDoodle per-slice ``_seg.dcm`` files.

Changes from the original:

* segments are identified by their *name* across files. DicomDoodle writes
  one SEG per slice and numbers that slice's segments 1..k, so "segment 1"
  can mean different labels on different slices; keying by name keeps every
  label distinct, colored consistently, and saved under its real name;
* outlines are filled per slice after every file is read, so an outline
  traced as several open pieces is filled as a whole (from the
  zero_shot_tumor version, see :func:`load_seg_as_labelmap`).
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pydicom
import SimpleITK as sitk

from ..labels import EMPTY_SEGMENT_NAMES
from .loading import find_series, read_slices_with_pydicom, series_files
from .registration import RegistrationResult

RGB = Tuple[int, int, int]
ColorLookup = Optional[Callable[[str], Optional[RGB]]]


@dataclass
class SegmentInfo:
    number: int
    label: str
    color_rgb: RGB


@dataclass
class Segmentation:
    """Label volume on exactly the grid of the image it annotates."""

    label_image: sitk.Image
    segments: Dict[int, SegmentInfo]

    @property
    def array(self) -> np.ndarray:
        return sitk.GetArrayFromImage(self.label_image)


_PALETTE = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
            (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212)]


def _palette_color(segment_number: int) -> RGB:
    return _PALETTE[(segment_number - 1) % len(_PALETTE)]


def _is_seg_file(path: str) -> bool:
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        return str(getattr(ds, "Modality", "")).upper() == "SEG"
    except Exception:
        return False


def find_seg_files(folder: str) -> List[str]:
    """SEG objects in ``folder``, detected by Modality (any file name)."""
    if not os.path.isdir(folder):
        return []
    candidates = sorted(p for p in glob.glob(os.path.join(folder, "*"))
                        if os.path.isfile(p) and p.lower().endswith(".dcm"))
    return [p for p in candidates if _is_seg_file(p)]


def find_series_seg_files(image: sitk.Image, folder: str) -> List[str]:
    """SEG files for an image from load_dicom_series: those beside its own
    files, else those in a subfolder holding single-frame copies of the same
    images (some exports keep an Enhanced multi-frame file next to a folder
    of single-frame copies and their SEGs: same SeriesInstanceUID, or that
    subfolder's only volume, same frame of reference and size), else those
    in ``folder`` (the folder the user picked)."""
    own = image.GetMetaData("dicom_folder") if image.HasMetaDataKey("dicom_folder") else folder
    segs = find_seg_files(own)
    if segs:
        return segs

    def meta(key: str) -> str:
        return image.GetMetaData(key) if image.HasMetaDataKey(key) else ""

    uid, frame = meta("series_key").split("#")[0], meta("frame_of_reference")
    dims = sorted(image.GetSize())
    if uid:
        for sub in sorted(glob.glob(os.path.join(own, "*", ""))):
            volumes = find_series(sub)
            if any(v.key.split("#")[0] == uid for v in volumes) or (
                    frame and len(volumes) == 1 and volumes[0].frame_of_reference == frame
                    and sorted(volumes[0].dims) == dims):
                segs = find_seg_files(sub)
                if segs:
                    return segs
    return find_seg_files(folder) if os.path.abspath(folder) != os.path.abspath(own) else []


def load_seg_as_labelmap(seg_paths: List[str], reference_image: sitk.Image, fill_contours: bool = True,
                         color_lookup: ColorLookup = None) -> Optional[Segmentation]:
    """Stack the frames of one or more SEG files into a label volume on the
    reference image's grid. Each frame is placed using its own position and
    orientation. Later files only fill background voxels.

    ``fill_contours`` fills outline-style annotations so the volume is solid
    in all three planes. The fill is done per slice once every file is read:
    annotators often trace one outline as several open pieces stored as
    separate segments on the same slice (AIIMS: "Tumor & Brain Boundary" +
    "Tumor & Dura Boundary", which only enclose the tumour together), and
    filling each piece alone filled nothing, so the reformatted views showed
    stripes and dots (00021 axial: 17.9 mL loaded instead of 83.9 mL). An
    outline that encloses its region on its own is filled as before; open
    pieces on a slice are filled together with small gaps bridged
    (:func:`_fill_traced_outline`), and the region takes the lowest label
    value among its pieces, so the structure has one label on every slice.
    """
    if not seg_paths:
        return None

    size = reference_image.GetSize()
    label_array = np.zeros((size[2], size[1], size[0]), dtype=np.uint16)
    ids: Dict[str, int] = {}                 # segment name -> label value in this volume
    segments: Dict[int, SegmentInfo] = {}
    traced: Dict[Tuple[int, int], list] = {}   # (axis, slice) -> [(value, row0, col0, cropped mask)]

    def global_id(name: str) -> int:
        key = " ".join(name.split()).casefold()
        if key not in ids:
            ids[key] = len(ids) + 1
            number = ids[key]
            color = (color_lookup(name) if color_lookup else None) or _palette_color(number)
            segments[number] = SegmentInfo(number, name, tuple(color))
        return ids[key]

    def place(mask: np.ndarray, position, iop, spacing, value: int) -> None:
        axis = _plane_axis_from_orientation(iop)
        mask = _reorient_frame(mask, position, iop, spacing, reference_image, axis)
        if not fill_contours:
            _write_frame(label_array, reference_image, position, mask, value, axis)
            return
        index = _frame_slice_index(reference_image, position, axis)
        if index is None or not mask.any():
            return
        mask = _fit(mask, _label_slice(label_array, axis, index).shape)
        if mask is None:
            return
        rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
        traced.setdefault((axis, index), []).append(
            (value, rows[0], cols[0], mask[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1].copy()))

    for path in seg_paths:
        try:
            ds = pydicom.dcmread(path)
            pixels = ds.pixel_array
        except Exception:
            continue
        if pixels.ndim == 2:
            pixels = pixels[np.newaxis, ...]
        local_names = {}
        for item in getattr(ds, "SegmentSequence", []):
            try:
                local_names[int(item.SegmentNumber)] = str(getattr(item, "SegmentLabel", f"Segment {item.SegmentNumber}"))
            except Exception:
                continue

        def value_for(local_number: int) -> int:
            name = local_names.get(local_number, f"Segment {local_number}")
            return 0 if " ".join(name.split()).casefold() in EMPTY_SEGMENT_NAMES else global_id(name)

        seg_type = str(getattr(ds, "SegmentationType", "BINARY")).upper()
        threshold = float(getattr(ds, "MaximumFractionalValue", 255)) * 0.5 if seg_type == "FRACTIONAL" else 0
        per_frame = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
        shared_seq = getattr(ds, "SharedFunctionalGroupsSequence", None)
        shared = shared_seq[0] if shared_seq else None
        dataset_iop = getattr(ds, "ImageOrientationPatient", None)
        dataset_spacing = getattr(ds, "PixelSpacing", None)
        shared_iop = shared_spacing = None
        if shared is not None:
            if getattr(shared, "PlaneOrientationSequence", None):
                shared_iop = shared.PlaneOrientationSequence[0].ImageOrientationPatient
            if getattr(shared, "PixelMeasuresSequence", None):
                shared_spacing = shared.PixelMeasuresSequence[0].PixelSpacing

        if per_frame:
            for i in range(min(len(per_frame), pixels.shape[0])):
                fg = per_frame[i]
                pos_seq = getattr(fg, "PlanePositionSequence", None) or (
                    getattr(shared, "PlanePositionSequence", None) if shared is not None else None)
                if not pos_seq:
                    continue
                position = [float(v) for v in pos_seq[0].ImagePositionPatient]
                local = int(fg.SegmentIdentificationSequence[0].ReferencedSegmentNumber) \
                    if getattr(fg, "SegmentIdentificationSequence", None) else 1
                value = value_for(local)
                if value == 0:
                    continue
                frame_po = getattr(fg, "PlaneOrientationSequence", None)
                iop = frame_po[0].ImageOrientationPatient if frame_po else (
                    shared_iop if shared_iop is not None else dataset_iop)
                frame_pm = getattr(fg, "PixelMeasuresSequence", None)
                spacing = frame_pm[0].PixelSpacing if frame_pm else (
                    shared_spacing if shared_spacing is not None else dataset_spacing)
                place(pixels[i] > threshold, position, iop, spacing, value)
        else:
            position = getattr(ds, "ImagePositionPatient", None)
            value = value_for(1)
            if position is not None and value:
                position = [float(v) for v in position]
                for i in range(pixels.shape[0]):
                    place(pixels[i] > threshold, position, dataset_iop, dataset_spacing, value)

    for (axis, index), frames in traced.items():
        _fill_traced_slice(_label_slice(label_array, axis, index), frames)

    if not np.any(label_array):
        return None
    present = {int(v) for v in np.unique(label_array) if v != 0}
    segments = {n: info for n, info in segments.items() if n in present}
    label_image = sitk.GetImageFromArray(label_array.astype(np.uint16))
    label_image.CopyInformation(reference_image)
    return Segmentation(label_image, segments)


def _reorient_frame(mask: np.ndarray, position, iop, pixel_spacing, reference_image: sitk.Image,
                    target_axis: int) -> np.ndarray:
    """Transpose/flip a SEG frame to match the volume's cross-section,
    derived from the frame's own orientation (works for any plane)."""
    if iop is None or len(iop) < 6 or pixel_spacing is None:
        return mask
    row_dir = np.array([float(v) for v in iop[0:3]])
    col_dir = np.array([float(v) for v in iop[3:6]])
    row_spacing, col_spacing = float(pixel_spacing[0]), float(pixel_spacing[1])
    p0 = np.array(position, dtype=float)
    idx0 = np.array(reference_image.TransformPhysicalPointToContinuousIndex(tuple(p0)))
    idx_col1 = np.array(reference_image.TransformPhysicalPointToContinuousIndex(tuple(p0 + col_spacing * row_dir)))
    idx_row1 = np.array(reference_image.TransformPhysicalPointToContinuousIndex(tuple(p0 + row_spacing * col_dir)))
    d_col, d_row = idx_col1 - idx0, idx_row1 - idx0
    first_xyz, second_xyz = {0: (1, 0), 1: (2, 0), 2: (2, 1)}[target_axis]
    natural = abs(d_row[first_xyz]) + abs(d_col[second_xyz])
    swapped = abs(d_row[second_xyz]) + abs(d_col[first_xyz])
    if swapped > natural:
        out, row_component, col_component = mask.T, d_row[second_xyz], d_col[first_xyz]
    else:
        out, row_component, col_component = mask, d_row[first_xyz], d_col[second_xyz]
    if row_component < 0:
        out = np.flip(out, axis=0)
    if col_component < 0:
        out = np.flip(out, axis=1)
    return np.ascontiguousarray(out)


def _plane_axis_from_orientation(iop) -> int:
    """Label-array axis a frame belongs on: 0 axial, 1 coronal, 2 sagittal."""
    if iop is None or len(iop) < 6:
        return 0
    normal = np.cross(np.array([float(v) for v in iop[0:3]]), np.array([float(v) for v in iop[3:6]]))
    return {0: 2, 1: 1, 2: 0}[int(np.argsort(-np.abs(normal))[0])]


def _frame_slice_index(reference_image: sitk.Image, position: List[float], target_axis: int) -> Optional[int]:
    """Index along ``target_axis`` of the slice a frame at ``position`` belongs on (None if off the grid)."""
    try:
        xi, yi, zi = reference_image.TransformPhysicalPointToIndex(position)
    except Exception:
        return None
    size = reference_image.GetSize()
    index, extent = {0: (zi, size[2]), 1: (yi, size[1]), 2: (xi, size[0])}[target_axis]
    return index if 0 <= index < extent else None


def _label_slice(label_array: np.ndarray, target_axis: int, index: int) -> np.ndarray:
    """Writable 2D view of ``label_array`` at ``index`` along ``target_axis``."""
    if target_axis == 0:
        return label_array[index, :, :]
    if target_axis == 1:
        return label_array[:, index, :]
    return label_array[:, :, index]


def _fit(mask: np.ndarray, shape) -> Optional[np.ndarray]:
    """``mask`` resized (nearest neighbour) to ``shape`` when a frame's size
    differs from its slice's; None if that isn't possible."""
    if mask.shape == tuple(shape):
        return mask
    try:
        from skimage.transform import resize as sk_resize
    except ImportError:
        return None
    return sk_resize(mask.astype(np.float32), shape, order=0, preserve_range=True,
                     anti_aliasing=False).astype(bool)


def _write_frame(label_array: np.ndarray, reference_image: sitk.Image, position: List[float],
                 mask: np.ndarray, value: int, target_axis: int = 0) -> None:
    """Write a frame at its physical position, only onto background voxels."""
    index = _frame_slice_index(reference_image, position, target_axis)
    if index is None:
        return
    target = _label_slice(label_array, target_axis, index)
    mask = _fit(mask, target.shape)
    if mask is not None:
        target[mask & (target == 0)] = value


# Largest gap (pixels, each side) _fill_traced_outline bridges in a traced
# outline; AIIMS outlines needed up to 12 (a ~24 px opening).
_MAX_OUTLINE_BRIDGE = 16


def _encloses(filled: np.ndarray, outline: np.ndarray) -> bool:
    """True if ``filled`` is a real interior for ``outline``, not just the
    line or a small pocket of it: at least twice the outline's pixel count
    and isoperimetrically plausible (4*pi*area / perimeter^2 >= 0.1, about
    0.5-0.9 for traced tumour outlines and 0.04 for a pocket)."""
    area, perimeter = int(filled.sum()), int(outline.sum())
    return area >= max(2 * perimeter, perimeter * perimeter / (40 * np.pi))


def _is_traced_line(mask: np.ndarray) -> bool:
    """True for a thin traced line (most pixels on its edge), False for a solid region."""
    from scipy.ndimage import binary_erosion
    return binary_erosion(mask, np.ones((3, 3), bool)).sum() < 0.5 * mask.sum()


def _fill_traced_outline(outline: np.ndarray) -> np.ndarray:
    """The region a traced outline encloses, bridging gaps left in it. A
    plain hole fill first; if that encloses nothing real, the outline is
    grown by d pixels, filled, and shrunk back by d (d = 1 .. 16, smallest
    that works) — filling before shrinking closes gaps a morphological
    closing reopens. Returns the plain fill if nothing works (e.g. a short
    scribble on a tumour's end slice)."""
    from scipy.ndimage import binary_fill_holes, distance_transform_cdt
    filled = binary_fill_holes(outline)
    if _encloses(filled, outline):
        return filled
    margin = _MAX_OUTLINE_BRIDGE + 2
    rows = np.flatnonzero(outline.any(axis=1))
    cols = np.flatnonzero(outline.any(axis=0))
    r0, c0 = max(rows[0] - margin, 0), max(cols[0] - margin, 0)
    r1, c1 = min(rows[-1] + margin + 1, outline.shape[0]), min(cols[-1] + margin + 1, outline.shape[1])
    # padded with background: near the image edge the grown outline must shrink back
    # from that side too (the distance transform does not count the array border)
    crop = np.pad(outline[r0:r1, c0:c1], margin)
    dist_out = distance_transform_cdt(~crop, metric="chessboard")   # = repeated 3x3 dilation, in one pass
    for d in range(1, _MAX_OUTLINE_BRIDGE + 1):
        grown = binary_fill_holes(dist_out <= d)
        bridged = (distance_transform_cdt(grown, metric="chessboard") > d) | crop
        if _encloses(bridged, crop):
            out = filled.copy()
            out[r0:r1, c0:c1] |= bridged[margin:-margin, margin:-margin]
            return out
    return filled


def _fill_traced_slice(target: np.ndarray, frames: list) -> None:
    """Write one slice's frames into ``target`` (a 2D view of the label
    array), filled. A segment whose outline encloses its region on its own
    (or is already solid) is filled and written in the order read. Segments
    that are only open pieces are filled together with every outline on the
    slice; each region takes the lowest label value traced around it (a
    fixed rule keeps the structure's label the same on every slice, so
    reformats show no false edges). Only background pixels are written."""
    from scipy.ndimage import binary_fill_holes, label as cc_label
    outline = np.zeros(target.shape, dtype=np.uint16)
    for value, r0, c0, crop in frames:
        region = outline[r0:r0 + crop.shape[0], c0:c0 + crop.shape[1]]
        region[crop & (region == 0)] = value

    pooled = False
    for value in dict.fromkeys(f[0] for f in frames):      # first-read order
        mask = outline == value
        if not mask.any():
            continue
        filled = binary_fill_holes(mask)
        if _encloses(filled, mask) or not _is_traced_line(mask):
            target[filled & (target == 0)] = value
        else:
            pooled = True
    if not pooled:
        return
    filled = _fill_traced_outline(outline > 0)
    regions, n = cc_label(filled, structure=np.ones((3, 3), bool))
    for region_id in range(1, n + 1):
        region = regions == region_id
        traced_by = np.unique(outline[region])
        traced_by = traced_by[traced_by > 0]
        if traced_by.size:
            target[region & (target == 0)] = int(traced_by[0])


def resample_label_like(label_image: sitk.Image, reference_image: sitk.Image,
                        transform: Optional[sitk.Transform] = None) -> sitk.Image:
    """Nearest-neighbour only: label values must never be blended."""
    return sitk.Resample(label_image, reference_image, transform or sitk.Transform(), sitk.sitkNearestNeighbor,
                         0, label_image.GetPixelID())


def resample_label_array_like(label_image: sitk.Image, reference_image: sitk.Image,
                              transform: Optional[sitk.Transform] = None) -> np.ndarray:
    return sitk.GetArrayFromImage(resample_label_like(label_image, reference_image, transform))


def transfer_labelmap(source: Segmentation, target_reference_image: sitk.Image,
                      registration_result: RegistrationResult, source_was_fixed: bool) -> Segmentation:
    """Carry an annotation from one registered series onto the other's native grid.

    The registration transform maps fixed (primary) points to moving
    (secondary) points; going primary -> secondary needs its inverse.
    Displacement fields are inverted numerically on the primary grid.
    """
    transform = registration_result.transform
    if source_was_fixed:
        try:
            effective = transform.GetInverse()
        except Exception:
            src = source.label_image
            field = sitk.TransformToDisplacementField(transform, sitk.sitkVectorFloat64, src.GetSize(),
                                                      src.GetOrigin(), src.GetSpacing(), src.GetDirection())
            effective = sitk.DisplacementFieldTransform(sitk.InvertDisplacementField(field))
    else:
        effective = transform
    warped = resample_label_like(source.label_image, target_reference_image, effective)
    return Segmentation(warped, dict(source.segments))


def save_transferred_seg_slice_wise(segmentation: Segmentation, source_folder: str, output_folder: str,
                                    series_id: Optional[str] = None) -> int:
    """Write one ``<slice>_seg.dcm`` per annotated slice of the source series
    (DicomDoodle's native format, with the real label names). Returns the count."""
    from ..seg_io import TRANSFERRED, write_segmentation

    slice_paths = series_files(source_folder, series_id)     # the files load_dicom_series read, in order
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(slice_paths)
    try:
        original_grid = reader.Execute()
    except RuntimeError:
        original_grid = read_slices_with_pydicom(slice_paths)[0]
    labels = resample_label_array_like(segmentation.label_image, original_grid).astype(np.uint8)
    names = {number: info.label for number, info in segmentation.segments.items()}

    saved = 0
    for z, slice_path in enumerate(slice_paths):
        mask = labels[z]
        if not np.any(mask):
            continue
        base = os.path.splitext(os.path.basename(slice_path))[0]
        write_segmentation(mask, slice_path, os.path.join(output_folder, f"{base}_seg.dcm"), z, labels=names,
                           method=TRANSFERRED)
        saved += 1
    return saved
