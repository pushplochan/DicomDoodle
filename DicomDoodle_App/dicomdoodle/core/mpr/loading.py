"""
Finding and loading DICOM image volumes for MPR (from the original
mpr_core.py; the loader of the zero_shot_tumor version, validated on 4,624
folders from seven hospitals).

:func:`find_series` decides which files make one volume from a header scan
of every file in the folder (any file name; SEG / SR / PR ... skipped):

* a multi-frame (Enhanced MR) file is one volume;
* one series is split into separate volumes when it mixes orientations,
  image sizes or pixel spacings (localisers, several stacks), when slice
  positions repeat (echoes, time points, diffusion volumes, magnitude +
  phase) and at gaps of more than 5x the usual slice spacing (blocks);
* the same image exported twice (same SOPInstanceUID) is read once;
* missing slices on an even grid become blank slices, so every slice keeps
  its true position.

:func:`load_dicom_series` reads exactly those files (GDCM, or pydicom when
GDCM refuses them) and reorients the volume to LPS, so array axis 0 is
always axial, 1 coronal and 2 sagittal whatever plane the scan was acquired
in (an exact permute/flip for on-axis scans — no interpolation).
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pydicom
import SimpleITK as sitk


@dataclass
class SeriesInfo:
    """One loadable 3D volume found in a folder (see :func:`find_series`)."""

    key: str                  # for load_dicom_series; the SeriesInstanceUID itself when the series is not split
    label: str                # readable, for a series chooser
    files: List[str]          # sorted along the slice normal; one file for a multi-frame object
    multi_frame: bool = False
    notes: List[str] = field(default_factory=list)   # e.g. "echo 2", "3 missing slice(s) shown blank ..."
    slots: Optional[List[int]] = None   # grid index of each file when slices are missing
    slot_spacing: Optional[float] = None
    frame_of_reference: str = ""
    dims: Tuple[int, int, int] = (0, 0, 0)   # columns, rows, slices (blank ones included)


# Objects that may sit next to an image series but are never the image series itself.
_NON_IMAGE_MODALITIES = {"SEG", "RTSTRUCT", "RTDOSE", "RTPLAN", "PR", "KO", "SR", "DOC"}

# Header fields that tell apart volumes sharing the same slice positions, tried in this order.
_SPLIT_TAGS = (("echo", "EchoNumbers"), ("time point", "TemporalPositionIdentifier"),
               ("trigger time", "TriggerTime"), ("image type", "ImageType"),
               ("b-value", (0x0018, 0x9087)), ("acquisition", "AcquisitionNumber"))
_ORIENTATION_TOL = 1e-2       # direction cosines closer than this = same orientation (~0.6 degrees)
_POSITION_DECIMALS = 2        # slice positions equal to 0.01 mm = same position


def _looks_like_dicom(path: str) -> bool:
    """A *.dcm file, or a file of any name with the 'DICM' marker at byte 128
    (some exports have no file extension at all)."""
    if path.lower().endswith(".dcm"):
        return True
    if os.path.basename(path).upper() == "DICOMDIR":
        return False
    try:
        with open(path, "rb") as fh:
            fh.seek(128)
            return fh.read(4) == b"DICM"
    except OSError:
        return False


def _tag_text(ds, tag) -> str:
    value = ds.get(tag)
    if value is None:
        return ""
    from pydicom.multival import MultiValue
    value = getattr(value, "value", value)
    if isinstance(value, (list, tuple, MultiValue)):
        return "\\".join(str(v) for v in value)
    return str(value)


def _floats(value, n: int) -> Optional[Tuple[float, ...]]:
    try:
        out = tuple(float(v) for v in value)
        return out if len(out) == n else None
    except Exception:
        return None


def _scan_folder_headers(folder: str) -> List[dict]:
    """One header-only pass over every DICOM file in ``folder``: one record
    per image file (SEG and other non-image objects skipped).

    (Asking GDCM per series is very slow for folders that hold one SEG file
    per slice, because every SEG file is its own "series".)
    """
    records = []
    for path in sorted(p for p in glob.glob(os.path.join(folder, "*")) if os.path.isfile(p)):
        if not _looks_like_dicom(path):
            continue
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        except Exception:
            continue
        uid = str(ds.get("SeriesInstanceUID", "") or "")
        modality = str(ds.get("Modality", "") or "").upper()
        if not uid or modality in _NON_IMAGE_MODALITIES or "Rows" not in ds or "Columns" not in ds:
            continue
        try:
            frames = int(ds.get("NumberOfFrames", 1) or 1)
        except (TypeError, ValueError):
            frames = 1
        try:
            instance = int(ds.get("InstanceNumber", 0) or 0)
        except (TypeError, ValueError):
            instance = 0
        records.append({
            "path": path, "uid": uid, "sop": str(ds.get("SOPInstanceUID", "") or ""),
            "for": str(ds.get("FrameOfReferenceUID", "") or ""),
            "desc": str(ds.get("SeriesDescription", "") or "").strip(),
            "size": (int(ds.Rows), int(ds.Columns)), "frames": frames, "instance": instance,
            "iop": _floats(ds.get("ImageOrientationPatient"), 6),
            "ipp": _floats(ds.get("ImagePositionPatient"), 3),
            "spacing": _floats(ds.get("PixelSpacing"), 2),
            "split": {name: _tag_text(ds, tag) for name, tag in _SPLIT_TAGS},
        })
    return records


def _plane_from_normal(k_axis) -> str:
    """Axial / Coronal / Sagittal / Oblique from the slice normal in LPS."""
    k_axis = np.asarray(k_axis, float)
    order = np.argsort(-np.abs(k_axis))
    dominant, second = np.abs(k_axis[order[0]]), np.abs(k_axis[order[1]])
    if dominant < 0.90 or (dominant - second) < 0.15:
        return "Oblique"
    return {0: "Sagittal", 1: "Coronal", 2: "Axial"}[int(order[0])]


def _unique_positions(recs: List[dict]) -> bool:
    pos = [r["pos"] for r in recs if r["pos"] is not None]
    return len(set(np.round(pos, _POSITION_DECIMALS))) == len(pos)


def _split_repeats(recs: List[dict]) -> List[Tuple[str, List[dict]]]:
    """A stack whose slice positions repeat holds several volumes. Split it
    by the first header field in _SPLIT_TAGS that gives exactly as many
    volumes as positions repeat, each with every position once; if none
    does, the k-th file (by InstanceNumber) at each position goes to volume k."""
    if _unique_positions(recs):
        return [("", recs)]
    counts: Dict[float, int] = {}
    for r in recs:
        p = round(r["pos"], _POSITION_DECIMALS)
        counts[p] = counts.get(p, 0) + 1
    repeat = max(counts.values())
    for name, _ in _SPLIT_TAGS:
        values = list(dict.fromkeys(r["split"][name] for r in recs))
        if len(values) != repeat:
            continue
        parts = [[r for r in recs if r["split"][name] == v] for v in values]
        if all(_unique_positions(p) for p in parts):
            return [(f"{name} {v[:24]}", p) for v, p in zip(values, parts)]
    by_position: Dict[float, List[dict]] = {}
    for r in sorted(recs, key=lambda r: (r["instance"], r["path"])):
        by_position.setdefault(round(r["pos"], _POSITION_DECIMALS), []).append(r)
    return [(f"volume {j + 1} of {repeat}", [v[j] for v in by_position.values() if len(v) > j])
            for j in range(repeat)]


def _spacing_notes(pos: np.ndarray) -> List[str]:
    """Warn when slices are unevenly spaced: the volume uses the average
    spacing (first to last / (n-1)), as ITK does."""
    if len(pos) < 3:
        return []
    gaps = np.diff(pos)
    mean = (pos[-1] - pos[0]) / (len(pos) - 1)
    off = np.abs(pos - (pos[0] + mean * np.arange(len(pos)))).max()
    if off <= 0.25 * mean:
        return []
    median = float(np.median(gaps))
    big = np.flatnonzero(gaps > 1.5 * median)
    if big.size:
        return [f"missing slice(s): {big.size} gap(s) up to {gaps.max():.1f} mm (usual {median:.2f} mm), "
                f"so slices are shown up to {off:.1f} mm from their true position"]
    return [f"uneven slice spacing ({gaps.min():.2f}-{gaps.max():.2f} mm), "
            f"slices shown up to {off:.1f} mm from their true position"]


def _split_blocks(part: List[dict]) -> List[Tuple[str, List[dict]]]:
    """Split a sorted stack at gaps of more than 5x the usual slice spacing:
    separate blocks (e.g. two slabs) are shown as separate volumes."""
    pos = np.array([r["pos"] for r in part])
    if len(pos) < 3:
        return [("", part)]
    gaps = np.diff(pos)
    median = float(np.median(gaps))
    cuts = np.flatnonzero(gaps > 5 * median) + 1 if median > 0 else []
    if not len(cuts):
        return [("", part)]
    blocks = np.split(np.arange(len(part)), cuts)
    return [(f"block {k + 1} of {len(blocks)}", [part[i] for i in b]) for k, b in enumerate(blocks)]


def _slice_slots(pos: np.ndarray) -> Tuple[Optional[List[int]], Optional[float]]:
    """Grid index of each slice and the grid spacing when the slices sit on
    an even grid with some positions empty (missing slices), so blank slices
    can fill the gaps; (None, None) without gaps or off-grid (> 20 % of a step)."""
    if len(pos) < 3:
        return None, None
    rel = pos - pos[0]
    step = float(np.median(np.diff(pos)))
    if step <= 0:
        return None, None
    k = np.round(rel / step)
    step = float(np.dot(rel, k) / np.dot(k, k))       # least-squares spacing for that grid
    k = np.round(rel / step)
    if len(np.unique(k)) != len(k) or np.abs(rel / step - k).max() > 0.2 or k[-1] + 1 == len(pos):
        return None, None
    return [int(x) for x in k], step


def _discover_series(folder: str) -> List[SeriesInfo]:
    by_uid: Dict[str, List[dict]] = {}
    for r in _scan_folder_headers(folder):
        by_uid.setdefault(r["uid"], []).append(r)

    found = []
    for uid, recs in by_uid.items():
        desc = next((r["desc"] for r in recs if r["desc"]), "") or "Series"
        unique, seen = [], set()                 # the same image exported twice under other names
        for r in recs:
            if r["sop"] and r["sop"] in seen:
                continue
            seen.add(r["sop"])
            unique.append(r)
        dup_note = ([f"{len(recs) - len(unique)} duplicate file(s) of the same images ignored"]
                    if len(unique) < len(recs) else [])
        recs = unique
        volumes = []      # (label extra, files, multi_frame, notes, n_slices, slots, step)
        n_multi = sum(1 for r in recs if r["frames"] > 1)
        for r in recs:
            if r["frames"] > 1:
                rows, cols = r["size"]
                name = f" | file {os.path.basename(r['path'])[-40:]}" if n_multi > 1 else ""
                volumes.append((f"multi-frame {cols}x{rows}x{r['frames']}{name}", [r["path"]], True, [],
                                r["frames"], None, None))
        singles = [r for r in recs if r["frames"] <= 1]

        stacks: List[List[dict]] = []            # same orientation, image size and pixel spacing
        for r in singles:
            iop = None if r["iop"] is None else np.array(r["iop"])
            for st in stacks:
                ref = st[0]
                ref_iop = None if ref["iop"] is None else np.array(ref["iop"])
                same_iop = (iop is None and ref_iop is None) or (
                    iop is not None and ref_iop is not None and np.abs(iop - ref_iop).max() < _ORIENTATION_TOL)
                same_spacing = (r["spacing"] is None) == (ref["spacing"] is None) and (
                    r["spacing"] is None or np.allclose(r["spacing"], ref["spacing"], atol=1e-3))
                if same_iop and r["size"] == ref["size"] and same_spacing:
                    st.append(r)
                    break
            else:
                stacks.append([r])

        loadable = [st for st in stacks if len(st) >= 2]
        left_out = sum(len(st) for st in stacks if len(st) < 2)
        left_note = ([f"{left_out} image(s) in another orientation or size left out (e.g. a localiser)"]
                     if left_out and loadable else [])
        multi_stack = len(loadable) > 1
        for k, st in enumerate(loadable):
            iop = st[0]["iop"]
            normal = np.cross(iop[:3], iop[3:]) if iop is not None else None
            for r in st:
                r["pos"] = float(np.dot(normal, r["ipp"])) if (normal is not None and r["ipp"] is not None) else None
            plane = _plane_from_normal(normal) if normal is not None else "unknown plane"
            rows, cols = st[0]["size"]
            stack_note = f"stack {k + 1} of {len(loadable)}" if multi_stack else ""
            if any(r["pos"] is None for r in st):
                ordered = sorted(st, key=lambda r: (r["instance"], r["path"]))
                notes = ["no slice positions in the headers: slices ordered by instance number"] + left_note + dup_note
                volumes.append((" | ".join(x for x in (f"{plane} {cols}x{rows}x{len(st)}", stack_note) if x),
                                [r["path"] for r in ordered], False, notes, len(st), None, None))
                continue
            for split_label, part in _split_repeats(st):
                part = sorted(part, key=lambda r: r["pos"])
                for block_label, block in _split_blocks(part):
                    if len(block) < 2:
                        continue
                    pos = np.array([r["pos"] for r in block])
                    slots, step = _slice_slots(pos)
                    if slots is not None:
                        notes = [f"{slots[-1] + 1 - len(block)} missing slice(s) shown blank, "
                                 "so the other slices keep their true positions"]
                    else:
                        notes = _spacing_notes(pos)
                    notes += left_note + dup_note
                    extra = " | ".join(x for x in (f"{plane} {cols}x{rows}x{len(block)}", stack_note, split_label,
                                                   block_label) if x)
                    volumes.append((extra, [r["path"] for r in block], False, notes, len(block), slots, step))

        volumes.sort(key=lambda v: -v[4])
        for k, (extra, files, multi, notes, n, slots, step) in enumerate(volumes):
            key = uid if len(volumes) == 1 else f"{uid}#{k + 1}"
            split_notes = [extra.split(" | ", 1)[1]] if (len(volumes) > 1 and " | " in extra) else []
            first = next(r for r in recs if r["path"] == files[0])
            dims = (first["size"][1], first["size"][0], (slots[-1] + 1) if slots else n)
            found.append((n, SeriesInfo(key=key, label=f"{desc} | {extra}", files=files, multi_frame=multi,
                                        notes=split_notes + notes, slots=slots, slot_spacing=step,
                                        frame_of_reference=first["for"], dims=dims)))

    found = [s for _, s in sorted(found, key=lambda x: -x[0])]
    labels = [s.label for s in found]
    for s in found:   # a chooser needs unique labels
        if labels.count(s.label) > 1:
            s.label = f"{s.label} | {s.key[-12:]}"
    return found


_SERIES_CACHE: Dict[tuple, List[SeriesInfo]] = {}


def find_series(folder: str) -> List[SeriesInfo]:
    """Every loadable 3D volume directly inside ``folder``, largest first
    (single-frame series of 2+ slices and multi-frame objects). Cached per
    folder state, so a series chooser and the following load scan once."""
    folder = os.path.abspath(folder)
    try:
        state = (folder, os.stat(folder).st_mtime_ns, len(os.listdir(folder)))
    except OSError:
        return []
    if state not in _SERIES_CACHE:
        if len(_SERIES_CACHE) > 32:
            _SERIES_CACHE.clear()
        _SERIES_CACHE[state] = _discover_series(folder)
    return list(_SERIES_CACHE[state])


def find_series_ids(folder: str) -> List[str]:
    """Keys of the image volumes in ``folder`` (see :func:`find_series`)."""
    return [s.key for s in find_series(folder)]


def detect_acquisition_plane(image: sitk.Image) -> str:
    """Axial / Coronal / Sagittal / Oblique from the slice normal (before reorienting)."""
    d = image.GetDirection()
    return _plane_from_normal([d[2], d[5], d[8]])


def get_acquisition_plane(image: sitk.Image) -> str:
    key = "detected_acquisition_plane"
    return image.GetMetaData(key) if image.HasMetaDataKey(key) else "Unknown"


def get_series_notes(image: sitk.Image) -> str:
    """What load_dicom_series noted about the volume ("" if nothing): which
    part of a split series it is, missing slices, uneven spacing ..."""
    key = "series_notes"
    return image.GetMetaData(key) if image.HasMetaDataKey(key) else ""


def _select_series(folder: str, series_id: Optional[str]) -> Tuple[str, SeriesInfo]:
    """(folder, volume): ``series_id`` is a key from find_series (a bare
    SeriesInstanceUID of a split series picks its largest volume); None picks
    the largest volume, looking one folder level down if needed."""
    infos = find_series(folder)
    if not infos and series_id is None:
        for sub in sorted(glob.glob(os.path.join(folder, "*", ""))):
            infos = find_series(sub)
            if infos:
                folder = sub
                break
    if not infos:
        raise RuntimeError(f"No DICOM image series found in {folder!r}. Make sure the folder directly contains "
                           "the DICOM slice files (or a multi-frame DICOM file); single-slice series can't be "
                           "shown in 3D.")
    if series_id is None:
        return folder, infos[0]
    for s in infos:
        if s.key == series_id:
            return folder, s
    for s in infos:
        if s.key.split("#")[0] == series_id:
            return folder, s
    raise RuntimeError(f"Series {series_id} not found in {folder!r}.")


def read_slices_with_pydicom(files: List[str]) -> Tuple[sitk.Image, int]:
    """Fallback when GDCM can't read a series: the same sorted single-frame
    files read with pydicom, with the geometry and rescale ITK would use. A
    slice even pydicom can't read keeps its place, filled with the volume's
    minimum. Returns (image, number of unreadable slices)."""
    slices, slopes, intercepts = [], set(), set()
    for path in files:
        try:
            ds = pydicom.dcmread(path, force=True)
            slices.append(ds.pixel_array)
            slopes.add(float(ds.get("RescaleSlope", 1) or 1))
            intercepts.add(float(ds.get("RescaleIntercept", 0) or 0))
        except Exception:
            slices.append(None)
    read = [s for s in slices if s is not None]
    if not read:
        raise RuntimeError("none of the slice files could be read")
    background = min(float(s.min()) for s in read)
    slices = [s if s is not None else np.full(read[0].shape, background, read[0].dtype) for s in slices]
    first = pydicom.dcmread(files[0], stop_before_pixels=True, force=True)
    last = pydicom.dcmread(files[-1], stop_before_pixels=True, force=True)
    array = np.stack(slices)
    if slopes != {1.0} or intercepts != {0.0}:
        slope, intercept = slopes.pop(), intercepts.pop()
        if slopes or intercepts:
            raise RuntimeError("slices with different RescaleSlope/Intercept")
        array = array * slope + intercept
        if float(slope).is_integer() and float(intercept).is_integer():
            array = array.astype(np.int32)
    row, col = np.array(first.ImageOrientationPatient[:3], float), np.array(first.ImageOrientationPatient[3:], float)
    normal = np.cross(row, col)
    p0, p1 = np.array(first.ImagePositionPatient, float), np.array(last.ImagePositionPatient, float)
    dz = float(np.dot(p1 - p0, normal)) / max(len(files) - 1, 1)
    image = sitk.GetImageFromArray(array)
    row_spacing, col_spacing = (float(x) for x in first.PixelSpacing)
    image.SetSpacing((col_spacing, row_spacing, abs(dz) if dz else 1.0))
    image.SetOrigin(tuple(p0))
    axis_z = normal if dz >= 0 else -normal
    image.SetDirection(tuple(np.column_stack([row, col, axis_z]).ravel()))
    return image, len(files) - len(read)


def _insert_blank_slices(image: sitk.Image, slots: List[int], spacing: float) -> sitk.Image:
    """Put the read slices at their grid positions with blank slices (the
    volume's minimum) in between; origin and direction are unchanged."""
    array = sitk.GetArrayFromImage(image)
    full = np.full((slots[-1] + 1,) + array.shape[1:], array.min(), array.dtype)
    full[slots] = array
    out = sitk.GetImageFromArray(full)
    sx, sy, _ = image.GetSpacing()
    out.SetSpacing((sx, sy, spacing))
    out.SetOrigin(image.GetOrigin())
    out.SetDirection(image.GetDirection())
    return out


def _slice_thickness(path: str) -> Optional[float]:
    """SliceThickness of a single-frame file, or of a multi-frame file's
    shared (else first per-frame) PixelMeasuresSequence."""
    ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    value = ds.get("SliceThickness")
    for group in ("SharedFunctionalGroupsSequence", "PerFrameFunctionalGroupsSequence"):
        if value is not None:
            break
        try:
            value = ds.get(group)[0].PixelMeasuresSequence[0].get("SliceThickness")
        except Exception:
            value = None
    return float(value) if value is not None and float(value) > 0 else None


def load_dicom_series(folder: str, series_id: Optional[str] = None) -> sitk.Image:
    """Read one volume (see :func:`find_series`) and reorient it to LPS.

    The image carries metadata: detected_acquisition_plane, dicom_folder
    (where its files are), series_key, frame_of_reference, series_notes
    (see :func:`get_series_notes`) and slice_thickness_mm.
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Not a folder: {folder}")
    folder, info = _select_series(folder, series_id)
    notes = list(info.notes)
    if info.multi_frame:
        image = sitk.ReadImage(info.files[0])
    else:
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(info.files)
        try:
            image = reader.Execute()
        except RuntimeError:
            # GDCM rejects some files pydicom reads fine (seen: 68 of 352 slices of one series)
            image, unreadable = read_slices_with_pydicom(info.files)
            notes.append("read with pydicom (GDCM could not read some files)")
            if unreadable:
                notes.append(f"{unreadable} corrupt slice file(s) shown blank")
    if info.slots is not None:
        image = _insert_blank_slices(image, info.slots, info.slot_spacing)
    if image.GetDimension() != 3 or image.GetSize()[2] < 2:
        raise RuntimeError(f"{info.label} is not a 3D stack (size {image.GetSize()}); "
                           "need a real 3D stack to build MPR views.")
    if image.GetNumberOfComponentsPerPixel() != 1:
        raise RuntimeError(f"{info.label} is a colour image ({image.GetNumberOfComponentsPerPixel()} "
                           "components per pixel); only grayscale volumes can be shown.")

    acquired_as = detect_acquisition_plane(image)
    image = sitk.DICOMOrient(image, "LPS")
    image.SetMetaData("detected_acquisition_plane", acquired_as)
    image.SetMetaData("dicom_folder", os.path.dirname(os.path.abspath(info.files[0])))
    image.SetMetaData("series_key", info.key)
    image.SetMetaData("frame_of_reference", info.frame_of_reference)
    image.SetMetaData("series_notes", "; ".join(notes))
    try:   # for the slice-profile step of the through-plane reconstruction (header only)
        thickness = _slice_thickness(info.files[0])
        if thickness is not None:
            image.SetMetaData("slice_thickness_mm", str(thickness))
    except Exception:
        pass
    return image


def series_files(folder: str, series_id: Optional[str] = None) -> List[str]:
    """The sorted single-frame files load_dicom_series reads for this volume
    (for writing per-slice objects that line up with them)."""
    folder, info = _select_series(folder, series_id)
    if info.multi_frame:
        raise RuntimeError(f"{info.label} is a single multi-frame file, not one file per slice.")
    return list(info.files)
