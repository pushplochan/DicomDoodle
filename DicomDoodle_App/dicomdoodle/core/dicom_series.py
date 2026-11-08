"""
Loading a folder of single-frame DICOM slices for slice-by-slice annotation.

Each slice is min-max normalised to an 8-bit image exactly like the original
DicomDoodle — that is also the image the YOLO and SAM models receive, so
detections are reproducible. Annotation files (``*_seg.dcm``,
``*_segmentation.dcm`` or anything whose Modality is SEG) are skipped.

Also home to the small helpers that know how annotation files are named.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pydicom

from .progress import Progress, ensure_progress

ANNOTATION_SUFFIX = "_seg.dcm"     # what DicomDoodle writes next to every slice


# --------------------------------------------------------------------------- #
# File-name helpers
# --------------------------------------------------------------------------- #

def is_dicom_filename(name: str) -> bool:
    return name.lower().endswith(".dcm")


def looks_like_annotation(name: str) -> bool:
    """True for '..._seg.dcm', '..._segmentation.dcm' and similar files."""
    return "_seg" in os.path.basename(name).lower()


def seg_path_for(dicom_path: str) -> str:
    """'<slice>.dcm' -> '<slice>_seg.dcm' (DicomDoodle's annotation file)."""
    return f"{os.path.splitext(dicom_path)[0]}{ANNOTATION_SUFFIX}"


class SegFileFilter(Enum):
    """Which annotation files a viewer/calculator should read.

    Folders sometimes hold annotations from two tools side by side
    (``_seg.dcm`` from DicomDoodle and ``_segmentation.dcm`` from another
    pipeline); this lets the user choose instead of mixing them silently.
    """

    NATIVE = "*_seg.dcm (DicomDoodle)"
    SEGMENTATION = "*_segmentation.dcm"
    ALL = "All annotation files (merged)"

    def matches(self, filename: str) -> bool:
        low = os.path.basename(filename).lower()
        if self is SegFileFilter.NATIVE:
            return low.endswith("_seg.dcm")
        if self is SegFileFilter.SEGMENTATION:
            return low.endswith("_segmentation.dcm")
        return low.endswith(".dcm") and "_seg" in low

    def candidates_for(self, dicom_path: str) -> List[str]:
        """Annotation file paths that would belong to ``dicom_path``."""
        base = os.path.splitext(dicom_path)[0]
        if self is SegFileFilter.NATIVE:
            return [base + "_seg.dcm"]
        if self is SegFileFilter.SEGMENTATION:
            return [base + "_segmentation.dcm"]
        return [base + "_seg.dcm", base + "_segmentation.dcm"]


def annotation_files_in(folder: str, seg_filter: SegFileFilter = SegFileFilter.ALL) -> List[str]:
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, n) for n in sorted(os.listdir(folder)) if seg_filter.matches(n)]


# --------------------------------------------------------------------------- #
# Pixel helpers
# --------------------------------------------------------------------------- #

def normalize_to_uint8(pixels: np.ndarray) -> np.ndarray:
    """Min-max scale to 0-255 (same formula as the original tool)."""
    pixels = np.asarray(pixels)
    lo, hi = pixels.min(), pixels.max()
    if hi == lo:
        return np.zeros(pixels.shape, dtype=np.uint8)
    return ((pixels - lo) / (hi - lo) * 255).astype(np.uint8)


def detect_plane(image_orientation: Optional[Sequence[float]]) -> str:
    """'Axial', 'Coronal', 'Sagittal' or 'Oblique' from ImageOrientationPatient."""
    if image_orientation is None or len(image_orientation) < 6:
        return "Unknown"
    row = np.asarray(image_orientation[:3], dtype=float)
    col = np.asarray(image_orientation[3:6], dtype=float)
    normal = np.abs(np.cross(row, col))
    order = np.argsort(-normal)
    if normal[order[0]] < 0.75:
        return "Oblique"
    return {0: "Sagittal", 1: "Coronal", 2: "Axial"}[int(order[0])]


# --------------------------------------------------------------------------- #
# Series
# --------------------------------------------------------------------------- #

@dataclass
class SeriesInfo:
    """Header details shown in the 'Series' panel."""

    patient_name: str = ""
    patient_id: str = ""
    study_date: str = ""
    modality: str = ""
    series_description: str = ""
    manufacturer: str = ""
    rows: int = 0
    columns: int = 0
    pixel_spacing: Optional[Tuple[float, float]] = None
    slice_thickness: Optional[float] = None
    plane: str = "Unknown"

    @classmethod
    def from_dataset(cls, ds) -> "SeriesInfo":
        spacing = ds.get("PixelSpacing")
        thickness = ds.get("SliceThickness")
        return cls(
            patient_name=str(ds.get("PatientName", "") or ""),
            patient_id=str(ds.get("PatientID", "") or ""),
            study_date=str(ds.get("StudyDate", "") or ""),
            modality=str(ds.get("Modality", "") or ""),
            series_description=str(ds.get("SeriesDescription", "") or ""),
            manufacturer=str(ds.get("Manufacturer", "") or ""),
            rows=int(ds.get("Rows", 0) or 0),
            columns=int(ds.get("Columns", 0) or 0),
            pixel_spacing=(float(spacing[0]), float(spacing[1])) if spacing else None,
            slice_thickness=float(thickness) if thickness not in (None, "") else None,
            plane=detect_plane(ds.get("ImageOrientationPatient")),
        )


class DicomSeries:
    """A folder of DICOM slices sorted by InstanceNumber, ready to annotate."""

    def __init__(self, folder: str, files: List[str], images: List[np.ndarray],
                 instance_numbers: List[Optional[int]], info: SeriesInfo, skipped: List[str]):
        self.folder = folder
        self.files = files
        self.images = images
        self.instance_numbers = instance_numbers
        self.info = info
        self.skipped = skipped            # files that could not be used, with reasons
        self._raw_cache: Tuple[int, Optional[np.ndarray]] = (-1, None)

    # -------------------------------------------------------------- loading
    @classmethod
    def load(cls, folder: str, progress: Optional[Progress] = None) -> "DicomSeries":
        progress = ensure_progress(progress)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Not a folder: {folder}")

        names = [n for n in sorted(os.listdir(folder))
                 if is_dicom_filename(n) and not looks_like_annotation(n)]
        if not names:
            raise RuntimeError("No DICOM (.dcm) slice files were found in this folder.")

        total = len(names) * 2
        headers, skipped = [], []
        for k, name in enumerate(names):
            progress(k, total, "Reading DICOM headers…")
            path = os.path.join(folder, name)
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True)
            except Exception as exc:  # unreadable / not DICOM
                skipped.append(f"{name}: {exc}")
                continue
            if str(ds.get("Modality", "")).upper() == "SEG":
                continue
            inst = ds.get("InstanceNumber")
            instance = int(inst) if inst not in (None, "") else None
            headers.append((instance, name, path, ds))

        # Same ordering as the original (InstanceNumber); file name breaks ties.
        headers.sort(key=lambda h: (h[0] is None, h[0] if h[0] is not None else 0, h[1]))

        files, images, instances = [], [], []
        for k, (instance, name, path, _hdr) in enumerate(headers):
            progress(len(names) + k, total, f"Loading slice {k + 1} of {len(headers)}…")
            try:
                pixels = pydicom.dcmread(path).pixel_array
            except Exception as exc:
                skipped.append(f"{name}: {exc}")
                continue
            if pixels.ndim != 2:
                skipped.append(f"{name}: multi-frame or color image "
                               "(use Tools ▸ Multi-frame to Single-frame Converter)")
                continue
            files.append(path)
            images.append(normalize_to_uint8(pixels))
            instances.append(instance)

        if not files:
            raise RuntimeError("None of the DICOM files in this folder could be used as slices.\n\n"
                               + "\n".join(skipped[:10]))
        info = SeriesInfo.from_dataset(headers[0][3])
        progress(total, total, "Done")
        return cls(folder, files, images, instances, info, skipped)

    # --------------------------------------------------------------- access
    def __len__(self) -> int:
        return len(self.files)

    @property
    def name(self) -> str:
        return os.path.basename(os.path.normpath(self.folder))

    def image(self, index: int) -> np.ndarray:
        return self.images[index]

    def file(self, index: int) -> str:
        return self.files[index]

    def seg_path(self, index: int) -> str:
        return seg_path_for(self.files[index])

    def has_seg_file(self, index: int) -> bool:
        return os.path.isfile(self.seg_path(index))

    def raw_pixels(self, index: int) -> np.ndarray:
        """Original (un-normalised) pixel values of one slice, cached."""
        cached_index, cached = self._raw_cache
        if cached_index != index or cached is None:
            cached = pydicom.dcmread(self.files[index]).pixel_array
            self._raw_cache = (index, cached)
        return cached

    def reload_image(self, index: int) -> np.ndarray:
        """Re-read one slice from disk (used by 'Reset slice')."""
        self.images[index] = normalize_to_uint8(pydicom.dcmread(self.files[index]).pixel_array)
        self._raw_cache = (-1, None)
        return self.images[index]
