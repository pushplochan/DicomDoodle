"""
Lesion volume from annotated series (the original "Lesion Volume
Calculator"; shown in the right-hand panel of the Multiplanar & 3D Viewer).

For every orientation folder (axial / coronal / sagittal) the annotation of
each slice is matched to its image, the outline is filled, and

    volume (cm³) = filled voxels × pixel spacing × pixel spacing × slice thickness / 1000

The average over the annotated series is reported as well — a series loaded
only for the 3D picture, without annotations, does not lower it.

Differences from the original, all bug fixes:
  * annotation files with several labels (several frames) are counted —
    the original skipped them because of a shape check;
  * slices are indexed by position, so gaps in InstanceNumber no longer crash;
  * if a slice has several annotation files they are merged (union) instead
    of using whichever file happened to be read last;
  * a series that cannot be read is reported instead of stopping the others.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pydicom

from .dicom_series import SegFileFilter, annotation_files_in, is_dicom_filename, looks_like_annotation
from .mask_ops import fill_closed_boundaries
from .progress import OperationCancelled, Progress, ensure_progress
from .seg_io import collect_series_masks

ORIENTATIONS = ("Axial", "Coronal", "Sagittal")


@dataclass
class OrientationVolume:
    orientation: str
    folder: str
    slices: int = 0
    annotated_slices: int = 0
    pixel_spacing: Tuple[float, float] = (1.0, 1.0)
    slice_thickness: float = 1.0
    voxels: int = 0
    warnings: List[str] = field(default_factory=list)
    error: str = ""                   # why the series could not be measured

    @property
    def counted(self) -> bool:
        """Measured and annotated, so it counts towards the average."""
        return not self.error and self.annotated_slices > 0

    @property
    def voxel_mm3(self) -> float:
        return self.pixel_spacing[0] * self.pixel_spacing[1] * self.slice_thickness

    @property
    def volume_mm3(self) -> float:
        return self.voxels * self.voxel_mm3

    @property
    def volume_cm3(self) -> float:
        return self.volume_mm3 / 1000.0


def compute_orientation_volume(folder: str, orientation: str,
                               seg_filter: SegFileFilter = SegFileFilter.ALL,
                               progress: Optional[Progress] = None) -> OrientationVolume:
    progress = ensure_progress(progress)
    result = OrientationVolume(orientation, folder)
    if not os.path.isdir(folder):
        raise RuntimeError(f"Folder not found: {folder}")
    names = [n for n in sorted(os.listdir(folder)) if is_dicom_filename(n) and not looks_like_annotation(n)]
    headers = []
    for k, name in enumerate(names):
        progress(k, len(names), f"{orientation}: reading headers")
        try:
            ds = pydicom.dcmread(os.path.join(folder, name), stop_before_pixels=True)
        except Exception as exc:
            result.warnings.append(f"Failed to read {name}: {exc}")
            continue
        if str(ds.get("Modality", "")).upper() == "SEG":
            continue
        instance = ds.get("InstanceNumber")
        headers.append((int(instance) if instance not in (None, "") else 0, name, ds))
    if not headers:
        raise RuntimeError(f"No valid DICOM files found in the {orientation.lower()} folder.")
    headers.sort(key=lambda h: (h[0], h[1]))
    result.slices = len(headers)

    thickness = next((float(h[2].SliceThickness) for h in headers if h[2].get("SliceThickness")), None)
    spacing = next(((float(h[2].PixelSpacing[0]), float(h[2].PixelSpacing[1]))
                    for h in headers if h[2].get("PixelSpacing")), None)
    if thickness is None:
        thickness = 1.0
        result.warnings.append("SliceThickness not found, using default 1.0 mm")
    if spacing is None:
        spacing = (1.0, 1.0)
        result.warnings.append("PixelSpacing not found, using default 1.0 mm x 1.0 mm")
    result.slice_thickness, result.pixel_spacing = thickness, spacing

    ref = headers[0][2]
    shape = (int(ref.Rows), int(ref.Columns))
    uids = [str(h[2].get("SOPInstanceUID", "")) for h in headers]
    instances = [h[0] for h in headers]
    seg_paths = annotation_files_in(folder, seg_filter)
    masks = collect_series_masks(uids, instances, seg_paths, shape, progress)

    voxels = 0
    for mask in masks.values():
        filled = fill_closed_boundaries(mask)
        voxels += int(np.count_nonzero(filled))
    result.voxels = voxels
    result.annotated_slices = len(masks)
    return result


def compute_volumes(folders: dict, seg_filter: SegFileFilter = SegFileFilter.ALL,
                    progress: Optional[Progress] = None) -> List[OrientationVolume]:
    """folders: {"Axial": path or "", ...} -> one result per selected orientation.
    A series that cannot be measured comes back with its ``error`` set."""
    results = []
    for orientation in ORIENTATIONS:
        folder = folders.get(orientation)
        if not folder:
            continue
        try:
            results.append(compute_orientation_volume(folder, orientation, seg_filter, progress))
        except OperationCancelled:
            raise
        except Exception as exc:
            results.append(OrientationVolume(orientation, folder, error=str(exc) or exc.__class__.__name__))
    return results


def average_volume_cm3(results: List[OrientationVolume]) -> float:
    """Average lesion volume of the annotated series (0 if there are none)."""
    counted = [r.volume_cm3 for r in results if r.counted]
    return float(np.mean(counted)) if counted else 0.0
