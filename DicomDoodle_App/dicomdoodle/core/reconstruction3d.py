"""
Building the 3D model of the Multiplanar & 3D Viewer (port of the two
background workers of the original 3D Model Viewer, create_3d.py).

Up to three series folders (axial / coronal / sagittal) are read with
SimpleITK, resampled onto the grid of the first one available (axial
preferred) and averaged. Annotation files become a filled 3D lesion mask on
the same grid, and the lesion is brightened in the "anatomy" volume so it
stands out (FLAIR folders: lesion x1.4 and the rest x0.2; otherwise lesion
x1.99) — the original's rules.

Differences from the original: the series are read once instead of twice
(one worker per view), images are always resampled with linear interpolation
(the original switched to nearest-neighbour after the first mask), and
multi-label annotation frames stay on their own slice.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .dicom_series import SegFileFilter, looks_like_annotation
from .mask_ops import fill_closed_boundaries
from .progress import Progress, ensure_progress
from .seg_io import collect_series_masks

ORDER = ("Axial", "Coronal", "Sagittal")


@dataclass
class Reconstruction:
    anatomy: np.ndarray                  # (z, y, x) float32, lesion-enhanced
    volume: np.ndarray                   # (z, y, x) float32, plain average
    lesion_mask: Optional[np.ndarray]    # (z, y, x) uint8 (0 / 255) or None
    spacing: Tuple[float, float, float]  # x, y, z (mm)
    origin: Tuple[float, float, float]
    reference: str
    series_used: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    direction: Tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)   # of the reference grid
    frame_of_reference: str = ""         # FrameOfReferenceUID of the reference series (its scanner space)

    def grid_image(self, array: Optional[np.ndarray] = None):
        """``array`` (default: the plain volume) as a SimpleITK image on the reference grid."""
        import SimpleITK as sitk

        image = sitk.GetImageFromArray(self.volume if array is None else array)
        image.SetSpacing(self.spacing)
        image.SetOrigin(self.origin)
        image.SetDirection(self.direction)
        return image


def _series_files(folder: str) -> List[Tuple[str, int, str]]:
    """(path, InstanceNumber, SOPInstanceUID) of every image file under ``folder``."""
    import pydicom

    entries = []
    for root, _dirs, files in os.walk(folder):
        for name in files:
            if not name.lower().endswith(".dcm") or looks_like_annotation(name):
                continue
            path = os.path.join(root, name)
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True)
            except Exception:
                continue
            if str(ds.get("Modality", "")).upper() == "SEG":
                continue
            entries.append((path, int(ds.get("InstanceNumber", 0) or 0), str(ds.get("SOPInstanceUID", ""))))
    entries.sort(key=lambda e: (e[1], e[0]))
    return entries


def _annotation_files(folder: str, seg_filter: SegFileFilter) -> List[str]:
    found = []
    for root, _dirs, files in os.walk(folder):
        found += [os.path.join(root, n) for n in sorted(files) if seg_filter.matches(n)]
    return found


def build_reconstruction(folders: Dict[str, str], seg_filter: SegFileFilter = SegFileFilter.ALL,
                         progress: Optional[Progress] = None) -> Reconstruction:
    import SimpleITK as sitk

    progress = ensure_progress(progress)
    warnings: List[str] = []
    loaded = []      # (name, image, lesion mask image or None)
    first_file: Dict[str, str] = {}
    names = [n for n in ORDER if folders.get(n)]
    if not names:
        raise ValueError("Please select at least one DICOM series folder.")

    for k, name in enumerate(names):
        folder = folders[name]
        progress(k, len(names) + 2, f"Reading the {name.lower()} series…")
        if not os.path.isdir(folder):
            warnings.append(f"{name} folder not found: {folder}")
            continue
        entries = _series_files(folder)
        if not entries:
            warnings.append(f"No image .dcm files found in the {name.lower()} folder.")
            continue
        first_file[name] = entries[0][0]
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames([e[0] for e in entries])
        try:
            image = reader.Execute()
        except RuntimeError as exc:
            warnings.append(f"Error reading the {name.lower()} series: {exc}")
            continue

        mask_image = None
        seg_paths = _annotation_files(folder, seg_filter)
        if seg_paths:
            progress(k, len(names) + 2, f"Reading {name.lower()} annotations…")
            size = image.GetSize()   # x, y, z
            masks = collect_series_masks([e[2] for e in entries], [e[1] for e in entries],
                                         seg_paths, (size[1], size[0]))
            if masks:
                volume_mask = np.zeros((size[2], size[1], size[0]), dtype=np.uint8)
                for z, mask in masks.items():
                    volume_mask[z] = fill_closed_boundaries(mask)
                mask_image = sitk.GetImageFromArray(volume_mask)
                mask_image.CopyInformation(image)
        loaded.append((name, image, mask_image))

    if not loaded:
        raise RuntimeError("No DICOM series could be loaded.\n" + "\n".join(warnings))

    reference_name, reference, _ = loaded[0]          # axial preferred (ORDER)
    frame_of_reference = _frame_of_reference(first_file[reference_name])
    progress(len(names), len(names) + 2, f"Resampling onto the {reference_name.lower()} grid…")
    image_resampler = sitk.ResampleImageFilter()
    image_resampler.SetReferenceImage(reference)
    image_resampler.SetInterpolator(sitk.sitkLinear)
    image_resampler.SetDefaultPixelValue(float(np.min(sitk.GetArrayViewFromImage(reference))))
    mask_resampler = sitk.ResampleImageFilter()
    mask_resampler.SetReferenceImage(reference)
    mask_resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    mask_resampler.SetDefaultPixelValue(0)

    arrays, masks = [], []
    for name, image, mask_image in loaded:
        as_float = sitk.Cast(image, sitk.sitkFloat32)
        try:
            resampled = as_float if image is reference else image_resampler.Execute(as_float)
            arrays.append(sitk.GetArrayFromImage(resampled))
            if mask_image is not None:
                masks.append(sitk.GetArrayFromImage(mask_resampler.Execute(mask_image)))
        except RuntimeError as exc:
            warnings.append(f"Error resampling the {name.lower()} series: {exc}")

    shape = arrays[0].shape
    compatible = [a for a in arrays if a.shape == shape]
    plain = np.mean(np.stack(compatible, axis=0), axis=0).astype(np.float32)
    anatomy = plain.copy()

    lesion = None
    compatible_masks = [m for m in masks if m.shape == shape]
    if compatible_masks:
        progress(len(names) + 1, len(names) + 2, "Combining lesion masks…")
        lesion = np.max(np.stack(compatible_masks, axis=0), axis=0).astype(np.uint8)
        inside = lesion > 0
        is_flair = any("flair" in folders[n].lower() for n in names)
        if is_flair:
            anatomy[inside] *= 1.4
            anatomy[~inside] *= 0.2
        else:
            anatomy[inside] *= 1.99
        if not inside.any():
            lesion = None

    progress(len(names) + 2, len(names) + 2, "Done")
    return Reconstruction(
        anatomy=np.ascontiguousarray(anatomy, dtype=np.float32),
        volume=np.ascontiguousarray(plain, dtype=np.float32),
        lesion_mask=None if lesion is None else np.ascontiguousarray(lesion),
        spacing=tuple(reference.GetSpacing()),
        origin=tuple(reference.GetOrigin()),
        reference=reference_name,
        series_used=[n for n, _i, _m in loaded],
        warnings=warnings,
        direction=tuple(reference.GetDirection()),
        frame_of_reference=frame_of_reference,
    )


def _frame_of_reference(path: str) -> str:
    """FrameOfReferenceUID of one image file ("" if it has none)."""
    import pydicom

    try:
        return str(pydicom.dcmread(path, stop_before_pixels=True).get("FrameOfReferenceUID", "") or "")
    except Exception:
        return ""
