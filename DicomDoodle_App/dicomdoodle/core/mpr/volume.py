"""
MPR volumes: resampling strategies, slice extraction and display windowing
(from the original mpr_core.py; through-plane reconstruction and the voxel
limit from its zero_shot_tumor version).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np
import SimpleITK as sitk

from .through_plane import SR_MIN_COVERAGE, reconstruct_through_plane


class InterpMethod(Enum):
    """How the volume is resampled for display.

    NONE_NATIVE  original voxels, no interpolation (zero data loss); the
                 display aspect ratio corrects anisotropic spacing
    NEAREST      isotropic grid, nearest neighbour (no new values, can look blocky)
    LINEAR       trilinear — the fast standard of most MPR viewers
    BSPLINE      cubic B-spline — smoothest, most accurate isotropic volume
    LANCZOS      windowed sinc — sharpest smooth interpolator, slowest
    THROUGH_PLANE  refines only the slice axis (B-spline + slice profile,
                 see through_plane.py) — the sharpest reformatted views; it
                 re-estimates the acquired slices slightly, so use
                 NONE_NATIVE when the bit-exact values are needed
    """

    NONE_NATIVE = "None (native voxels, no interpolation - zero data loss)"
    NEAREST = "Nearest Neighbor"
    LINEAR = "Trilinear (Linear)"
    BSPLINE = "Cubic B-Spline (recommended, high accuracy)"
    LANCZOS = "Lanczos / Windowed-Sinc (slowest, sharpest)"
    THROUGH_PLANE = "Through-plane reconstruction (B-spline + slice profile, sharpest reformats)"

    @classmethod
    def from_label(cls, label: str) -> "InterpMethod":
        for method in cls:
            if method.value == label:
                return method
        raise ValueError(f"Unknown interpolation method label: {label!r}")


# Largest grid an interpolating method builds by default (int16: 2 GB, plus the
# float copies of the 3D view and processing). Only series with very fine
# pixels and thick slices reach it; their grid is coarsened just enough.
MAX_RESAMPLED_VOXELS = 1.0e9

_SITK_INTERPOLATOR = {
    InterpMethod.NEAREST: sitk.sitkNearestNeighbor,
    InterpMethod.LINEAR: sitk.sitkLinear,
    InterpMethod.BSPLINE: sitk.sitkBSpline,
    InterpMethod.LANCZOS: sitk.sitkLanczosWindowedSinc,
}


@dataclass
class Volume:
    """A 3D volume plus what is needed to slice and display it correctly."""

    array: np.ndarray                          # (Z, Y, X)
    spacing_zyx: Tuple[float, float, float]
    origin_xyz: Tuple[float, float, float]
    direction: Tuple[float, ...]
    is_isotropic: bool
    interp_method: InterpMethod
    sitk_image: sitk.Image

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.array.shape


def build_volume(image: sitk.Image, method: InterpMethod, target_spacing: Optional[float] = None) -> Volume:
    """Volume for display. NONE_NATIVE keeps the exact voxels; THROUGH_PLANE
    refines only the slice axis; any other method resamples onto an
    isotropic grid at the finest existing spacing (within MAX_RESAMPLED_VOXELS)."""
    if method is InterpMethod.NONE_NATIVE:
        sx, sy, sz = image.GetSpacing()
        return Volume(sitk.GetArrayFromImage(image), (sz, sy, sx), image.GetOrigin(), image.GetDirection(),
                      False, method, image)
    if method is InterpMethod.THROUGH_PLANE:
        return _build_through_plane_volume(image, target_spacing)

    sx, sy, sz = image.GetSpacing()
    size_x, size_y, size_z = image.GetSize()
    if target_spacing is None:
        target_spacing = min(sx, sy, sz)
        # e.g. 1680 x 1680 pixels of 0.14 mm with 3 mm slices would need 2.9 G voxels
        n_voxels = size_x * sx * size_y * sy * size_z * sz / target_spacing ** 3
        if n_voxels > MAX_RESAMPLED_VOXELS:
            target_spacing *= (n_voxels / MAX_RESAMPLED_VOXELS) ** (1 / 3)
    new_size = [max(1, int(round(size_x * sx / target_spacing))),
                max(1, int(round(size_y * sy / target_spacing))),
                max(1, int(round(size_z * sz / target_spacing)))]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing((target_spacing,) * 3)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(float(np.min(sitk.GetArrayViewFromImage(image))))
    resampler.SetInterpolator(_SITK_INTERPOLATOR[method])
    resampled = resampler.Execute(image)
    return Volume(sitk.GetArrayFromImage(resampled), (target_spacing,) * 3, resampled.GetOrigin(),
                  resampled.GetDirection(), True, method, resampled)


def _build_through_plane_volume(image: sitk.Image, target_spacing: Optional[float]) -> Volume:
    """The coarsest axis (the slice axis) refined to the finest in-plane
    spacing (or ``target_spacing``) by a whole factor, so every acquired
    slice keeps its position and the extent is unchanged. A series already
    (nearly) as fine through-plane as in-plane is returned unchanged."""
    sx, sy, sz = image.GetSpacing()
    spacing_zyx = [sz, sy, sx]
    axis = int(np.argmax(spacing_zyx))
    gap = spacing_zyx[axis]
    array = sitk.GetArrayFromImage(image)
    if target_spacing is None:
        target_spacing = min(s for a, s in enumerate(spacing_zyx) if a != axis)
        factor = int(round(gap / target_spacing))
        n = array.shape[axis]
        per_slice = array.size // max(n, 1)
        factor = min(factor, int((MAX_RESAMPLED_VOXELS / per_slice - 1) // max(n - 1, 1)))
    else:
        factor = int(round(gap / target_spacing))

    if factor < 2 or array.shape[axis] < 2:
        volume = build_volume(image, InterpMethod.NONE_NATIVE)
        volume.interp_method = InterpMethod.THROUGH_PLANE
        return volume

    profile_width = None
    if image.HasMetaDataKey("slice_thickness_mm"):
        thickness = float(image.GetMetaData("slice_thickness_mm"))
        # contiguous slices only (see through_plane.py); slices thicker than the
        # gap overlap, and their extra blur can't be undone stably: cap at the gap
        if thickness >= SR_MIN_COVERAGE * gap:
            profile_width = min(thickness, gap) / (gap / factor)
    fine = reconstruct_through_plane(array, axis, factor, profile_width)
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        np.rint(fine, out=fine)
        np.clip(fine, info.min, info.max, out=fine)
    fine = np.ascontiguousarray(fine, dtype=array.dtype if np.issubdtype(array.dtype, np.integer) else np.float32)

    spacing_zyx[axis] = gap / factor
    resampled = sitk.GetImageFromArray(fine)
    resampled.SetSpacing((spacing_zyx[2], spacing_zyx[1], spacing_zyx[0]))
    resampled.SetOrigin(image.GetOrigin())
    resampled.SetDirection(image.GetDirection())
    for key in image.GetMetaDataKeys():
        resampled.SetMetaData(key, image.GetMetaData(key))
    return Volume(fine, tuple(spacing_zyx), resampled.GetOrigin(), resampled.GetDirection(),
                  max(spacing_zyx) <= 1.01 * min(spacing_zyx), InterpMethod.THROUGH_PLANE, resampled)


class Plane(Enum):
    AXIAL = "axial"
    CORONAL = "coronal"
    SAGITTAL = "sagittal"


def plane_extent(volume: Volume, plane: Plane) -> int:
    z, y, x = volume.shape
    return {Plane.AXIAL: z, Plane.CORONAL: y, Plane.SAGITTAL: x}[plane]


def _slice_array(array: np.ndarray, plane: Plane, index: int) -> np.ndarray:
    z, y, x = array.shape
    extent = {Plane.AXIAL: z, Plane.CORONAL: y, Plane.SAGITTAL: x}[plane]
    index = int(np.clip(index, 0, extent - 1))
    if plane is Plane.AXIAL:
        img = array[index, :, :]
    elif plane is Plane.CORONAL:
        img = array[:, index, :]
    else:
        img = array[:, :, index]
    return np.flipud(img)


def extract_slice(volume: Volume, plane: Plane, index: int) -> np.ndarray:
    """2D slice by pure indexing (radiological display, superior/anterior up)."""
    return _slice_array(volume.array, plane, index)


def extract_label_slice(label_array: np.ndarray, plane: Plane, index: int) -> np.ndarray:
    return _slice_array(label_array, plane, index)


def display_aspect(volume: Volume, plane: Plane) -> float:
    """row_spacing / col_spacing, so native (anisotropic) slices look right."""
    sz, sy, sx = volume.spacing_zyx
    if plane is Plane.AXIAL:
        return sy / sx
    if plane is Plane.CORONAL:
        return sz / sx
    return sz / sy


def window_to_uint8(img: np.ndarray, center: Optional[float] = None, width: Optional[float] = None) -> np.ndarray:
    """Window/level to 8-bit for display; auto 1st-99th percentile if not given."""
    img = img.astype(np.float32)
    if center is None or width is None:
        lo, hi = np.percentile(img, [1, 99])
        if hi <= lo:
            lo, hi = float(img.min()), float(img.max() + 1e-3)
        center = (lo + hi) / 2.0
        width = max(hi - lo, 1e-3)
    lo = center - width / 2.0
    hi = center + width / 2.0
    img = np.clip(img, lo, hi)
    img = (img - lo) / max(hi - lo, 1e-6)
    return (img * 255.0).astype(np.uint8)


def intensity_range(volume: Volume) -> Tuple[float, float]:
    return float(volume.array.min()), float(volume.array.max())
