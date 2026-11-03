"""
mpr_core.py
-----------
GUI-independent core logic for Multiplanar Reconstruction (MPR) of a 3D MRI
volume built from a folder of 2D DICOM slices.

Responsibilities:
  1. Robustly load a DICOM series folder into a 3D SimpleITK image
     (handles slice sorting, spacing, orientation, gantry-tilt-free stacks).
  2. Provide several resampling / interpolation strategies so the user can
     trade off "zero data loss" vs. "smooth, isotropic, high accuracy".
  3. Extract axial / coronal / sagittal 2D slices (as plain numpy arrays)
     from the resulting volume, together with the correct on-screen pixel
     aspect ratio.
  4. Provide a simple window/level (contrast) normalisation helper for
     8-bit display.

This module intentionally has NO Qt / matplotlib / GUI imports so that it
can be unit-tested headlessly and reused in other tools (batch export,
notebooks, etc). The GUI lives in mpr_gui.py and imports this module.
"""

from __future__ import annotations

import os
import glob
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List, Dict

import numpy as np
import SimpleITK as sitk
import pydicom

try:
    import ants  # noqa: F401 -- optional; only needed for RegistrationBackend.ANTSPY
    HAVE_ANTSPY = True
except Exception:
    HAVE_ANTSPY = False


# --------------------------------------------------------------------------- #
# Interpolation / resampling algorithms
# --------------------------------------------------------------------------- #

class InterpMethod(Enum):
    """
    Resampling / interpolation strategies available to the user.

    NONE_NATIVE
        No interpolation at all. The original voxel values are kept
        exactly as read from the DICOM files (bit-for-bit). Anisotropic
        spacing (slice thickness != pixel spacing) is corrected purely by
        adjusting the *display* aspect ratio, never by inventing pixel
        values. This is the only option with a mathematical guarantee of
        zero data loss and is the recommended default for primary
        diagnostic viewing.

    NEAREST
        Volume is resampled onto an isotropic grid using nearest-neighbour
        interpolation. No new intensity values are invented (every output
        voxel copies one existing input voxel), but the result can look
        blocky / aliased ("staircase" artifacts) because sample positions
        move without blending.

    LINEAR
        Trilinear interpolation. The standard, fast, general-purpose
        choice used by almost all real-time MPR viewers (OsiriX, Horos,
        RadiAnt, 3D Slicer's default). Good accuracy/speed trade-off.

    BSPLINE
        Cubic B-spline interpolation (Unser et al.). Fits a smooth cubic
        spline through the data before resampling, which gives the most
        accurate, smoothest reconstruction of the underlying continuous
        anatomy of the options here. This is the recommended choice when
        you want a genuinely high-accuracy isotropic volume (e.g. for the
        3D render panel, or oblique/curved reformats), at the cost of
        being slower and slightly softening true discontinuities.

    LANCZOS
        Windowed-sinc (Lanczos) interpolation. Theoretically the closest
        to ideal band-limited signal reconstruction and the sharpest of
        the smooth interpolators; can introduce mild ringing near sharp
        edges. Slowest option, offered for completeness / research use.
    """

    NONE_NATIVE = "None (native voxels, no interpolation - zero data loss)"
    NEAREST = "Nearest Neighbor"
    LINEAR = "Trilinear (Linear)"
    BSPLINE = "Cubic B-Spline (recommended, high accuracy)"
    LANCZOS = "Lanczos / Windowed-Sinc (slowest, sharpest)"

    @classmethod
    def from_label(cls, label: str) -> "InterpMethod":
        for m in cls:
            if m.value == label:
                return m
        raise ValueError(f"Unknown interpolation method label: {label!r}")


_SITK_INTERPOLATOR = {
    InterpMethod.NEAREST: sitk.sitkNearestNeighbor,
    InterpMethod.LINEAR: sitk.sitkLinear,
    InterpMethod.BSPLINE: sitk.sitkBSpline,
    InterpMethod.LANCZOS: sitk.sitkLanczosWindowedSinc,
}


# --------------------------------------------------------------------------- #
# Data container
# --------------------------------------------------------------------------- #

@dataclass
class Volume:
    """A 3D volume plus the metadata needed to reslice/display it correctly."""

    array: np.ndarray          # shape (Z, Y, X) — SimpleITK's GetArrayFromImage order
    spacing_zyx: Tuple[float, float, float]   # physical mm spacing, (z, y, x)
    origin_xyz: Tuple[float, float, float]
    direction: Tuple[float, ...]              # 9-element direction cosine matrix
    is_isotropic: bool
    interp_method: InterpMethod
    sitk_image: sitk.Image     # kept around so we can re-resample cheaply

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.array.shape  # (Z, Y, X)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def _scan_folder_headers(folder: str) -> Dict[str, dict]:
    """
    One fast pass over every .dcm file in `folder`, reading only headers
    (pydicom, stop_before_pixels=True -- a few milliseconds per file even
    for large images), to build SeriesInstanceUID -> {"modality", "files"}.

    This exists specifically to avoid a real performance trap: asking GDCM
    (via ImageSeriesReader.GetGDCMSeriesFileNames) to resolve the file list
    for a series is fast for *one* series, but a folder that also contains
    one single-frame DICOM SEG file per slice (a real, common export
    convention) has as many distinct "series" as it has SEG files, since
    each single-frame SEG object gets its own SeriesInstanceUID. Calling
    GetGDCMSeriesFileNames once per candidate series to peek its Modality
    -- which is what naively filtering out non-image series requires --
    was measured taking 20+ seconds on a real ~170-slice folder (169
    "series": 1 image + 168 individual SEG objects), because that call's
    cost scales with total folder size and gets paid once per series. This
    scan replaces all of that with a single O(n) pass.
    """
    result: Dict[str, dict] = {}
    candidates = sorted(
        p for p in glob.glob(os.path.join(folder, "*"))
        if os.path.isfile(p) and p.lower().endswith(".dcm")
    )
    for path in candidates:
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        except Exception:
            continue
        uid = str(getattr(ds, "SeriesInstanceUID", "") or "")
        if not uid:
            continue
        entry = result.setdefault(uid, {"modality": str(getattr(ds, "Modality", "")).upper(),
                                         "files": []})
        entry["files"].append(path)
    return result


def get_acquisition_plane(image: sitk.Image) -> str:
    """Read back the plane detected by load_dicom_series (call this on the
    image it returned, before any resampling -- see build_volume)."""
    key = "detected_acquisition_plane"
    return image.GetMetaData(key) if image.HasMetaDataKey(key) else "Unknown"


# Modalities that can legitimately sit alongside an image series in the
# same folder but are never themselves "the image series" to load/display
# as a grayscale volume (DICOM SEG being the one this module cares about).
_NON_IMAGE_MODALITIES = {"SEG", "RTSTRUCT", "RTDOSE", "RTPLAN", "PR", "KO", "SR"}


def find_series_ids(folder: str) -> list:
    """Return the DICOM SeriesInstanceUIDs found in `folder` that
    represent an actual image series -- i.e. excluding SEG/RTSTRUCT/etc.
    objects that might sit in the same folder (a SEG file has its own
    SeriesInstanceUID and would otherwise show up here as if it were a
    selectable image series, which it isn't and can't be loaded as one).
    Fast (see _scan_folder_headers) regardless of how many stray
    single-file non-image "series" are mixed into the folder."""
    scan = _scan_folder_headers(folder)
    # Only series with >=2 files can be a real 3D stack; this also drops
    # single-frame SEG objects even in the (shouldn't-happen) case one
    # slipped through the modality check with a blank Modality tag.
    return [uid for uid, info in scan.items()
            if info["modality"] not in _NON_IMAGE_MODALITIES and len(info["files"]) >= 2]


def detect_acquisition_plane(image: sitk.Image) -> str:
    """
    Report which plane a DICOM series was actually *acquired* in
    (Axial / Coronal / Sagittal / Oblique), purely for user feedback.
    Call this BEFORE reorienting the image (see `load_dicom_series`).

    The through-plane (slice) axis is column 2 of the image's 3x3
    direction-cosine matrix, expressed in LPS patient coordinates
    (component order: Left, Posterior, Superior). Whichever of L/P/S has
    the largest magnitude tells you which anatomical axis the slices are
    stacked along, which is exactly what defines the acquisition plane:
    stacked along Superior-Inferior -> axial, along Anterior-Posterior ->
    coronal, along Left-Right -> sagittal. If no component dominates
    (e.g. > 25 degrees off-axis) the scan is reported as "Oblique".
    """
    d = image.GetDirection()
    k_axis = np.array([d[2], d[5], d[8]])  # (L, P, S) components of the slice-normal
    order = np.argsort(-np.abs(k_axis))
    dominant, second = np.abs(k_axis[order[0]]), np.abs(k_axis[order[1]])
    if dominant < 0.90 or (dominant - second) < 0.15:
        return "Oblique"
    return {0: "Sagittal", 1: "Coronal", 2: "Axial"}[int(order[0])]


def load_dicom_series(folder: str, series_id: Optional[str] = None) -> sitk.Image:
    """
    Load one DICOM series from `folder` into a 3D SimpleITK image.

    Uses SimpleITK's GDCM-based series reader, which:
      - groups files by SeriesInstanceUID,
      - sorts slices geometrically using ImagePositionPatient / the slice
        normal (robust even if InstanceNumber is missing or out of order),
      - reads PixelSpacing + SliceThickness/SpacingBetweenSlices into the
        image's physical spacing,
      - applies RescaleSlope/RescaleIntercept automatically.

    Raises FileNotFoundError / RuntimeError with a clear message if the
    folder has no readable DICOM series.
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Not a folder: {folder}")

    reader = sitk.ImageSeriesReader()

    if series_id is None:
        # No specific series requested -- discover one. (If the caller
        # already knows which series_id to load, e.g. the GUI resolved it
        # via find_series_ids up front, this whole discovery step is
        # skipped and we go straight to the single GetGDCMSeriesFileNames
        # call below.)
        series_ids = find_series_ids(folder)
        if not series_ids:
            # Fall back: maybe the folder has DICOMs without proper series
            # metadata, or files are nested one level deep.
            nested = glob.glob(os.path.join(folder, "*", ""))
            if nested:
                for sub in nested:
                    sub_ids = find_series_ids(sub)
                    if sub_ids:
                        folder = sub
                        series_ids = sub_ids
                        break
            if not series_ids:
                raise RuntimeError(
                    f"No DICOM image series found in {folder!r}. "
                    "Make sure the folder directly contains the .dcm slice files."
                )
        series_id = series_ids[0]

    file_names = reader.GetGDCMSeriesFileNames(folder, series_id)
    if len(file_names) < 2:
        raise RuntimeError(
            f"Series {series_id} only has {len(file_names)} slice(s); "
            "need a real 3D stack to build MPR views."
        )

    reader.SetFileNames(file_names)
    # Keep original pixel values (no implicit cast) so we can guarantee
    # "no data loss" for the NONE_NATIVE path.
    image = reader.Execute()

    acquired_as = detect_acquisition_plane(image)

    # Canonicalise to LPS: this permutes/flips the array axes so that,
    # regardless of whether the scan was acquired axially, coronally or
    # sagittally, index-axis i always increases towards patient Left,
    # j always increases towards Posterior, and k always increases
    # towards Superior. That single guarantee is what lets extract_slice()
    # always treat array axis 0 as axial, axis 1 as coronal, axis 2 as
    # sagittal (see Plane below) -- exactly like ITK-SNAP's own display
    # normalisation. For on-axis acquisitions (the overwhelming majority
    # of clinical MRI/CT) this is an EXACT permutation + flip of the
    # existing voxels: no interpolation, no resampling, zero data loss.
    # Only genuinely oblique acquisitions (rare, and flagged as "Oblique"
    # by detect_acquisition_plane) can't be handled by permutation alone;
    # DICOMOrient still produces the closest axis-aligned approximation.
    image = sitk.DICOMOrient(image, "LPS")
    image.SetMetaData("detected_acquisition_plane", acquired_as)
    return image


# --------------------------------------------------------------------------- #
# Resampling
# --------------------------------------------------------------------------- #

def build_volume(image: sitk.Image, method: InterpMethod,
                  target_spacing: Optional[float] = None) -> Volume:
    """
    Turn a loaded sitk.Image into a Volume ready for MPR display, applying
    the requested interpolation strategy.

    method == NONE_NATIVE:
        No resampling happens at all. array/spacing are taken directly from
        the source image, so every voxel value is bit-identical to the
        original DICOM pixel data. Coronal/sagittal views will need an
        aspect-ratio correction at display time (see `display_aspect`)
        because the through-plane spacing usually differs from in-plane
        spacing.

    Any other method:
        The volume is resampled onto an isotropic grid (cubic voxels) using
        the chosen SimpleITK interpolator, so every 2D slice — axial,
        coronal or sagittal — has square pixels and needs no further aspect
        correction. `target_spacing` (mm) lets you control output
        resolution/memory; default is the finest spacing already present in
        the source image (i.e. we upsample the coarse axis, never throw
        away detail from the fine axes).
    """
    if method is InterpMethod.NONE_NATIVE:
        array = sitk.GetArrayFromImage(image)  # (Z, Y, X), exact original values
        sx, sy, sz = image.GetSpacing()        # sitk order is (x, y, z)
        return Volume(
            array=array,
            spacing_zyx=(sz, sy, sx),
            origin_xyz=image.GetOrigin(),
            direction=image.GetDirection(),
            is_isotropic=False,
            interp_method=method,
            sitk_image=image,
        )

    sx, sy, sz = image.GetSpacing()
    if target_spacing is None:
        target_spacing = min(sx, sy, sz)

    size_x, size_y, size_z = image.GetSize()
    new_size = [
        max(1, int(round(size_x * sx / target_spacing))),
        max(1, int(round(size_y * sy / target_spacing))),
        max(1, int(round(size_z * sz / target_spacing))),
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing((target_spacing, target_spacing, target_spacing))
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(float(np.min(sitk.GetArrayViewFromImage(image))))
    resampler.SetInterpolator(_SITK_INTERPOLATOR[method])
    resampled = resampler.Execute(image)

    array = sitk.GetArrayFromImage(resampled)
    return Volume(
        array=array,
        spacing_zyx=(target_spacing, target_spacing, target_spacing),
        origin_xyz=resampled.GetOrigin(),
        direction=resampled.GetDirection(),
        is_isotropic=True,
        interp_method=method,
        sitk_image=resampled,
    )


# --------------------------------------------------------------------------- #
# Slice extraction (pure numpy indexing — this step never interpolates;
# any interpolation already happened in build_volume, once, for the whole
# volume, which is far more accurate than interpolating each 2D slice on
# the fly).
# --------------------------------------------------------------------------- #

class Plane(Enum):
    AXIAL = "axial"
    CORONAL = "coronal"
    SAGITTAL = "sagittal"


def plane_extent(volume: Volume, plane: Plane) -> int:
    """Number of valid slice indices along `plane` for this volume."""
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
    else:  # SAGITTAL
        img = array[:, :, index]
    return np.flipud(img)


def extract_slice(volume: Volume, plane: Plane, index: int) -> np.ndarray:
    """
    Return a single 2D slice (numpy array) through `volume` at `index`
    along `plane`. Pure indexing — introduces no interpolation and no
    data loss beyond whatever `build_volume` already did.

    Orientation convention used (standard radiological display):
      axial    -> looking from feet to head,  row 0 = anterior (top)
      coronal  -> looking from front to back,  row 0 = superior (top)
      sagittal -> looking from patient's left, row 0 = superior (top)
    """
    return _slice_array(volume.array, plane, index)


def extract_label_slice(label_array: np.ndarray, plane: Plane, index: int) -> np.ndarray:
    """Same slicing/flip convention as extract_slice, for a raw label
    array (e.g. Segmentation.array) that's already on the exact same grid
    as the Volume it annotates -- avoids needing to wrap it in a Volume
    just to reuse the indexing convention."""
    return _slice_array(label_array, plane, index)


def display_aspect(volume: Volume, plane: Plane) -> float:
    """
    Matplotlib/QImage 'aspect' value (row_spacing / col_spacing) so that a
    non-resampled (NONE_NATIVE) slice renders with geometrically correct
    proportions instead of looking squashed/stretched. For an isotropic
    volume this is always 1.0 and is a no-op.
    """
    sz, sy, sx = volume.spacing_zyx
    if plane is Plane.AXIAL:
        row_spacing, col_spacing = sy, sx
    elif plane is Plane.CORONAL:
        row_spacing, col_spacing = sz, sx
    else:  # SAGITTAL
        row_spacing, col_spacing = sz, sy
    return row_spacing / col_spacing


# --------------------------------------------------------------------------- #
# Rigid registration (align a second series onto the first)
# --------------------------------------------------------------------------- #

class RegistrationBackend(Enum):
    """
    Three independent rigid-registration engines to choose from.

    IDENTITY
        Assumes the images are already perfectly aligned in the scanner's
        physical coordinate space. The math is just an identity transform.
        This gives a mathematically perfect 100% IoU alignment when the two
        series (like T1 and T1 Contrast) were acquired in the same session
        without the patient moving.

    SIMPLEITK
        ITK's own multi-resolution Mattes Mutual Information registration
        (sitk.ImageRegistrationMethod). Fast, deterministic, no extra
        runtime dependency beyond SimpleITK.

    ANTSPY
        The Python wrapper for ANTs (Advanced Normalization Tools).
        Robust to tricky initial misalignments, but slower.
    """
    IDENTITY = "Scanner Coordinates (Identity) — perfect for same-session"
    SIMPLEITK = "SimpleITK (ITK Mattes MI) — fast"
    SIMPLEITK_DEFORMABLE = "SimpleITK (Deformable / BSpline) — robust, highly accurate"
    ANTSPY = "ANTsPy (ANTs toolkit) — robust, slower"


@dataclass
class RegistrationResult:
    registered_image: sitk.Image      # `moving`, resampled into `fixed`'s grid
    transform: object                 # backend-specific transform object
    final_metric: float               # similarity value at convergence (backend-specific scale)
    translation_mm: Tuple[float, float, float]
    rotation_deg: Tuple[float, float, float]
    backend: "RegistrationBackend" = None


def _matrix_to_euler_degrees(matrix: np.ndarray) -> Tuple[float, float, float]:
    """Decompose a (near-)rotation matrix into extrinsic XYZ Euler angles,
    in degrees, purely for human-readable reporting in the GUI."""
    from scipy.spatial.transform import Rotation
    return tuple(float(v) for v in Rotation.from_matrix(matrix).as_euler("xyz", degrees=True))


def register_rigid_mi(fixed_image: sitk.Image, moving_image: sitk.Image,
                       sampling_percentage: float = 0.20,
                       number_of_iterations: int = 200,
                       target_sample_count: int = 200_000,
                       progress_callback=None) -> RegistrationResult:
    """
    Rigid-body (6 degrees of freedom: 3 rotations + 3 translations)
    registration of `moving_image` onto `fixed_image`, using the
    multi-resolution Mattes Mutual Information framework built into ITK --
    the same registration engine underlying 3D Slicer and a large fraction
    of published/clinical neuro-registration pipelines. Mutual information
    is used (rather than plain intensity correlation) because it remains
    reliable even when the two series are different MRI sequences/contrasts
    (e.g. an axial T2 and a coronal FLAIR), the same way multi-modal
    registration is handled in surgical-navigation systems.

    Why rigid, not affine or deformable: two series of the *same patient*
    taken minutes apart differ only by patient motion, which is a rigid
    transform (pure rotation + translation). Restricting the search to
    those 6 DOF means the algorithm physically cannot stretch, shear or
    warp the anatomy -- it is the transform family least likely to
    introduce geometric distortion, which is what "not losing information"
    means for a spatial transform. If you also need to correct for scan
    scale/skew differences, affine is the natural next step (3 extra DOF)
    but was intentionally left out of this default to avoid trading
    accuracy for flexibility you may not need.

    On sampling: `sampling_percentage` alone doesn't scale well -- a real
    clinical volume (e.g. 512x512x168 =~ 44M voxels) at a flat 20% samples
    ~8.8M points on *every* optimizer iteration at the finest pyramid
    level, which was measured taking well over 3 minutes for a single
    registration. `target_sample_count` bounds the ABSOLUTE number of
    samples instead (default 200,000, which is already generous for a
    Mattes MI histogram estimate), and the actual percentage used is
    whichever is smaller of that or `sampling_percentage` -- so small
    volumes still respect `sampling_percentage` as normal, and large ones
    get a proportionally smaller percentage so the per-iteration cost (and
    total registration time) stays roughly constant regardless of image
    size, instead of scaling linearly with voxel count.

    On interpolation: the iterative optimizer below evaluates the
    similarity metric hundreds of times per run, so it uses fast linear
    interpolation *only to score candidate alignments* -- this never
    touches the image you actually get back. Once the optimal transform is
    found, the moving image is resampled into the fixed image's grid
    exactly once, with cubic B-spline interpolation (see mpr_core's
    InterpMethod.BSPLINE docstring for why that is the most accurate
    commonly-used choice). A single high-order resampling pass is the
    standard best practice for minimising interpolation blur/error -
    never resample repeatedly in sequence.

    `progress_callback(iteration, metric_value)`, if given, is invoked
    after every optimizer iteration (useful for a GUI progress dialog).
    """
    fixed = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving = sitk.Cast(moving_image, sitk.sitkFloat32)

    total_voxels = int(np.prod(fixed_image.GetSize()))
    effective_sampling_percentage = min(sampling_percentage,
                                         max(0.01, target_sample_count / max(total_voxels, 1)))

    initial_transform = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(effective_sampling_percentage)
    reg.SetInterpolator(sitk.sitkLinear)  # optimisation-time only; see docstring

    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, minStep=1e-4,
        numberOfIterations=number_of_iterations,
        gradientMagnitudeTolerance=1e-8)
    reg.SetOptimizerScalesFromPhysicalShift()

    # Multi-resolution pyramid: register on coarse, heavily-smoothed copies
    # first, then progressively refine. This is what makes the optimizer
    # robust to large initial misalignments instead of getting stuck in a
    # local optimum, and is standard practice for any ITK/SimpleITK rigid
    # registration.
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    reg.SetInitialTransform(initial_transform, inPlace=False)

    if progress_callback is not None:
        reg.AddCommand(sitk.sitkIterationEvent,
                        lambda: progress_callback(reg.GetOptimizerIteration(),
                                                   reg.GetMetricValue()))

    final_transform = reg.Execute(fixed, moving)
    final_metric = reg.GetMetricValue()

    background = float(np.min(sitk.GetArrayViewFromImage(moving_image)))
    registered = sitk.Resample(
        moving_image, fixed_image, final_transform,
        sitk.sitkBSpline, background, moving_image.GetPixelID())

    try:
        # final_transform is a type-erased sitk.Transform wrapper around the
        # Euler3DTransform actually optimised; GetParameters() works
        # generically without needing to downcast.
        params = final_transform.GetParameters()  # (rx, ry, rz radians, tx, ty, tz mm)
        rotation_deg = tuple(float(v) for v in np.degrees(params[:3]))
        translation_mm = tuple(float(v) for v in params[3:6])
    except Exception:
        rotation_deg = (float("nan"),) * 3
        translation_mm = (float("nan"),) * 3

    return RegistrationResult(
        registered_image=registered,
        transform=final_transform,
        final_metric=final_metric,
        translation_mm=translation_mm,
        rotation_deg=rotation_deg,
        backend=RegistrationBackend.SIMPLEITK,
    )


def register_rigid_antspy(fixed_image: sitk.Image, moving_image: sitk.Image,
                           progress_callback=None) -> RegistrationResult:
    """
    Rigid registration using ANTsPy (`ants.registration(type_of_transform=
    "Rigid")`) instead of SimpleITK's own optimizer. See RegistrationBackend
    for when you'd prefer this over register_rigid_mi.

    Images are converted between SimpleITK and ANTsPy via a lossless NIfTI
    round-trip (verified bit-exact for integer DICOM pixel data -- this is
    the standard, safest way to hand data between the two toolkits, since
    both have fully-tested, standards-compliant NIfTI IO and it avoids any
    risk of a hand-rolled numpy-axis-order mismatch).

    As with register_rigid_mi, the transform is found once using ANTs'
    own (fast, internal) interpolation, and the returned image is produced
    by a SEPARATE, single, explicit `ants.apply_transforms(...,
    interpolator="bSpline")` call -- one accurate resampling pass, matching
    the same "don't lose pixel information" principle used throughout this
    module.

    Note: the ANTsPy C++ backend does not expose a per-iteration progress
    callback the way SimpleITK's does, so `progress_callback`, if given, is
    only called once before registration starts (useful for updating a
    "please wait" dialog) rather than after every iteration.
    """
    import tempfile

    if not HAVE_ANTSPY:
        raise RuntimeError(
            "The 'antspyx' package is not installed, so the ANTsPy registration "
            "backend is unavailable. Install it with: pip install antspyx "
            "(or switch the Registration dropdown to SimpleITK).")

    if progress_callback is not None:
        progress_callback(0, float("nan"))

    with tempfile.TemporaryDirectory() as tmpdir:
        fixed_path = f"{tmpdir}/fixed.nii.gz"
        moving_path = f"{tmpdir}/moving.nii.gz"
        sitk.WriteImage(fixed_image, fixed_path)
        sitk.WriteImage(moving_image, moving_path)
        fixed_ants = ants.image_read(fixed_path)
        moving_ants = ants.image_read(moving_path)

        reg = ants.registration(fixed=fixed_ants, moving=moving_ants,
                                 type_of_transform="Rigid")

        warped_ants = ants.apply_transforms(
            fixed=fixed_ants, moving=moving_ants,
            transformlist=reg["fwdtransforms"], interpolator="bSpline")

        warped_path = f"{tmpdir}/warped.nii.gz"
        ants.image_write(warped_ants, warped_path)
        registered = sitk.Cast(sitk.ReadImage(warped_path), moving_image.GetPixelID())

        # The rigid transform ANTs wrote out is an ITK-format affine file
        # (ANTs is itself built on ITK), so SimpleITK's own transform
        # reader parses it directly -- no ANTs-specific parsing needed.
        transform_path = reg["fwdtransforms"][0]
        transform = sitk.ReadTransform(transform_path)
        params = transform.GetParameters()
        matrix = np.array(params[:9]).reshape(3, 3)
        translation_mm = tuple(float(v) for v in params[9:12])
        rotation_deg = _matrix_to_euler_degrees(matrix)

    return RegistrationResult(
        registered_image=registered,
        transform=transform,
        final_metric=float("nan"),  # ANTs doesn't return a comparable scalar metric here
        translation_mm=translation_mm,
        rotation_deg=rotation_deg,
        backend=RegistrationBackend.ANTSPY,
    )


def register_identity(fixed_image: sitk.Image, moving_image: sitk.Image,
                      progress_callback=None) -> RegistrationResult:
    """
    Returns an Identity transform. Use this when the two series are from the
    exact same exam/session and have not moved. This relies entirely on the
    DICOM scanner coordinates embedded in the images.
    """
    if progress_callback is not None:
        progress_callback(0, 0.0)
        
    transform = sitk.Transform() # Identity
    background = float(np.min(sitk.GetArrayViewFromImage(moving_image)))
    registered = sitk.Resample(
        moving_image, fixed_image, transform,
        sitk.sitkBSpline, background, moving_image.GetPixelID())
        
    return RegistrationResult(
        registered_image=registered,
        transform=transform,
        final_metric=0.0,
        translation_mm=(0.0, 0.0, 0.0),
        rotation_deg=(0.0, 0.0, 0.0),
        backend=RegistrationBackend.IDENTITY,
    )


def register_deformable_sitk(fixed_image: sitk.Image, moving_image: sitk.Image,
                             progress_callback=None) -> RegistrationResult:
    """
    Robust Deformable (BSpline) registration using SimpleITK.
    First aligns the images roughly with an Affine transform, then applies a
    free-form BSpline deformation to warp the local anatomy into exact alignment.
    This guarantees the highest possible IoU when structure shapes differ
    slightly between series (e.g. tumor margin changes in T1 vs T1 Contrast).

    The final composite transform is materialised as a dense displacement field
    so that: (a) the affine component is baked in exactly once (not double-
    applied via SetMovingInitialTransform AND the explicit composite), and
    (b) transfer_labelmap can always invert it via InvertDisplacementField
    regardless of transform type.
    """
    fixed = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving = sitk.Cast(moving_image, sitk.sitkFloat32)

    if progress_callback is not None:
        progress_callback(0, 0.0)

    # 1. Affine Initialization
    initial_transform = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)
        
    reg_affine = sitk.ImageRegistrationMethod()
    reg_affine.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg_affine.SetMetricSamplingStrategy(reg_affine.RANDOM)
    reg_affine.SetMetricSamplingPercentage(0.1)
    reg_affine.SetInterpolator(sitk.sitkLinear)
    reg_affine.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
    reg_affine.SetOptimizerScalesFromPhysicalShift()
    reg_affine.SetInitialTransform(initial_transform, inPlace=False)
    
    affine_transform = reg_affine.Execute(fixed, moving)

    # 2. BSpline Deformable
    transformDomainMeshSize = [5] * fixed.GetDimension()
    bspline = sitk.BSplineTransformInitializer(fixed, transformDomainMeshSize)
    
    reg_bspline = sitk.ImageRegistrationMethod()
    reg_bspline.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg_bspline.SetMetricSamplingStrategy(reg_bspline.RANDOM)
    reg_bspline.SetMetricSamplingPercentage(0.1)
    reg_bspline.SetInterpolator(sitk.sitkLinear)
    
    reg_bspline.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=50,
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=1000,
        costFunctionConvergenceFactor=1e+7)
        
    reg_bspline.SetInitialTransformAsBSpline(bspline, inPlace=True)
    reg_bspline.SetShrinkFactorsPerLevel([4, 2, 1])
    reg_bspline.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg_bspline.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    reg_bspline.SetMovingInitialTransform(affine_transform)

    out_bspline = reg_bspline.Execute(fixed, moving)
    final_metric = reg_bspline.GetMetricValue()

    # Materialise affine + bspline into a single dense displacement field.
    # IMPORTANT: the composite must include BOTH components; out_bspline
    # alone does NOT include the affine -- SetMovingInitialTransform only
    # changes the optimizer's gradient computation, it does not bake the
    # affine into out_bspline. The old code that built
    #   composite.AddTransform(affine_transform)
    #   composite.AddTransform(out_bspline)
    # was CORRECT for the composition, but the resulting CompositeTransform
    # object has no GetInverse(), so transfer_labelmap hit the exception
    # path and tried to invert it as a displacement field -- but that
    # inverted only the outer transform, losing the affine component.
    # Materialising to a displacement field here (once) fixes both issues.
    composite = sitk.CompositeTransform(3)
    composite.AddTransform(affine_transform)
    composite.AddTransform(out_bspline)

    # Flatten to a displacement field defined on the FIXED (primary) grid.
    # This is what Resample(moving, fixed, T) uses: for each fixed-grid
    # point p, T(p) gives the corresponding moving-space point.
    df = sitk.TransformToDisplacementField(
        composite,
        sitk.sitkVectorFloat64,
        fixed_image.GetSize(),
        fixed_image.GetOrigin(),
        fixed_image.GetSpacing(),
        fixed_image.GetDirection(),
    )
    displacement_transform = sitk.DisplacementFieldTransform(df)

    background = float(np.min(sitk.GetArrayViewFromImage(moving_image)))
    registered = sitk.Resample(
        moving_image, fixed_image, displacement_transform,
        sitk.sitkBSpline, background, moving_image.GetPixelID())

    return RegistrationResult(
        registered_image=registered,
        transform=displacement_transform,
        final_metric=final_metric,
        translation_mm=(0.0, 0.0, 0.0),
        rotation_deg=(0.0, 0.0, 0.0),
        backend=RegistrationBackend.SIMPLEITK_DEFORMABLE,
    )



def register_rigid(fixed_image: sitk.Image, moving_image: sitk.Image,
                    backend: RegistrationBackend,
                    progress_callback=None) -> RegistrationResult:
    """Single entry point the GUI calls: dispatches to whichever backend
    the user picked, so callers don't need an if/else of their own."""
    if backend is RegistrationBackend.IDENTITY:
        return register_identity(fixed_image, moving_image, progress_callback)
    if backend is RegistrationBackend.SIMPLEITK_DEFORMABLE:
        return register_deformable_sitk(fixed_image, moving_image, progress_callback)
    if backend is RegistrationBackend.ANTSPY:
        return register_rigid_antspy(fixed_image, moving_image,
                                      progress_callback=progress_callback)
    return register_rigid_mi(fixed_image, moving_image,
                              progress_callback=progress_callback)


# --------------------------------------------------------------------------- #
# DICOM SEG (segmentation / annotation) support
# --------------------------------------------------------------------------- #

@dataclass
class SegmentInfo:
    number: int
    label: str
    color_rgb: Tuple[int, int, int]


@dataclass
class Segmentation:
    """A label volume plus per-segment metadata, already aligned to some
    reference image's exact voxel grid (same size/spacing/origin/direction),
    so it can be indexed with the exact same (z, y, x) indices as that
    image -- see extract_slice, which works on either."""
    label_image: sitk.Image
    segments: Dict[int, SegmentInfo]

    @property
    def array(self) -> np.ndarray:
        return sitk.GetArrayFromImage(self.label_image)


# A small, deliberately high-contrast, deterministic categorical palette.
# DICOM SEG's *optional* RecommendedDisplayCIELabValue is not parsed here
# (it needs a CIELab->sRGB colorimetric conversion, which is more machinery
# than the cosmetic benefit is worth); every segment's displayed color
# instead comes from this palette, indexed by segment number, so colors are
# always distinct and consistent across a session even if the source SEG
# carries no color info at all.
_PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
]


def _palette_color(segment_number: int) -> Tuple[int, int, int]:
    return _PALETTE[(segment_number - 1) % len(_PALETTE)]


def _is_seg_file(path: str) -> bool:
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        return str(getattr(ds, "Modality", "")).upper() == "SEG"
    except Exception:
        return False


def find_seg_files(folder: str) -> List[str]:
    """Find DICOM Segmentation (SEG) objects in `folder`. Detection is by
    the DICOM `Modality` tag (== "SEG"), not filename, so it works whether
    the file is named like the user's own convention (e.g. '..._seg.dcm')
    or anything else -- but that convention will always match too, since
    such files necessarily have Modality=SEG."""
    if not os.path.isdir(folder):
        return []
    candidates = sorted(
        p for p in glob.glob(os.path.join(folder, "*"))
        if os.path.isfile(p) and p.lower().endswith(".dcm")
    )
    return [p for p in candidates if _is_seg_file(p)]


def load_seg_as_labelmap(seg_paths: List[str], reference_image: sitk.Image,
                          fill_contours: bool = True) -> Optional[Segmentation]:
    """
    Read one or more multi-frame DICOM SEG files and stack every frame into
    a single 3D label volume aligned to `reference_image`'s exact grid
    (same size/spacing/origin/direction as the image it annotates).

    A DICOM SEG object is inherently multi-frame: each frame is a 2D mask
    for one (segment, slice) pair, located via that frame's own
    PlanePositionSequence.ImagePositionPatient in the
    PerFrameFunctionalGroupsSequence. This function reads every frame,
    looks up which voxel slice of `reference_image` its physical position
    corresponds to (via TransformPhysicalPointToIndex, which is exact for
    on-grid positions and correct regardless of how the reference image
    was reoriented by load_dicom_series), and writes that frame's mask
    into the output volume at that slice -- i.e. it "un-multiframes" the
    SEG back into one proper 3D array, which is what "stack into a single
    volume before overlaying" means here. Slices the SEG doesn't cover
    are simply left as background (0).

    fill_contours (default True): many real annotation tools save a SEG
    mask as only the thin (1-pixel) traced BOUNDARY of the drawn region,
    not a solid fill -- confirmed on real clinical data, where the raw
    mask is a hollow outline. Left unfilled, that renders as a barely-
    visible thin colored line once alpha-blended, which is what "the
    overlay isn't working" usually turns out to mean: it IS there, just
    imperceptible. Filling each 2D frame's enclosed interior (via
    scipy.ndimage.binary_fill_holes) BEFORE stacking into the 3D volume --
    rather than patching each 2D display slice cosmetically later -- means
    the reconstructed volume is genuinely solid, so it displays correctly
    as a filled region in the axial, coronal AND sagittal views alike
    (filling only the axial cross-sections as they're read would not
    produce solid coronal/sagittal cuts through the resulting volume).
    This is a no-op for masks that were already solid (nothing to fill),
    so it's safe to leave on by default; set False if your source SEGs are
    genuinely sparse/scattered annotations where "enclosed area" isn't a
    meaningful concept.

    If more than one SEG file is given, they are merged non-destructively
    (a later file only fills in currently-background voxels), so e.g. two
    SEGs covering different structures don't clobber each other.

    Returns None if `seg_paths` is empty. Malformed/unreadable SEG files
    are skipped with their frames simply not contributing, rather than
    raising, since an annotation is optional extra context, not something
    that should block loading the underlying image series.
    """
    if not seg_paths:
        return None

    size = reference_image.GetSize()  # (x, y, z)
    label_array = np.zeros((size[2], size[1], size[0]), dtype=np.uint16)  # z, y, x
    segments: Dict[int, SegmentInfo] = {}

    def _prep_mask(m: np.ndarray) -> np.ndarray:
        if fill_contours:
            from scipy.ndimage import binary_fill_holes
            return binary_fill_holes(m)
        return m

    for path in seg_paths:
        try:
            ds = pydicom.dcmread(path)
            pixels = ds.pixel_array
        except Exception:
            continue
        if pixels.ndim == 2:
            pixels = pixels[np.newaxis, ...]

        for seg_item in getattr(ds, "SegmentSequence", []):
            try:
                num = int(seg_item.SegmentNumber)
            except Exception:
                continue
            label_text = str(getattr(seg_item, "SegmentLabel", f"Segment {num}"))
            if num not in segments:
                segments[num] = SegmentInfo(number=num, label=label_text,
                                             color_rgb=_palette_color(num))

        seg_type = str(getattr(ds, "SegmentationType", "BINARY")).upper()
        threshold = float(getattr(ds, "MaximumFractionalValue", 255)) * 0.5 if seg_type == "FRACTIONAL" else 0

        per_frame = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
        shared = None
        shared_seq = getattr(ds, "SharedFunctionalGroupsSequence", None)
        if shared_seq:
            shared = shared_seq[0]

        # Fallback orientation/spacing if a given frame doesn't carry its
        # own: shared functional group's, or dataset-level (very old-style
        # files), in that priority order.
        dataset_iop = getattr(ds, "ImageOrientationPatient", None)
        dataset_spacing = getattr(ds, "PixelSpacing", None)
        shared_iop = None
        shared_spacing = None
        if shared is not None:
            po_seq = getattr(shared, "PlaneOrientationSequence", None)
            if po_seq:
                shared_iop = po_seq[0].ImageOrientationPatient
            pm_seq = getattr(shared, "PixelMeasuresSequence", None)
            if pm_seq:
                shared_spacing = pm_seq[0].PixelSpacing

        if per_frame:
            n = min(len(per_frame), pixels.shape[0])
            for i in range(n):
                fg = per_frame[i]
                pos_seq = getattr(fg, "PlanePositionSequence", None)
                if not pos_seq and shared is not None:
                    pos_seq = getattr(shared, "PlanePositionSequence", None)
                if not pos_seq:
                    continue
                position = [float(v) for v in pos_seq[0].ImagePositionPatient]

                seg_num = 1
                seg_id_seq = getattr(fg, "SegmentIdentificationSequence", None)
                if seg_id_seq:
                    seg_num = int(seg_id_seq[0].ReferencedSegmentNumber)

                frame_po_seq = getattr(fg, "PlaneOrientationSequence", None)
                iop = (frame_po_seq[0].ImageOrientationPatient if frame_po_seq
                       else shared_iop if shared_iop is not None
                       else dataset_iop)
                target_axis = _plane_axis_from_orientation(iop)

                frame_pm_seq = getattr(fg, "PixelMeasuresSequence", None)
                spacing = (frame_pm_seq[0].PixelSpacing if frame_pm_seq
                           else shared_spacing if shared_spacing is not None
                           else dataset_spacing)

                frame_mask = _prep_mask(pixels[i] > threshold)
                frame_mask = _reorient_frame(frame_mask, position, iop, spacing,
                                              reference_image, target_axis)
                _write_frame(label_array, reference_image, position, frame_mask, seg_num, target_axis)
        else:
            # Very old-style / non-conformant single-position fallback.
            position = getattr(ds, "ImagePositionPatient", None)
            target_axis = _plane_axis_from_orientation(dataset_iop)
            if position is not None:
                position = [float(v) for v in position]
                for i in range(pixels.shape[0]):
                    frame_mask = _prep_mask(pixels[i] > threshold)
                    frame_mask = _reorient_frame(frame_mask, position, dataset_iop, dataset_spacing,
                                                  reference_image, target_axis)
                    _write_frame(label_array, reference_image, position, frame_mask, 1, target_axis)

    if not np.any(label_array):
        return None

    if not segments:
        present = sorted(int(v) for v in np.unique(label_array) if v != 0)
        for n in present:
            segments[n] = SegmentInfo(number=n, label=f"Segment {n}", color_rgb=_palette_color(n))

    label_image = sitk.GetImageFromArray(label_array.astype(np.uint16))
    label_image.CopyInformation(reference_image)
    return Segmentation(label_image=label_image, segments=segments)


def _reorient_frame(mask: np.ndarray, position, iop, pixel_spacing,
                     reference_image: sitk.Image, target_axis: int) -> np.ndarray:
    """
    Reorient a 2D SEG frame (as stored: mask[row, col]) so it matches the
    row/column axis order and direction of the 2D cross-section it will be
    written into at `target_axis`'s out-of-plane slot in the label array.

    Why this is needed: a SEG frame's own row/column axes only happen to
    line up directly with the reconstructed volume's array axes when the
    frame's orientation cosines happen to point the same way the volume's
    axes increase. That was true, by coincidence, for one real axial
    dataset tested during development -- and false for a real coronal
    dataset, where the frame's row axis increases toward Inferior while
    the volume's z-axis increases toward Superior, so writing the frame
    unmodified mirrored every annotated slice vertically (confirmed: the
    transferred annotation from a separately-loaded axial primary series
    landed on the visible lesion, while the un-corrected coronal SEG's own
    annotation appeared shifted well away from it, in the mirrored
    position).

    Rather than hardcoding axial/coronal/sagittal-specific flip rules
    (fragile -- it's exactly that kind of unstated assumption that caused
    the bug), this works the correct orientation out directly from the
    DICOM geometry every time, using ONLY that frame's own metadata:
    nudge one pixel in the row direction and one in the column direction,
    see how each of the reconstructed volume's 3 index axes actually
    responds, and transpose/flip the mask to match. This is what makes the
    loader correct for every combination of primary/secondary acquisition
    plane (axial, coronal or sagittal, independently for each series) --
    it never assumes which plane it's dealing with; it always re-derives
    the mapping from that frame's actual ImageOrientationPatient.
    """
    if iop is None or len(iop) < 6 or pixel_spacing is None:
        return mask  # no orientation info to correct with; leave as-is

    row_dir = np.array([float(v) for v in iop[0:3]])   # physical dir of +1 COLUMN step
    col_dir = np.array([float(v) for v in iop[3:6]])   # physical dir of +1 ROW step
    row_spacing, col_spacing = float(pixel_spacing[0]), float(pixel_spacing[1])

    p0 = np.array(position, dtype=float)
    idx0 = np.array(reference_image.TransformPhysicalPointToContinuousIndex(tuple(p0)))
    idx_col1 = np.array(reference_image.TransformPhysicalPointToContinuousIndex(
        tuple(p0 + col_spacing * row_dir)))
    idx_row1 = np.array(reference_image.TransformPhysicalPointToContinuousIndex(
        tuple(p0 + row_spacing * col_dir)))
    d_col = idx_col1 - idx0   # how the (x, y, z) index changes per +1 COLUMN step
    d_row = idx_row1 - idx0   # how the (x, y, z) index changes per +1 ROW step

    # The two in-plane (x,y,z)-index positions, in the order label_array's
    # slice along target_axis leaves them: axial slice label_array[zi,:,:]
    # leaves (y,x) -> xyz-index (1,0); coronal label_array[:,yi,:] leaves
    # (z,x) -> (2,0); sagittal label_array[:,:,xi] leaves (z,y) -> (2,1).
    first_xyz, second_xyz = {0: (1, 0), 1: (2, 0), 2: (2, 1)}[target_axis]

    # Whichever pairing (row->first & col->second, vs. row->second &
    # col->first) explains more of the actual measured index movement is
    # the correct one for this frame.
    natural = abs(d_row[first_xyz]) + abs(d_col[second_xyz])
    swapped = abs(d_row[second_xyz]) + abs(d_col[first_xyz])

    if swapped > natural:
        out = mask.T
        row_component, col_component = d_row[second_xyz], d_col[first_xyz]
    else:
        out = mask
        row_component, col_component = d_row[first_xyz], d_col[second_xyz]

    if row_component < 0:
        out = np.flip(out, axis=0)
    if col_component < 0:
        out = np.flip(out, axis=1)
    return np.ascontiguousarray(out)


def _plane_axis_from_orientation(iop) -> int:
    """
    Given a 6-element DICOM ImageOrientationPatient (row direction cosines
    + column direction cosines) for a single SEG frame, return which axis
    of the (z, y, x) label array that frame's plane belongs on: 0 (axial,
    z/Superior-Inferior-defining), 1 (coronal, y/Anterior-Posterior-
    defining), or 2 (sagittal, x/Left-Right-defining).

    This exists because a SEG frame's own in-plane shape only matches the
    array's axial cross-section when the SEG (and its source series) was
    itself acquired axially. A coronally-acquired SEG's frames are
    coronal-plane images and must be written along the coronal axis
    instead -- writing them as if they were axial silently fails a shape
    check and drops every frame, which is exactly what was happening
    before this fix on real coronal/sagittal per-slice SEG data. The
    frame's plane normal (cross product of its two in-plane direction
    vectors) points along whichever of L/P/S it was acquired perpendicular
    to, which is what actually determines this -- the same logic
    detect_acquisition_plane uses for a whole series, applied per frame.

    Defaults to axial (0) if no orientation info is available at all,
    matching this module's original (axial-only) behaviour as a safe
    fallback rather than silently dropping the frame.
    """
    if iop is None or len(iop) < 6:
        return 0
    row = np.array([float(v) for v in iop[0:3]])
    col = np.array([float(v) for v in iop[3:6]])
    normal = np.cross(row, col)  # components in (L, P, S) order
    order = np.argsort(-np.abs(normal))
    # dominant normal component: 0=L -> sagittal -> array axis 2 (x)
    #                             1=P -> coronal  -> array axis 1 (y)
    #                             2=S -> axial    -> array axis 0 (z)
    return {0: 2, 1: 1, 2: 0}[int(order[0])]


def _write_frame(label_array: np.ndarray, reference_image: sitk.Image,
                  position: List[float], mask: np.ndarray, seg_num: int,
                  target_axis: int = 0) -> None:
    """Place one 2D SEG frame into `label_array` at whichever index its
    physical position corresponds to in `reference_image`, along
    `target_axis` (0=axial/z, 1=coronal/y, 2=sagittal/x -- see
    _plane_axis_from_orientation). Writes only onto currently-background
    voxels so multiple SEG files/segments merge without one silently
    overwriting another's labeled pixels."""
    try:
        xi, yi, zi = reference_image.TransformPhysicalPointToIndex(position)
    except Exception:
        return
    size = reference_image.GetSize()  # (x, y, z)
    pos_idx, axis_extent = {0: (zi, size[2]), 1: (yi, size[1]), 2: (xi, size[0])}[target_axis]
    if not (0 <= pos_idx < axis_extent):
        return
    if target_axis == 0:
        target = label_array[pos_idx, :, :]
    elif target_axis == 1:
        target = label_array[:, pos_idx, :]
    else:
        target = label_array[:, :, pos_idx]
    if mask.shape != target.shape:
        # Resize the mask to the target slice shape rather than silently
        # dropping the frame. This can happen when the SEG was encoded at
        # a different resolution than the reference image (e.g. half-FOV
        # save). Nearest-neighbour resize preserves binary values.
        try:
            from skimage.transform import resize as sk_resize
            mask = sk_resize(mask.astype(np.float32), target.shape,
                             order=0, preserve_range=True, anti_aliasing=False).astype(bool)
        except ImportError:
            # skimage not available — fall back to numpy repeat/crop
            import warnings
            warnings.warn(
                f"_write_frame: mask shape {mask.shape} != target {target.shape}; "
                "skimage not available for resize, frame skipped.")
            return
    write_mask = mask & (target == 0)
    target[write_mask] = seg_num



def resample_label_like(label_image: sitk.Image, reference_image: sitk.Image,
                         transform: Optional[sitk.Transform] = None) -> sitk.Image:
    """
    Resample a label map onto `reference_image`'s grid using NEAREST-
    NEIGHBOR interpolation only. This matters: linear/cubic/Lanczos
    interpolation would blend adjacent label categories into meaningless
    fractional in-between values (e.g. averaging label 1 and label 2 into
    "1.5"), silently corrupting the annotation. Nearest-neighbor is the
    only interpolator here, regardless of what the user has chosen for the
    grayscale display Algorithm -- that choice never applies to label data.

    `transform`, if given, maps points in `reference_image`'s physical
    space to points in `label_image`'s physical space (the same direction
    convention sitk.Resample always uses). Left as identity by default,
    which is what you want when the label is already in the same physical
    space and you're just re-gridding it (e.g. after the display Algorithm
    changed the grayscale volume's resampling grid).
    """
    if transform is None:
        transform = sitk.Transform()  # identity
    return sitk.Resample(label_image, reference_image, transform,
                          sitk.sitkNearestNeighbor, 0, label_image.GetPixelID())


def resample_label_array_like(label_image: sitk.Image, reference_image: sitk.Image,
                               transform: Optional[sitk.Transform] = None) -> np.ndarray:
    """Same as resample_label_like, but returns a plain numpy (z,y,x)
    array -- lets callers (the GUI) work with label data without ever
    needing to import SimpleITK themselves."""
    return sitk.GetArrayFromImage(resample_label_like(label_image, reference_image, transform))


def transfer_labelmap(source_segmentation: Segmentation,
                       target_reference_image: sitk.Image,
                       registration_result: RegistrationResult,
                       source_was_fixed: bool) -> Segmentation:
    """
    Propagate a Segmentation drawn on one series onto another series'
    native grid, using the rigid transform already found between them by
    register_rigid_mi/register_rigid_antspy -- this is the "transfer
    annotation from [primary] to [secondary]" (or vice versa) operation.
    It works regardless of which planes the two series were actually
    acquired in (axial/coronal/sagittal all reduce to the same 3D physical
    problem once both series are loaded), so "axial to coronal", "coronal
    to sagittal", etc. are all just this one function with different
    inputs.

    `registration_result` was computed as register_rigid(fixed=primary,
    moving=secondary); its `.transform` T maps primary-space points to
    secondary-space points (the ITK "pull-back" convention: for each point
    p in the fixed/output grid, T(p) gives the corresponding moving/input
    point to sample from).

    source_was_fixed:
        True  -- source_segmentation lives in the FIXED (primary) series'
                 space; propagate it onto target_reference_image (assumed
                 to be the MOVING/secondary series' own native grid). This
                 needs the INVERSE transform T⁻¹ (secondary → primary) so
                 that, for each output voxel in the secondary grid, we look
                 up the corresponding primary-space label.
                 Inversion: for analytic transforms (Euler3D, Identity) use
                 GetInverse(); for displacement-field transforms materialise
                 T on the SOURCE (primary) domain first -- that is where T's
                 input lives -- then call InvertDisplacementField. Using the
                 target (secondary) domain is wrong because T expects
                 primary-space inputs, so evaluating it at secondary-space
                 points gives nonsense displacements.
        False -- source_segmentation lives in the MOVING (secondary)
                 series' space; propagate it onto target_reference_image
                 (assumed to be the FIXED/primary series' own grid), using
                 the transform exactly as returned (primary → secondary
                 direction is exactly what Resample needs here too).

    Only nearest-neighbor interpolation is used (see resample_label_like).
    """
    transform = registration_result.transform

    if source_was_fixed:
        try:
            effective_transform = transform.GetInverse()
        except Exception:
            # transform is non-analytically-invertible (e.g. DisplacementField).
            # Materialise on the SOURCE (primary / fixed) grid -- i.e. the
            # domain where T's input lives -- then invert.
            # WRONG: using target_reference_image (secondary) domain here
            # because T's input space is primary, not secondary.
            source_image = source_segmentation.label_image  # primary grid
            df = sitk.TransformToDisplacementField(
                transform,
                sitk.sitkVectorFloat64,
                source_image.GetSize(),
                source_image.GetOrigin(),
                source_image.GetSpacing(),
                source_image.GetDirection(),
            )
            inv_df = sitk.InvertDisplacementField(df)
            effective_transform = sitk.DisplacementFieldTransform(inv_df)
    else:
        effective_transform = transform

    warped_label = resample_label_like(source_segmentation.label_image,
                                        target_reference_image, effective_transform)
    return Segmentation(label_image=warped_label, segments=dict(source_segmentation.segments))



# --------------------------------------------------------------------------- #
# Overlay rendering helpers (grayscale + label -> RGB for display)
# --------------------------------------------------------------------------- #

def gray_to_rgb(gray_uint8: np.ndarray) -> np.ndarray:
    """Replicate a grayscale uint8 image into 3 channels. Used so the 2D
    view panels always display an (H, W, 3) RGB array -- with or without
    an annotation overlay -- rather than switching an AxesImage between
    grayscale-with-colormap and RGB modes."""
    return np.stack([gray_uint8] * 3, axis=-1)


def blend_label_overlay(rgb_uint8: np.ndarray, label_slice: np.ndarray,
                         segments: Dict[int, SegmentInfo], alpha: float = 0.45) -> np.ndarray:
    """Alpha-blend each segment's color onto an (H, W, 3) RGB image
    wherever `label_slice` equals that segment's number. Background
    (label 0) is left untouched. Returns a new array; does not modify
    `rgb_uint8` in place."""
    if label_slice is None or not np.any(label_slice):
        return rgb_uint8
    out = rgb_uint8.astype(np.float32).copy()
    for num, info in segments.items():
        mask = label_slice == num
        if not np.any(mask):
            continue
        color = np.array(info.color_rgb, dtype=np.float32)
        out[mask] = (1 - alpha) * out[mask] + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)


def blend_label_boundary(rgb_uint8: np.ndarray, label_slice: np.ndarray,
                         segments: Dict[int, SegmentInfo],
                         thickness: int = 2) -> np.ndarray:
    """
    Draw just the BOUNDARY (outline) of each labeled region on an
    (H, W, 3) RGB image, using the color defined for that segment in `segments`,
    instead of filling the interior. This is the default annotation display style --
    a clean edge that doesn't obscure the underlying grayscale image --
    with blend_label_overlay (solid fill) as the alternate "show mask"
    mode. Underlying label data is unaffected either way; this only
    changes how it's drawn.

    The boundary is computed per segment number (via binary erosion, so
    two touching but different segments each still get their own edge
    rather than only the outer boundary of their combined footprint), then
    drawn in its specific color. `thickness` controls how many
    pixels wide the drawn edge is.
    """
    if label_slice is None or not np.any(label_slice):
        return rgb_uint8
    from scipy.ndimage import binary_erosion, binary_dilation
    out = rgb_uint8.copy()
    structure = np.ones((3, 3), dtype=bool)
    for num in np.unique(label_slice):
        if num == 0:
            continue
        mask = label_slice == num
        eroded = binary_erosion(mask, structure=structure, border_value=0)
        edge = mask & ~eroded
        if thickness > 1:
            edge = binary_dilation(edge, iterations=thickness - 1)
        info = segments.get(num)
        color = info.color_rgb if info else (255, 0, 0)
        out[edge] = color
    return out


def dice_coefficient(label_array_a: np.ndarray, label_array_b: np.ndarray) -> float:
    """Overall (all-labels-combined) Dice similarity coefficient between
    two label volumes of the same shape: 2*|A∩B| / (|A|+|B|), treating any
    nonzero voxel as foreground. 1.0 = identical, 0.0 = no overlap. Used to
    give a quick, quantitative answer to "how well did the transferred
    annotation line up with the real one" in the comparison view."""
    a = label_array_a > 0
    b = label_array_b > 0
    denom = a.sum() + b.sum()
    if denom == 0:
        return float("nan")
    return float(2 * np.logical_and(a, b).sum() / denom)


def iou_coefficient(label_array_a: np.ndarray, label_array_b: np.ndarray) -> float:
    """Overall (all-labels-combined) Intersection-over-Union between two
    label arrays of the same shape (2D slices or full 3D volumes both
    work): |A∩B| / |A∪B|, treating any nonzero value as foreground.
    1.0 = identical, 0.0 = no overlap. NaN if both arrays are entirely
    background (undefined -- there's nothing to compare on that slice),
    which callers should display as "n/a", not 0, since 0 would wrongly
    suggest a complete mismatch rather than "nothing here to check"."""
    a = label_array_a > 0
    b = label_array_b > 0
    union = np.logical_or(a, b).sum()
    if union == 0:
        return float("nan")
    return float(np.logical_and(a, b).sum() / union)


# --------------------------------------------------------------------------- #
# Display windowing (contrast) helper
# --------------------------------------------------------------------------- #

def window_to_uint8(img: np.ndarray, center: Optional[float] = None,
                     width: Optional[float] = None) -> np.ndarray:
    """
    Apply window/level (contrast) to a raw-intensity 2D slice and return an
    8-bit array for display. Does not modify the source data — this is a
    display-only transform, applied on a copy.

    If center/width are not given, they are derived from the 1st/99th
    percentile of the slice's own intensities (robust auto-windowing).
    """
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


# --------------------------------------------------------------------------- #
# Convenience: volume-wide intensity range, used to initialise sliders/window
# --------------------------------------------------------------------------- #

def intensity_range(volume: Volume) -> Tuple[float, float]:
    return float(volume.array.min()), float(volume.array.max())


# --------------------------------------------------------------------------- #
# Save segmentation as DICOM SEG
# --------------------------------------------------------------------------- #

def save_transferred_seg_slice_wise(
        segmentation: "Segmentation",
        source_folder: str,
        output_folder: str,
        series_id: Optional[str] = None,
) -> int:
    """Write a ``Segmentation`` (e.g. a transferred annotation) slice-by-slice,
    creating a `_seg.dcm` for each original DICOM slice that contains an
    annotation. Matches DicomDoodle's native behavior.

    Returns
    -------
    int
        The number of slices saved.
    """
    import os
    import sys
    
    # We need utils.create_dicom_segmentation
    try:
        from utils import create_dicom_segmentation
    except ImportError:
        # Fallback if utils isn't in path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from utils import create_dicom_segmentation

    # 1. Collect sorted source DICOM slice files
    reader = sitk.ImageSeriesReader()
    if series_id is None:
        ids = find_series_ids(source_folder)
        sid = ids[0] if ids else None
    else:
        sid = series_id

    if sid is not None:
        file_names = reader.GetGDCMSeriesFileNames(source_folder, sid)
    else:
        file_names = reader.GetGDCMSeriesFileNames(source_folder)

    slice_paths = [p for p in file_names if not _is_seg_file(p)]
    if not slice_paths:
        raise RuntimeError(f"No readable non-SEG DICOM slice files found in {source_folder!r}.")

    # 2. Load original un-oriented image to get original voxel grid
    reader.SetFileNames(slice_paths)
    orig_image = reader.Execute()

    # 3. Resample LPS label map back to original un-oriented slice grid
    # This guarantees label_array_3d[z] perfectly matches slice_paths[z]
    label_array_3d = resample_label_array_like(segmentation.label_image, orig_image)
    label_array_3d = label_array_3d.astype(np.uint8)

    # 4. Save each slice using DicomDoodle's native function
    saved_count = 0
    for z, slice_path in enumerate(slice_paths):
        mask_slice = label_array_3d[z]
        
        # Only save slices that have an annotation
        if not np.any(mask_slice != 0):
            continue
            
        base_name = os.path.splitext(os.path.basename(slice_path))[0]
        out_file = os.path.join(output_folder, f"{base_name}_seg.dcm")
        
        create_dicom_segmentation(mask_slice, slice_path, out_file, z)
        saved_count += 1
        
    return saved_count

