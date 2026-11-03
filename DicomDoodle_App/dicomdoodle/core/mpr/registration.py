"""
Aligning a second series onto the primary one (from the original mpr_core.py).

Four engines:
    IDENTITY              scanner coordinates as-is (same session, no motion)
    SIMPLEITK             rigid, multi-resolution Mattes mutual information
    SIMPLEITK_DEFORMABLE  affine + B-spline, materialised as a displacement field
    ANTSPY                rigid with ANTs (optional ``antspyx`` package)

In every case the optimiser uses fast linear interpolation only to score
candidate alignments; the moving image is resampled once at the end with
cubic B-spline interpolation.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np
import SimpleITK as sitk

# optional, only for RegistrationBackend.ANTSPY; imported when used (importing it takes about a second)
HAVE_ANTSPY = importlib.util.find_spec("ants") is not None


class RegistrationBackend(Enum):
    IDENTITY = "Scanner Coordinates (Identity) — perfect for same-session"
    SIMPLEITK = "SimpleITK (ITK Mattes MI) — fast"
    SIMPLEITK_DEFORMABLE = "SimpleITK (Deformable / BSpline) — robust, highly accurate"
    ANTSPY = "ANTsPy (ANTs toolkit) — robust, slower"


@dataclass
class RegistrationResult:
    registered_image: sitk.Image          # moving image resampled into the fixed grid
    transform: object
    final_metric: float
    translation_mm: Tuple[float, float, float]
    rotation_deg: Tuple[float, float, float]
    backend: Optional[RegistrationBackend] = None


def _matrix_to_euler_degrees(matrix: np.ndarray) -> Tuple[float, float, float]:
    from scipy.spatial.transform import Rotation
    return tuple(float(v) for v in Rotation.from_matrix(matrix).as_euler("xyz", degrees=True))


def register_rigid_mi(fixed_image: sitk.Image, moving_image: sitk.Image, sampling_percentage: float = 0.20,
                      number_of_iterations: int = 200, target_sample_count: int = 200_000,
                      progress_callback=None) -> RegistrationResult:
    """Rigid (6 DOF) Mattes-MI registration; the sample count is capped so
    large volumes don't make every iteration slower."""
    fixed = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving = sitk.Cast(moving_image, sitk.sitkFloat32)
    total_voxels = int(np.prod(fixed_image.GetSize()))
    sampling = min(sampling_percentage, max(0.01, target_sample_count / max(total_voxels, 1)))

    initial = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(),
                                                sitk.CenteredTransformInitializerFilter.GEOMETRY)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(sampling)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0, minStep=1e-4,
                                                 numberOfIterations=number_of_iterations,
                                                 gradientMagnitudeTolerance=1e-8)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(initial, inPlace=False)
    if progress_callback is not None:
        reg.AddCommand(sitk.sitkIterationEvent,
                       lambda: progress_callback(reg.GetOptimizerIteration(), reg.GetMetricValue()))

    final_transform = reg.Execute(fixed, moving)
    final_metric = reg.GetMetricValue()
    background = float(np.min(sitk.GetArrayViewFromImage(moving_image)))
    registered = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkBSpline, background,
                               moving_image.GetPixelID())
    try:
        params = final_transform.GetParameters()   # rx, ry, rz (rad), tx, ty, tz (mm)
        rotation = tuple(float(v) for v in np.degrees(params[:3]))
        translation = tuple(float(v) for v in params[3:6])
    except Exception:
        rotation = translation = (float("nan"),) * 3
    return RegistrationResult(registered, final_transform, final_metric, translation, rotation,
                              RegistrationBackend.SIMPLEITK)


def register_rigid_antspy(fixed_image: sitk.Image, moving_image: sitk.Image,
                          progress_callback=None) -> RegistrationResult:
    """Rigid registration with ANTsPy (images exchanged through lossless NIfTI)."""
    import tempfile

    if not HAVE_ANTSPY:
        raise RuntimeError("The 'antspyx' package is not installed, so the ANTsPy registration backend is "
                           "unavailable. Install it with: pip install antspyx (or choose SimpleITK).")
    import ants

    if progress_callback is not None:
        progress_callback(0, float("nan"))
    with tempfile.TemporaryDirectory() as tmpdir:
        fixed_path, moving_path = f"{tmpdir}/fixed.nii.gz", f"{tmpdir}/moving.nii.gz"
        sitk.WriteImage(fixed_image, fixed_path)
        sitk.WriteImage(moving_image, moving_path)
        fixed_ants, moving_ants = ants.image_read(fixed_path), ants.image_read(moving_path)
        reg = ants.registration(fixed=fixed_ants, moving=moving_ants, type_of_transform="Rigid")
        warped = ants.apply_transforms(fixed=fixed_ants, moving=moving_ants,
                                       transformlist=reg["fwdtransforms"], interpolator="bSpline")
        warped_path = f"{tmpdir}/warped.nii.gz"
        ants.image_write(warped, warped_path)
        registered = sitk.Cast(sitk.ReadImage(warped_path), moving_image.GetPixelID())
        transform = sitk.ReadTransform(reg["fwdtransforms"][0])
        params = transform.GetParameters()
        translation = tuple(float(v) for v in params[9:12])
        rotation = _matrix_to_euler_degrees(np.array(params[:9]).reshape(3, 3))
    return RegistrationResult(registered, transform, float("nan"), translation, rotation,
                              RegistrationBackend.ANTSPY)


def register_identity(fixed_image: sitk.Image, moving_image: sitk.Image,
                      progress_callback=None) -> RegistrationResult:
    """No movement: rely on the scanner coordinates of both series."""
    if progress_callback is not None:
        progress_callback(0, 0.0)
    transform = sitk.Transform()
    background = float(np.min(sitk.GetArrayViewFromImage(moving_image)))
    registered = sitk.Resample(moving_image, fixed_image, transform, sitk.sitkBSpline, background,
                               moving_image.GetPixelID())
    return RegistrationResult(registered, transform, 0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                              RegistrationBackend.IDENTITY)


def register_deformable_sitk(fixed_image: sitk.Image, moving_image: sitk.Image,
                             progress_callback=None) -> RegistrationResult:
    """Affine initialisation + B-spline deformation, flattened into one
    displacement field on the fixed grid (so it can always be inverted)."""
    fixed = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving = sitk.Cast(moving_image, sitk.sitkFloat32)
    if progress_callback is not None:
        progress_callback(0, 0.0)

    initial = sitk.CenteredTransformInitializer(fixed, moving, sitk.AffineTransform(3),
                                                sitk.CenteredTransformInitializerFilter.GEOMETRY)
    reg_affine = sitk.ImageRegistrationMethod()
    reg_affine.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg_affine.SetMetricSamplingStrategy(reg_affine.RANDOM)
    reg_affine.SetMetricSamplingPercentage(0.1)
    reg_affine.SetInterpolator(sitk.sitkLinear)
    reg_affine.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
    reg_affine.SetOptimizerScalesFromPhysicalShift()
    reg_affine.SetInitialTransform(initial, inPlace=False)
    affine_transform = reg_affine.Execute(fixed, moving)

    bspline = sitk.BSplineTransformInitializer(fixed, [5] * fixed.GetDimension())
    reg_bspline = sitk.ImageRegistrationMethod()
    reg_bspline.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg_bspline.SetMetricSamplingStrategy(reg_bspline.RANDOM)
    reg_bspline.SetMetricSamplingPercentage(0.1)
    reg_bspline.SetInterpolator(sitk.sitkLinear)
    reg_bspline.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=50,
                                     maximumNumberOfCorrections=5, maximumNumberOfFunctionEvaluations=1000,
                                     costFunctionConvergenceFactor=1e+7)
    reg_bspline.SetInitialTransformAsBSpline(bspline, inPlace=True)
    reg_bspline.SetShrinkFactorsPerLevel([4, 2, 1])
    reg_bspline.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg_bspline.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg_bspline.SetMovingInitialTransform(affine_transform)
    out_bspline = reg_bspline.Execute(fixed, moving)
    final_metric = reg_bspline.GetMetricValue()

    composite = sitk.CompositeTransform(3)
    composite.AddTransform(affine_transform)
    composite.AddTransform(out_bspline)
    field = sitk.TransformToDisplacementField(composite, sitk.sitkVectorFloat64, fixed_image.GetSize(),
                                              fixed_image.GetOrigin(), fixed_image.GetSpacing(),
                                              fixed_image.GetDirection())
    displacement = sitk.DisplacementFieldTransform(field)
    background = float(np.min(sitk.GetArrayViewFromImage(moving_image)))
    registered = sitk.Resample(moving_image, fixed_image, displacement, sitk.sitkBSpline, background,
                               moving_image.GetPixelID())
    return RegistrationResult(registered, displacement, final_metric, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                              RegistrationBackend.SIMPLEITK_DEFORMABLE)


def register_rigid(fixed_image: sitk.Image, moving_image: sitk.Image, backend: RegistrationBackend,
                   progress_callback=None) -> RegistrationResult:
    """Single entry point: dispatch to the chosen engine."""
    if backend is RegistrationBackend.IDENTITY:
        return register_identity(fixed_image, moving_image, progress_callback)
    if backend is RegistrationBackend.SIMPLEITK_DEFORMABLE:
        return register_deformable_sitk(fixed_image, moving_image, progress_callback)
    if backend is RegistrationBackend.ANTSPY:
        return register_rigid_antspy(fixed_image, moving_image, progress_callback=progress_callback)
    return register_rigid_mi(fixed_image, moving_image, progress_callback=progress_callback)
