"""
Lesion shape and texture metrics of the 3D model (Multiplanar & 3D Viewer;
port of the original ``_calculate_tumor_features``):

  * Volume (cc)          voxel count x voxel volume
  * Surface Area (mm²)   marching-cubes surface of the mask
  * Sphericity           pi^(1/3) (6V)^(2/3) / A   (1.0 = perfect sphere)
  * Texture Contrast / Homogeneity  (GLCM, on the slice with the largest
    lesion area; needs scikit-image, otherwise "N/A")

Returns the metrics and the lesion surface (vtkPolyData) for display.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple, Union

import numpy as np

Metric = Union[float, str]


def compute_tumor_metrics(mask: np.ndarray, spacing, origin, anatomy: np.ndarray) -> Tuple[Dict[str, Metric], object]:
    import vtk

    metrics: Dict[str, Metric] = {}
    voxel_cc = spacing[0] * spacing[1] * spacing[2] / 1000.0
    volume_cc = int(np.count_nonzero(mask)) * voxel_cc
    metrics["Volume (cc)"] = volume_cc

    importer = vtk.vtkImageImport()
    data = np.ascontiguousarray(mask, dtype=np.uint8).tobytes("C")
    importer.CopyImportVoidPointer(data, len(data))
    importer.SetDataScalarTypeToUnsignedChar()
    importer.SetNumberOfScalarComponents(1)
    z, y, x = mask.shape
    importer.SetDataExtent(0, x - 1, 0, y - 1, 0, z - 1)
    importer.SetWholeExtent(0, x - 1, 0, y - 1, 0, z - 1)
    importer.SetDataSpacing(spacing)
    importer.SetDataOrigin(origin)
    importer.Update()

    cubes = vtk.vtkMarchingCubes()
    cubes.SetInputConnection(importer.GetOutputPort())
    cubes.SetValue(0, 1.0)
    cubes.ComputeNormalsOn()
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputConnection(cubes.GetOutputPort())
    connectivity.SetExtractionModeToAllRegions()
    connectivity.Update()

    mass = vtk.vtkMassProperties()
    mass.SetInputConnection(connectivity.GetOutputPort())
    area = mass.GetSurfaceArea()
    metrics["Surface Area (mm²)"] = area
    metrics["Sphericity"] = ((math.pi ** (1 / 3) * (6 * volume_cc * 1000.0) ** (2 / 3)) / area) if area > 0 else 0.0

    try:
        from skimage.feature import graycomatrix, graycoprops
        areas = np.sum(mask, axis=(1, 2))
        if np.max(areas) > 0:
            best = int(np.argmax(areas))
            image = anatomy[best]
            lo, hi = float(np.min(image)), float(np.max(image))
            image8 = ((image - lo) * (255.0 / (hi - lo))).astype(np.uint8) if hi > lo else np.zeros_like(image, np.uint8)
            glcm = graycomatrix(image8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            metrics["Texture Contrast"] = float(graycoprops(glcm, "contrast")[0, 0])
            metrics["Texture Homogeneity"] = float(graycoprops(glcm, "homogeneity")[0, 0])
    except ImportError:
        metrics["Texture Contrast"] = "N/A"
        metrics["Texture Homogeneity"] = "N/A"
    except Exception:
        metrics["Texture Contrast"] = "N/A"
        metrics["Texture Homogeneity"] = "N/A"

    surface = vtk.vtkPolyData()
    surface.DeepCopy(connectivity.GetOutput())
    return metrics, surface


def format_metric(name: str, value: Metric) -> str:
    if not isinstance(value, float):
        return str(value)
    if name == "Volume (cc)":
        return f"{value:.3f} cc"
    if name == "Surface Area (mm²)":
        return f"{value:.2f} mm²"
    return f"{value:.3f}"
