"""
The 3D view (VTK) of the Multiplanar & 3D Viewer: the volume, its transfer
functions, colormaps, clipping planes and lights — the rendering logic of the
original 3D Model Viewer (``update_anatomy_visualization`` /
``update_volume_visualization``, written once). The quadrant around it
(mpr/panels.py ThreeDPanel) adds the header and its buttons.

Mouse: left-drag rotates, right-drag / wheel zooms, middle-drag pans.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PyQt5.QtWidgets import QFileDialog, QVBoxLayout, QWidget

import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from .controls import RenderSettings

vtk.vtkObject.GlobalWarningDisplayOff()

# (fraction of the scalar range, r, g, b) control points — the original's 13 colormaps.
COLORMAPS: Dict[str, List[Tuple[float, float, float, float]]] = {
    "Grayscale": [(0, 0, 0, 0), (0.1, 0.2, 0.2, 0.2), (0.5, 0.5, 0.5, 0.5), (1, 1, 1, 1)],
    "Bone": [(0, 0, 0, 0), (0.1, 0.8, 0.8, 0.8), (0.5, 0.9, 0.9, 0.8), (1, 1, 1, 0.9)],
    "Hot": [(0, 0, 0, 0), (0.1, 1, 0, 0), (0.5, 1, 1, 0), (1, 1, 1, 1)],
    "Jet": [(0, 0, 0, 1), (0.33, 0, 1, 1), (0.66, 1, 1, 0), (1, 1, 0, 0)],
    "CoolWarm": [(0, 0, 0, 1), (0.5, 1, 1, 1), (1, 1, 0, 0)],
    "Viridis": [(0, 0.267, 0, 0.329), (0.25, 0, 0.247, 0.541), (0.5, 0, 0.506, 0.529),
                (0.75, 0.471, 0.886, 0.298), (1, 0.992, 0.906, 0.144)],
    "Magma": [(0, 0, 0, 0), (0.25, 0.188, 0.071, 0.271), (0.5, 0.557, 0.173, 0.408),
              (0.75, 0.969, 0.471, 0.349), (1, 1, 0.988, 0.706)],
    "Rainbow": [(0, 1, 0, 0), (0.2, 1, 0.647, 0), (0.4, 1, 1, 0), (0.6, 0, 1, 0), (0.8, 0, 0, 1),
                (1, 0.294, 0, 0.51)],
    "Spectral": [(0, 0.619, 0, 0.278), (0.25, 0.957, 0.427, 0.263), (0.5, 1, 1, 0.749),
                 (0.75, 0, 0.557, 0.58), (1, 0, 0, 0.502)],
    "BrainTumorSpecific": [(0, 0.2, 0.2, 0.2), (0.5, 0.5, 0.5, 0.5), (0.9, 0.8, 0.8, 0.8), (1, 1, 0, 0)],
    "Parula": [(0, 0.208, 0.169, 0.647), (0.25, 0, 0.408, 0.784), (0.5, 0, 0.659, 0.663),
               (0.75, 0.631, 0.851, 0), (1, 0.996, 0.878, 0)],
    "Ocean": [(0, 0, 0, 0.2), (0.25, 0, 0.2, 0.4), (0.5, 0, 0.4, 0.6), (0.75, 0, 0.6, 0.4), (1, 0, 0.8, 0.2)],
    "Autumn": [(0, 1, 0, 0), (0.5, 1, 0.5, 0), (1, 1, 1, 0)],
}


def color_function(name: str, smin: float, smax: float):
    function = vtk.vtkColorTransferFunction()
    delta = (smax - smin) or 1.0
    for fraction, r, g, b in COLORMAPS.get(name, COLORMAPS["Grayscale"]):
        function.AddRGBPoint(smin + fraction * delta, r, g, b)
    return function


class VolumeView(QWidget):
    """VTK render window for one volume."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.vtk_widget, 1)

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.06, 0.07, 0.09)
        self.renderer.SetBackground2(0.16, 0.18, 0.22)
        self.renderer.GradientBackgroundOn()
        self.render_window = self.vtk_widget.GetRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.interactor = self.render_window.GetInteractor()
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

        self.volume = vtk.vtkVolume()
        self.has_data = False
        self.data_range = (0.0, 1.0)
        self.planes = {axis: vtk.vtkPlane() for axis in "xyz"}
        self.lights: List = []
        self.surface_actor = None
        self.metrics_actor = None
        self._importer = None
        self.message = vtk.vtkTextActor()             # shown while there is nothing to render
        self.message.GetTextProperty().SetJustificationToCentered()
        self.message.GetTextProperty().SetVerticalJustificationToCentered()
        self.message.GetTextProperty().SetFontSize(15)
        self.message.GetTextProperty().SetColor(0.62, 0.66, 0.72)
        self.message.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        self.message.SetPosition(0.5, 0.5)
        self.renderer.AddViewProp(self.message)
        self.vtk_widget.Initialize()

    # ------------------------------------------------------------------ data
    def set_message(self, text: str) -> None:
        """Text in the middle of the empty view (before the first volume)."""
        self.message.SetInput(text)
        self.render()

    def set_volume(self, array: np.ndarray, spacing: Sequence[float], origin: Sequence[float],
                   reset_camera: bool = True) -> None:
        """Show ``array`` (z, y, x). ``reset_camera`` False keeps the view, for
        another volume on the same grid."""
        self.renderer.RemoveAllViewProps()
        nz, ny, nx = array.shape
        importer = vtk.vtkImageImport()
        data = np.ascontiguousarray(array, dtype=np.float32).tobytes("C")
        importer.CopyImportVoidPointer(data, len(data))
        importer.SetDataScalarTypeToFloat()
        importer.SetNumberOfScalarComponents(1)
        importer.SetDataExtent(0, nx - 1, 0, ny - 1, 0, nz - 1)
        importer.SetWholeExtent(0, nx - 1, 0, ny - 1, 0, nz - 1)
        importer.SetDataSpacing(tuple(spacing))
        importer.SetDataOrigin(tuple(origin))
        importer.Update()
        self._importer = importer
        mapper = vtk.vtkSmartVolumeMapper()
        mapper.SetInputData(importer.GetOutput())
        self.volume = vtk.vtkVolume()
        self.volume.SetMapper(mapper)
        self.renderer.AddVolume(self.volume)
        self.data_range = (float(array.min()), float(array.max()))
        self.has_data = True
        if reset_camera:
            self.reset_camera()

    def set_lesion(self, surface, metrics_text: str) -> None:
        """Red lesion surface + metrics text (shown with 'measurement').

        The surface is opaque: VTK draws volumes after translucent geometry, so
        a see-through surface (the original used 40 % opacity) is painted over
        by the volume and never shows. Opaque, it is hidden only by tissue in
        front of it — cut the model open with a clipping plane to see it.
        """
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(surface)
        mapper.ScalarVisibilityOff()
        self.surface_actor = vtk.vtkActor()
        self.surface_actor.SetMapper(mapper)
        self.surface_actor.GetProperty().SetColor(1.0, 0.0, 0.0)
        self.metrics_actor = vtk.vtkCornerAnnotation()
        self.metrics_actor.SetLinearFontScaleFactor(2)
        self.metrics_actor.SetNonlinearFontScaleFactor(1)
        self.metrics_actor.SetMaximumFontSize(20)
        self.metrics_actor.SetText(vtk.vtkCornerAnnotation.UpperRight, metrics_text)
        self.metrics_actor.GetTextProperty().SetColor(1.0, 1.0, 0.0)
        self.metrics_actor.GetTextProperty().BoldOn()

    # ------------------------------------------------------------ appearance
    def apply(self, s: RenderSettings) -> None:
        if not self.has_data:
            return
        smin, smax = self.data_range
        if smin == smax:
            smax = smin + 1
        smin, smax = max(smin, s.scalar_min), min(smax, s.scalar_max)
        if smin >= smax:
            smin = s.scalar_min
            smax = s.scalar_max if s.scalar_max > s.scalar_min else s.scalar_min + 1

        threshold = smin + s.opacity_threshold * (smax - smin)
        contrast_range = (smax - smin) * (1.0 - s.contrast * 0.8)
        opacity = vtk.vtkPiecewiseFunction()
        opacity.AddPoint(smin, 0.0)
        opacity.AddPoint(threshold, 0.0)
        opacity.AddPoint(threshold + 0.1 * contrast_range, 0.05)
        opacity.AddPoint(threshold + 0.3 * contrast_range, 0.3 * s.tissue_density)
        opacity.AddPoint(smin + contrast_range, 0.8 * s.tissue_density)
        gradient = vtk.vtkPiecewiseFunction()
        gradient.AddPoint(0, 0.0)
        gradient.AddPoint(10, s.gradient_opacity * (1.0 + s.edge_enhancement))
        gradient.AddPoint(255, s.gradient_opacity * (1.0 + s.edge_enhancement))

        prop = self.volume.GetProperty()
        prop.SetColor(color_function(s.colormap, smin, smax))
        prop.SetScalarOpacity(opacity)
        prop.SetGradientOpacity(gradient)
        prop.SetAmbient(s.ambience)
        prop.SetDiffuse(0.8)
        prop.SetSpecular(0.1 + s.edge_enhancement * 0.2)
        prop.SetSpecularPower(s.specular_power)
        prop.SetShade(1 if s.shading else 0)

        for actor in (self.surface_actor, self.metrics_actor):
            if actor is None:
                continue
            if s.measurement and not self.renderer.HasViewProp(actor):
                self.renderer.AddViewProp(actor)
            elif not s.measurement and self.renderer.HasViewProp(actor):
                self.renderer.RemoveViewProp(actor)

        mapper = self.volume.GetMapper()
        mapper.RemoveAllClippingPlanes()
        bounds = self.volume.GetBounds()
        for i, (axis, enabled, position) in enumerate((("x", s.clip_x, s.clip_x_pos), ("y", s.clip_y, s.clip_y_pos),
                                                       ("z", s.clip_z, s.clip_z_pos))):
            if not enabled:
                continue
            lo, hi = bounds[2 * i], bounds[2 * i + 1]
            origin, normal = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
            origin[i] = lo + position * (hi - lo)
            normal[i] = 1.0
            self.planes[axis].SetOrigin(*origin)
            self.planes[axis].SetNormal(*normal)
            mapper.AddClippingPlane(self.planes[axis])
        self.render()

    def apply_lighting(self, s: RenderSettings) -> None:
        for light in self.lights:
            self.renderer.RemoveLight(light)
        self.lights.clear()
        az, el = math.radians(s.light_azimuth), math.radians(s.light_elevation)
        key = vtk.vtkLight()
        key.SetPosition(math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))
        key.SetFocalPoint(0, 0, 0)
        key.SetIntensity(s.light_intensity)
        fill = vtk.vtkLight()
        fill.SetPosition(0, 1, 0)
        fill.SetFocalPoint(0, 0, 0)
        fill.SetIntensity(s.light_intensity * 0.4)
        for light in (key, fill):
            self.renderer.AddLight(light)
            self.lights.append(light)
        self.render()

    # ---------------------------------------------------------------- camera
    def render(self) -> None:
        self.render_window.Render()

    def reset_camera(self) -> None:
        """Back to the starting view: along the slice axis, the whole volume in view (VTK's ResetCamera
        alone keeps the current viewing direction)."""
        camera = self.renderer.GetActiveCamera()
        camera.SetFocalPoint(0.0, 0.0, 0.0)
        camera.SetPosition(0.0, 0.0, 1.0)
        camera.SetViewUp(0.0, 1.0, 0.0)
        self.renderer.ResetCamera()
        self.render()

    def save_screenshot(self, path: Optional[str] = None) -> None:
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, "Save screenshot", "3d_view.png", "PNG image (*.png)")
            if not path:
                return
        grabber = vtk.vtkWindowToImageFilter()
        grabber.SetInput(self.render_window)
        grabber.ReadFrontBufferOff()
        grabber.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(path)
        writer.SetInputConnection(grabber.GetOutputPort())
        writer.Write()

    def shutdown(self) -> None:
        try:
            self.render_window.Finalize()
            self.interactor.TerminateApp()
        except Exception:
            pass
