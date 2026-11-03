"""
Settings and controls of a 3D view.

:class:`RenderSettings` holds every option of the original visualization
dialogs (same names, ranges and defaults). :class:`ControlsPanel` edits one
RenderSettings and emits ``changed`` — one panel per view, shown as tabs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout, QHBoxLayout,
                             QLabel, QSlider, QVBoxLayout, QWidget)

from ...qt_utils import label, make_button
from ...widgets.cards import Card

COLORMAP_NAMES = ["Grayscale", "Bone", "Hot", "Jet", "CoolWarm", "Viridis", "Magma", "Rainbow", "Spectral",
                  "BrainTumorSpecific", "Parula", "Ocean", "Autumn"]


@dataclass
class RenderSettings:
    contrast: float = 0.5
    ambience: float = 0.2
    opacity_threshold: float = 0.0
    gradient_opacity: float = 0.5
    specular_power: int = 10
    scalar_min: float = -10000.0
    scalar_max: float = 10000.0
    colormap: str = "Grayscale"
    shading: bool = True
    edge_enhancement: float = 0.2
    tissue_density: float = 0.5
    annotations: bool = False
    measurement: bool = False          # show the lesion surface + metrics (anatomy view)
    clip_x: bool = False
    clip_x_pos: float = 0.5
    clip_y: bool = False
    clip_y_pos: float = 0.5
    clip_z: bool = False
    clip_z_pos: float = 0.5
    light_azimuth: int = 30
    light_elevation: int = 30
    light_intensity: float = 0.6


class _Slider(QWidget):
    """Slider + live value label."""

    valueChanged = pyqtSignal(int)

    def __init__(self, lo: int, hi: int, value: int, suffix: str = "", parent=None):
        super().__init__(parent)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(lo, hi)
        self.slider.setValue(value)
        self.text = QLabel()
        self.text.setObjectName("Muted")
        self.text.setMinimumWidth(42)
        self.text.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.suffix = suffix
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.text)
        self.slider.valueChanged.connect(self._changed)
        self._changed(value)

    def _changed(self, value: int) -> None:
        self.text.setText(f"{value}{self.suffix}")
        self.valueChanged.emit(value)

    def value(self) -> int:
        return self.slider.value()

    def setValue(self, value: int) -> None:
        self.slider.setValue(int(value))


class ControlsPanel(QWidget):
    changed = pyqtSignal(object)          # RenderSettings
    lightingChanged = pyqtSignal(object)  # RenderSettings

    def __init__(self, show_measurement: bool, parent=None):
        super().__init__(parent)
        s = RenderSettings()
        self.contrast = _Slider(0, 100, int(s.contrast * 100), "%")
        self.ambience = _Slider(0, 100, int(s.ambience * 100), "%")
        self.opacity_threshold = _Slider(0, 100, int(s.opacity_threshold * 100), "%")
        self.gradient_opacity = _Slider(0, 100, int(s.gradient_opacity * 100), "%")
        self.specular_power = _Slider(1, 100, s.specular_power)
        self.scalar_min = QDoubleSpinBox()
        self.scalar_max = QDoubleSpinBox()
        for spin, value in ((self.scalar_min, 0), (self.scalar_max, 1000)):
            spin.setRange(-10000, 10000)
            spin.setDecimals(1)
            spin.setValue(value)
            spin.setKeyboardTracking(False)
        self.colormap = QComboBox()
        self.colormap.addItems(COLORMAP_NAMES)
        self.shading = QCheckBox("Advanced shading")
        self.shading.setChecked(True)
        self.edge = _Slider(0, 100, int(s.edge_enhancement * 100), "%")
        self.density = _Slider(0, 100, int(s.tissue_density * 100), "%")
        self.annotations = QCheckBox("Show annotations")
        self.measurement = QCheckBox("Show lesion surface && metrics")
        self.measurement.setVisible(show_measurement)
        self.clip_toggles: Dict[str, QCheckBox] = {}
        self.clip_positions: Dict[str, _Slider] = {}
        self.azimuth = _Slider(-180, 180, s.light_azimuth, "°")
        self.elevation = _Slider(-90, 90, s.light_elevation, "°")
        self.intensity = _Slider(0, 100, int(s.light_intensity * 100), "%")

        basic = Card("Appearance")
        form = QFormLayout()
        form.addRow(label("Colormap", "Muted"), self.colormap)
        form.addRow(label("Contrast", "Muted"), self.contrast)
        form.addRow(label("Ambience", "Muted"), self.ambience)
        form.addRow(label("Opacity threshold", "Muted"), self.opacity_threshold)
        form.addRow(label("Gradient opacity", "Muted"), self.gradient_opacity)
        form.addRow(label("Specular power", "Muted"), self.specular_power)
        form.addRow(label("Scalar min", "Muted"), self.scalar_min)
        form.addRow(label("Scalar max", "Muted"), self.scalar_max)
        basic.add_layout(form)

        advanced = Card("Surgery planning")
        form2 = QFormLayout()
        form2.addRow(label("Edge enhancement", "Muted"), self.edge)
        form2.addRow(label("Tissue density", "Muted"), self.density)
        advanced.add_layout(form2)
        advanced.add(self.shading)
        advanced.add(self.annotations)
        advanced.add(self.measurement)

        clipping = Card("Clipping planes")
        grid = QGridLayout()
        for row, axis in enumerate("xyz"):
            toggle = QCheckBox(f"{axis.upper()} axis")
            position = _Slider(0, 100, 50, "%")
            position.setEnabled(False)
            toggle.toggled.connect(position.setEnabled)
            self.clip_toggles[axis], self.clip_positions[axis] = toggle, position
            grid.addWidget(toggle, row, 0)
            grid.addWidget(position, row, 1)
        grid.setColumnStretch(1, 1)
        clipping.add_layout(grid)

        lighting = Card("Lighting")
        form3 = QFormLayout()
        form3.addRow(label("Azimuth", "Muted"), self.azimuth)
        form3.addRow(label("Elevation", "Muted"), self.elevation)
        form3.addRow(label("Intensity", "Muted"), self.intensity)
        lighting.add_layout(form3)

        reset = make_button("Reset to defaults", "reset", role="ghost")
        reset.clicked.connect(self.reset)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        for card in (basic, advanced, clipping, lighting):
            layout.addWidget(card)
        layout.addWidget(reset)
        layout.addStretch(1)

        for widget in (self.contrast, self.ambience, self.opacity_threshold, self.gradient_opacity,
                       self.specular_power, self.edge, self.density, *self.clip_positions.values()):
            widget.valueChanged.connect(self._emit)
        for spin in (self.scalar_min, self.scalar_max):
            spin.valueChanged.connect(self._emit)
        self.colormap.currentIndexChanged.connect(self._emit)
        for box in (self.shading, self.annotations, self.measurement, *self.clip_toggles.values()):
            box.toggled.connect(self._emit)
        for widget in (self.azimuth, self.elevation, self.intensity):
            widget.valueChanged.connect(lambda _v: self.lightingChanged.emit(self.settings()))
        self._data_range = (0.0, 1000.0)

    def settings(self) -> RenderSettings:
        return RenderSettings(
            contrast=self.contrast.value() / 100, ambience=self.ambience.value() / 100,
            opacity_threshold=self.opacity_threshold.value() / 100,
            gradient_opacity=self.gradient_opacity.value() / 100, specular_power=self.specular_power.value(),
            scalar_min=self.scalar_min.value(), scalar_max=self.scalar_max.value(),
            colormap=self.colormap.currentText(), shading=self.shading.isChecked(),
            edge_enhancement=self.edge.value() / 100, tissue_density=self.density.value() / 100,
            annotations=self.annotations.isChecked(), measurement=self.measurement.isChecked(),
            clip_x=self.clip_toggles["x"].isChecked(), clip_x_pos=self.clip_positions["x"].value() / 100,
            clip_y=self.clip_toggles["y"].isChecked(), clip_y_pos=self.clip_positions["y"].value() / 100,
            clip_z=self.clip_toggles["z"].isChecked(), clip_z_pos=self.clip_positions["z"].value() / 100,
            light_azimuth=self.azimuth.value(), light_elevation=self.elevation.value(),
            light_intensity=self.intensity.value() / 100,
        )

    def set_data_range(self, lo: float, hi: float) -> None:
        """After loading: limit and preset the scalar range to the data (as the original)."""
        self._data_range = (lo, hi)
        for spin in (self.scalar_min, self.scalar_max):
            spin.blockSignals(True)
            spin.setRange(lo, hi)
        self.scalar_min.setValue(lo)
        self.scalar_max.setValue(hi)
        for spin in (self.scalar_min, self.scalar_max):
            spin.blockSignals(False)

    def reset(self) -> None:
        s = RenderSettings()
        widgets = [self.contrast, self.ambience, self.opacity_threshold, self.gradient_opacity, self.specular_power,
                   self.edge, self.density, self.azimuth, self.elevation, self.intensity]
        for w in widgets:
            w.blockSignals(True)
        self.contrast.setValue(int(s.contrast * 100))
        self.ambience.setValue(int(s.ambience * 100))
        self.opacity_threshold.setValue(0)
        self.gradient_opacity.setValue(int(s.gradient_opacity * 100))
        self.specular_power.setValue(s.specular_power)
        self.edge.setValue(int(s.edge_enhancement * 100))
        self.density.setValue(int(s.tissue_density * 100))
        self.azimuth.setValue(s.light_azimuth)
        self.elevation.setValue(s.light_elevation)
        self.intensity.setValue(int(s.light_intensity * 100))
        for w in widgets:
            w.blockSignals(False)
        self.colormap.setCurrentIndex(0)
        for box in self.clip_toggles.values():
            box.setChecked(False)
        self.set_data_range(*self._data_range)
        self._emit()

    def _emit(self, *_args) -> None:
        self.changed.emit(self.settings())
