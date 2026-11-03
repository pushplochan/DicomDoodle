"""
Multiplanar & 3D Viewer — the original Multiplanar MRI View (mpr_gui) and 3D
Model Viewer (create_3d) in one window.

Centre: axial / coronal / sagittal / 3D views in a 2x2 grid (any view can be
maximized), whatever plane the series was acquired in.

Left panel, two tabs:

* Multiplanar  Series: load the primary series (a folder holding several
               volumes — echoes, stacks, a localiser … — offers them one by
               one), register a second series onto it (identity / rigid /
               deformable / ANTs) and switch between them. Display: resampling
               algorithm (incl. through-plane reconstruction) and the image —
               original, or skull-stripped with HD-BET. Annotation: SEG overlay
               (outline or filled), transfer of the primary's annotation onto
               the registered series, comparison with Dice / IoU, and saving
               the transferred annotation as _seg.dcm files.
* 3D view      what the 3D view shows — the original MRI, the MRI with the
               annotated lesion brightened, or the skull-stripped MRI — and
               how (colormap, opacity, clipping planes, lighting …): the
               original 3D Model Viewer's settings, now for one view.

Right panel, as in the 3D Model Viewer: the axial / coronal / sagittal series
folders and "Build 3D model", the lesion volume of each series and their
average, and the 3D lesion metrics.

"Detect" on each 2D view finds the tumour in 3D from the slice shown
(core/mpr/tumour_detect.py); the result is the "Detected tumour" overlay.
Skull stripping (core/skull_strip.py) runs once per scanner space (frame of
reference) and its brain mask serves every series and grid in it: the 2D
views, the registered second series and the 3D model.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import traceback
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDockWidget, QFileDialog, QGridLayout,
                             QInputDialog, QMainWindow, QMessageBox, QProgressDialog, QScrollArea, QSizePolicy,
                             QStackedWidget, QTabWidget, QToolBar, QVBoxLayout, QWidget)

from ....core import mpr, skull_strip
from ....core.dicom_series import detect_plane
from ....core.progress import OperationCancelled
from ....core.reconstruction3d import Reconstruction, build_reconstruction
from ....core.tumor_metrics import compute_tumor_metrics, format_metric
from ....core.volume_calculator import OrientationVolume, average_volume_cm3, compute_volumes
from ....workers import run_task
from ...dialogs.progress_dialog import TaskProgressDialog
from ...guide import help_action, help_button
from ...icons import bind_icon
from ...label_manager import LabelManager
from ...qt_utils import label, make_button, show_error, show_info
from ...widgets.cards import Card, panel_body
from ..viewer3d.controls import ControlsPanel, RenderSettings
from ..viewer3d.panels import SidePanel
from .dialogs import AnnotationCompareDialog
from .panels import BasePanel, MPR2DPanel, ThreeDPanel

Plane, InterpMethod = mpr.Plane, mpr.InterpMethod
HAVE_ULTRALYTICS = importlib.util.find_spec("ultralytics") is not None
log = logging.getLogger("dicomdoodle")

ORIGINAL, ANNOTATED, SKULL_STRIPPED = "original", "annotated", "skull_stripped"
IMAGE_CHOICES = (("Original MRI", ORIGINAL), ("Skull stripped (HD-BET)", SKULL_STRIPPED))          # 2D views
VIEW_CHOICES = (("Original MRI", ORIGINAL), ("MRI with annotation", ANNOTATED),                    # 3D view
                ("Skull stripped (HD-BET)", SKULL_STRIPPED))
SKULL_NOTE = "Skull stripped: the brain alone (HD-BET), computed once per scan — about a minute without a GPU."
NO_HDBET_NOTE = f"HD-BET is not installed ({skull_strip.INSTALL_HINT}), so only the original image can be shown."
MAX_BRAIN_MASKS = 3                  # HD-BET masks kept (one per scanner space)


@dataclass
class ModelBuild:
    volumes: List[OrientationVolume]           # lesion volume of each series
    model: Optional[Reconstruction] = None     # None if the 3D model could not be built
    error: str = ""                            # ... and why
    details: str = ""


def _build_job(ctx, paths: Dict[str, str], seg_filter) -> ModelBuild:
    """Measure the lesion volume of each series, then build the 3D model."""
    build = ModelBuild(compute_volumes(paths, seg_filter, ctx))
    try:
        build.model = build_reconstruction(paths, seg_filter, ctx)
    except OperationCancelled:
        raise
    except Exception as exc:
        build.error, build.details = str(exc) or exc.__class__.__name__, traceback.format_exc()
    return build


def _detect_job(ctx, detector, volume, plane, index, acquired, pick):
    from ....core.mpr.tumour_detect import NoTumourFound
    try:
        return detector.detect(volume, plane, index, acquired, pick=pick, progress=ctx)
    except NoTumourFound as exc:       # an answer, not a failure
        return exc


def _brain_mask_job(ctx, image):
    return skull_strip.brain_mask(image, ctx)


def _set_choice_enabled(combo: QComboBox, data: str, enabled: bool, tooltip: str = "") -> None:
    item = combo.model().item(combo.findData(data))
    item.setEnabled(enabled)
    item.setToolTip("" if enabled else tooltip)


def _select(combo: QComboBox, data: str) -> None:
    """Show ``data``'s item without emitting a change."""
    combo.blockSignals(True)
    combo.setCurrentIndex(combo.findData(data))
    combo.blockSignals(False)


class MPRWindow(QMainWindow):
    def __init__(self, labels: Optional[LabelManager] = None, initial_folder: str = "", models=None,
                 tab: str = "multiplanar"):
        """``initial_folder``: start of the folder dialogs, and the first series of the
        right-hand panel; ``tab``: "multiplanar" or "3d", the left panel's tab to show."""
        super().__init__()
        self.setWindowTitle("Multiplanar & 3D Viewer")
        self.resize(1640, 940)
        self.labels = labels
        self.models = models                 # the app's ModelManager (for Detect)
        self.start_dir = initial_folder or os.path.expanduser("~")

        self.primary_image = None
        self.primary_plane = ""
        self.primary_folder = ""
        self.secondary_image = None          # registered onto the primary grid
        self.secondary_plane = ""
        self.secondary_folder = ""
        self.secondary_native_image = None   # before registration (for comparison / saving)
        self.registration_result: Optional[mpr.RegistrationResult] = None
        self.primary_seg: Optional[mpr.Segmentation] = None
        self.secondary_seg_native: Optional[mpr.Segmentation] = None
        self.transferred_seg: Optional[mpr.Segmentation] = None
        self.overlay_choice = None
        self._active_label_array: Optional[np.ndarray] = None
        self._active_segments: Dict[int, mpr.SegmentInfo] = {}
        self.active_source = "primary"
        self.volume: Optional[mpr.Volume] = None
        self.frac = [0.5, 0.5, 0.5]          # cursor (z, y, x) as fractions of the extent
        # Detect: the result belongs to the image it was computed on and is only
        # offered/drawn while that image is shown; _last_pick remembers the view
        # clicked last (a click on the tumour tells Detect on that view which box to use)
        self.detection = None                # tumour_detect.DetectionResult
        self.detected_seg: Optional[mpr.Segmentation] = None
        self._detection_source = None        # the sitk.Image the detection ran on
        self._detector = None                # tumour_detect.TumourDetector, created on first use
        self._last_pick = None               # (plane, sitk.Image) of the last click
        # 3D model (right-hand panel) and skull stripping
        self.model: Optional[Reconstruction] = None
        self._brain_masks: Dict[str, object] = {}     # HD-BET brain mask (sitk.Image) of each scanner space
        self._model_brain: Optional[np.ndarray] = None   # ... resampled onto the 3D model's grid
        self._view_mode = ORIGINAL           # what the 3D view shows now

        self._build_central()
        self._build_controls()
        self._build_side_panel()
        self._build_toolbar()
        if initial_folder:
            plane = self._guess_plane(initial_folder)
            self.side.series.set_folder(plane if plane in self.side.series.edits else "Axial", initial_folder)
        if tab == "3d":
            self.control_tabs.setCurrentIndex(1)
        self.statusBar().showMessage("Load a primary series (left) for the slices; build the 3D model (right) "
                                     "for the 3D view and the lesion volume.")
        self.maximized_panel: Optional[BasePanel] = None
        self.panel_positions: Dict[BasePanel, Tuple[int, int]] = {
            self.axial_panel: (0, 0), self.coronal_panel: (0, 1),
            self.sagittal_panel: (1, 0), self.three_d_panel: (1, 1)}

    # ================================================================ layout
    def _build_central(self) -> None:
        self.info_banner = label("Load a primary series to begin.", "Muted", wrap=True)
        banner = QWidget()
        banner.setObjectName("DockBody")
        banner_layout = QVBoxLayout(banner)
        banner_layout.setContentsMargins(12, 6, 12, 6)
        banner_layout.addWidget(self.info_banner)

        self.axial_panel = MPR2DPanel(Plane.AXIAL)
        self.coronal_panel = MPR2DPanel(Plane.CORONAL)
        self.sagittal_panel = MPR2DPanel(Plane.SAGITTAL)
        self.three_d_panel = ThreeDPanel()
        if self.three_d_panel.view is not None:
            self.three_d_panel.view.set_message("Choose the series in the panel on the right\n"
                                                "and click Build 3D model")
        for panel in (self.axial_panel, self.coronal_panel, self.sagittal_panel):
            panel.sliceChanged.connect(self.on_slice_changed)
            panel.positionPicked.connect(self.on_position_picked)
            panel.positionPicked.connect(self._remember_pick)
            panel.detectRequested.connect(self.on_detect_requested)
            if self.models is None or not HAVE_ULTRALYTICS:
                panel.set_detect_available(False, "Tumour detection needs the AI models and the 'ultralytics' "
                                                  "package (pip install ultralytics).")
        for panel in (self.axial_panel, self.coronal_panel, self.sagittal_panel, self.three_d_panel):
            panel.toggleRequested.connect(self.on_toggle_maximize)

        self.grid_page = QWidget()
        self.grid_layout = QGridLayout(self.grid_page)
        self.grid_layout.setContentsMargins(2, 2, 2, 2)
        self.grid_layout.setSpacing(2)
        for i in range(2):
            self.grid_layout.setRowStretch(i, 1)
            self.grid_layout.setColumnStretch(i, 1)
        self.grid_layout.addWidget(self.axial_panel, 0, 0)
        self.grid_layout.addWidget(self.coronal_panel, 0, 1)
        self.grid_layout.addWidget(self.sagittal_panel, 1, 0)
        self.grid_layout.addWidget(self.three_d_panel, 1, 1)
        self.max_page = QWidget()
        self.max_layout = QVBoxLayout(self.max_page)
        self.max_layout.setContentsMargins(0, 0, 0, 0)
        self.stacked = QStackedWidget()
        self.stacked.addWidget(self.grid_page)
        self.stacked.addWidget(self.max_page)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(banner)
        layout.addWidget(self.stacked, 1)
        self.setCentralWidget(container)

    def _build_controls(self) -> None:
        self.control_tabs = QTabWidget()
        self.control_tabs.addTab(self._scroll(self._multiplanar_tab()), "Multiplanar")
        self.control_tabs.addTab(self._scroll(self._three_d_tab()), "3D view")
        self.controls_dock = self._dock("Controls", "MprControls", self.control_tabs, Qt.LeftDockWidgetArea, 330)

    def _multiplanar_tab(self) -> QWidget:
        body = panel_body()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Series ------------------------------------------------------------
        series = Card("Series")
        series.add_header_widget(help_button("mpr"))
        load_btn = make_button("Load primary series…", "folder-open", role="primary")
        load_btn.clicked.connect(self.on_load_primary)
        self.backend_combo = QComboBox()
        for backend in mpr.RegistrationBackend:
            text = backend.value + ("  [not installed]" if backend is mpr.RegistrationBackend.ANTSPY
                                    and not mpr.HAVE_ANTSPY else "")
            self.backend_combo.addItem(text, backend)
        self.backend_combo.setToolTip("Used the next time you press 'Register second series'.")
        self.register_btn = make_button("Register second series…", "merge")
        self.register_btn.setEnabled(False)
        self.register_btn.clicked.connect(self.on_register_second_series)
        self.source_combo = QComboBox()
        self.source_combo.addItem("Primary series")
        self.source_combo.setEnabled(False)
        self.source_combo.currentIndexChanged.connect(self.on_source_changed)
        self.reg_info = label("", "Faint", wrap=True)
        series.add(load_btn)
        series.add(label("Registration engine", "Muted"))
        series.add(self.backend_combo)
        series.add(self.register_btn)
        series.add(label("Show", "Muted"))
        series.add(self.source_combo)
        series.add(self.reg_info)

        # Display -----------------------------------------------------------
        display = Card("Display")
        self.algo_combo = QComboBox()
        for method in InterpMethod:
            self.algo_combo.addItem(method.value, method)
        self.algo_combo.setToolTip("How the shown volume is resampled; applies immediately.")
        self.algo_combo.currentIndexChanged.connect(lambda _i: self._rebuild_volume())
        self.image_combo = QComboBox()
        for text, choice in IMAGE_CHOICES:
            self.image_combo.addItem(text, choice)
        self.image_combo.setToolTip("The original image, or the brain alone with skull and scalp removed by HD-BET")
        _set_choice_enabled(self.image_combo, SKULL_STRIPPED, skull_strip.HAVE_HDBET, NO_HDBET_NOTE)
        self.image_combo.currentIndexChanged.connect(self.on_image_choice_changed)
        display.add(label("Resampling algorithm", "Muted"))
        display.add(self.algo_combo)
        display.add(label("Image", "Muted"))
        display.add(self.image_combo)
        display.add(label(SKULL_NOTE if skull_strip.HAVE_HDBET else NO_HDBET_NOTE, "Faint", wrap=True))

        # Annotation --------------------------------------------------------
        annotation = Card("Annotation")
        self.annotation_combo = QComboBox()
        self.annotation_combo.addItem("None", None)
        self.annotation_combo.currentIndexChanged.connect(self.on_annotation_choice_changed)
        self.show_filled_check = QCheckBox("Show filled mask")
        self.show_filled_check.setToolTip("Off: outline only. On: translucent filled regions.")
        self.show_filled_check.stateChanged.connect(lambda _s: self._refresh_2d_panels())
        self.fill_contours_check = QCheckBox("Fill SEG contours")
        self.fill_contours_check.setChecked(True)
        self.fill_contours_check.setToolTip("Fill outline-style annotations so they are solid in every view; "
                                            "an outline saved in several open pieces on one slice is filled as "
                                            "one region (re-loads all annotations).")
        self.fill_contours_check.stateChanged.connect(self.on_fill_contours_changed)
        self.compare_btn = make_button("Compare annotations…", "columns")
        self.compare_btn.setEnabled(False)
        self.compare_btn.clicked.connect(self.on_compare_annotations)
        self.save_transferred_btn = make_button("Save transferred annotation…", "save")
        self.save_transferred_btn.setEnabled(False)
        self.save_transferred_btn.clicked.connect(self.on_save_transferred_annotation)
        self.annotation_info = label("", "Faint", wrap=True)
        annotation.add(label("Overlay", "Muted"))
        annotation.add(self.annotation_combo)
        annotation.add(self.show_filled_check)
        annotation.add(self.fill_contours_check)
        annotation.add(self.compare_btn)
        annotation.add(self.save_transferred_btn)
        annotation.add(self.annotation_info)

        for card in (series, display, annotation):
            layout.addWidget(card)
        layout.addStretch(1)
        return body

    def _three_d_tab(self) -> QWidget:
        body = panel_body()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        show = Card("3D view")
        show.add_header_widget(help_button("viewer3d"))
        self.view_combo = QComboBox()
        for text, choice in VIEW_CHOICES:
            self.view_combo.addItem(text, choice)
        self.view_combo.setToolTip("What the 3D view shows: the series as it is, with the annotated lesion made "
                                   "brighter, or with skull and scalp removed (HD-BET)")
        _set_choice_enabled(self.view_combo, ANNOTATED, False, "Build the 3D model from series with annotation files")
        _set_choice_enabled(self.view_combo, SKULL_STRIPPED, skull_strip.HAVE_HDBET, NO_HDBET_NOTE)
        self.view_combo.currentIndexChanged.connect(self.on_view_choice_changed)
        show.add(label("Show", "Muted"))
        show.add(self.view_combo)
        show.add(label("The 3D view shows the model built in the panel on the right; the settings below change "
                       "how it looks.", "Faint", wrap=True))
        self.view_controls = ControlsPanel(show_measurement=True)
        self.view_controls.changed.connect(self._apply_view_settings)
        self.view_controls.lightingChanged.connect(self._apply_view_lighting)
        self.view_controls.setEnabled(self.three_d_panel.view is not None)
        layout.addWidget(show)
        layout.addWidget(self.view_controls)
        return body

    def _build_side_panel(self) -> None:
        self.side = SidePanel()
        holder = panel_body()
        holder_layout = QVBoxLayout(holder)
        holder_layout.setContentsMargins(10, 10, 10, 10)
        holder_layout.addWidget(self.side)
        self.side_dock = self._dock("Series & Measurements", "MprMeasurements", self._scroll(holder),
                                    Qt.RightDockWidgetArea, 320)
        self.side.series.buildRequested.connect(self.build_model)
        surface = self.side.metrics.show_surface
        surface.toggled.connect(self.view_controls.measurement.setChecked)
        self.view_controls.measurement.toggled.connect(surface.setChecked)

    def _build_toolbar(self) -> None:
        bar = QToolBar("Panels", self)
        bar.setObjectName("MprToolbar")
        bar.setMovable(False)
        bar.setIconSize(QSize(20, 20))
        bar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        for dock, icon, text in ((self.controls_dock, "sliders", "Controls"),
                                 (self.side_dock, "bar-chart", "Series && Measurements")):
            action = dock.toggleViewAction()
            action.setText(text)
            action.setToolTip(f"Show or hide the {text.replace('&&', '&')} panel")
            bind_icon(action, icon)
            bar.addAction(action)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        bar.addWidget(spacer)
        bar.addAction(help_action(self, "mpr"))
        self.addToolBar(bar)

    def _dock(self, title: str, name: str, widget: QWidget, area, width: int) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(name)
        dock.setWidget(widget)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetClosable
                         | QDockWidget.DockWidgetFloatable)
        dock.setMinimumWidth(width)
        self.addDockWidget(area, dock)
        return dock

    @staticmethod
    def _scroll(widget: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QScrollArea.NoFrame)
        area.setWidget(widget)
        return area

    @staticmethod
    def _guess_plane(folder: str) -> str:
        try:
            import pydicom
            name = next(n for n in sorted(os.listdir(folder)) if n.lower().endswith(".dcm") and "_seg" not in n.lower())
            ds = pydicom.dcmread(os.path.join(folder, name), stop_before_pixels=True)
            return detect_plane(ds.get("ImageOrientationPatient"))
        except Exception:
            return "Axial"

    def _update_info_banner(self) -> None:
        if self.volume is None:
            self.info_banner.setText("Load a primary series to begin.")
            return
        algo: InterpMethod = self.algo_combo.currentData()
        source = "PRIMARY series" if self.active_source == "primary" else "REGISTERED second series"
        if self.active_source == "secondary" and self.registration_result is not None:
            reg = self.registration_result
            metric = f", metric {reg.final_metric:.3f}" if np.isfinite(reg.final_metric) else ""
            reg_text = f"aligned with {reg.backend.value}{metric}"
        else:
            reg_text = "no registration (primary)"
        image = "skull stripped (HD-BET)" if self._image_choice() == SKULL_STRIPPED else "original"
        annot = self.annotation_combo.currentText() if self.overlay_choice is not None else "none"
        self.info_banner.setText(f"Showing: <b>{source}</b>  ·  Algorithm: {algo.value}  ·  {reg_text}  ·  "
                                 f"Image: {image}  ·  Annotation: {annot}")

    # ============================================================== loading
    def _color_lookup(self, name: str):
        if self.labels is None:
            return None
        lab = self.labels.registry.find_by_name(name)
        return lab.color if lab else None

    def _load_segmentation(self, folder: str, image):
        seg_paths = mpr.find_series_seg_files(image, folder)
        if not seg_paths:
            return None
        return mpr.load_seg_as_labelmap(seg_paths, image, fill_contours=self.fill_contours_check.isChecked(),
                                        color_lookup=self._color_lookup)

    def _pick_and_load_series(self, title: str):
        folder = QFileDialog.getExistingDirectory(self, title, self.start_dir)
        if not folder:
            return None, None, None, None
        self.start_dir = os.path.dirname(folder)
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)       # a header scan of every file in the folder
            try:
                series = mpr.find_series(folder)
            finally:
                QApplication.restoreOverrideCursor()
            if len(series) > 1:
                names = [s.label for s in series]
                item, ok = QInputDialog.getItem(self, "Several image volumes found",
                                                "This folder contains more than one DICOM image volume.\n"
                                                "Choose one:", names, 0, False)
                if not ok:
                    return None, None, None, None
                series_id = series[names.index(item)].key
            else:
                series_id = series[0].key if series else None
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                image = mpr.load_dicom_series(folder, series_id)
                try:
                    segmentation = self._load_segmentation(folder, image)
                except Exception:
                    segmentation = None   # an unreadable SEG must not block the image
            finally:
                QApplication.restoreOverrideCursor()
            return image, mpr.get_acquisition_plane(image), folder, segmentation
        except Exception as exc:
            show_error(self, "Failed to load DICOM series", str(exc), traceback.format_exc())
            return None, None, None, None

    def on_load_primary(self) -> None:
        image, plane, folder, segmentation = self._pick_and_load_series("Select the PRIMARY DICOM series folder")
        if image is None:
            return
        self.primary_image, self.primary_plane, self.primary_seg, self.primary_folder = image, plane, segmentation, folder
        self.secondary_image = self.secondary_native_image = None
        self.secondary_plane = self.secondary_folder = ""
        self.secondary_seg_native = self.transferred_seg = self.registration_result = None
        self.detection = self.detected_seg = self._detection_source = self._last_pick = None
        self.active_source = "primary"
        self.frac = [0.5, 0.5, 0.5]
        _select(self.image_combo, ORIGINAL)      # HD-BET runs only when asked for
        self._forget_masks("image-")
        if not any(self.side.series.paths().values()):
            self.side.series.set_folder(plane if plane in self.side.series.edits else "Axial", folder)
        self.source_combo.blockSignals(True)
        self.source_combo.clear()
        self.source_combo.addItem("Primary series")
        self.source_combo.setEnabled(False)
        self.source_combo.blockSignals(False)
        self.reg_info.setText("")
        self.register_btn.setEnabled(True)
        self._refresh_annotation_combo()
        self._rebuild_volume()
        note = f" — annotation loaded ({len(segmentation.segments)} label(s))" if segmentation else ""
        series_note = mpr.get_series_notes(image)
        note += f" — note: {series_note}" if series_note else ""
        self.statusBar().showMessage(f"Loaded {folder} — acquired as {plane}, shown as axial / coronal / "
                                     f"sagittal{note}")

    # ========================================================= registration
    def on_register_second_series(self) -> None:
        if self.primary_image is None:
            show_info(self, "Load a primary series first", "Load the primary series before registering another.")
            return
        moving, plane, folder, moving_seg = self._pick_and_load_series("Select the SECOND series folder to register")
        if moving is None:
            return
        backend: mpr.RegistrationBackend = self.backend_combo.currentData()
        progress = QProgressDialog(f"Registering the second series onto the primary using {backend.value}…",
                                   None, 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()

        def on_iteration(iteration, metric):
            if backend is mpr.RegistrationBackend.ANTSPY:
                progress.setLabelText("Registering with ANTsPy — this can take a while…")
            else:
                progress.setLabelText(f"Registering…\niteration {iteration}, similarity metric {metric:.4f}")
            QApplication.processEvents()

        try:
            result = mpr.register_rigid(self.primary_image, moving, backend=backend, progress_callback=on_iteration)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            progress.close()
            show_error(self, "Registration failed", str(exc), traceback.format_exc())
            return
        QApplication.restoreOverrideCursor()
        progress.close()

        self.secondary_image, self.secondary_plane, self.secondary_folder = result.registered_image, plane, folder
        self.registration_result, self.secondary_native_image, self.secondary_seg_native = result, moving, moving_seg
        self.transferred_seg = None
        if self.primary_seg is not None:
            try:
                self.transferred_seg = mpr.transfer_labelmap(self.primary_seg, moving, result, source_was_fixed=True)
            except Exception as exc:
                QMessageBox.warning(self, "Annotation transfer failed",
                                    f"Registration succeeded but transferring the annotation failed:\n{exc}")
        self.source_combo.blockSignals(True)
        self.source_combo.clear()
        self.source_combo.addItems(["Primary series", "Registered second series"])
        self.source_combo.setEnabled(True)
        self.source_combo.setCurrentIndex(1)
        self.source_combo.blockSignals(False)
        tx, ty, tz = result.translation_mm
        rx, ry, rz = result.rotation_deg
        metric = f"metric {result.final_metric:.3f}" if np.isfinite(result.final_metric) else "metric n/a"
        self.reg_info.setText(f"[{result.backend.name}] translation ({tx:.2f}, {ty:.2f}, {tz:.2f}) mm · "
                              f"rotation ({rx:.2f}, {ry:.2f}, {rz:.2f})° · {metric}")
        self.active_source = "secondary"
        self._refresh_annotation_combo()
        self._rebuild_volume()
        notes = []
        if mpr.get_series_notes(moving):
            notes.append(f"note: {mpr.get_series_notes(moving)}")
        if moving_seg is not None:
            notes.append(f"secondary has its own annotation ({len(moving_seg.segments)} label(s))")
        if self.transferred_seg is not None:
            notes.append("the primary's annotation was transferred onto it")
        self.statusBar().showMessage(f"Registered {folder} (acquired as {plane}) onto the primary series"
                                     + (" — " + "; ".join(notes) if notes else ""))

    def on_source_changed(self, index: int) -> None:
        self.active_source = "primary" if index == 0 else "secondary"
        self._refresh_annotation_combo()
        self._rebuild_volume()

    # ============================================================ annotation
    def _refresh_annotation_combo(self) -> None:
        prior = self.overlay_choice
        self.annotation_combo.blockSignals(True)
        self.annotation_combo.clear()
        self.annotation_combo.addItem("None", None)
        if self.active_source == "primary":
            if self.primary_seg is not None:
                self.annotation_combo.addItem(f"Primary annotation ({len(self.primary_seg.segments)} label(s))",
                                              "primary")
        else:
            if self.secondary_seg_native is not None:
                self.annotation_combo.addItem(
                    f"Secondary's own annotation ({len(self.secondary_seg_native.segments)} label(s))",
                    "secondary_native")
            if self.transferred_seg is not None:
                self.annotation_combo.addItem("Transferred from primary", "transferred")
        if self.detected_seg is not None and self._detection_source is self._active_source_image():
            self.annotation_combo.addItem("Detected tumour (best_flair + SAM 2.1)", "detected")
        index = self.annotation_combo.findData(prior)
        self.annotation_combo.setCurrentIndex(index if index >= 0 else 0)
        self.overlay_choice = self.annotation_combo.currentData()
        self.annotation_combo.blockSignals(False)
        self.compare_btn.setEnabled(self.secondary_seg_native is not None or self.transferred_seg is not None)
        self.save_transferred_btn.setEnabled(self.transferred_seg is not None)
        self._rebuild_active_label_array()

    def on_annotation_choice_changed(self, _index: int) -> None:
        self.overlay_choice = self.annotation_combo.currentData()
        self._rebuild_active_label_array()
        self._refresh_2d_panels()
        self._update_info_banner()

    def on_fill_contours_changed(self, _state: int) -> None:
        """Re-load all annotations with the new fill setting (keeps Dice comparable)."""
        try:
            if self.primary_image is not None and self.primary_folder:
                self.primary_seg = self._load_segmentation(self.primary_folder, self.primary_image)
            if self.secondary_native_image is not None and self.secondary_folder:
                self.secondary_seg_native = self._load_segmentation(self.secondary_folder,
                                                                    self.secondary_native_image)
            if self.primary_seg is not None and self.secondary_native_image is not None \
                    and self.registration_result is not None:
                self.transferred_seg = mpr.transfer_labelmap(self.primary_seg, self.secondary_native_image,
                                                             self.registration_result, source_was_fixed=True)
        except Exception as exc:
            show_error(self, "Annotation", str(exc), traceback.format_exc())
        self._refresh_annotation_combo()
        self._refresh_2d_panels()

    def _rebuild_active_label_array(self) -> None:
        self._active_label_array, self._active_segments = None, {}
        segmentation = {"primary": self.primary_seg, "secondary_native": self.secondary_seg_native,
                        "transferred": self.transferred_seg, "detected": self.detected_seg}.get(self.overlay_choice)
        if self.volume is None or segmentation is None:
            self.annotation_info.setText("")
            return
        try:
            self._active_label_array = mpr.resample_label_array_like(segmentation.label_image, self.volume.sitk_image)
            self._active_segments = segmentation.segments
            self.annotation_info.setText("Labels: " + ", ".join(i.label for i in segmentation.segments.values()))
        except Exception as exc:
            self.annotation_info.setText(f"Overlay unavailable: {exc}")

    def on_compare_annotations(self) -> None:
        if self.secondary_native_image is None:
            show_info(self, "No secondary series", "Register a second series first.")
            return
        if self.secondary_seg_native is None and self.transferred_seg is None:
            show_info(self, "Nothing to compare",
                      "Neither the secondary series' own annotation nor a transferred one is available.")
            return
        AnnotationCompareDialog(self.secondary_native_image, self.secondary_seg_native, self.transferred_seg,
                                parent=self, initial_plane=self.secondary_plane).exec_()

    def on_save_transferred_annotation(self) -> None:
        if self.transferred_seg is None or not self.secondary_folder:
            show_info(self, "No transferred annotation", "Load a primary series with an annotation, then register "
                                                         "a second series onto it.")
            return
        out_folder = QFileDialog.getExistingDirectory(self, "Folder for the slice-wise _seg.dcm files",
                                                      self.secondary_folder)
        if not out_folder:
            return
        image = self.secondary_native_image

        def meta(key: str, default):
            return image.GetMetaData(key) if image is not None and image.HasMetaDataKey(key) else default

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            count = mpr.save_transferred_seg_slice_wise(self.transferred_seg, meta("dicom_folder", self.secondary_folder),
                                                        out_folder, series_id=meta("series_key", None))
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            show_error(self, "Save failed", f"Could not save the transferred annotation:\n{exc}",
                       traceback.format_exc())
            return
        QApplication.restoreOverrideCursor()
        show_info(self, "Annotation saved", f"Transferred annotation saved.\n\nFolder: {out_folder}\n"
                                            f"Slices saved: {count}\nLabels: {len(self.transferred_seg.segments)}")
        self.statusBar().showMessage(f"Transferred annotation saved → {count} slices in {out_folder}")

    # =========================================================== volume
    def _rebuild_volume(self) -> None:
        image = self.primary_image if self.active_source == "primary" else self.secondary_image
        if image is None:
            return
        method: InterpMethod = self.algo_combo.currentData()
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.statusBar().showMessage(f"Building volume: {method.value} …")
            QApplication.processEvents()
            self.volume = mpr.build_volume(image, method)
        except Exception as exc:
            show_error(self, "Resampling failed", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        mask = self._brain_masks.get(self._space_key(self.primary_image))
        if self._image_choice() == SKULL_STRIPPED and mask is not None:
            # the registered second series lies on the primary grid: the primary's brain mask fits it too
            self.volume.array = skull_strip.apply_mask(self.volume.array,
                                                       mpr.resample_label_array_like(mask, self.volume.sitk_image))

        z, y, x = self.volume.shape
        sz, sy, sx = self.volume.spacing_zyx
        plane = self.primary_plane if self.active_source == "primary" else self.secondary_plane
        source = "Primary" if self.active_source == "primary" else "Registered second series"
        self.statusBar().showMessage(f"{source} (acquired as {plane}) · {x}×{y}×{z} voxels · spacing "
                                     f"({sx:.3f}, {sy:.3f}, {sz:.3f}) mm · {method.value}")
        self.axial_panel.set_range(z, self._idx(0))
        self.coronal_panel.set_range(y, self._idx(1))
        self.sagittal_panel.set_range(x, self._idx(2))
        self._rebuild_active_label_array()
        self._refresh_2d_panels()
        self._update_info_banner()

    # ======================================================= skull stripping
    def _image_choice(self) -> str:
        return self.image_combo.currentData()

    def on_image_choice_changed(self, _index: int) -> None:
        """Original or skull-stripped image in the 2D views (HD-BET the first time)."""
        if self._image_choice() == SKULL_STRIPPED:
            if self.primary_image is None:
                show_info(self, "Load a series first", "Load a primary series before skull stripping it.")
                _select(self.image_combo, ORIGINAL)
                return
            if self._brain_mask(self._space_key(self.primary_image), lambda: self.primary_image,
                                "the primary series") is None:
                _select(self.image_combo, ORIGINAL)
                return
        self._rebuild_volume()

    @staticmethod
    def _space_key(image) -> str:
        """The scanner space an image lies in: its FrameOfReferenceUID, which every series of one session
        shares (an image without one is a space of its own)."""
        if image is None:
            return ""
        frame = image.GetMetaData("frame_of_reference") if image.HasMetaDataKey("frame_of_reference") else ""
        return frame or f"image-{id(image)}"

    def _brain_mask(self, key: str, make_image: Callable, what: str):
        """The HD-BET brain mask of one scanner space, computed the first time from ``make_image()``
        — the brain is the same region in every series and grid of that space."""
        if key not in self._brain_masks:
            mask = self._run_hdbet(make_image(), what)
            if mask is None:
                return None
            self._brain_masks[key] = mask
            while len(self._brain_masks) > MAX_BRAIN_MASKS:
                self._brain_masks.pop(next(iter(self._brain_masks)))
        return self._brain_masks[key]

    def _forget_masks(self, prefix: str) -> None:
        """Drop masks keyed by object identity (series without a FrameOfReferenceUID): a new object may get
        the same id."""
        self._brain_masks = {k: v for k, v in self._brain_masks.items() if not k.startswith(prefix)}

    def _run_hdbet(self, image, what: str):
        dialog = TaskProgressDialog("Skull stripping (HD-BET)", self)
        dialog.set_progress(0, 0, f"HD-BET is removing the skull from {what}…")
        state = {}
        task = run_task(_brain_mask_job, image,
                        on_result=lambda m: (state.update(mask=m), dialog.accept()),
                        on_error=lambda m, d: (state.update(error=(m, d)), dialog.accept()),
                        on_progress=dialog.set_progress, on_cancelled=dialog.accept)
        dialog.cancelRequested.connect(task.cancel)
        dialog.exec_()
        if "error" in state:
            show_error(self, "Skull stripping failed", *state["error"])
            return None
        mask = state.get("mask")
        if mask is None:
            self.statusBar().showMessage("Skull stripping cancelled.")
            return None
        import SimpleITK as sitk
        brain_ml = np.count_nonzero(sitk.GetArrayViewFromImage(mask)) * np.prod(mask.GetSpacing()) / 1000.0
        self.statusBar().showMessage(f"HD-BET removed the skull from {what}: brain {brain_ml:.0f} mL")
        log.info("HD-BET brain mask of %s: %.0f mL", what, brain_ml)
        return mask

    # =========================================================== 3D model
    def build_model(self, paths: Optional[Dict[str, str]] = None, seg_filter=None) -> None:
        """Measure the lesion volume and build the 3D model of the right-hand panel's series."""
        card = self.side.series
        paths = paths if paths is not None else card.paths()
        if not any(paths.values()):
            show_info(self, "Build 3D model", "Please choose at least one series folder.")
            return
        progress = TaskProgressDialog("Building the 3D model", self)
        state = {}
        task = run_task(_build_job, paths, seg_filter or card.seg_filter(),
                        on_result=lambda r: (state.update(build=r), progress.accept()),
                        on_error=lambda m, d: (progress.accept(), show_error(self, "3D processing error", m, d)),
                        on_progress=progress.set_progress, on_cancelled=progress.accept)
        progress.cancelRequested.connect(task.cancel)
        progress.exec_()
        if "build" in state:
            self._show_build(state["build"])

    def _show_build(self, build: ModelBuild) -> None:
        side = self.side
        side.volume.show_results(build.volumes)
        counted = [v for v in build.volumes if v.counted]
        if counted:
            log.info("Lesion volume: %.2f cm³ (%s)", average_volume_cm3(build.volumes),
                     ", ".join(f"{v.orientation} {v.volume_cm3:.2f}" for v in counted))
        if build.model is None:
            side.series.show_info(f"The 3D model could not be built: {build.error}")
            side.metrics.show_metrics(None, "The 3D model could not be built.")
            self.statusBar().showMessage("Lesion volume measured — the 3D model could not be built.")
            show_error(self, "3D model", f"The 3D model could not be built:\n{build.error}", build.details)
            return
        result = self.model = build.model
        self._model_brain = None
        self._forget_masks("model-")
        view = self.three_d_panel.view
        if view is not None:
            view.surface_actor = view.metrics_actor = None
        if result.lesion_mask is not None:
            try:
                metrics, surface = compute_tumor_metrics(result.lesion_mask, result.spacing, result.origin,
                                                         result.anatomy)
                text = "Tumor Metrics:\n\n" + "\n".join(f"{k}: {format_metric(k, v)}" for k, v in metrics.items())
                if view is not None:
                    view.set_lesion(surface, text)
                side.metrics.show_metrics(metrics, "Measured on the 3D model (all loaded series on one grid), so "
                                                   "its volume can differ a little from the lesion volume.")
                log.info("3D lesion volume: %s", format_metric("Volume (cc)", metrics["Volume (cc)"]))
            except Exception as exc:
                side.metrics.show_metrics(None, f"Metrics unavailable: {exc}")
        else:
            side.metrics.show_metrics(None, "No lesion annotation was found in the loaded folders.")
        _set_choice_enabled(self.view_combo, ANNOTATED, result.lesion_mask is not None,
                            "No annotation files with a lesion were found in the series folders")
        if self.view_combo.currentData() == ANNOTATED and result.lesion_mask is None:
            _select(self.view_combo, ORIGINAL)
        series = ", ".join(result.series_used)
        side.series.show_info(f"Loaded: {series} · reference grid {result.reference}\n"
                              f"Voxel spacing {result.spacing[0]:.3f} × {result.spacing[1]:.3f} × "
                              f"{result.spacing[2]:.3f} mm"
                              + ("\n\nWarnings:\n" + "\n".join(result.warnings) if result.warnings else ""))
        if not self._show_3d(reset_camera=True):
            _select(self.view_combo, ORIGINAL)
            self._show_3d(reset_camera=True)
        self.statusBar().showMessage(f"3D model ready — {series}" + (" (with warnings)" if result.warnings else ""))

    def on_view_choice_changed(self, _index: int) -> None:
        """Original / with annotation / skull stripped in the 3D view (keeps the camera)."""
        if self.model is not None and not self._show_3d():
            _select(self.view_combo, self._view_mode)      # skull stripping cancelled: back to what is shown

    def _show_3d(self, reset_camera: bool = False) -> bool:
        """Render the 3D model as the "Show" choice asks; False if that is not possible now (skull
        stripping cancelled or failed)."""
        view, model = self.three_d_panel.view, self.model
        if view is None or model is None:
            return True
        choice = self.view_combo.currentData()
        if choice == ANNOTATED and model.lesion_mask is not None:
            array = model.anatomy
        elif choice == SKULL_STRIPPED:
            mask = self._model_mask()
            if mask is None:
                return False
            array = skull_strip.apply_mask(model.volume, mask)
        else:
            array = model.volume
        view.set_volume(array, model.spacing, model.origin, reset_camera=reset_camera)
        self.view_controls.set_data_range(*view.data_range)
        view.apply(self.view_controls.settings())
        self._view_mode = choice
        nz, ny, nx = array.shape
        self.three_d_panel.info_label.setText(f"  {self.view_combo.currentText()} · {nx}×{ny}×{nz} voxels")
        return True

    def _model_mask(self) -> Optional[np.ndarray]:
        """The brain mask on the 3D model's grid. It is the mask of the model's scanner space, so one HD-BET
        run serves the 2D views and the model alike when both show series of the same session."""
        if self._model_brain is None:
            model = self.model
            key = model.frame_of_reference or f"model-{id(model)}"
            mask = self._brain_mask(key, model.grid_image, "the 3D model")
            if mask is None:
                return None
            grid = model.grid_image(np.zeros(model.volume.shape, np.uint8))
            self._model_brain = mpr.resample_label_array_like(mask, grid)
        return self._model_brain

    def _apply_view_settings(self, settings: RenderSettings) -> None:
        if self.three_d_panel.view is not None:
            self.three_d_panel.view.apply(settings)

    def _apply_view_lighting(self, settings: RenderSettings) -> None:
        if self.three_d_panel.view is not None:
            self.three_d_panel.view.apply_lighting(settings)

    # =========================================================== cursor
    def _idx(self, axis: int) -> int:
        if self.volume is None:
            return 0
        return int(round(self.frac[axis] * (self.volume.shape[axis] - 1)))

    def on_slice_changed(self, plane: mpr.Plane, value: int) -> None:
        if self.volume is None:
            return
        axis = {Plane.AXIAL: 0, Plane.CORONAL: 1, Plane.SAGITTAL: 2}[plane]
        self.frac[axis] = value / max(self.volume.shape[axis] - 1, 1)
        self._refresh_2d_panels()

    def on_position_picked(self, plane: mpr.Plane, row: int, col: int) -> None:
        if self.volume is None:
            return
        z, y, x = self.volume.shape
        if plane is Plane.AXIAL:
            self.frac[1] = np.clip((y - 1) - row, 0, y - 1) / max(y - 1, 1)
            self.frac[2] = np.clip(col, 0, x - 1) / max(x - 1, 1)
        elif plane is Plane.CORONAL:
            self.frac[0] = np.clip((z - 1) - row, 0, z - 1) / max(z - 1, 1)
            self.frac[2] = np.clip(col, 0, x - 1) / max(x - 1, 1)
        else:
            self.frac[0] = np.clip((z - 1) - row, 0, z - 1) / max(z - 1, 1)
            self.frac[1] = np.clip(col, 0, y - 1) / max(y - 1, 1)
        self.axial_panel.set_range(z, self._idx(0))
        self.coronal_panel.set_range(y, self._idx(1))
        self.sagittal_panel.set_range(x, self._idx(2))
        self._refresh_2d_panels()

    def _refresh_2d_panels(self) -> None:
        if self.volume is None:
            return
        z, y, x = self.volume.shape
        cz, cy, cx = self._idx(0), self._idx(1), self._idx(2)
        images = {}
        for plane, index in ((Plane.AXIAL, cz), (Plane.CORONAL, cy), (Plane.SAGITTAL, cx)):
            rgb = mpr.gray_to_rgb(mpr.window_to_uint8(mpr.extract_slice(self.volume, plane, index)))
            if self._active_label_array is not None:
                label_slice = mpr.extract_label_slice(self._active_label_array, plane, index)
                rgb = (mpr.blend_label_overlay(rgb, label_slice, self._active_segments)
                       if self.show_filled_check.isChecked()
                       else mpr.blend_label_boundary(rgb, label_slice, self._active_segments,
                                                     aspect=mpr.display_aspect(self.volume, plane)))
            images[plane] = rgb
        self.axial_panel.show_image(images[Plane.AXIAL], mpr.display_aspect(self.volume, Plane.AXIAL),
                                    f"slice {cz + 1}/{z}")
        self.coronal_panel.show_image(images[Plane.CORONAL], mpr.display_aspect(self.volume, Plane.CORONAL),
                                      f"slice {cy + 1}/{y}")
        self.sagittal_panel.show_image(images[Plane.SAGITTAL], mpr.display_aspect(self.volume, Plane.SAGITTAL),
                                       f"slice {cx + 1}/{x}")
        self.axial_panel.set_crosshair(row=(y - 1) - cy, col=cx)
        self.coronal_panel.set_crosshair(row=(z - 1) - cz, col=cx)
        self.sagittal_panel.set_crosshair(row=(z - 1) - cz, col=cy)
        self._update_detection_boxes(cz, cy, cx)

    # ======================================================= tumour detection
    def _active_source_image(self):
        return self.primary_image if self.active_source == "primary" else self.secondary_image

    def _remember_pick(self, plane: mpr.Plane, _row: int, _col: int) -> None:
        self._last_pick = (plane, self._active_source_image())

    def on_detect_requested(self, plane: mpr.Plane) -> None:
        """'Detect' on one 2D view (core/mpr/tumour_detect.py): starts from this
        view's slice — from the box under the crosshair if this view was the
        last one clicked — and segments the tumour on every slice of the
        acquired series inside the 3D box the three views agree on. The mask
        becomes the "Detected tumour" overlay, its box is outlined on all three
        views, and the crosshair jumps to the tumour."""
        if self.volume is None:
            show_info(self, "Load a series first", "Load a primary series before detecting.")
            return
        from ....core.mpr import tumour_detect as tdet
        image = self._active_source_image()
        # always the series' own slices and values, whatever display algorithm or
        # image (skull stripped) is chosen (the overlay resamples the mask for display)
        source = mpr.build_volume(image, InterpMethod.NONE_NATIVE)

        def at(axis: int) -> int:
            return int(round(self.frac[axis] * (source.shape[axis] - 1)))

        slice_axis, row_axis, col_axis = tdet.PLANE_AXES[plane]
        index = at(slice_axis)
        pick = None
        if self._last_pick is not None and self._last_pick[0] is plane and self._last_pick[1] is image:
            pick = (at(row_axis), at(col_axis))
        acquired = self.primary_plane if self.active_source == "primary" else self.secondary_plane
        if self._detector is None:
            self._detector = tdet.TumourDetector(self.models)

        dialog = TaskProgressDialog("Detecting the tumour", self)
        dialog.set_progress(0, 0, "Loading the tumour detector and SAM 2.1 (only the first time)…")
        state = {}
        task = run_task(_detect_job, self._detector, source, plane, index, acquired, pick,
                        on_result=lambda r: (state.update(result=r), dialog.accept()),
                        on_error=lambda m, d: (state.update(error=(m, d)), dialog.accept()),
                        on_progress=dialog.set_progress,
                        on_cancelled=lambda: (state.update(cancelled=True), dialog.accept()))
        dialog.cancelRequested.connect(task.cancel)
        dialog.exec_()
        result = state.get("result")
        if "error" in state:
            show_error(self, "Detection failed", *state["error"])
            return
        if result is None:
            self.statusBar().showMessage("Detection cancelled.")
            return
        if isinstance(result, tdet.NoTumourFound):
            show_info(self, "No tumour found", f"{result}\n\nScroll to a slice where the tumour is clearly "
                                               "visible, click on it, and press Detect again.")
            return

        self.detection = result
        self.detected_seg = result.to_segmentation(source.sitk_image)
        self._detection_source = image
        if result.mask.any():
            z, y, x = self.volume.shape
            for axis, c in enumerate(result.centre):
                self.frac[axis] = c / max(result.shape[axis] - 1, 1)
            self.axial_panel.set_range(z, self._idx(0))
            self.coronal_panel.set_range(y, self._idx(1))
            self.sagittal_panel.set_range(x, self._idx(2))
        self._refresh_annotation_combo()
        detected = self.annotation_combo.findData("detected")
        self.annotation_combo.blockSignals(True)
        self.annotation_combo.setCurrentIndex(detected)
        self.annotation_combo.blockSignals(False)
        self.on_annotation_choice_changed(detected)        # also refreshes if it was already selected

        dice_note = ""
        if self.active_source == "primary" and self.primary_seg is not None:
            reference = mpr.resample_label_array_like(self.primary_seg.label_image, source.sitk_image)
            dice_note = f"  ·  Dice vs. the loaded annotation: {mpr.dice_coefficient(result.mask, reference):.3f}"
        self.annotation_info.setToolTip(result.details_text())
        self.statusBar().showMessage(result.summary_text() + dice_note)

    def _update_detection_boxes(self, cz: int, cy: int, cx: int) -> None:
        """Outline the detected tumour's 3D box on each view whose slice passes
        through it (only while the image it was detected on is shown)."""
        det = self.detection if self._detection_source is self._active_source_image() else None
        for panel, plane, index in ((self.axial_panel, Plane.AXIAL, cz), (self.coronal_panel, Plane.CORONAL, cy),
                                    (self.sagittal_panel, Plane.SAGITTAL, cx)):
            panel.set_detection_box(None if det is None else det.display_box(plane, index, self.volume.shape))

    # =========================================================== maximize
    def on_toggle_maximize(self, panel: BasePanel) -> None:
        if self.maximized_panel is panel:
            self.max_layout.removeWidget(panel)
            row, col = self.panel_positions[panel]
            self.grid_layout.addWidget(panel, row, col)
            self.stacked.setCurrentIndex(0)
            panel.set_maximized(False)
            self.maximized_panel = None
        elif self.maximized_panel is None:
            self.grid_layout.removeWidget(panel)
            self.max_layout.addWidget(panel)
            self.stacked.setCurrentIndex(1)
            panel.set_maximized(True)
            self.maximized_panel = panel
        if isinstance(panel, ThreeDPanel):
            panel.refresh_after_resize()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape and self.maximized_panel is not None:
            self.on_toggle_maximize(self.maximized_panel)
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self.three_d_panel.shutdown()
        super().closeEvent(event)
