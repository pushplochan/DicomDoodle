#!/usr/bin/env python
"""
Makes the screenshots of the User Guide: dicomdoodle/assets/guide/*.png.

The screenshots show real, anonymized scans: an axial contrast-enhanced T1
series with a tumour (plus its coronal and sagittal series for the 3D
Model Viewer) and an Enhanced MR multi-frame file for the converter. The
slices are copied into a temporary work folder with every date and the
patient name and IDs replaced by "ANONYMIZED", so no identifier reaches the pictures; the
original folders are only read. The lesion outlined in the series' existing
annotation files tells the script where to draw boxes and put markers.

Orange numbered markers are drawn on the widgets that the guide's tables
explain — when you move a marker here, check the matching table in
dicomdoodle/help/part_*.py. The slice numbers quoted in the guide (lesion
on slices 114-151, Auto Detect on 133 …) are those of the default series.

    python scripts/make_guide_images.py [--axial DIR --coronal DIR --sagittal DIR --multiframe FILE]
    xvfb-run -a -s "-screen 0 1600x1000x24" python scripts/make_guide_images.py   # Linux without a desktop

The AI models really run (Auto Detect, SAM and a batch over the lesion), so
this takes about ten minutes on a CPU. The ontology screenshot needs
internet access (BioPortal); without it the existing screenshot is kept.
Settings and data go to a temporary folder, so your own settings are not
touched.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
import traceback
import urllib.request
from typing import Callable, Dict, List, Optional, Sequence, Tuple

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP_DIR)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pydicom  # noqa: E402
from PyQt5.QtCore import QEvent, QEventLoop, QPoint, QPointF, QRect, QRectF, Qt, QTimer  # noqa: E402
from PyQt5.QtGui import QImage, QMouseEvent, QPainter  # noqa: E402
from PyQt5.QtWidgets import (QApplication, QFileDialog, QLabel, QMessageBox, QPushButton,  # noqa: E402
                             QTabWidget, QToolBar, QVBoxLayout, QWidget)

from dicomdoodle import resources  # noqa: E402
from dicomdoodle.app import _setup_logging  # noqa: E402
from dicomdoodle.core.ai_models import ModelManager  # noqa: E402
from dicomdoodle.core import mask_ops  # noqa: E402
from dicomdoodle.core.dicom_series import DicomSeries, SegFileFilter  # noqa: E402
from dicomdoodle.core.labels import LabelRegistry  # noqa: E402
from dicomdoodle.core.seg_io import is_blank_segmentation, read_segmentation  # noqa: E402
from dicomdoodle.settings import Settings  # noqa: E402
from dicomdoodle.ui.dialogs.progress_dialog import TaskProgressDialog  # noqa: E402
from dicomdoodle.ui.guide.drawing import paint_marker  # noqa: E402
from dicomdoodle.ui.qt_utils import numpy_to_qimage  # noqa: E402
from dicomdoodle.ui.theme import install_theme_manager, theme_manager  # noqa: E402

OUT_DIR = resources.GUIDE_DIR
MAIN_SIZE = (1200, 760)          # main window and most tool windows
WIDE_SIZE = (1640, 940)          # multiplanar & 3D viewer: two side panels
MARKER_RADIUS = 13

PATIENT = "/data2/icmr/pushp/data_AIIMS/00014AIIMSD281023"
DEFAULT_AXIAL = os.path.join(PATIENT, "RECON AX 3D T1 FS+C")
DEFAULT_CORONAL = os.path.join(PATIENT, "RECON COR 3D T1 FS+C")
DEFAULT_SAGITTAL = os.path.join(PATIENT, "SAG 3D T1 FS+C (IGS)")
DEFAULT_MULTIFRAME = "/data2/icmr/pushp/DicomDoodle/AXIAL/PGICHIMG-0012-00001.dcm"

Marker = Tuple[int, QPoint]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def wait(ms: int = 150) -> None:
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec_()


def wait_until(condition: Callable[[], bool], what: str, timeout: float = 600.0) -> None:
    start = time.time()
    while not condition():
        if time.time() - start > timeout:
            raise TimeoutError(f"timed out waiting for {what}")
        wait(100)


def at(widget: QWidget, root: QWidget, fx: float = 0.5, fy: float = 0.5, dx: int = 0, dy: int = 0) -> QPoint:
    """Point of ``widget`` (fractions of its size plus an offset) in ``root`` coordinates."""
    return widget.mapTo(root, QPoint(int(widget.width() * fx) + dx, int(widget.height() * fy) + dy))


def on_axes(panel, root: QWidget, x: float, y: float, dx: int = 0, dy: int = 0) -> QPoint:
    """Point (x, y) in the data coordinates of a Multiplanar view (matplotlib), in ``root`` coordinates."""
    px, py = panel.ax.transData.transform((x, y))
    ratio = getattr(panel.canvas, "device_pixel_ratio", 1) or 1
    return panel.canvas.mapTo(root, QPoint(int(px / ratio) + dx, int(panel.canvas.height() - py / ratio) + dy))


def on_image(canvas, root: QWidget, x: float, y: float, dx: int = 0, dy: int = 0) -> QPoint:
    """Image pixel (x, y) of an image canvas, in ``root`` coordinates."""
    view = canvas.mapFromScene(QPointF(x, y))
    return canvas.viewport().mapTo(root, view + QPoint(dx, dy))


def on_hud(canvas, root: QWidget, corner: str, fx: float, fy: float, dx: int = 0, dy: int = 0) -> QPoint:
    """Point of a canvas corner text (see ImageCanvas.hud_rects), in ``root`` coordinates."""
    r = canvas.hud_rects[corner]
    return canvas.viewport().mapTo(root, QPoint(int(r.left() + r.width() * fx) + dx,
                                                int(r.top() + r.height() * fy) + dy))


def tool_button(window: QWidget, action) -> QWidget:
    for bar in window.findChildren(QToolBar):
        button = bar.widgetForAction(action)
        if button is not None:
            return button
    raise LookupError(action.text())


def toolbar_action(window: QWidget, text: str):
    for bar in window.findChildren(QToolBar):
        for action in bar.actions():
            if action.text().replace("&&", "&") == text:
                return action
    raise LookupError(text)


def button(window: QWidget, text: str) -> QPushButton:
    return next(b for b in window.findChildren(QPushButton) if b.text().replace("&&", "&").strip() == text)


def top_level(kind) -> QWidget:
    return next(w for w in QApplication.topLevelWidgets() if isinstance(w, kind) and w.isVisible())


# --------------------------------------------------------------------------- #
# The example data
# --------------------------------------------------------------------------- #

_PRIVATE_KEYWORDS = ("AdmissionID", "OtherPatientIDs", "OtherPatientIDsSequence", "PatientBirthDate")


def redacted_copy(source: str, target: str) -> str:
    """Copy a DICOM file without its dates and IDs (the screenshots show headers)."""
    ds = pydicom.dcmread(source)
    for element in list(ds):
        if element.VR in ("DA", "DT") or element.keyword in _PRIVATE_KEYWORDS:
            element.value = ""
    ds.PatientName = ds.PatientID = "ANONYMIZED"
    ds.save_as(target)
    return target


def copy_series(source: str, work: str) -> str:
    """The image slices of ``source`` (not its annotation files) in ``work/<folder name>``."""
    folder = os.path.join(work, os.path.basename(os.path.normpath(source)))
    os.makedirs(folder, exist_ok=True)
    series = DicomSeries.load(source)
    for i in range(len(series)):
        redacted_copy(series.file(i), os.path.join(folder, os.path.basename(series.file(i))))
    return os.path.basename(folder)          # relative: the window shows the folder's name, not a temp path


class Example:
    """The series of the guide, and the lesion of its existing annotation files (filled, per slice)."""

    def __init__(self, source: str, work: str):
        self.folder = copy_series(source, work)
        series = DicomSeries.load(source)
        self.size = series.image(0).shape[:2]
        registry = LabelRegistry()
        self.masks: Dict[int, np.ndarray] = {}
        for i in range(len(series)):
            path = series.seg_path(i)
            if os.path.isfile(path) and not is_blank_segmentation(path):
                outline = read_segmentation(path, registry, self.size).label_map
                self.masks[i] = mask_ops.fill_closed_boundaries(outline) > 0
        if not self.masks:
            raise RuntimeError(f"{source}: no annotation files with a lesion; the script needs them to find it")
        self.slices: List[int] = sorted(self.masks)
        ys, xs = np.where(np.any([self.masks[i] for i in self.slices], axis=0))
        cx, cy, half = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2, max(np.ptp(xs), np.ptp(ys)) * 0.85
        self.view = QRectF(cx - half, cy - half, 2 * half, 2 * half)      # image region around the lesion

    @property
    def middle(self) -> int:
        return self.slices[len(self.slices) // 2]

    def box(self, index: int, margin: int) -> Tuple[int, int, int, int]:
        ys, xs = np.where(self.masks[index])
        return xs.min() - margin, ys.min() - margin, xs.max() + margin, ys.max() + margin

    def centre(self, index: int) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.box(index, 0)
        return (x1 + x2) / 2, (y1 + y2) / 2


def zoom_on(canvas, view: QRectF) -> None:
    canvas.fitInView(view, Qt.KeepAspectRatio)
    canvas.zoom_by(1.0)               # leaves "fit to window" mode, like a user zoom


def hide_toasts(root: QWidget) -> None:
    from dicomdoodle.ui.widgets.toast import Toast
    for toast in root.findChildren(Toast):
        toast.hide()


def grab(root: QWidget) -> QImage:
    hide_toasts(root)
    wait(250)
    image = root.grab().toImage()
    ratio = image.devicePixelRatio()
    if ratio != 1:                     # keep screenshots at logical size on HiDPI screens
        image = image.scaled(int(image.width() / ratio), int(image.height() / ratio),
                             Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        image.setDevicePixelRatio(1)
    return image.convertToFormat(QImage.Format_RGB32)


def paint_vtk(image: QImage, root: QWidget, vtk_widget) -> None:
    """QWidget.grab() cannot see OpenGL content: copy each VTK frame into the screenshot."""
    import vtk
    from vtkmodules.util.numpy_support import vtk_to_numpy
    window = vtk_widget.GetRenderWindow()
    window.Render()
    grabber = vtk.vtkWindowToImageFilter()
    grabber.SetInput(window)
    grabber.ReadFrontBufferOff()
    grabber.Update()
    data = grabber.GetOutput()
    width, height, _depth = data.GetDimensions()
    pixels = vtk_to_numpy(data.GetPointData().GetScalars()).reshape(height, width, -1)[::-1, :, :3]
    painter = QPainter(image)
    painter.setRenderHint(QPainter.SmoothPixmapTransform)
    painter.drawImage(QRect(vtk_widget.mapTo(root, QPoint(0, 0)), vtk_widget.size()),
                      numpy_to_qimage(np.ascontiguousarray(pixels)))
    painter.end()


def save(name: str, image: QImage, markers: Sequence[Marker] = (), crop: Optional[QRect] = None) -> None:
    offset = QPoint(0, 0)
    if crop is not None:
        image = image.copy(crop)
        offset = crop.topLeft()
    painter = QPainter(image)
    for number, point in markers:
        paint_marker(painter, QPointF(point - offset), str(number), MARKER_RADIUS)
    painter.end()
    image.save(os.path.join(OUT_DIR, f"{name}.png"), "PNG")
    print(f"  {name}.png  {image.width()} × {image.height()}", flush=True)


def rect_of(widget: QWidget, root: QWidget, margin: int = 0) -> QRect:
    return QRect(widget.mapTo(root, QPoint(0, 0)), widget.size()).adjusted(-margin, -margin, margin, margin)


def send_mouse(widget: QWidget, kind, pos: QPoint, button_, buttons) -> None:
    QApplication.sendEvent(widget, QMouseEvent(kind, QPointF(pos), button_, buttons, Qt.NoModifier))


def draw_outline(canvas, mask: np.ndarray) -> None:
    """Trace the outline of ``mask`` with the mouse on a drawing canvas."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    points = [canvas.mapFromScene(QPointF(x + 0.5, y + 0.5)) for x, y in contours[0][:, 0, :]]
    points.append(points[0])
    viewport = canvas.viewport()
    send_mouse(viewport, QEvent.MouseButtonPress, points[0], Qt.LeftButton, Qt.LeftButton)
    for point in points[1:]:
        send_mouse(viewport, QEvent.MouseMove, point, Qt.NoButton, Qt.LeftButton)
    send_mouse(viewport, QEvent.MouseButtonRelease, points[-1], Qt.LeftButton, Qt.NoButton)


class ModalWatcher:
    """Handles modal windows that appear while a job runs (they would block the script)."""

    def __init__(self):
        self.on_summary: Optional[Callable[[TaskProgressDialog], None]] = None
        self.timer = QTimer()
        self.timer.setInterval(200)
        self.timer.timeout.connect(self._check)
        self.timer.start()

    def _check(self) -> None:
        widget = QApplication.activeModalWidget()
        if isinstance(widget, TaskProgressDialog) and not widget.running:
            callback, self.on_summary = self.on_summary, None
            (callback or (lambda d: d.accept()))(widget)
        elif isinstance(widget, QMessageBox):
            print(f"  (closed an unexpected message: {widget.text()!r})", flush=True)
            widget.reject()


# --------------------------------------------------------------------------- #
# The screenshots
# --------------------------------------------------------------------------- #

class GuideScreenshots:
    def __init__(self, work: str, args):
        self.app = QApplication.instance() or QApplication(sys.argv)
        install_theme_manager(self.app, "dark")
        self.work = work
        os.chdir(work)                  # folders show by name, not as a temp path
        print("copying the example series …", flush=True)
        self.ax = Example(args.axial, work)
        self.coronal = copy_series(args.coronal, work) if args.coronal else ""
        self.sagittal = copy_series(args.sagittal, work) if args.sagittal else ""
        self.multiframe = redacted_copy(args.multiframe, os.path.join(work, os.path.basename(args.multiframe)))
        self.auto_slice = self.ax.slices[0] + 19                 # slice 133: Quick Start, Auto Detect
        self.box_slice = self.auto_slice - 5                     # slice 128: the box
        self.draw_slice = self.auto_slice + 5                    # slice 138: the pen
        self.settings = Settings(os.path.join(work, "settings.json"))
        self.settings.update(preload_sam=False, recent_folders=[])
        from dicomdoodle.ui.label_manager import LabelManager
        from dicomdoodle.ui.main import MainWindow
        self.models = ModelManager(self.settings.get("models_dir"), self.settings.get("device"))
        self.labels = LabelManager(self.settings)
        self.window = MainWindow(self.settings, self.labels, self.models, _setup_logging())
        self.window.resize(*MAIN_SIZE)
        self.window.show()
        self.watcher = ModalWatcher()
        self.last_detection = None
        canvas = self.window.canvas
        show_detection = canvas.show_detection

        def remember_detection(box=None, text: str = "") -> None:   # where Auto Detect found the lesion
            self.last_detection = box
            show_detection(box, text)
        canvas.show_detection = remember_detection
        wait(500)

    # ------------------------------------------------------------- start page
    def welcome(self) -> None:
        w, page = self.window, self.window.welcome
        examples = [os.path.join(os.sep, "data", "mri", "Example study A", "T1 contrast axial"),
                    os.path.join(os.sep, "data", "mri", "Example study A", "FLAIR axial"),
                    os.path.join(os.sep, "data", "mri", "Example study B", "T1 contrast axial")]
        real_isdir = os.path.isdir
        os.path.isdir = lambda p: p in examples or real_isdir(p)     # example entries for the screenshot
        try:
            page.set_recent(examples)
        finally:
            os.path.isdir = real_isdir
        w.resize(MAIN_SIZE[0], 1000)                 # tall enough for the whole start page
        wait(400)
        card = next(c for c in page.findChildren(QWidget) if c.objectName() == "Card")
        recent = page.recent_box.itemAt(0).widget()
        first_tool = page.tools_grid.itemAt(0).widget()
        save("welcome", grab(w), [
            (1, at(page.open_btn, w, 0.0, 0.5, dx=-20)),
            (2, at(page.guide_btn, w, 1.0, 0.5, dx=20)),
            (3, at(recent, w, 0.0, 0.5, dx=-20)),
            (4, at(first_tool, w, 0.0, 0.5, dx=-20)),
        ], rect_of(card, w, 1))
        w.resize(*MAIN_SIZE)

    # ------------------------------------------------- box, SAM and saving
    def box_and_segment(self) -> None:
        w = self.window
        w.open_folder(self.ax.folder)
        wait_until(lambda: w.session is not None, "the example series")
        index = self.box_slice
        w.go_to_slice(index)
        zoom_on(w.canvas, self.ax.view)
        x1, y1, x2, y2 = self.ax.box(index, 10)
        w.canvas.set_box(x1, y1, x2, y2)
        wait(300)
        crop = rect_of(w.canvas, w)
        save("box", grab(w), [
            (1, on_image(w.canvas, w, (x1 + x2) / 2, y1, dy=-24)),
            (2, on_image(w.canvas, w, x2, y2, dx=20, dy=20)),
            (3, on_hud(w.canvas, w, "bottom_left", 1.0, 0.5, dx=20)),
        ], crop)

        w.annotation.segment_box()
        wait_until(lambda: not w.busy and w.session.annotation(index).label_map.any(), "SAM")
        wait(400)
        timeline = w.navigator.timeline
        tick = timeline.mapTo(w, QPoint(int(timeline.x_of(index)), int(timeline.height() / 2 + 8)))
        cy = (y1 + y2) / 2
        save("segmented", grab(w), [
            (1, on_image(w.canvas, w, x2 - 5, cy, dx=24)),
            (2, on_hud(w.canvas, w, "top_left", 1.0, 0.5, dx=20)),
            (3, on_hud(w.canvas, w, "top_right", 0.0, 0.5, dx=-20)),
            (4, tick + QPoint(34, 0)),
        ], crop.united(rect_of(w.navigator, w)))
        w.annotation.save_current()
        wait_until(lambda: not w.busy and w.session.is_saved(index), f"saving slice {index + 1}")

    # ------------------------------------------------------------ batch panel
    def batch_panel(self) -> None:
        from dicomdoodle.ui.widgets.batch_panel import BatchPanel
        w = self.window
        w.go_to_slice(self.auto_slice)               # Find tumour range from slice 133
        w.batch.find_tumour_range()
        wait_until(lambda: not w.busy and w.batch.found is not None, "Find tumour range")
        found, s = w.batch.found, w.session
        print(f"  {found.summary_text()} ({found.seconds:.1f} s on {found.device}); the annotation files: "
              f"slices {self.ax.slices[0] + 1}-{self.ax.slices[-1] + 1}", flush=True)
        # the panel on its own, with room around it for the markers
        holder = QWidget()
        holder.setObjectName("DockBody")
        holder.setAttribute(Qt.WA_StyledBackground, True)
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(36, 18, 36, 18)
        panel = BatchPanel()
        layout.addWidget(panel)
        panel.show_auto_range(s.auto_range, len(s.auto_targets()), w.label_panel.active_label().display_name)
        panel.show_found_range(found)
        panel.show_blank_masks(True, (len(s.files - s.blank_files), len(s.blank_files), len(s.series) - len(s.files)))
        holder.resize(380, max(holder.sizeHint().height(), holder.heightForWidth(380)))   # wrapped text, preview
        holder.show()
        wait(400)
        row = panel.auto_row
        save("batch", grab(holder), [
            (1, at(row.start_btn, holder, 0.0, 0.5, dx=-19)),
            (2, at(row.clear_btn, holder, 1.0, 0.5, dx=19)),
            (3, at(panel.find_range_btn, holder, 0.0, 0.5, dx=-19)),
            (4, at(panel.preview, holder, 0.0, 0.4, dx=-19)),
            (5, at(panel.auto_label_hint, holder, 0.0, 0.5, dx=-19)),
            (6, at(panel.run_auto_btn, holder, 0.0, 0.5, dx=-19)),
            (7, at(panel.auto_blank_check, holder, 0.0, 0.5, dx=-19)),
            (8, at(panel.stats["annotated"], holder, 0.0, 0.5, dx=-19)),
        ])
        holder.close()
        # the rest of the guide (batch, review, volumes) uses the lesion's own slices: 114 to 151
        w.batch.show_found(None)
        w.go_to_slice(self.ax.slices[0])
        w.batch.range_action("start")
        w.go_to_slice(self.ax.slices[-1])
        w.batch.range_action("end")

    # ------------------------------------------------ main window, Auto Detect
    def overview_and_autodetect(self) -> None:
        w = self.window
        index = self.auto_slice
        w.go_to_slice(index)
        w.batch_dock.raise_()
        w.annotation.auto_detect()
        wait_until(lambda: not w.busy, "Auto Detect")
        if not w.session.annotation(index).label_map.any():
            raise RuntimeError(f"Auto Detect found nothing on slice {index + 1}")
        w.canvas.fit_to_window()
        wait(500)
        bar = w.menuBar()
        help_menu = bar.actionGeometry(bar.actions()[-1])
        labels_card = w.label_panel.parentWidget()
        display_card = w.style_combo.parentWidget()
        timeline = w.navigator.timeline
        save("overview", grab(w), [
            (1, bar.mapTo(w, QPoint(help_menu.right() + 22, help_menu.center().y()))),
            (2, at(tool_button(w, w.actions.zoom_in), w, 1.0, 0.5, dx=22)),
            (3, at(tool_button(w, w.actions.tool_actions["pan"]), w, 1.0, 0.5, dx=20)),
            (4, at(labels_card.title_label, w, 1.0, 0.5, dx=20)),
            (5, at(display_card.title_label, w, 1.0, 0.5, dx=20)),
            (6, at(w.canvas, w, 0.08, 0.5)),
            (7, on_hud(w.canvas, w, "top_right", 0.0, 0.5, dx=-20)),
            (8, timeline.mapTo(w, QPoint(int(timeline.x_of(self.ax.slices[0] - 2)), timeline.height() // 2))),
            (9, at(w.batch_panel, w, 0.62, 0.0, dy=-24)),
            (10, at(w.zoom_label, w, 0.0, 0.5, dx=-18)),
        ])

        zoom_on(w.canvas, self.ax.view)
        wait(300)
        x1, y1, x2, y2 = self.last_detection
        save("autodetect", grab(w), [
            (1, on_image(w.canvas, w, x1, y1, dx=-18, dy=-18)),
            (2, on_image(w.canvas, w, *self.ax.centre(index))),
            (3, on_hud(w.canvas, w, "top_right", 0.0, 0.5, dx=-20)),
            (4, on_hud(w.canvas, w, "bottom_left", 1.0, 0.5, dx=20)),
        ], rect_of(w.canvas, w))
        w.canvas.fit_to_window()

    # --------------------------------------------------------------- results
    def results(self) -> None:
        from dicomdoodle.ui.dialogs.results_dialog import ResultsDialog
        w = self.window
        index = w.current
        ann = w.session.annotation(index)
        registry = w.labels.registry
        legend = [(registry.get(v).display_name, registry.color_for(v)) for v in ann.label_values()]
        stem = os.path.splitext(os.path.basename(w.session.series.file(index)))[0]
        dialog = ResultsDialog(w.session.series.image(index), ann.filled_map(), ann.outline_map(), registry.color_lut(),
                               legend, f"Slice {index + 1}", stem, w)
        dialog.resize(1060, 560)
        dialog.show()
        wait(400)
        for view in (dialog.filled_view, dialog.boundary_view):
            zoom_on(view, self.ax.view)
        wait(300)
        left, right = dialog.filled_view.parentWidget(), dialog.boundary_view.parentWidget()
        legend_text = next(l for l in dialog.findChildren(QLabel) if l.text() == legend[0][0])
        save_button = button(left, "Save image…")
        save("results", grab(dialog), [
            (1, at(left.title_label, dialog, 1.0, 0.5, dx=20)),
            (2, at(right.title_label, dialog, 1.0, 0.5, dx=20)),
            (3, at(dialog.opacity, dialog, 0.5, 1.0, dy=10)),
            (4, at(legend_text, dialog, 1.0, 0.5, dx=18)),
            (5, at(save_button, dialog, 0.0, 0.5, dx=-20)),
        ])
        dialog.close()

    # ----------------------------------------------------------- label editor
    def label_editor(self) -> None:
        from dicomdoodle.ui.dialogs.label_editor import LabelEditorDialog, _ColorChip
        w = self.window
        dialog = LabelEditorDialog(self.labels, parent=w)
        dialog.name_edit.setText("Enhancing Tumor Boundary")
        dialog.short_edit.setText("Enh. Tumor")
        chips = dialog.findChildren(_ColorChip)
        suggested = [c for c in chips if c.isEnabled()]
        used = [c for c in chips if not c.isEnabled()]
        suggested[0].click()
        dialog.show()
        dialog.name_edit.deselect()
        wait(400)
        save("label_editor", grab(dialog), [
            (1, at(dialog.name_edit, dialog, 1.0, 0.5, dx=18)),
            (2, at(dialog.short_edit, dialog, 1.0, 0.5, dx=18)),
            (3, at(dialog.preview, dialog, 1.0, 0.5, dx=18)),
            (4, at(suggested[5], dialog, 1.0, 0.5, dx=20)),
            (5, at(used[-1], dialog, 1.0, 0.5, dx=20)),
            (6, at(dialog.picker, dialog, 1.0, 0.0, dx=-16, dy=-14)),
            (7, at(dialog.ok_button, dialog, 0.0, 0.5, dx=-20)),
        ])
        dialog.reject()

    # ------------------------------------------------------- drawing by hand
    def manual_draw(self) -> None:
        w = self.window
        index = self.draw_slice
        w.go_to_slice(index)
        w.drawing.set_tool("pen")
        w.tool_strip.size_spin.setValue(2)
        zoom_on(w.canvas, self.ax.view)
        wait(200)
        draw_outline(w.canvas, self.ax.masks[index])
        w.annotation.save_current()                  # saved: "Edit saved outline" is now available
        wait_until(lambda: not w.busy and w.session.is_saved(index), f"saving slice {index + 1}")
        wait(400)
        strip, canvas = w.tool_strip, w.canvas
        ys, xs = np.where(self.ax.masks[index])
        save("manual_draw", grab(w), [
            (1, at(tool_button(w, w.actions.tool_actions["box"]), w, 0.5, 1.0, dy=14)),
            (2, at(tool_button(w, w.actions.tool_actions["pen"]), w, 0.5, 1.0, dy=14)),
            (3, at(tool_button(w, w.actions.tool_actions["eraser"]), w, 0.5, 1.0, dy=14)),
            (4, at(tool_button(w, w.actions.tool_actions["pan"]), w, 0.5, 1.0, dy=14)),
            (5, at(strip.size_spin, w, 0.5, 1.0, dy=14)),
            (6, at(tool_button(w, w.actions.edit_saved), w, 0.5, 1.0, dy=14)),
            (7, on_image(canvas, w, xs.max(), (ys.min() + ys.max()) / 2, dx=26)),
            (8, on_hud(canvas, w, "bottom_left", 1.0, 0.5, dx=20)),
        ])
        w.drawing.set_tool("box")
        canvas.fit_to_window()

    # ------------------------------------------------------------ preferences
    def preferences(self) -> None:
        from dicomdoodle.ui.dialogs.settings_dialog import SettingsDialog
        dialog = SettingsDialog(self.settings, self.models, self.window)
        dialog.findChild(QTabWidget).setCurrentIndex(1)       # Display
        dialog.show()
        wait(400)
        save("preferences", grab(dialog))
        dialog.reject()

    # ------------------------------------------------------------ light theme
    def light(self) -> None:
        self.window.go_to_slice(self.auto_slice)
        self.window.canvas.fit_to_window()
        theme_manager().apply("light")
        wait(600)
        save("light", grab(self.window))
        theme_manager().apply("dark")
        wait(400)

    # ------------------------------------------------------- batch processing
    def batch_run(self) -> None:
        w = self.window
        index = self.auto_slice
        w.go_to_slice(index)
        w.annotation.save_current()                  # saved slices are skipped by batches
        wait_until(lambda: not w.busy and w.session.is_saved(index), f"saving slice {index + 1}")
        self.watcher.on_summary = lambda dialog: (save("progress", grab(dialog)), dialog.accept())
        w.batch.run_auto_detect()
        wait(300)

    def review(self) -> None:
        """Check the batch like a user would: where the AI outline misses the lesion (slices 114 and 142-151
        of the default series), the outline is drawn by hand — the routine the guide recommends."""
        from dicomdoodle.core.seg_io import HAND_DRAWN, write_segmentation
        w = self.window
        wait_until(lambda: not w.busy, "the auto-detect batch")
        series, registry, corrected = w.session.series, LabelRegistry(), []
        for index in self.ax.slices:
            path = series.seg_path(index)
            found = np.zeros(self.ax.size, bool)
            if os.path.isfile(path) and not is_blank_segmentation(path):
                found = mask_ops.fill_closed_boundaries(read_segmentation(path, registry, self.ax.size).label_map) > 0
            truth = self.ax.masks[index]
            if 2 * (found & truth).sum() / max(1, found.sum() + truth.sum()) < 0.8:
                write_segmentation(mask_ops.boundary_map(truth.astype(np.uint8)), series.file(index), path, index,
                                   labels=w.labels.snapshot(), method=HAND_DRAWN)
                corrected.append(index + 1)
        print(f"  corrected by hand (the AI outline missed the lesion): slices {corrected}", flush=True)

    # ------------------------------------------------------- annotated viewer
    def viewer(self) -> None:
        from dicomdoodle.ui.tools.overlay_viewer import OverlayViewerWindow
        w = self.window
        w.tools.overlay_viewer()
        viewer = top_level(OverlayViewerWindow)
        viewer.resize(*MAIN_SIZE)
        wait_until(lambda: viewer.series is not None, "the annotated viewer")
        viewer.show_slice(self.auto_slice)
        viewer.mask_action.setChecked(True)
        viewer.opacity.setValue(55)
        viewer.render_overlay()
        wait(500)
        file_card = viewer.file_label.parentWidget()
        segments_card = viewer.segments_table.parentWidget()
        summary_card = viewer.summary_label.parentWidget()
        save("viewer", grab(viewer), [
            (1, at(tool_button(viewer, toolbar_action(viewer, "Load DICOM Folder")), viewer, 0.5, 1.0, dy=14)),
            (2, at(tool_button(viewer, viewer.zoom_action), viewer, 0.5, 1.0, dy=14)),
            (3, at(tool_button(viewer, viewer.mask_action), viewer, 0.5, 1.0, dy=14)),
            (4, at(viewer.opacity, viewer, 0.5, 1.0, dy=14)),
            (5, at(viewer.filter_combo, viewer, 0.85, 1.0, dy=14)),
            (6, at(tool_button(viewer, viewer.ontology_action), viewer, 0.5, 1.0, dy=14)),
            (7, at(file_card.title_label, viewer, 1.0, 0.5, dx=20)),
            (8, at(segments_card.title_label, viewer, 1.0, 0.5, dx=20)),
            (9, at(summary_card.title_label, viewer, 1.0, 0.5, dx=20)),
        ])
        self.viewer_window = viewer

    # ----------------------------------------------------------- ontology
    def ontology(self) -> None:
        from dicomdoodle.ui.tools.ontology_search import OntologySearchWindow
        try:
            urllib.request.urlopen("https://data.bioontology.org", timeout=10)
        except Exception as exc:                     # noqa: BLE001 — any network problem
            if "401" not in str(exc) and "403" not in str(exc):
                print(f"  ontology: BioPortal not reachable ({exc}); keeping the old screenshot", flush=True)
                return
        self.viewer_window.open_ontology_search()
        onto = top_level(OntologySearchWindow)
        onto.resize(1160, 720)
        onto.segment_combo.setCurrentIndex(max(0, onto.segment_combo.findText("Tumor & Brain Boundary")))
        onto.category_combo.setCurrentIndex(onto.category_combo.findText("Morphologically Abnormal Structure",
                                                                         Qt.MatchStartsWith))
        onto.search_edit.setText("lesion of brain")                # one DICOM term, then BioPortal's
        onto.search()
        onto.search_bioportal()
        try:
            wait_until(lambda: onto._bioportal_done, "BioPortal results", timeout=60)
        except TimeoutError:
            print("  ontology: BioPortal did not answer; keeping the old screenshot", flush=True)
            onto.close()
            return
        onto.list.setCurrentRow(0)
        onto.definition_btn.click()
        wait_until(lambda: onto.definition.toPlainText() not in ("", "Loading from BioPortal…"), "definition",
                   timeout=60)
        onto.as_type_btn.click()
        wait(500)
        details = onto.as_type_btn.parentWidget()
        choice = onto.apply_btn.parentWidget()
        save("ontology", grab(onto), [
            (1, at(onto.search_edit, onto, 0.82, 0.5)),
            (2, at(onto.search_btn, onto, 0.5, 1.0, dy=14)),
            (3, at(onto.list, onto, 0.6, 0.55)),
            (4, at(details.title_label, onto, 1.0, 0.5, dx=20)),
            (5, at(onto.as_category_btn, onto, 0.0, 0.5, dx=-17)),
            (6, at(choice.title_label, onto, 1.0, 0.5, dx=20)),
        ])
        onto.close()

    # ---------------------------------------------- multiplanar & 3D viewer
    def mpr(self) -> None:
        from dicomdoodle.ui.tools.mpr.window import MPRWindow, Plane
        w = self.window
        w.tools.mpr_viewer()
        mpr = self.mpr_window = top_level(MPRWindow)
        mpr.resize(*WIDE_SIZE)
        wait(400)
        original = QFileDialog.getExistingDirectory
        QFileDialog.getExistingDirectory = staticmethod(lambda *args, **kwargs: self.ax.folder)
        try:
            mpr.on_load_primary()
        finally:
            QFileDialog.getExistingDirectory = original
        if mpr.annotation_combo.count() > 1:
            mpr.annotation_combo.setCurrentIndex(1)
        # put the crosshair on the lesion (the volume is resampled, so work in fractions)
        index, slices = self.auto_slice, len(self.window.session.series)
        cx, cy = self.ax.centre(index)
        depth, rows, columns = mpr.volume.shape
        fy, fx = cy / (self.ax.size[0] - 1), cx / (self.ax.size[1] - 1)
        mpr.on_slice_changed(Plane.AXIAL, int(round(index / (slices - 1) * (depth - 1))))
        click = (int(round((1 - fy) * (rows - 1))), int(round(fx * (columns - 1))))
        mpr.axial_panel.positionPicked.emit(Plane.AXIAL, *click)      # as a click on the lesion does

        # the 3D model of the axial, coronal and sagittal series (right-hand panel)
        for plane, folder in (("Coronal", self.coronal), ("Sagittal", self.sagittal)):
            if folder:
                mpr.side.series.set_folder(plane, folder)
        mpr.build_model(mpr.side.series.paths(), SegFileFilter.ALL)
        wait_until(lambda: mpr.model is not None and bool(mpr.side.volume.results), "the 3D model")
        print(f"  (lesion volume from the axial annotations: {mpr.side.volume.results[0].volume_cm3:.2f} cm³)",
              flush=True)
        mpr.view_combo.setCurrentIndex(mpr.view_combo.findData("annotated"))
        self._turn_camera(mpr, -60, 1.5)
        wait(1500)
        save("mpr", self._grab_mpr(), [
            (1, at(button(mpr, "Load primary series…"), mpr, 1.0, 0.5, dx=-16)),
            (2, at(mpr.register_btn, mpr, 1.0, 0.5, dx=-16)),
            (3, at(mpr.source_combo, mpr, 1.0, 0.5, dx=-36)),
            (4, at(mpr.algo_combo, mpr, 1.0, 0.5, dx=-36)),
            (5, at(mpr.image_combo, mpr, 1.0, 0.5, dx=-36)),
            (6, at(mpr.annotation_combo, mpr, 1.0, 0.5, dx=-36)),
            (7, at(mpr.axial_panel.body, mpr, 0.08, 0.12)),
            (8, at(mpr.coronal_panel.toggle_btn, mpr, 0.5, 1.0, dy=16)),
            (9, at(mpr.three_d_panel.body, mpr, 0.08, 0.12)),
            (10, at(mpr.axial_panel.detect_btn, mpr, 0.5, 1.0, dy=16)),
            (11, at(mpr.control_tabs.tabBar(), mpr, 1.0, 0.5, dx=18)),
            (12, at(mpr.side.series.title_label, mpr, 1.0, 0.5, dx=20)),
        ])

        # Detect from the axial view, after the click on the lesion (best_flair + SAM 2.1, minutes on a CPU)
        started = time.time()
        mpr.on_detect_requested(Plane.AXIAL)
        if mpr.detection is None:
            raise RuntimeError("Detect gave no result")
        print(f"   Detect: {mpr.detection.summary_text()} ({time.time() - started:.0f} s)\n   "
              f"{mpr.statusBar().currentMessage().rsplit('·', 1)[-1].strip()}", flush=True)
        wait(1500)
        box = mpr.detection.display_box(Plane.CORONAL, mpr._idx(1), mpr.volume.shape)
        save("mpr_detect", self._grab_mpr(), [
            (1, at(mpr.axial_panel.detect_btn, mpr, 0.5, 1.0, dy=16)),
            (2, on_axes(mpr.coronal_panel, mpr, box[0], box[1], dx=-14, dy=-14) if box is not None
             else at(mpr.coronal_panel.body, mpr, 0.1, 0.1)),
            (3, at(mpr.annotation_combo, mpr, 1.0, 0.5, dx=-36)),
        ])

    def _grab_mpr(self) -> QImage:
        mpr = self.mpr_window
        image = grab(mpr)
        if mpr.three_d_panel.view is not None:
            paint_vtk(image, mpr, mpr.three_d_panel.view.vtk_widget)
        return image

    def _turn_camera(self, mpr, elevation: float, zoom: float) -> None:
        view = mpr.three_d_panel.view
        view.reset_camera()
        camera = view.renderer.GetActiveCamera()
        camera.Elevation(elevation)
        camera.OrthogonalizeViewUp()
        camera.Zoom(zoom)
        view.renderer.ResetCameraClippingRange()
        view.render()

    def viewer3d(self) -> None:
        """The 3D view enlarged, cut open through the lesion, with its settings and the measurements."""
        mpr = self.mpr_window
        mpr.control_tabs.setCurrentIndex(1)
        mpr.on_toggle_maximize(mpr.three_d_panel)
        side, controls = mpr.side, mpr.view_controls
        side.metrics.show_surface.setChecked(True)
        controls.clip_positions["z"].setValue(round(100 * self.auto_slice / (len(self.window.session.series) - 1)))
        controls.clip_toggles["z"].setChecked(True)
        self._turn_camera(mpr, 150, 1.7)
        wait(1500)
        panel = mpr.three_d_panel
        save("viewer3d", self._grab_mpr(), [
            (1, at(mpr.view_combo, mpr, 1.0, 0.5, dx=-36)),
            (2, at(controls, mpr, 1.0, 0.08, dx=-18)),
            (3, at(panel.view.vtk_widget, mpr, 0.06, 0.08)),
            (4, at(panel.reset_btn, mpr, 0.0, 0.5, dx=-18)),
            (5, at(side.series.title_label, mpr, 1.0, 0.5, dx=20)),
            (6, at(side.volume.title_label, mpr, 1.0, 0.5, dx=20)),
            (7, at(side.metrics.title_label, mpr, 1.0, 0.5, dx=20)),
            (8, at(controls.measurement, mpr, 1.0, 0.5, dx=-24)),      # the right panel's twin is below the fold
        ])
        self.volume(mpr)
        mpr.on_toggle_maximize(mpr.three_d_panel)          # back to the 2 x 2 layout
        controls.clip_toggles["z"].setChecked(False)
        side.metrics.show_surface.setChecked(False)
        mpr.control_tabs.setCurrentIndex(0)

    def volume(self, mpr) -> None:
        """The right-hand panel on its own, with room around it for the markers."""
        from dicomdoodle.ui.tools.viewer3d.panels import SidePanel
        holder = QWidget()
        holder.setObjectName("DockBody")
        holder.setAttribute(Qt.WA_StyledBackground, True)
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(36, 18, 36, 18)
        panel = SidePanel()
        layout.addWidget(panel)
        source = mpr.side
        for name, path in source.series.paths().items():
            panel.series.set_folder(name, path)
        panel.series.show_info(source.series.info.text())
        panel.volume.show_results(source.volume.results)
        panel.metrics.show_metrics(source.metrics.metrics, source.metrics.hint.text())
        panel.metrics.show_surface.setChecked(True)
        holder.resize(400, max(holder.sizeHint().height(), holder.heightForWidth(400)))
        holder.show()
        wait(400)
        first_row = panel.volume.rows.itemAt(0).widget()
        save("volume", grab(holder), [
            (1, QPoint(panel.series.mapTo(holder, QPoint(0, 0)).x() - 17,
                       at(panel.series.edits["Axial"], holder).y())),
            (2, at(panel.series.filter_combo, holder, 0.0, 0.5, dx=-19)),
            (3, at(panel.series.build_btn, holder, 0.0, 0.5, dx=-19)),
            (4, at(panel.volume.total, holder, 0.0, 0.5, dx=-19)),
            (5, at(first_row, holder, 0.0, 0.3, dx=-19)),
            (6, at(panel.volume.copy_btn, holder, 0.0, 0.5, dx=-19)),
            (7, at(panel.metrics.title_label, holder, 0.0, 0.5, dx=-19)),
        ])
        holder.close()

    def skull(self) -> None:
        """Skull stripping (HD-BET, about 40 s on a CPU) in the slice views and the 3D view."""
        mpr = self.mpr_window
        started = time.time()
        mpr.image_combo.setCurrentIndex(mpr.image_combo.findData("skull_stripped"))
        if mpr.image_combo.currentData() != "skull_stripped":
            print("  HD-BET is not available: the skull screenshot is kept as it was", flush=True)
            mpr.close()
            return
        mpr.view_combo.setCurrentIndex(mpr.view_combo.findData("skull_stripped"))   # the same brain mask
        print(f"  HD-BET and the views: {time.time() - started:.0f} s", flush=True)
        self._turn_camera(mpr, 0, 1.5)                         # from the top: the brain's folds
        wait(1500)
        save("skull", self._grab_mpr(), [
            (1, at(mpr.image_combo, mpr, 1.0, 0.5, dx=-36)),
            (2, at(mpr.axial_panel.body, mpr, 0.08, 0.12)),
            (3, at(mpr.three_d_panel.body, mpr, 0.08, 0.12)),
        ])
        mpr.close()

    # ------------------------------------------------------------- converter
    def converter(self) -> None:
        from dicomdoodle.ui.tools.multiframe_converter import MultiFrameConverterWindow
        path = os.path.abspath(self.multiframe)
        w = self.window
        w.tools.multiframe_converter()
        conv = top_level(MultiFrameConverterWindow)
        conv.resize(*MAIN_SIZE)
        original = QFileDialog.getOpenFileName
        QFileDialog.getOpenFileName = staticmethod(lambda *args, **kwargs: (path, ""))
        try:
            conv.open_file()
        finally:
            QFileDialog.getOpenFileName = original
        conv.show_frame(15)
        wait(500)
        details = conv.info_values["File"].parentWidget()
        save("converter", grab(conv), [
            (1, at(tool_button(conv, toolbar_action(conv, "Open DICOM File…")), conv, 0.5, 1.0, dy=14)),
            (2, at(tool_button(conv, conv.save_action), conv, 0.5, 1.0, dy=14)),
            (3, on_hud(conv.canvas, conv, "top_left", 1.0, 0.5, dx=20)),
            (4, at(conv.navigator.timeline, conv, 0.3, 0.5)),
            (5, at(details.title_label, conv, 1.0, 0.5, dx=20)),
        ])
        conv.close()

    # ----------------------------------------------------------------- run
    def run(self) -> None:
        steps = [self.welcome, self.box_and_segment, self.batch_panel, self.overview_and_autodetect,
                 self.results, self.label_editor, self.manual_draw, self.preferences, self.light,
                 self.batch_run, self.review, self.viewer, self.ontology, self.mpr, self.viewer3d, self.skull,
                 self.converter]
        for step in steps:
            print(f"{step.__name__} …", flush=True)
            step()
        self.window.tools.close_all()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1].strip(),
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--axial", default=DEFAULT_AXIAL, help="series with a lesion and its annotation files")
    parser.add_argument("--coronal", default=DEFAULT_CORONAL, help="coronal series of the same study (3D viewer)")
    parser.add_argument("--sagittal", default=DEFAULT_SAGITTAL, help="sagittal series of the same study (3D viewer)")
    parser.add_argument("--multiframe", default=DEFAULT_MULTIFRAME, help="an Enhanced MR multi-frame file")
    parser.add_argument("--keep", action="store_true", help="keep the temporary work folder")
    args = parser.parse_args()
    for path in (args.axial, args.multiframe):
        if not os.path.exists(path):
            parser.error(f"not found: {path}")
    sys.excepthook = lambda *exc: traceback.print_exception(*exc)   # report, don't let PyQt abort
    os.makedirs(OUT_DIR, exist_ok=True)
    work = tempfile.mkdtemp(prefix="dicomdoodle_guide_")
    print(f"Writing screenshots to {OUT_DIR}\n(work folder: {work})", flush=True)
    try:
        GuideScreenshots(work, args).run()
    finally:
        os.chdir(APP_DIR)
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
