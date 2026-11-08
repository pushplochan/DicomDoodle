"""
Multi-frame to Single-frame Converter window (original "Convert Multi Frame
Dicom to Single Dicom"): open a multi-frame DICOM, browse its frames, and
save every frame as its own standard single-frame DICOM file (core/multiframe.py).
"""
from __future__ import annotations

import os
from typing import Optional

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import (QAction, QDockWidget, QFileDialog, QFormLayout, QMainWindow, QSizePolicy, QToolBar,
                             QVBoxLayout, QWidget)

from ...core.multiframe import MultiFrameDicom
from ...workers import run_task
from ..dialogs.progress_dialog import TaskProgressDialog
from ..guide import help_action
from ..icons import bind_icon
from ..qt_utils import label, show_error
from ..widgets.cards import Card, panel_body
from ..widgets.image_canvas import ImageCanvas
from ..widgets.slice_navigator import SliceNavigator


class MultiFrameConverterWindow(QMainWindow):
    def __init__(self, start_dir: str = ""):
        super().__init__()
        self.setWindowTitle("Multi-frame to Single-frame DICOM Converter")
        self.resize(1100, 820)
        self.start_dir = start_dir or os.path.expanduser("~")
        self.dicom: Optional[MultiFrameDicom] = None
        self.index = 0

        bar = QToolBar("Converter", self)
        bar.setMovable(False)
        bar.setIconSize(QSize(20, 20))
        bar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(bar)
        open_action = QAction("Open DICOM File…", self)
        bind_icon(open_action, "folder-open")
        open_action.triggered.connect(lambda: self.open_file())
        self.save_action = QAction("Save Slices…", self)
        bind_icon(self.save_action, "save")
        self.save_action.setEnabled(False)
        self.save_action.triggered.connect(lambda: self.save_slices())
        bar.addAction(open_action)
        bar.addAction(self.save_action)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        bar.addWidget(spacer)
        bar.addAction(help_action(self, "converter"))

        self.canvas = ImageCanvas()
        self.canvas.placeholder = "Open a multi-frame DICOM file"
        self.canvas.sliceScrolled.connect(lambda step: self.show_frame(self.index - step))
        self.navigator = SliceNavigator()
        self.navigator.valueChanged.connect(self.show_frame)
        center = QWidget()
        layout = QVBoxLayout(center)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.canvas, 1)
        holder = QWidget()
        holder.setObjectName("DockBody")
        holder_layout = QVBoxLayout(holder)
        holder_layout.setContentsMargins(0, 0, 0, 0)
        holder_layout.addWidget(self.navigator)
        layout.addWidget(holder)
        self.setCentralWidget(center)

        body = panel_body()
        side = QVBoxLayout(body)
        side.setContentsMargins(10, 10, 10, 10)
        self.info_values = {}
        card = Card("File")
        form = QFormLayout()
        for key in ("File", "Patient", "Study date", "Dimensions", "Frames", "Stack position", "Frame"):
            value = label("—", "Value", wrap=True)
            self.info_values[key] = value
            form.addRow(label(key, "Muted"), value)
        card.add_layout(form)
        side.addWidget(card)
        hint = Card("How it works")
        hint.add(label("Every frame is saved as <file>_<frame>.dcm, a standard single-frame DICOM image (MR, CT or "
                       "Secondary Capture) with its own position, spacing, window and scan settings, ready to be "
                       "opened in the annotator.", "Faint", wrap=True))
        side.addWidget(hint)
        side.addStretch(1)
        dock = QDockWidget("Details", self)
        dock.setWidget(body)
        dock.setFeatures(QDockWidget.DockWidgetMovable)
        dock.setMinimumWidth(280)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    def open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open DICOM File", self.start_dir,
                                              "DICOM Files (*.dcm);;All Files (*)")
        if not path:
            return
        try:
            self.dicom = MultiFrameDicom(path)
        except Exception as exc:
            self.dicom = None
            self.save_action.setEnabled(False)
            show_error(self, "Error", f"Could not read or process the DICOM file.\n{exc}")
            return
        self.start_dir = os.path.dirname(path)
        self.save_action.setEnabled(True)
        self.navigator.set_count(len(self.dicom))
        self.info_values["File"].setText(os.path.basename(path))
        self.info_values["Frames"].setText(str(len(self.dicom)))
        self.index = -1
        self.show_frame(len(self.dicom) // 2)
        self.canvas.fit_to_window()

    def show_frame(self, index: int) -> None:
        if self.dicom is None:
            return
        index = max(0, min(len(self.dicom) - 1, int(index)))
        if index == self.index:
            return
        self.index = index
        self.navigator.set_value(index, emit=False)
        self.canvas.set_image(self.dicom.frame_for_display(index))
        info = self.dicom.frame_info(index)
        for key, value in info.items():
            self.info_values[key].setText(value)
        self.canvas.set_hud(top_left=[(f"Frame {index + 1} / {len(self.dicom)}", True),
                                      (f"Stack position {info['Stack position']}", False)])

    def save_slices(self) -> None:
        if self.dicom is None:
            return
        default = self.dicom.default_output_folder()
        folder = QFileDialog.getExistingDirectory(self, "Folder for the single-frame files",
                                                  os.path.dirname(default))
        if not folder:
            return
        target = os.path.join(folder, os.path.basename(default)) if folder == os.path.dirname(default) else folder
        dialog = TaskProgressDialog("Saving slices", self)

        def done(count: int) -> None:
            kind = self.dicom.target_sop_class.name.replace(" Storage", "")
            dialog.finish("Slices saved", [f"✓ {count} single-frame DICOM files ({kind})", f"in {target}"])

        task = run_task(lambda ctx: self.dicom.split(target, ctx), on_result=done,
                        on_error=lambda m, d: dialog.finish("Could not save slices", [m], "error"),
                        on_cancelled=lambda: dialog.finish("Cancelled", ["Slice saving was cancelled."], "warning"),
                        on_progress=dialog.set_progress)
        dialog.cancelRequested.connect(task.cancel)
        dialog.exec_()
