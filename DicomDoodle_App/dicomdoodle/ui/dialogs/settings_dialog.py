"""
Preferences: appearance, annotation display, files, AI models and ontology.
Changes are written to the settings file when the user clicks Save.
"""
from __future__ import annotations

import os

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                             QFileDialog, QFormLayout, QHBoxLayout, QLineEdit, QSpinBox,
                             QTabWidget, QVBoxLayout, QWidget)

from ...core.ai_models import ModelManager
from ...settings import Settings
from ..qt_utils import label, make_button, set_role


def _form_page() -> tuple:
    page = QWidget()
    form = QFormLayout(page)
    form.setContentsMargins(16, 16, 16, 16)
    form.setVerticalSpacing(12)
    form.setHorizontalSpacing(18)
    return page, form


class SettingsDialog(QDialog):
    def __init__(self, settings: Settings, models: ModelManager, parent=None):
        super().__init__(parent)
        self.settings, self.models = settings, models
        self.setWindowTitle("Preferences")
        self.setMinimumWidth(620)

        tabs = QTabWidget()

        # Appearance ---------------------------------------------------------------
        page, form = _form_page()
        self.theme = QComboBox()
        self.theme.addItem("Dark", "dark")
        self.theme.addItem("Light", "light")
        self.theme.setCurrentIndex(max(0, self.theme.findData(settings.get("theme"))))
        form.addRow("Theme", self.theme)
        form.addRow("", label("Image viewports stay dark in both themes (radiology convention).", "Faint", wrap=True))
        tabs.addTab(page, "Appearance")

        # Annotation display ---------------------------------------------------------
        page, form = _form_page()
        self.overlay_mode = QComboBox()
        self.overlay_mode.addItem("Boundary outline", "boundary")
        self.overlay_mode.addItem("Filled regions + outline", "filled")
        self.overlay_mode.setCurrentIndex(max(0, self.overlay_mode.findData(settings.get("overlay_mode"))))
        self.line_width = QSpinBox()
        self.line_width.setRange(1, 6)
        self.line_width.setSuffix(" px")
        self.line_width.setValue(int(settings.get("line_width")))
        self.fill_opacity = QDoubleSpinBox()
        self.fill_opacity.setRange(0.05, 1.0)
        self.fill_opacity.setSingleStep(0.05)
        self.fill_opacity.setValue(float(settings.get("fill_opacity")))
        self.show_saved = QCheckBox("Show annotation files already saved on disk (faint layer)")
        self.show_saved.setChecked(bool(settings.get("show_saved_overlay")))
        form.addRow("Annotation style", self.overlay_mode)
        form.addRow("Outline width", self.line_width)
        form.addRow("Fill opacity", self.fill_opacity)
        form.addRow("", self.show_saved)
        tabs.addTab(page, "Display")

        # Files ----------------------------------------------------------------------
        page, form = _form_page()
        self.confirm_overwrite = QCheckBox("Ask before replacing annotation files from earlier sessions")
        self.confirm_overwrite.setChecked(bool(settings.get("confirm_overwrite")))
        self.auto_blank = QCheckBox("Save blank masks automatically for slices without an annotation")
        self.auto_blank.setChecked(bool(settings.get("auto_blank_masks")))
        form.addRow("Saving", self.confirm_overwrite)
        form.addRow("", self.auto_blank)
        form.addRow("", label("Annotations are saved next to each slice as <slice>_seg.dcm "
                              "(DICOM Segmentation), exactly like the original DicomDoodle. With automatic "
                              "blank masks, every save also gives each slice that has no annotation file a "
                              "blank mask (“No Segmentation”), so only the slices with a lesion need "
                              "annotating.", "Faint", wrap=True))
        tabs.addTab(page, "Files")

        # AI models ---------------------------------------------------------------
        page, form = _form_page()
        self.models_dir = QLineEdit(settings.get("models_dir") or "")
        self.models_dir.setPlaceholderText("Default: the DicomDoodle folder")
        browse = make_button("Browse…", "folder")
        browse.clicked.connect(self._browse_models)
        row = QHBoxLayout()
        row.addWidget(self.models_dir, 1)
        row.addWidget(browse)
        self.device = QComboBox()
        self.device.addItem("Automatic (GPU when available)", "auto")
        self.device.addItem("CPU only", "cpu")
        self.device.addItem("CUDA GPU", "cuda")
        self.device.setCurrentIndex(max(0, self.device.findData(settings.get("device"))))
        self.preload = QCheckBox("Load SAM in the background when the app starts")
        self.preload.setChecked(bool(settings.get("preload_sam")))
        self.model_status = label("", "Muted", wrap=True)
        self.model_status.setTextFormat(Qt.RichText)
        form.addRow("Models folder", row)
        form.addRow("Device", self.device)
        form.addRow("", self.preload)
        form.addRow("Weights found", self.model_status)
        self.models_dir.textChanged.connect(self._refresh_model_status)
        tabs.addTab(page, "AI Models")

        # Ontology ---------------------------------------------------------------
        page, form = _form_page()
        self.api_key = QLineEdit(settings.get("bioportal_api_key") or "")
        self.api_key.setEchoMode(QLineEdit.PasswordEchoOnEdit)
        self.api_key.setPlaceholderText("Empty: DicomDoodle's built-in key")
        form.addRow("BioPortal API key", self.api_key)
        form.addRow("", label("Used by Annotated DICOM Viewer ▸ Add Ontology to search biomedical terms.",
                              "Faint", wrap=True))
        tabs.addTab(page, "Ontology")

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        set_role(buttons.button(QDialogButtonBox.Save), "primary")
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 14)
        root.addWidget(tabs)
        root.addWidget(buttons)
        self._refresh_model_status()

    def _browse_models(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Folder containing the .pt model files",
                                                  self.models_dir.text() or os.path.expanduser("~"))
        if folder:
            self.models_dir.setText(folder)

    def _refresh_model_status(self) -> None:
        probe = ModelManager(self.models_dir.text().strip())
        rows = []
        for name, path in probe.available().items():
            mark = "✓" if path else "✗ missing"
            rows.append(f"<b>{name}</b> — {mark}" + (f"<br><span style='font-size:8pt'>{path}</span>" if path else ""))
        self.model_status.setText("<br>".join(rows))

    def _save(self) -> None:
        self.settings.update(
            theme=self.theme.currentData(),
            overlay_mode=self.overlay_mode.currentData(),
            line_width=self.line_width.value(),
            fill_opacity=round(self.fill_opacity.value(), 2),
            show_saved_overlay=self.show_saved.isChecked(),
            confirm_overwrite=self.confirm_overwrite.isChecked(),
            auto_blank_masks=self.auto_blank.isChecked(),
            models_dir=self.models_dir.text().strip(),
            device=self.device.currentData(),
            preload_sam=self.preload.isChecked(),
            bioportal_api_key=self.api_key.text().strip(),
        )
        self.accept()
