"""
Preferences: appearance, annotation display, files, AI models and ontology.
Changes are written to the settings file when the user clicks Save; the AI
page's Load model card… and Remove act at once (through the main window's
AIController, which also reports back when an installation has finished).
"""
from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
                             QHBoxLayout, QLineEdit, QSpinBox, QTabWidget, QVBoxLayout, QWidget)

from ...core.ai import devices
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
    def __init__(self, settings: Settings, models: ModelManager, parent=None, ai=None, tab: str = ""):
        super().__init__(parent)
        self.settings, self.models, self.ai = settings, models, ai
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
        self.card_status = label("", wrap=True)
        self.card_status.setTextFormat(Qt.RichText)
        self.load_card = make_button("Load model card…", "download",
                                     tooltip="Install a DicomDoodle model card file (….ddcard)")
        self.remove_card = make_button("Remove", "trash", role="ghost",
                                       tooltip="Delete the installed model card: manual annotation only")
        self.load_card.clicked.connect(lambda: self.ai.choose_card(self))
        self.remove_card.clicked.connect(lambda: self.ai.remove_card(self))
        card_buttons = QHBoxLayout()
        card_buttons.addWidget(self.load_card)
        card_buttons.addWidget(self.remove_card)
        card_buttons.addStretch(1)
        self.model_status = label("", "Muted", wrap=True)
        self.model_status.setTextFormat(Qt.RichText)
        self.device = QComboBox()
        self.device.addItem("Processor (CPU)", devices.CPU)
        self.device.addItem("Graphics card", devices.GPU)
        self.device.setCurrentIndex(1 if devices.normalize(settings.get("device")) == devices.GPU else 0)
        self.device_note = label("", "Faint", wrap=True)
        self.preload = QCheckBox("Load the AI models when DicomDoodle starts (otherwise when first used)")
        self.preload.setChecked(bool(settings.get("preload_sam")))
        form.addRow("Model card", self.card_status)
        if ai is not None:
            form.addRow("", card_buttons)
        form.addRow("Models", self.model_status)
        form.addRow("Run the AI on", self.device)
        form.addRow("", self.device_note)
        form.addRow("", self.preload)
        tabs.addTab(page, "AI Models")
        ai_page = page
        if ai is not None:
            ai.changed.connect(self._refresh_model_status)

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
        if tab == "ai":
            tabs.setCurrentWidget(ai_page)

    def _refresh_model_status(self) -> None:
        """The AI page: the installed card, what runs each purpose, the graphics card."""
        card = self.models.card
        if card is None:
            self.card_status.setText("<b>None</b> — manual annotation only.<br>The AI needs the DicomDoodle model "
                                     "card, a separate download: put the file in your Downloads folder and "
                                     "restart DicomDoodle, or load it here.")
            self.model_status.setText("—")
        else:
            self.card_status.setText(f"<b>{card.label}</b> — installed {card.installed[:10]}"
                                     f"<br><span style='font-size:8pt'>{card.folder}</span>")
            rows = []
            for row in self.models.info():
                if not row.title:
                    rows.append(f"<b>{row.purpose}</b> — not in this card")
                    continue
                state = f"cannot run here: {row.problem}" if row.problem else row.engine
                if row.loaded:
                    state += f", loaded on the {'graphics card' if row.device == 'GPU' else 'processor'}"
                rows.append(f"<b>{row.purpose}</b> — {row.title} · {state}")
            self.model_status.setText("<br>".join(rows))
        self.remove_card.setEnabled(card is not None)

        gpu = devices.known_gpu()
        item = self.device.model().item(1)
        chosen = self.device.currentData() == devices.GPU
        if gpu is None:
            item.setText("Graphics card (looking for one…)")
            note = ""
        elif gpu.available:
            item.setText(f"Graphics card: {gpu.describe()}")
            note = ("The graphics card makes the AI faster. (HD-BET skull stripping uses it only with an "
                    "NVIDIA card and a CUDA PyTorch; the Windows version runs HD-BET on the processor.)")
        else:
            item.setText("Graphics card (none that can be used)")
            note = gpu.problem
        item.setEnabled(gpu is not None and gpu.available or chosen)
        self.device_note.setText(note)
        self.device_note.setVisible(bool(note))

    def _save(self) -> None:
        self.settings.update(
            theme=self.theme.currentData(),
            overlay_mode=self.overlay_mode.currentData(),
            line_width=self.line_width.value(),
            fill_opacity=round(self.fill_opacity.value(), 2),
            show_saved_overlay=self.show_saved.isChecked(),
            confirm_overwrite=self.confirm_overwrite.isChecked(),
            auto_blank_masks=self.auto_blank.isChecked(),
            device=self.device.currentData(),
            preload_sam=self.preload.isChecked(),
            bioportal_api_key=self.api_key.text().strip(),
        )
        self.accept()
