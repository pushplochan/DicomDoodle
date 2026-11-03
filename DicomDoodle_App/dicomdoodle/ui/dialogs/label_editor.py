"""
Create or edit a label: a name plus any color from a full color palette.

The palette is Qt's complete color picker (basic colors, a hue/saturation
field, RGB / HSV / HTML entry and a screen color picker), embedded in the
dialog, plus one-click suggestions of colors that are not used by any label
yet. The dialog checks the name and color live: names must be unique and
DICOM-safe, colors must differ from other labels (a warning appears when a
color merely looks similar).
"""
from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import (QColorDialog, QDialog, QDialogButtonBox, QFormLayout, QGridLayout,
                             QHBoxLayout, QLineEdit, QPushButton, QVBoxLayout, QWidget)

from ...core.labels import MAX_NAME_LENGTH, LabelError, SegLabel
from ..label_manager import LabelManager
from ..qt_utils import label, set_role
from ..theme import theme_color


class _Preview(QWidget):
    """How the label will look in the label list."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.color = QColor(255, 140, 0)
        self.title = "New label"
        self.setMinimumHeight(52)

    def sizeHint(self) -> QSize:
        return QSize(260, 52)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.setPen(QPen(theme_color("border"), 1))
        painter.setBrush(theme_color("surface"))
        painter.drawRoundedRect(rect, 10, 10)
        painter.setPen(QPen(QColor(255, 255, 255, 80), 1))
        painter.setBrush(self.color)
        painter.drawRoundedRect(rect.left() + 14, rect.center().y() - 11, 22, 22, 6, 6)
        painter.setPen(theme_color("text"))
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(rect.adjusted(48, 0, -10, 0), Qt.AlignVCenter | Qt.AlignLeft, self.title)


class _ColorChip(QPushButton):
    def __init__(self, rgb, parent=None):
        super().__init__(parent)
        self.rgb = rgb
        hex_color = "#{:02X}{:02X}{:02X}".format(*rgb)
        self.setFixedSize(26, 26)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(hex_color)
        self.setStyleSheet(f"QPushButton {{ background: {hex_color}; border: 2px solid rgba(255,255,255,0.25); "
                           f"border-radius: 7px; padding: 0; }} QPushButton:hover {{ border-color: white; }}")


class LabelEditorDialog(QDialog):
    def __init__(self, manager: LabelManager, editing: Optional[SegLabel] = None, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.editing = editing
        self.result_label: Optional[SegLabel] = None
        self.setWindowTitle("Edit label" if editing else "New label")
        self.setModal(True)

        registry = manager.registry
        start_color = QColor(*editing.color) if editing else QColor(*(registry.suggested_colors(1) or [(255, 140, 0)])[0])

        # ---- left column: details -------------------------------------------
        self.name_edit = QLineEdit(editing.name if editing else "")
        self.name_edit.setMaxLength(MAX_NAME_LENGTH)
        self.name_edit.setMinimumWidth(260)
        self.name_edit.setPlaceholderText("e.g. Enhancing Tumor Boundary")
        self.short_edit = QLineEdit(editing.short_name if editing else "")
        self.short_edit.setMaxLength(32)
        self.short_edit.setPlaceholderText("Optional, e.g. Enh. Tumor")
        value = editing.value if editing else registry.next_free_value()
        value_label = label(f"{value}", "Value")
        value_label.setToolTip("Pixel value of this label inside the annotation (assigned automatically).")

        form = QFormLayout()
        form.setVerticalSpacing(10)
        form.addRow("Name (saved in DICOM)", self.name_edit)
        form.addRow("Short name", self.short_edit)
        form.addRow("Label value", value_label)

        self.preview = _Preview()
        self.message = label("", wrap=True)
        self.message.setMinimumHeight(36)

        suggestions = QGridLayout()
        suggestions.setSpacing(6)
        for i, rgb in enumerate(registry.suggested_colors(12)):
            chip = _ColorChip(rgb)
            chip.clicked.connect(lambda _c=False, c=rgb: self.picker.setCurrentColor(QColor(*c)))
            suggestions.addWidget(chip, i // 6, i % 6)

        used = QHBoxLayout()
        used.setSpacing(4)
        for lab in registry.labels():
            chip = _ColorChip(lab.color)
            chip.setFixedSize(18, 18)
            chip.setEnabled(False)
            chip.setToolTip(f"{lab.display_name} ({lab.hex})")
            used.addWidget(chip)
        used.addStretch(1)

        left = QVBoxLayout()
        left.setSpacing(10)
        left.addWidget(label("Label details", "Heading"))
        left.addLayout(form)
        left.addSpacing(6)
        left.addWidget(label("Preview", "Muted"))
        left.addWidget(self.preview)
        left.addWidget(self.message)
        left.addSpacing(6)
        left.addWidget(label("Suggested colors (not used yet)", "Muted"))
        left.addLayout(suggestions)
        left.addWidget(label("Colors already in use", "Muted"))
        left.addLayout(used)
        left.addStretch(1)

        # ---- right column: full color palette --------------------------------
        self.picker = QColorDialog(self)
        self.picker.setWindowFlags(Qt.Widget)
        self.picker.setOptions(QColorDialog.DontUseNativeDialog | QColorDialog.NoButtons)
        self.picker.setCurrentColor(start_color)
        right = QVBoxLayout()
        right.addWidget(label("Color palette", "Heading"))
        right.addWidget(self.picker)

        # ---- buttons -------------------------------------------------------------
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.ok_button = self.buttons.button(QDialogButtonBox.Ok)
        self.ok_button.setText("Save changes" if editing else "Create label")
        set_role(self.ok_button, "primary")
        self.buttons.accepted.connect(self._accept)
        self.buttons.rejected.connect(self.reject)

        columns = QHBoxLayout()
        columns.setSpacing(24)
        columns.addLayout(left, 0)
        columns.addLayout(right, 1)
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 16)
        root.addLayout(columns)
        root.addWidget(self.buttons)

        self.name_edit.textChanged.connect(self._validate)
        self.short_edit.textChanged.connect(self._validate)
        self.picker.currentColorChanged.connect(self._validate)
        self._validate()
        self.name_edit.setFocus()

    # ---------------------------------------------------------------- logic
    def _current_rgb(self):
        c = self.picker.currentColor()
        return c.red(), c.green(), c.blue()

    def _validate(self, *_args) -> bool:
        rgb = self._current_rgb()
        self.preview.color = QColor(*rgb)
        self.preview.title = (self.short_edit.text().strip() or self.name_edit.text().strip() or "New label")
        self.preview.update()
        exclude = self.editing.value if self.editing else None
        try:
            self.manager.registry.validate(self.name_edit.text(), rgb, exclude_value=exclude)
        except LabelError as exc:
            if self.name_edit.text().strip() or "name" not in str(exc).lower():
                self._show_message(str(exc), "danger")
            else:
                self._show_message("Give the label a name, then pick a color.", "text_muted")
            self.ok_button.setEnabled(False)
            return False
        similar = self.manager.registry.similar_labels(rgb, exclude_value=exclude)
        if similar:
            names = ", ".join(lab.display_name for lab in similar[:3])
            self._show_message(f"Looks similar to: {names}. It is allowed, but may be hard to tell apart.",
                               "warning")
        else:
            self._show_message("✓ Ready to save.", "success")
        self.ok_button.setEnabled(True)
        return True

    def _show_message(self, text: str, token: str) -> None:
        self.message.setText(text)
        self.message.setStyleSheet(f"color: {theme_color(token).name()};")

    def _accept(self) -> None:
        if not self._validate():
            return
        try:
            if self.editing:
                self.result_label = self.manager.update(self.editing.value, self.name_edit.text(),
                                                        self._current_rgb(), self.short_edit.text())
            else:
                self.result_label = self.manager.add(self.name_edit.text(), self._current_rgb(),
                                                     self.short_edit.text())
        except LabelError as exc:
            self._show_message(str(exc), "danger")
            return
        self.accept()
