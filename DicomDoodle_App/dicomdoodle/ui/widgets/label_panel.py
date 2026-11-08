"""
The label list: built-in labels plus the user's own, with color swatches.

Clicking a label makes it the *active* label (the color/segment used for the
next detection or drawing) — the replacement for the original row of color
buttons. Keys 1-9 select labels by position. Custom labels can be added with
a full color palette, edited, removed, imported and exported.
"""
from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import QRectF, QSize, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PyQt5.QtWidgets import (QAbstractItemView, QFileDialog, QHBoxLayout, QListWidget,
                             QListWidgetItem, QMenu, QMessageBox, QStyle, QStyledItemDelegate,
                             QToolButton, QVBoxLayout, QWidget)

from ...core.labels import LabelError, SegLabel
from ..icons import icon
from ..label_manager import LabelManager
from ..qt_utils import ask_yes_no, label, make_button, make_tool_button, qcolor
from ..theme import theme_color, theme_manager

VALUE_ROLE = Qt.UserRole + 1


class _LabelDelegate(QStyledItemDelegate):
    """Paints: [swatch]  Short name                    [key]
                         full DICOM name · #RRGGBB"""

    ROW_HEIGHT = 46

    def __init__(self, panel: "LabelPanel"):
        super().__init__(panel)
        self.panel = panel

    def sizeHint(self, option, index) -> QSize:
        return QSize(option.rect.width(), self.ROW_HEIGHT)

    def paint(self, painter: QPainter, option, index) -> None:
        lab: Optional[SegLabel] = self.panel.manager.get(index.data(VALUE_ROLE))
        if lab is None:
            return
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(option.rect).adjusted(3, 2, -3, -2)
        selected = bool(option.state & QStyle.State_Selected)
        hovered = bool(option.state & QStyle.State_MouseOver)
        if selected:
            bg = theme_color("accent")
            bg.setAlpha(46)
            painter.setPen(Qt.NoPen)
            painter.setBrush(bg)
            painter.drawRoundedRect(rect, 8, 8)
            painter.setBrush(theme_color("accent"))
            painter.drawRoundedRect(QRectF(rect.left(), rect.top() + 8, 3, rect.height() - 16), 1.5, 1.5)
        elif hovered:
            painter.setPen(Qt.NoPen)
            painter.setBrush(theme_color("hover"))
            painter.drawRoundedRect(rect, 8, 8)

        swatch = QRectF(rect.left() + 12, rect.center().y() - 9, 18, 18)
        painter.setPen(QPen(QColor(255, 255, 255, 70), 1))
        painter.setBrush(qcolor(lab.color))
        painter.drawRoundedRect(swatch, 5, 5)

        position = index.row() + 1
        key_text = str(position) if position <= 9 else ""
        text_left = swatch.right() + 10
        text_right = rect.right() - (30 if key_text else 10)

        title_font = QFont(option.font)
        title_font.setBold(True)
        if lab.transient:
            title_font.setItalic(True)
        painter.setFont(title_font)
        painter.setPen(theme_color("text"))
        fm = QFontMetrics(title_font)
        title = lab.display_name + ("  (from file)" if lab.transient else "")
        painter.drawText(QRectF(text_left, rect.top() + 5, text_right - text_left, fm.height()),
                         Qt.AlignLeft | Qt.AlignVCenter, fm.elidedText(title, Qt.ElideRight, int(text_right - text_left)))

        small = QFont(option.font)
        small.setPointSizeF(max(7.0, option.font.pointSizeF() - 1.0))
        painter.setFont(small)
        painter.setPen(theme_color("text_muted"))
        fm_small = QFontMetrics(small)
        detail = f"{lab.name}  ·  {lab.hex}" if lab.short_name else lab.hex
        detail += "" if lab.builtin else "  ·  custom"
        painter.drawText(QRectF(text_left, rect.bottom() - fm_small.height() - 5, text_right - text_left, fm_small.height()),
                         Qt.AlignLeft | Qt.AlignVCenter,
                         fm_small.elidedText(detail, Qt.ElideRight, int(text_right - text_left)))

        if key_text:
            key_rect = QRectF(rect.right() - 26, rect.center().y() - 10, 18, 20)
            painter.setPen(QPen(theme_color("border_strong"), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(key_rect, 4, 4)
            painter.setPen(theme_color("text_faint"))
            painter.drawText(key_rect, Qt.AlignCenter, key_text)
        painter.restore()


class LabelPanel(QWidget):
    """List of labels; the selected one is the active label."""

    activeChanged = pyqtSignal(int)

    def __init__(self, manager: LabelManager, active_value: int = 1, parent=None, show_hint: bool = True):
        super().__init__(parent)
        self.manager = manager
        self._active = active_value

        self.list = QListWidget(self)
        self.list.setItemDelegate(_LabelDelegate(self))
        self.list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list.setMouseTracking(True)
        self.list.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list.setMinimumHeight(6 * _LabelDelegate.ROW_HEIGHT + 12)
        self.list.currentItemChanged.connect(self._current_changed)
        self.list.itemDoubleClicked.connect(lambda item: self.edit_label(item.data(VALUE_ROLE)))
        self.list.customContextMenuRequested.connect(self._context_menu)

        self.add_btn = make_button("Add label", "plus", tooltip="Create a new label with any color")
        self.menu_btn = make_tool_button("sliders", "More label actions")
        self.menu_btn.setPopupMode(QToolButton.InstantPopup)
        more = QMenu(self)
        more.addAction(icon("edit"), "Edit selected label…", lambda: self.edit_label(self.active_value))
        more.addAction(icon("trash"), "Remove selected label", lambda: self.remove_label(self.active_value))
        more.addSeparator()
        more.addAction(icon("upload"), "Import labels…", self.import_labels)
        more.addAction(icon("download"), "Export my labels…", self.export_labels)
        self.menu_btn.setMenu(more)

        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        buttons.addWidget(self.add_btn, 1)
        buttons.addWidget(self.menu_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self.list, 1)
        layout.addLayout(buttons)
        if show_hint:
            layout.addWidget(label("Click a label to use it · keys 1–9 · double-click a custom label to edit",
                                   "Faint", wrap=True))

        self.add_btn.clicked.connect(self.add_label)
        manager.labelsChanged.connect(self.rebuild)
        theme_manager().themeChanged.connect(self._theme_changed)
        self.rebuild()

    def _theme_changed(self, *_args) -> None:
        self.list.viewport().update()

    # ------------------------------------------------------------ public API
    @property
    def active_value(self) -> int:
        return self._active

    def active_label(self) -> SegLabel:
        return self.manager.get(self._active) or self.manager.labels()[0]

    def set_active(self, value: int) -> None:
        for row in range(self.list.count()):
            if self.list.item(row).data(VALUE_ROLE) == value:
                self.list.setCurrentRow(row)
                return

    def select_position(self, position: int) -> None:
        """Select the label at 1-based list position (number-key shortcut)."""
        if 1 <= position <= self.list.count():
            self.list.setCurrentRow(position - 1)

    def rebuild(self) -> None:
        self.list.blockSignals(True)
        self.list.clear()
        for lab in self.manager.labels():
            item = QListWidgetItem()
            item.setData(VALUE_ROLE, lab.value)
            item.setToolTip(f"{lab.name}\nValue {lab.value} · {lab.hex}"
                            + ("\nBuilt-in label" if lab.builtin else "\nYour label (double-click to edit)"))
            self.list.addItem(item)
        self.list.blockSignals(False)
        if self.manager.get(self._active) is None:
            self._active = self.manager.labels()[0].value
        self.set_active(self._active)
        self.activeChanged.emit(self._active)

    # -------------------------------------------------------------- actions
    def add_label(self) -> None:
        from ..dialogs.label_editor import LabelEditorDialog
        dialog = LabelEditorDialog(self.manager, parent=self.window())
        if dialog.exec_():
            self.set_active(dialog.result_label.value)

    def edit_label(self, value: int) -> None:
        lab = self.manager.get(value)
        if lab is None:
            return
        if lab.builtin:
            QMessageBox.information(self.window(), "Built-in label",
                                    f"“{lab.display_name}” is one of DicomDoodle's built-in labels and "
                                    "cannot be changed, so existing annotation files keep their meaning.\n\n"
                                    "Use “Add label” to create your own.")
            return
        from ..dialogs.label_editor import LabelEditorDialog
        LabelEditorDialog(self.manager, editing=lab, parent=self.window()).exec_()

    def remove_label(self, value: int) -> None:
        lab = self.manager.get(value)
        if lab is None or lab.builtin:
            if lab is not None:
                self.edit_label(value)   # explains why it can't be removed
            return
        if ask_yes_no(self.window(), "Remove label",
                      f"Remove the label “{lab.display_name}”?\n\nAnnotation files already saved keep "
                      "their segment name; unsaved regions with this label will be saved as "
                      f"“Label {lab.value}”.", yes_text="Remove", default_yes=False):
            try:
                self.manager.remove(value)
            except LabelError as exc:
                QMessageBox.warning(self.window(), "Remove label", str(exc))

    def import_labels(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self.window(), "Import labels", "", "Label files (*.json)")
        if path:
            try:
                count = self.manager.import_from(path)
                QMessageBox.information(self.window(), "Import labels", f"{count} label(s) added.")
            except Exception as exc:
                QMessageBox.warning(self.window(), "Import labels", f"Could not import labels:\n{exc}")

    def export_labels(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self.window(), "Export labels", "dicomdoodle_labels.json",
                                              "Label files (*.json)")
        if path:
            try:
                count = self.manager.export_to(path)
                QMessageBox.information(self.window(), "Export labels", f"{count} custom label(s) exported.")
            except Exception as exc:
                QMessageBox.warning(self.window(), "Export labels", f"Could not export labels:\n{exc}")

    # -------------------------------------------------------------- internal
    def _current_changed(self, current: Optional[QListWidgetItem], _previous) -> None:
        if current is None:
            return
        value = current.data(VALUE_ROLE)
        if value != self._active:
            self._active = value
            self.activeChanged.emit(value)

    def _context_menu(self, pos) -> None:
        item = self.list.itemAt(pos)
        menu = QMenu(self)
        if item is not None:
            lab = self.manager.get(item.data(VALUE_ROLE))
            menu.addAction(icon("check"), f"Use “{lab.display_name}”", lambda: self.set_active(lab.value))
            if lab.transient:
                menu.addAction(icon("plus"), "Add to my labels", lambda: self.manager.promote(lab.value))
            if not lab.builtin:
                menu.addAction(icon("edit"), "Edit…", lambda: self.edit_label(lab.value))
                menu.addAction(icon("trash"), "Remove", lambda: self.remove_label(lab.value))
            menu.addSeparator()
        menu.addAction(icon("plus"), "Add label…", self.add_label)
        menu.addAction(icon("upload"), "Import labels…", self.import_labels)
        menu.addAction(icon("download"), "Export my labels…", self.export_labels)
        menu.exec_(self.list.viewport().mapToGlobal(pos))
