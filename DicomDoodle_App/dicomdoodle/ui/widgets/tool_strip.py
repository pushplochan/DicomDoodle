"""
The row of annotation tools above the image: what the left mouse button
does (Box for SAM, Pen, Eraser, Pan), the pen / eraser size, a one-line
hint for the chosen tool, and "Edit saved outline". The tool buttons are
the main window's actions, so they stay in step with the Annotate menu
and the keyboard shortcuts.
"""
from __future__ import annotations

from typing import List, Sequence

from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtWidgets import QAction, QLabel, QSizePolicy, QSpinBox, QToolBar

from ..qt_utils import label


class ElidedLabel(QLabel):
    """One line of text that ends in "…" when space is short, and disappears when it gets too narrow."""

    MIN_WIDTH = 200                       # narrower than this, a cut-off hint would not help

    def __init__(self, text: str = "", object_name: str = "", parent=None):
        super().__init__(parent)
        if object_name:
            self.setObjectName(object_name)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setIndent(12)
        self._full = ""
        self.set_full_text(text)

    def set_full_text(self, text: str) -> None:
        self._full = text
        self.setToolTip(text)
        self._elide()

    def sizeHint(self) -> QSize:          # never asks for room: it takes what is left
        return QSize(0, super().sizeHint().height())

    def minimumSizeHint(self) -> QSize:
        return QSize(0, super().minimumSizeHint().height())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        width = self.width() - self.indent() - 6
        text = "" if width < self.MIN_WIDTH else self.fontMetrics().elidedText(self._full, Qt.ElideRight, width)
        if text != self.text():
            self.setText(text)


class ToolStrip(QToolBar):
    sizeChanged = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__("Annotation tools", parent)
        self.setObjectName("ToolStrip")
        self.setMovable(False)
        self.setFloatable(False)
        self.setIconSize(QSize(18, 18))
        self.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.size_spin = QSpinBox()
        self.size_spin.setRange(1, 40)
        self.size_spin.setSuffix(" px")
        self.size_spin.setFixedWidth(88)
        self.size_spin.setToolTip("Size of the pen or eraser  ( [  and  ] )")
        self.size_spin.valueChanged.connect(self.sizeChanged.emit)
        self.hint = ElidedLabel(object_name="Faint")
        self._size_actions: List[QAction] = []

    def populate(self, tools: Sequence[QAction], edit_saved: QAction) -> None:
        """Tool buttons, then the size, the hint and "Edit saved outline" (right)."""
        for action in tools:
            self.addAction(action)
        caption = label("  Size", "Muted")
        self._size_actions = [self.addSeparator(), self.addWidget(caption), self.addWidget(self.size_spin)]
        self.addWidget(self.hint)
        self.addAction(edit_saved)

    def show_size(self, visible: bool, value: int) -> None:
        """Show the size box (pen and eraser only) with the tool's size."""
        for action in self._size_actions:
            action.setVisible(visible)
        self.size_spin.blockSignals(True)
        self.size_spin.setValue(value)
        self.size_spin.blockSignals(False)

    def set_hint(self, text: str) -> None:
        self.hint.set_full_text(text)
