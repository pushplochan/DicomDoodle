"""Card containers used to group controls in the side panels."""
from __future__ import annotations

from typing import Optional

from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget


class Card(QFrame):
    """Rounded panel with an optional small uppercase title and header widgets."""

    def __init__(self, title: str = "", parent: Optional[QWidget] = None, spacing: int = 8):
        super().__init__(parent)
        self.setObjectName("Card")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 12)
        outer.setSpacing(spacing)
        self.header = QHBoxLayout()
        self.header.setSpacing(4)
        self.title_label = QLabel(title.upper(), self)
        self.title_label.setObjectName("SectionTitle")
        self.header.addWidget(self.title_label)
        self.header.addStretch(1)
        if title:
            outer.addLayout(self.header)
        self.body = QVBoxLayout()
        self.body.setSpacing(spacing)
        outer.addLayout(self.body)

    def add(self, widget: QWidget, stretch: int = 0) -> QWidget:
        self.body.addWidget(widget, stretch)
        return widget

    def add_layout(self, layout) -> None:
        self.body.addLayout(layout)

    def add_header_widget(self, widget: QWidget) -> None:
        self.header.addWidget(widget)


def panel_body(parent: Optional[QWidget] = None) -> QWidget:
    """Background widget for dock contents."""
    body = QWidget(parent)
    body.setObjectName("DockBody")
    return body
