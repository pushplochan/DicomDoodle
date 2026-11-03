"""Keyboard and mouse reference for the main annotator."""
from __future__ import annotations

from PyQt5.QtWidgets import QDialog, QHBoxLayout, QHeaderView, QTableWidget, QTableWidgetItem, QVBoxLayout

from ...help.shortcuts import SHORTCUTS
from ..qt_utils import label, make_button


class ShortcutsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.resize(640, 620)
        rows = sum(len(items) + 1 for _, items in SHORTCUTS)
        table = QTableWidget(rows, 2)
        table.setHorizontalHeaderLabels(["Keys", "Action"])
        table.verticalHeader().hide()
        table.setShowGrid(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionMode(QTableWidget.NoSelection)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)
        row = 0
        for group, items in SHORTCUTS:
            head = QTableWidgetItem(group.upper())
            font = head.font()
            font.setBold(True)
            head.setFont(font)
            table.setItem(row, 0, head)
            table.setSpan(row, 0, 1, 2)
            row += 1
            for keys, action in items:
                key_item = QTableWidgetItem(keys)
                key_font = key_item.font()
                key_font.setBold(True)
                key_item.setFont(key_font)
                table.setItem(row, 0, key_item)
                table.setItem(row, 1, QTableWidgetItem(action))
                row += 1
        close_btn = make_button("Close", role="primary")
        close_btn.clicked.connect(self.accept)
        footer = QHBoxLayout()
        footer.addStretch(1)
        footer.addWidget(close_btn)
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 14)
        root.addWidget(label("Keyboard & mouse", "Heading"))
        root.addWidget(table, 1)
        root.addLayout(footer)
