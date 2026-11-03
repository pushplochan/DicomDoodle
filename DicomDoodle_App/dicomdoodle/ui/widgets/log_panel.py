"""
Activity log: everything the app does, with timestamps.

The original tool printed messages such as "Auto Detect Range Start set to
slice 12" to the console where users never saw them; here they are logged
through Python's ``logging`` and shown in this panel.
"""
from __future__ import annotations

import html
import logging
import time

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QHBoxLayout, QPlainTextEdit, QVBoxLayout, QWidget

from ..qt_utils import make_tool_button
from ..theme import current_theme


class _Relay(QObject):
    record = pyqtSignal(int, str)     # level, message (crosses threads safely)


class QtLogHandler(logging.Handler):
    """logging.Handler that forwards records to the GUI thread."""

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.relay = _Relay()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.relay.record.emit(record.levelno, record.getMessage())
        except Exception:
            pass


class LogPanel(QWidget):
    def __init__(self, handler: QtLogHandler, parent=None):
        super().__init__(parent)
        self.view = QPlainTextEdit(self)
        self.view.setReadOnly(True)
        self.view.setMaximumBlockCount(2000)
        self.view.setPlaceholderText("Activity will appear here.")
        clear_btn = make_tool_button("trash", "Clear the log")
        clear_btn.clicked.connect(self.view.clear)

        top = QHBoxLayout()
        top.addStretch(1)
        top.addWidget(clear_btn)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 8)
        layout.addLayout(top)
        layout.addWidget(self.view)
        handler.relay.record.connect(self.append)

    def append(self, level: int, message: str) -> None:
        theme = current_theme()
        color = {logging.WARNING: theme.warning, logging.ERROR: theme.danger,
                 logging.CRITICAL: theme.danger}.get(level, theme.text)
        stamp = time.strftime("%H:%M:%S")
        self.view.appendHtml(f'<span style="color:{theme.text_faint}">{stamp}</span>&nbsp;&nbsp;'
                             f'<span style="color:{color}">{html.escape(message)}</span>')
