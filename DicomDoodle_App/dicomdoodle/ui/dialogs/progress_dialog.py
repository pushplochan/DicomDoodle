"""
Progress window for long jobs (batch masks, loading, conversions).

Shows "Processing X/Y", a progress bar, elapsed time and an estimate of the
time left, with a Cancel button. When the job ends the same window turns
into a summary (the original showed "Finished - Total time: Ns").
"""
from __future__ import annotations

import time
from typing import Iterable

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import QDialog, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout

from ..icons import pixmap
from ..qt_utils import label, set_role
from ..theme import current_theme


def format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


class TaskProgressDialog(QDialog):
    cancelRequested = pyqtSignal()

    def __init__(self, title: str, parent=None, cancellable: bool = True):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModal)
        self.setMinimumWidth(480)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self._start = time.time()
        self._running = True
        self._done, self._total = 0, 0

        self.icon_label = QLabel()
        self.icon_label.setPixmap(pixmap("activity", current_theme().accent, 28))
        self.heading = label(title, "Heading")
        self.message = label("Starting…", "Muted", wrap=True)
        self.bar = QProgressBar()
        self.bar.setRange(0, 0)
        self.bar.setTextVisible(True)
        self.time_label = label("", "Faint")
        self.summary = label("", wrap=True)
        self.summary.setTextFormat(Qt.RichText)
        self.summary.hide()

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setVisible(cancellable)
        self.close_btn = QPushButton("Close")
        set_role(self.close_btn, "primary")
        self.close_btn.hide()

        head = QHBoxLayout()
        head.setSpacing(12)
        head.addWidget(self.icon_label, 0, Qt.AlignTop)
        text = QVBoxLayout()
        text.setSpacing(4)
        text.addWidget(self.heading)
        text.addWidget(self.message)
        head.addLayout(text, 1)

        buttons = QHBoxLayout()
        buttons.addWidget(self.time_label)
        buttons.addStretch(1)
        buttons.addWidget(self.cancel_btn)
        buttons.addWidget(self.close_btn)

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 16)
        root.setSpacing(12)
        root.addLayout(head)
        root.addWidget(self.bar)
        root.addWidget(self.summary)
        root.addLayout(buttons)

        self.cancel_btn.clicked.connect(self._request_cancel)
        self.close_btn.clicked.connect(self.accept)
        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._update_time)
        self._timer.start()

    # ------------------------------------------------------------ progress
    def set_progress(self, done: int, total: int, message: str = "") -> None:
        if not self._running:
            return
        self._done, self._total = done, total
        if total > 0:
            self.bar.setRange(0, total)
            self.bar.setValue(min(done, total))
            self.bar.setFormat(f"{min(done, total)} / {total}")
        if message:
            self.message.setText(message)
        self._update_time()

    def _update_time(self) -> None:
        elapsed = time.time() - self._start
        text = f"Elapsed {format_seconds(elapsed)}"
        if self._running and self._total and self._done:
            remaining = elapsed / self._done * (self._total - self._done)
            text += f"  ·  about {format_seconds(remaining)} left"
        self.time_label.setText(text)

    def _request_cancel(self) -> None:
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setText("Cancelling…")
        self.message.setText("Finishing the current slice, then stopping…")
        self.cancelRequested.emit()

    @property
    def running(self) -> bool:
        """False once the job has ended and the window shows its summary."""
        return self._running

    # --------------------------------------------------------------- finish
    def finish(self, title: str, lines: Iterable[str] = (), kind: str = "success") -> None:
        """Turn the dialog into a summary. kind: success / warning / error."""
        self._running = False
        self._timer.stop()
        self._update_time()
        theme = current_theme()
        icon_name, color = {"success": ("check-circle", theme.success), "warning": ("alert", theme.warning),
                            "error": ("x-circle", theme.danger)}.get(kind, ("info", theme.accent))
        self.icon_label.setPixmap(pixmap(icon_name, color, 28))
        self.heading.setText(title)
        self.message.setText(f"Total time: {format_seconds(time.time() - self._start)}")
        if self._total:
            self.bar.setRange(0, self._total)
            self.bar.setValue(self._total if kind == "success" else self._done)
            self.bar.setFormat(f"{self.bar.value()} / {self._total}")
        else:
            self.bar.setRange(0, 1)
            self.bar.setValue(1)
        lines = list(lines)
        if lines:
            self.summary.setText("<br>".join(lines))
            self.summary.show()
        self.cancel_btn.hide()
        self.close_btn.show()
        self.close_btn.setFocus()

    def reject(self) -> None:  # Esc / window close
        if not self._running:
            super().reject()
        elif not self.cancel_btn.isHidden() and self.cancel_btn.isEnabled():
            self._request_cancel()
