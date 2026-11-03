"""
Toast notifications: short messages that slide in at the bottom-right of a
window and fade away by themselves (e.g. "Saved Img101_seg.dcm").

ToastManager.progress() makes one that stays while work runs — a title, the
current step and a progress bar (e.g. loading the AI models at start) — until
finish() lets it fade.
"""
from __future__ import annotations

from typing import List

from PyQt5.QtCore import QEasingCurve, QEvent, QObject, QPropertyAnimation, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QPainter
from PyQt5.QtWidgets import (QFrame, QGraphicsOpacityEffect, QHBoxLayout, QLabel, QProgressBar, QVBoxLayout,
                             QWidget)

from ..icons import pixmap
from ..theme import current_theme

KINDS = {"info": ("info", "accent"), "success": ("check-circle", "success"),
         "warning": ("alert", "warning"), "error": ("x-circle", "danger")}


class Toast(QFrame):
    resized = pyqtSignal()           # the text changed size: the manager re-stacks the toasts

    def __init__(self, parent: QWidget, message: str, kind: str = "info", duration_ms: int = 3500,
                 title: str = "", progress: bool = False):
        super().__init__(parent)
        theme = current_theme()
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setStyleSheet(f"QFrame {{ background: {theme.surface_alt}; border: 1px solid {theme.border_strong}; "
                           f"border-radius: 10px; }} QLabel {{ border: none; background: transparent; }}")
        self._icon = QLabel(self)
        self._title = QLabel(f"<b>{title}</b>", self)
        self._title.setVisible(bool(title))
        self._text = QLabel(message, self)
        self._text.setWordWrap(True)
        self._text.setMaximumWidth(380)
        self._bar = QProgressBar(self)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(6)
        self._bar.setStyleSheet(f"QProgressBar {{ background: {theme.border}; border: none; border-radius: 3px; "
                                f"min-height: 6px; }} QProgressBar::chunk {{ background: {theme.accent}; "
                                f"border-radius: 3px; }}")
        self._bar.setRange(0, 0)                     # busy until a step count arrives
        self._bar.setVisible(progress)
        column = QVBoxLayout()
        column.setSpacing(5)
        column.addWidget(self._title)
        column.addWidget(self._text)
        column.addWidget(self._bar)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 16, 10)
        layout.setSpacing(10)
        layout.addWidget(self._icon, 0, Qt.AlignTop)
        layout.addLayout(column, 1)
        self._set_kind(kind)
        if progress:
            self.setMinimumWidth(340)
        self.adjustSize()

        self._effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._effect)
        self._fade = QPropertyAnimation(self._effect, b"opacity", self)
        self._fade.setDuration(220)
        self._fade.setEasingCurve(QEasingCurve.OutCubic)
        if duration_ms:
            QTimer.singleShot(duration_ms, self.dismiss)

    def _set_kind(self, kind: str) -> None:
        icon_name, token = KINDS.get(kind, KINDS["info"])
        theme = current_theme()
        self._accent = QColor(getattr(theme, token))
        self._icon.setPixmap(pixmap(icon_name, getattr(theme, token), 18))
        self.update()

    def set_title(self, title: str) -> None:
        if title and self._title.text() != f"<b>{title}</b>":
            self._title.setText(f"<b>{title}</b>")
            self._title.show()
            self.adjustSize()
            self.resized.emit()

    def set_message(self, message: str) -> None:
        if message != self._text.text():
            self._text.setText(message)
            self.adjustSize()
            self.resized.emit()

    def set_progress(self, done: int, total: int) -> None:
        """Step ``done`` of ``total`` (total 0: busy, no steps known)."""
        self._bar.setRange(0, max(0, total))
        if total:
            self._bar.setValue(max(0, min(done, total)))

    def finish(self, message: str, kind: str = "success", duration_ms: int = 2500) -> None:
        """The work is done: show the outcome, then fade away."""
        self._set_kind(kind)
        self._bar.hide()
        self.set_message(message)
        QTimer.singleShot(duration_ms, self.dismiss)

    def mousePressEvent(self, event) -> None:   # click to dismiss early
        self.dismiss()

    def appear(self) -> None:
        self.show()
        self.raise_()
        self._fade.setStartValue(0.0)
        self._fade.setEndValue(1.0)
        self._fade.start()

    def dismiss(self) -> None:
        if not self.isVisible():
            return
        self._fade.stop()
        self._fade.setStartValue(self._effect.opacity())
        self._fade.setEndValue(0.0)
        self._fade.finished.connect(self.deleteLater)
        self._fade.start()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(self._accent)
        painter.drawRoundedRect(1, 8, 3, self.height() - 16, 1.5, 1.5)


class ToastManager(QObject):
    """Stacks toasts in the bottom-right corner of ``host``."""

    def __init__(self, host: QWidget, bottom_offset: int = 46):
        super().__init__(host)
        self._host = host
        self._bottom = bottom_offset
        self._toasts: List[Toast] = []
        host.installEventFilter(self)

    def show(self, message: str, kind: str = "info", duration_ms: int = 3500) -> Toast:
        return self._add(Toast(self._host, message, kind, duration_ms))

    def progress(self, title: str, message: str = "", kind: str = "info") -> Toast:
        """A toast that stays until its finish() (or a click): for work in the background."""
        return self._add(Toast(self._host, message, kind, 0, title=title, progress=True))

    def _add(self, toast: Toast) -> Toast:
        toast.destroyed.connect(lambda *_: self._forget(toast))
        toast.resized.connect(self._layout)
        self._toasts.append(toast)
        self._layout()
        toast.appear()
        return toast

    def _forget(self, toast: Toast) -> None:
        if toast in self._toasts:
            self._toasts.remove(toast)
        self._layout()

    def _layout(self) -> None:
        try:
            y = self._host.height() - self._bottom
        except RuntimeError:      # host window already destroyed (app shutting down)
            return
        for toast in reversed(self._toasts):
            try:
                toast.adjustSize()
                y -= toast.height()
                toast.move(self._host.width() - toast.width() - 18, y)
                y -= 8
            except RuntimeError:
                continue

    def eventFilter(self, obj, event) -> bool:
        if obj is self._host and event.type() == QEvent.Resize:
            self._layout()
        return False
