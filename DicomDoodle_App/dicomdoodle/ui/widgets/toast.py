"""
Toast notifications: short messages that slide in at the bottom-right of a
window and fade away by themselves (e.g. "Saved Img101_seg.dcm").
"""
from __future__ import annotations

from typing import List

from PyQt5.QtCore import QEasingCurve, QEvent, QObject, QPropertyAnimation, Qt, QTimer
from PyQt5.QtGui import QColor, QPainter
from PyQt5.QtWidgets import QFrame, QGraphicsOpacityEffect, QHBoxLayout, QLabel, QWidget

from ..icons import pixmap
from ..theme import current_theme

KINDS = {"info": ("info", "accent"), "success": ("check-circle", "success"),
         "warning": ("alert", "warning"), "error": ("x-circle", "danger")}


class Toast(QFrame):
    def __init__(self, parent: QWidget, message: str, kind: str = "info", duration_ms: int = 3500):
        super().__init__(parent)
        icon_name, token = KINDS.get(kind, KINDS["info"])
        theme = current_theme()
        self._accent = QColor(getattr(theme, token))
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setStyleSheet(f"QFrame {{ background: {theme.surface_alt}; border: 1px solid {theme.border_strong}; "
                           f"border-radius: 10px; }} QLabel {{ border: none; background: transparent; }}")
        icon_label = QLabel(self)
        icon_label.setPixmap(pixmap(icon_name, getattr(theme, token), 18))
        text = QLabel(message, self)
        text.setWordWrap(True)
        text.setMaximumWidth(380)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 16, 10)
        layout.setSpacing(10)
        layout.addWidget(icon_label, 0, Qt.AlignTop)
        layout.addWidget(text, 1)
        self.adjustSize()

        self._effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._effect)
        self._fade = QPropertyAnimation(self._effect, b"opacity", self)
        self._fade.setDuration(220)
        self._fade.setEasingCurve(QEasingCurve.OutCubic)
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

    def show(self, message: str, kind: str = "info", duration_ms: int = 3500) -> None:
        toast = Toast(self._host, message, kind, duration_ms)
        toast.destroyed.connect(lambda *_: self._forget(toast))
        self._toasts.append(toast)
        self._layout()
        toast.appear()

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
