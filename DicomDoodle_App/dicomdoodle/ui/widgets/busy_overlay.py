"""
A translucent "working…" layer with a spinner, laid over a widget while a
background job (e.g. SAM segmentation) runs. It blocks clicks on the widget
underneath so the user cannot start conflicting edits.
"""
from __future__ import annotations

from PyQt5.QtCore import QEvent, QRectF, Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import QWidget

from ..theme import theme_color


class BusyOverlay(QWidget):
    def __init__(self, target: QWidget):
        super().__init__(target)
        self._target = target
        self._angle = 0
        self.message = ""
        self.detail = ""
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        target.installEventFilter(self)
        self.hide()

    def start(self, message: str, detail: str = "") -> None:
        self.message, self.detail = message, detail
        self.setGeometry(self._target.rect())
        self.show()
        self.raise_()
        self._timer.start()

    def set_message(self, message: str) -> None:
        self.message = message
        self.update()

    def stop(self) -> None:
        self._timer.stop()
        self.hide()

    @property
    def active(self) -> bool:
        return self.isVisible()

    def _tick(self) -> None:
        self._angle = (self._angle + 6) % 360
        self.update()

    def eventFilter(self, obj, event) -> bool:
        if obj is self._target and event.type() == QEvent.Resize:
            self.setGeometry(self._target.rect())
        return False

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(5, 7, 10, 150))

        card_w, card_h = min(380, self.width() - 40), 120
        card = QRectF((self.width() - card_w) / 2, (self.height() - card_h) / 2, card_w, card_h)
        painter.setPen(QPen(theme_color("border_strong"), 1))
        painter.setBrush(theme_color("surface_alt"))
        painter.drawRoundedRect(card, 14, 14)

        spinner = QRectF(card.left() + 26, card.center().y() - 18, 36, 36)
        painter.setPen(QPen(theme_color("border_strong"), 4))
        painter.drawEllipse(spinner)
        pen = QPen(theme_color("accent"), 4)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.drawArc(spinner, -self._angle * 16, 100 * 16)

        text_rect = card.adjusted(84, 26, -20, -26)
        font = QFont(self.font())
        font.setBold(True)
        font.setPointSizeF(font.pointSizeF() + 1)
        painter.setFont(font)
        painter.setPen(theme_color("text"))
        painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap, self.message)
        if self.detail:
            painter.setFont(self.font())
            painter.setPen(theme_color("text_muted"))
            painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignBottom | Qt.TextWordWrap, self.detail)

    def mousePressEvent(self, event) -> None:
        event.accept()

    def wheelEvent(self, event) -> None:
        event.accept()
