"""Startup splash screen with the DicomDoodle artwork and loading messages."""
from __future__ import annotations

from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QApplication, QSplashScreen

from .. import APP_NAME, APP_TAGLINE, APP_VERSION, resources
from .theme import current_theme

WIDTH, HEIGHT = 680, 420


def _render() -> QPixmap:
    theme = current_theme()
    ratio = 2.0
    pix = QPixmap(int(WIDTH * ratio), int(HEIGHT * ratio))
    pix.setDevicePixelRatio(ratio)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    p.setRenderHint(QPainter.SmoothPixmapTransform)

    card = QRectF(8, 8, WIDTH - 16, HEIGHT - 16)
    gradient = QLinearGradient(0, 0, 0, HEIGHT)
    top, bottom = QColor(theme.surface_alt), QColor(theme.window)
    gradient.setColorAt(0.0, top)
    gradient.setColorAt(1.0, bottom)
    p.setPen(QPen(QColor(theme.border_strong), 1))
    p.setBrush(gradient)
    p.drawRoundedRect(card, 18, 18)

    image_path = resources.find_resource(resources.SPLASH_IMAGE) or resources.find_resource(resources.LOGO_IMAGE)
    if image_path:
        art = QPixmap(image_path)
        if not art.isNull():
            art = art.scaled(int(250 * ratio), int(250 * ratio), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            art.setDevicePixelRatio(ratio)
            p.drawPixmap(int(card.left() + 30), int(card.center().y() - art.height() / ratio / 2 - 10), art)

    text_left = card.left() + 310
    p.setPen(QColor(theme.text))
    title = QFont(QApplication.font())
    title.setPointSize(30)
    title.setBold(True)
    p.setFont(title)
    p.drawText(QRectF(text_left, 110, 340, 60), Qt.AlignLeft | Qt.AlignVCenter, APP_NAME)
    sub = QFont(QApplication.font())
    sub.setPointSize(12)
    p.setFont(sub)
    p.setPen(QColor(theme.text_muted))
    p.drawText(QRectF(text_left, 172, 330, 60), Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap, APP_TAGLINE)
    p.setPen(QColor(theme.accent))
    p.drawText(QRectF(text_left, 236, 330, 24), Qt.AlignLeft | Qt.AlignVCenter, f"Version {APP_VERSION}")
    # accent strip
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(theme.accent))
    p.drawRoundedRect(QRectF(text_left, 98, 46, 4), 2, 2)
    p.end()
    return pix


class SplashScreen(QSplashScreen):
    def __init__(self):
        super().__init__(_render())
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)

    def set_status(self, text: str) -> None:
        self.showMessage(text)
        QApplication.processEvents()

    def drawContents(self, painter: QPainter) -> None:
        painter.setPen(QColor(current_theme().text_muted))
        rect = self.rect().adjusted(330, 0, -34, -30)
        painter.drawText(rect, Qt.AlignBottom | Qt.AlignLeft, self.message())
