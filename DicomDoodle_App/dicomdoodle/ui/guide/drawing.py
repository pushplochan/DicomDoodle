"""
Drawing helpers for the guide's diagrams: cards, labelled boxes, arrows,
text and a stylised MRI slice (head, ventricles, lesion) that can show a
SAM box, a detection box, an outline or a filled region.

All coordinates are in "design units"; diagrams.py scales the painter so
diagrams stay sharp at any width.

The numbered markers (``paint_marker``) are shared with the screenshot
script, so the numbers on screenshots and in the guide's tables look alike.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen, QPixmap, QRadialGradient

from ..icons import pixmap
from ..theme import Theme

LESION_RED = QColor(255, 72, 72)
DETECT_GREEN = QColor(64, 220, 120)
CALLOUT_ORANGE = "#f97316"


def paint_marker(p: QPainter, center: QPointF, label: str, radius: float, color: str = CALLOUT_ORANGE,
                 ring: bool = True) -> None:
    """Numbered disc: white ring and soft shadow (readable on any image), white number."""
    p.save()
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    inner = radius
    if ring:
        p.setBrush(QColor(0, 0, 0, 80))
        p.drawEllipse(center + QPointF(0, radius * 0.1), radius + 0.8, radius + 0.8)
        p.setBrush(QColor("#ffffff"))
        p.drawEllipse(center, radius, radius)
        inner = radius * 0.84
    p.setBrush(QColor(color))
    p.drawEllipse(center, inner, inner)
    f = QFont()
    f.setBold(True)
    f.setPixelSize(max(6, int(round(radius * (1.0 if len(label) < 2 else 0.85)))))
    p.setFont(f)
    p.setPen(QColor("#ffffff"))
    p.drawText(QRectF(center.x() - radius, center.y() - radius, 2 * radius, 2 * radius), Qt.AlignCenter, label)
    p.restore()


def marker_pixmap(label: str, size: int, ratio: float, color: str = CALLOUT_ORANGE, ring: bool = True) -> QPixmap:
    out = QPixmap(int(round(size * ratio)), int(round(size * ratio)))
    out.fill(Qt.transparent)
    p = QPainter(out)
    p.scale(ratio, ratio)
    paint_marker(p, QPointF(size / 2, size / 2), label, size / 2 - (1.2 if ring else 0.5), color, ring)
    p.end()
    out.setDevicePixelRatio(ratio)
    return out


def swatch_pixmap(hex_color: str, size: int, ratio: float, border: str) -> QPixmap:
    out = QPixmap(int(round(size * ratio)), int(round(size * ratio)))
    out.fill(Qt.transparent)
    p = QPainter(out)
    p.setRenderHint(QPainter.Antialiasing)
    p.scale(ratio, ratio)
    p.setPen(QPen(QColor(border), 1))
    p.setBrush(QColor(f"#{hex_color}"))
    p.drawRoundedRect(QRectF(1, 1, size - 2, size - 2), 4, 4)
    p.end()
    out.setDevicePixelRatio(ratio)
    return out


DESIGN_DPI = 96.0      # diagram text sizes are points on a screen of this resolution


def font(size: float, bold: bool = False, dpi: float = DESIGN_DPI) -> QFont:
    """``size`` points as on a 96 dpi screen, i.e. a fixed size in design
    units. Plain point sizes grow with the screen's dpi (or 125 % display
    scaling) while the diagram's boxes don't, so captions ran over the
    pictures around them."""
    f = QFont()
    f.setPointSizeF(size * DESIGN_DPI / max(dpi, 1.0))
    f.setBold(bold)
    return f


def card(p: QPainter, rect: QRectF, t: Theme) -> None:
    p.setPen(QPen(QColor(t.border), 1))
    p.setBrush(QColor(t.surface_alt))
    p.drawRoundedRect(rect, 12, 12)


def text(p: QPainter, rect: QRectF, value: str, color: str, size: float = 9.5, bold: bool = False,
         align=Qt.AlignCenter) -> None:
    """Wrapped text inside ``rect``. Fonts differ between systems, so text
    that would not fit is drawn smaller (down to 70 %) rather than over the
    drawing next to it."""
    flags = int(align) | Qt.TextWordWrap
    dpi = p.device().logicalDpiY() if p.device() is not None else DESIGN_DPI
    p.setPen(QColor(color))            # first: with Qt.NoPen, boundingRect measures nothing
    for shrink in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7):
        p.setFont(font(size * shrink, bold, dpi))
        needed = p.boundingRect(rect, flags, value)
        if needed.width() <= rect.width() + 0.5 and needed.height() <= rect.height() + 0.5:
            break
    p.drawText(rect, flags, value)


def box(p: QPainter, rect: QRectF, t: Theme, icon: str, title: str, subtitle: str = "",
        accent: Optional[str] = None) -> None:
    """Rounded tile: icon on top, bold title, muted subtitle."""
    color = accent or t.accent
    p.setPen(QPen(QColor(t.border_strong), 1))
    p.setBrush(QColor(t.raised))
    p.drawRoundedRect(rect, 10, 10)
    p.setPen(Qt.NoPen)
    tint = QColor(color)
    tint.setAlpha(40)
    p.setBrush(tint)
    circle = QRectF(rect.center().x() - 17, rect.top() + 12, 34, 34)
    p.drawEllipse(circle)
    p.drawPixmap(circle.adjusted(8, 8, -8, -8).toRect(), pixmap(icon, color, 36))
    text(p, QRectF(rect.left() + 4, rect.top() + 52, rect.width() - 8, 20), title, t.text, 9.5, True)
    if subtitle:
        text(p, QRectF(rect.left() + 6, rect.top() + 72, rect.width() - 12, rect.height() - 76),
             subtitle, t.text_muted, 8, False, Qt.AlignHCenter | Qt.AlignTop)


def arrow(p: QPainter, x1: float, y1: float, x2: float, y2: float, color: str, label: str = "",
          t: Optional[Theme] = None) -> None:
    pen = QPen(QColor(color), 2)
    pen.setCapStyle(Qt.RoundCap)
    p.setPen(pen)
    p.drawLine(QPointF(x1, y1), QPointF(x2, y2))
    angle = math.atan2(y2 - y1, x2 - x1)
    head = QPainterPath(QPointF(x2, y2))
    for delta in (2.6, -2.6):
        head.lineTo(QPointF(x2 - 9 * math.cos(angle + delta / 6), y2 - 9 * math.sin(angle + delta / 6)))
    head.closeSubpath()
    p.setBrush(QColor(color))
    p.drawPath(head)
    if label and t is not None:
        text(p, QRectF(min(x1, x2) - 30, min(y1, y2) - 24, abs(x2 - x1) + 60, 18), label, color, 8.5, True)


def chip(p: QPainter, x: float, y: float, color: Sequence[int], label: str, t: Theme, width: float = 150) -> None:
    rect = QRectF(x, y, width, 26)
    p.setPen(QPen(QColor(t.border), 1))
    p.setBrush(QColor(t.raised))
    p.drawRoundedRect(rect, 7, 7)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(*color))
    p.drawRoundedRect(QRectF(x + 8, y + 7, 12, 12), 3, 3)
    text(p, QRectF(x + 26, y, width - 30, 26), label, t.text, 8.8, True, Qt.AlignVCenter | Qt.AlignLeft)


def number(p: QPainter, center: QPointF, value: int, color: str) -> None:
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(color))
    p.drawEllipse(center, 10, 10)
    text(p, QRectF(center.x() - 10, center.y() - 10, 20, 20), str(value), "#ffffff", 8.5, True)


def mri_slice(p: QPainter, rect: QRectF, lesion: bool = True, sam_box: bool = False, detect_box: bool = False,
              outline: Optional[Sequence[int]] = None, filled: Optional[Sequence[int]] = None,
              faint: bool = False) -> QRectF:
    """Stylised axial MRI slice in ``rect``; returns the lesion rectangle."""
    p.save()
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(10, 11, 14))
    p.drawRoundedRect(rect, 8, 8)
    cx, cy = rect.center().x(), rect.center().y()
    rx, ry = rect.width() * 0.33, rect.height() * 0.41
    head = QRectF(cx - rx, cy - ry, 2 * rx, 2 * ry)
    p.setBrush(QColor(205, 205, 210) if not faint else QColor(120, 120, 125))
    p.drawEllipse(head)
    p.setBrush(QColor(28, 28, 32))
    p.drawEllipse(head.adjusted(4, 4, -4, -4))
    gradient = QRadialGradient(QPointF(cx, cy), rx)
    gradient.setColorAt(0.0, QColor(150, 152, 158))
    gradient.setColorAt(1.0, QColor(105, 107, 113))
    p.setBrush(QBrush(gradient))
    p.drawEllipse(head.adjusted(8, 8, -8, -8))
    p.setBrush(QColor(42, 43, 48))
    for side in (-1, 1):
        p.drawEllipse(QRectF(cx + side * rx * 0.17 - rx * 0.07, cy - ry * 0.25, rx * 0.14, ry * 0.5))
    lr = rect.width() * 0.085
    lesion_rect = QRectF(cx + rx * 0.35 - lr, cy - ry * 0.32 - lr, 2 * lr, 2 * lr)
    if lesion:
        p.setBrush(QColor(125, 125, 130))
        p.drawEllipse(lesion_rect.adjusted(-lr * 0.6, -lr * 0.6, lr * 0.6, lr * 0.6))
        p.setBrush(QColor(250, 250, 250))
        p.drawEllipse(lesion_rect)
        p.setBrush(QColor(95, 95, 100))
        p.drawEllipse(lesion_rect.adjusted(lr * 0.45, lr * 0.45, -lr * 0.45, -lr * 0.45))
    if filled is not None:
        fill = QColor(*filled)
        fill.setAlpha(150)
        p.setBrush(fill)
        p.setPen(Qt.NoPen)
        p.drawEllipse(lesion_rect.adjusted(-2, -2, 2, 2))
    if outline is not None:
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(*outline), 2.2))
        p.drawEllipse(lesion_rect.adjusted(-2, -2, 2, 2))
    if detect_box:
        pen = QPen(DETECT_GREEN, 1.8, Qt.DashLine)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawRect(lesion_rect.adjusted(-7, -7, 7, 7))
    if sam_box:
        p.setPen(QPen(LESION_RED, 1.8, Qt.DashLine))
        tint = QColor(LESION_RED)
        tint.setAlpha(35)
        p.setBrush(tint)
        b = lesion_rect.adjusted(-9, -9, 9, 9)
        p.drawRect(b)
        p.setPen(QPen(LESION_RED, 1.5))
        p.setBrush(QColor(255, 255, 255))
        for corner in (b.topLeft(), b.topRight(), b.bottomLeft(), b.bottomRight()):
            p.drawRect(QRectF(corner.x() - 3, corner.y() - 3, 6, 6))
    p.restore()
    return lesion_rect
