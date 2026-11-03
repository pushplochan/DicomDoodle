"""
Zoomable, pannable image view used by every 2D viewer in the app.

* fits the image to the window until the user zooms, then keeps the zoom
* Ctrl + mouse wheel zooms around the cursor; the plain wheel emits
  ``sliceScrolled`` so the owner can change slices (original behaviour) —
  or zooms too when *zoom mode* is on (Annotated DICOM Viewer)
* middle-button drag, or Space + left drag, pans
* an optional RGBA overlay layer (e.g. saved annotations)
* a heads-up display painted in the corners (slice number, status badges ...)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont, QFontMetrics, QPainter, QPixmap
from PyQt5.QtWidgets import QFrame, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

from ..qt_utils import numpy_to_pixmap
from ..theme import theme_color, theme_manager

ZOOM_STEP = 1.15
MIN_ZOOM, MAX_ZOOM = 0.05, 40.0

Badge = Tuple[str, str]   # (text, theme color token)


class ImageCanvas(QGraphicsView):
    zoomChanged = pyqtSignal(float)
    sliceScrolled = pyqtSignal(int)       # +1 = next slice, -1 = previous slice
    cursorMoved = pyqtSignal(int, int)    # image pixel under the mouse, (-1, -1) outside

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self._image_item = QGraphicsPixmapItem()
        self._image_item.setTransformationMode(Qt.SmoothTransformation)
        self._image_item.setAcceptedMouseButtons(Qt.NoButton)
        self._overlay_item = QGraphicsPixmapItem()
        self._overlay_item.setZValue(5)
        self._overlay_item.setAcceptedMouseButtons(Qt.NoButton)
        self.scene().addItem(self._image_item)
        self.scene().addItem(self._overlay_item)

        self._has_image = False
        self._fit_mode = True
        self.zoom_mode = False            # wheel zooms instead of changing slice
        self._pan_origin = None
        self._space_held = False
        self.placeholder = "No image loaded"
        self._hud_top_left: List[Tuple[str, bool]] = []
        self._hud_top_right: List[Badge] = []
        self._hud_bottom_left: Optional[Tuple[str, Optional[Sequence[int]]]] = None
        self.show_zoom_hud = True
        self.hud_rects: Dict[str, QRectF] = {}   # where each corner text was last drawn (viewport px)

        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setFrameShape(QFrame.NoFrame)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self._apply_theme()
        theme_manager().themeChanged.connect(self._apply_theme)

    # ------------------------------------------------------------- content
    def set_image(self, array: np.ndarray) -> None:
        pixmap = numpy_to_pixmap(array)
        size_changed = (not self._has_image) or pixmap.size() != self._image_item.pixmap().size()
        self._image_item.setPixmap(pixmap)
        self._has_image = True
        self.scene().setSceneRect(QRectF(pixmap.rect()))
        if size_changed or self._fit_mode:
            self.fit_to_window()
        self.viewport().update()

    def clear_image(self) -> None:
        self._image_item.setPixmap(QPixmap())
        self._overlay_item.setPixmap(QPixmap())
        self._has_image = False
        self.viewport().update()

    @property
    def has_image(self) -> bool:
        return self._has_image

    def image_rect(self) -> QRectF:
        return QRectF(self._image_item.pixmap().rect())

    def set_overlay(self, rgba: Optional[np.ndarray]) -> None:
        """Show (or clear with None) an RGBA layer aligned with the image."""
        self._overlay_item.setPixmap(QPixmap() if rgba is None else numpy_to_pixmap(rgba))

    # ---------------------------------------------------------------- zoom
    @property
    def zoom(self) -> float:
        return self.transform().m11()

    def fit_to_window(self) -> None:
        if not self._has_image:
            return
        self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)
        self._fit_mode = True
        self.zoomChanged.emit(self.zoom)
        self.viewport().update()

    def zoom_by(self, factor: float) -> None:
        if not self._has_image:
            return
        new_zoom = min(MAX_ZOOM, max(MIN_ZOOM, self.zoom * factor))
        factor = new_zoom / self.zoom
        self.scale(factor, factor)
        self._fit_mode = False
        self.zoomChanged.emit(self.zoom)
        self.viewport().update()

    def zoom_in(self) -> None:
        self.zoom_by(ZOOM_STEP ** 2)

    def zoom_out(self) -> None:
        self.zoom_by(1 / ZOOM_STEP ** 2)

    def zoom_actual(self) -> None:
        self.zoom_by(1.0 / self.zoom)

    # ------------------------------------------------------------------ HUD
    def set_hud(self, top_left: Optional[List[Tuple[str, bool]]] = None,
                top_right: Optional[List[Badge]] = None,
                bottom_left: Optional[Tuple[str, Optional[Sequence[int]]]] = None) -> None:
        """Corner texts. top_left: [(text, bold)], top_right: [(text, color token)],
        bottom_left: (text, swatch rgb or None)."""
        self._hud_top_left = top_left or []
        self._hud_top_right = top_right or []
        self._hud_bottom_left = bottom_left
        self.viewport().update()

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
        painter.save()
        painter.resetTransform()
        painter.setRenderHint(QPainter.Antialiasing)
        if not self._has_image:
            painter.setPen(theme_color("text_faint"))
            painter.drawText(self.viewport().rect(), Qt.AlignCenter, self.placeholder)
        else:
            self._paint_hud(painter)
        painter.restore()

    def _paint_hud(self, painter: QPainter) -> None:
        self.hud_rects = {}
        base = QFont(self.font())
        bold = QFont(base)
        bold.setBold(True)
        margin, pad = 12, 8
        bg = QColor(8, 10, 14, 170)

        # top-left: text block
        if self._hud_top_left:
            lines = [(text, bold if is_bold else base) for text, is_bold in self._hud_top_left]
            width = max(QFontMetrics(f).horizontalAdvance(t) for t, f in lines) + 2 * pad
            line_h = QFontMetrics(base).height() + 2
            box = QRectF(margin, margin, width, len(lines) * line_h + pad)
            self.hud_rects["top_left"] = box
            painter.setPen(Qt.NoPen)
            painter.setBrush(bg)
            painter.drawRoundedRect(box, 8, 8)
            y = margin + pad / 2
            for i, (text, font) in enumerate(lines):
                painter.setFont(font)
                painter.setPen(QColor(235, 238, 244) if i == 0 else QColor(165, 172, 186))
                painter.drawText(QRectF(margin + pad, y, width, line_h), Qt.AlignVCenter | Qt.AlignLeft, text)
                y += line_h

        # top-right: status badges
        x_right = self.viewport().width() - margin
        painter.setFont(bold)
        fm = QFontMetrics(bold)
        y = margin
        for text, token in self._hud_top_right:
            w = fm.horizontalAdvance(text) + 2 * pad + 14
            h = fm.height() + 8
            box = QRectF(x_right - w, y, w, h)
            self.hud_rects.setdefault("top_right", box)
            painter.setPen(Qt.NoPen)
            painter.setBrush(bg)
            painter.drawRoundedRect(box, h / 2, h / 2)
            color = theme_color(token)
            painter.setBrush(color)
            painter.drawEllipse(QPointF(box.left() + pad + 3, box.center().y()), 4, 4)
            painter.setPen(color.lighter(130) if token != "text_muted" else QColor(200, 205, 215))
            painter.drawText(box.adjusted(pad + 12, 0, -pad, 0), Qt.AlignVCenter | Qt.AlignLeft, text)
            y += h + 6

        bottom = self.viewport().height() - margin
        # bottom-left: active label chip
        if self._hud_bottom_left is not None:
            text, rgb = self._hud_bottom_left
            painter.setFont(base)
            fm = QFontMetrics(base)
            w = fm.horizontalAdvance(text) + 2 * pad + (18 if rgb else 0)
            h = fm.height() + 10
            box = QRectF(margin, bottom - h, w, h)
            self.hud_rects["bottom_left"] = box
            painter.setPen(Qt.NoPen)
            painter.setBrush(bg)
            painter.drawRoundedRect(box, 8, 8)
            text_left = box.left() + pad
            if rgb is not None:
                painter.setBrush(QColor(*rgb[:3]))
                painter.drawRoundedRect(QRectF(box.left() + pad, box.center().y() - 6, 12, 12), 3, 3)
                text_left += 18
            painter.setPen(QColor(225, 229, 236))
            painter.drawText(QRectF(text_left, box.top(), w, h), Qt.AlignVCenter | Qt.AlignLeft, text)

        # bottom-right: zoom
        if self.show_zoom_hud:
            painter.setFont(base)
            text = f"{self.zoom * 100:.0f}%"
            fm = QFontMetrics(base)
            w, h = fm.horizontalAdvance(text) + 2 * pad, fm.height() + 10
            box = QRectF(self.viewport().width() - margin - w, bottom - h, w, h)
            painter.setPen(Qt.NoPen)
            painter.setBrush(bg)
            painter.drawRoundedRect(box, 8, 8)
            painter.setPen(QColor(200, 205, 215))
            painter.drawText(box, Qt.AlignCenter, text)

    # --------------------------------------------------------------- events
    def wheelEvent(self, event) -> None:
        if not self._has_image:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        if self.zoom_mode or event.modifiers() & Qt.ControlModifier:
            self.zoom_by(ZOOM_STEP if delta > 0 else 1 / ZOOM_STEP)
        else:
            # wheel up -> previous slice, wheel down -> next slice (as the original)
            self.sliceScrolled.emit(-1 if delta > 0 else 1)
        event.accept()

    def _start_pan(self, event) -> bool:
        if event.button() == Qt.MiddleButton or (event.button() == Qt.LeftButton and self._space_held):
            self._pan_origin = event.pos()
            self.viewport().setCursor(Qt.ClosedHandCursor)
            return True
        return False

    def mousePressEvent(self, event) -> None:
        if self._has_image and self._start_pan(event):
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._pan_origin is not None:
            delta = event.pos() - self._pan_origin
            self._pan_origin = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self._fit_mode = False
            event.accept()
            return
        if self._has_image:
            pos = self.mapToScene(event.pos())
            rect = self.image_rect()
            if rect.contains(pos):
                self.cursorMoved.emit(int(pos.x()), int(pos.y()))
            else:
                self.cursorMoved.emit(-1, -1)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._pan_origin is not None and event.button() in (Qt.MiddleButton, Qt.LeftButton):
            self._pan_origin = None
            self.restore_cursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton:
            self.fit_to_window()
            return
        super().mouseDoubleClickEvent(event)

    def leaveEvent(self, event) -> None:
        self.cursorMoved.emit(-1, -1)
        super().leaveEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_held = True
            self.viewport().setCursor(Qt.OpenHandCursor)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_held = False
            self.restore_cursor()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def restore_cursor(self) -> None:
        """The cursor after panning (subclasses with tools show the tool's cursor)."""
        self.viewport().setCursor(Qt.OpenHandCursor if self._space_held else Qt.ArrowCursor)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._fit_mode:
            self.fit_to_window()

    def _apply_theme(self, *_args) -> None:
        self.setBackgroundBrush(QBrush(theme_color("canvas")))
        self.viewport().update()
