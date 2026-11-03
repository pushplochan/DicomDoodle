"""
The bounding box the user draws to guide SAM.

Same interactions as the original tool — drag inside to move it, drag a
corner handle to resize it — with handles that keep a constant on-screen
size at any zoom, and the box always kept inside the image.
"""
from __future__ import annotations

from typing import Callable, Optional

from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QBrush, QColor, QPen
from PyQt5.QtWidgets import QGraphicsItem, QGraphicsRectItem

BOX_COLOR = QColor(255, 82, 82)
HANDLE_SIZE = 10.0
CORNERS = ("top-left", "top-right", "bottom-left", "bottom-right")
OPPOSITE = {"top-left": "bottom-right", "bottom-right": "top-left",
            "top-right": "bottom-left", "bottom-left": "top-right"}


def _corner_point(rect: QRectF, corner: str) -> QPointF:
    return {"top-left": rect.topLeft(), "top-right": rect.topRight(),
            "bottom-left": rect.bottomLeft(), "bottom-right": rect.bottomRight()}[corner]


class _Handle(QGraphicsRectItem):
    """A corner grip; ignores view scaling so it is always HANDLE_SIZE pixels."""

    def __init__(self, box: "BoxROI", corner: str):
        half = HANDLE_SIZE / 2
        super().__init__(-half, -half, HANDLE_SIZE, HANDLE_SIZE, box)
        self.box, self.corner = box, corner
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations)
        self.setBrush(QBrush(QColor(255, 255, 255)))
        pen = QPen(BOX_COLOR, 2)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setCursor(Qt.SizeFDiagCursor if corner in ("top-left", "bottom-right") else Qt.SizeBDiagCursor)
        self.setZValue(1)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.box.begin_resize(self.corner)
            event.accept()
        else:
            event.ignore()

    def mouseMoveEvent(self, event) -> None:
        self.box.resize_to(event.scenePos())

    def mouseReleaseEvent(self, event) -> None:
        self.box.finish_interaction()


class BoxROI(QGraphicsRectItem):
    """Movable, resizable rectangle in image (scene) coordinates."""

    def __init__(self, rect: QRectF, bounds: QRectF, on_changed: Optional[Callable[[], None]] = None):
        super().__init__()
        self._bounds = QRectF(bounds)
        self._on_changed = on_changed
        self._anchor: Optional[QPointF] = None
        self._press_pos: Optional[QPointF] = None
        self._press_rect: Optional[QRectF] = None
        pen = QPen(BOX_COLOR, 2, Qt.DashLine)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setBrush(QBrush(QColor(BOX_COLOR.red(), BOX_COLOR.green(), BOX_COLOR.blue(), 28)))
        self.setCursor(Qt.SizeAllCursor)
        self.setZValue(20)
        self._handles = {corner: _Handle(self, corner) for corner in CORNERS}
        self.set_box(rect)

    # ------------------------------------------------------------- geometry
    def set_box(self, rect: QRectF) -> None:
        rect = rect.normalized().intersected(self._bounds)
        self.setRect(rect)
        for corner, handle in self._handles.items():
            handle.setPos(_corner_point(rect, corner))

    def box(self):
        """(x1, y1, x2, y2) — the same values the original passed to SAM."""
        r = self.rect()
        return r.x(), r.y(), r.right(), r.bottom()

    def is_usable(self, min_size: float = 3.0) -> bool:
        return self.rect().width() >= min_size and self.rect().height() >= min_size

    def _clamp(self, point: QPointF) -> QPointF:
        return QPointF(min(max(point.x(), self._bounds.left()), self._bounds.right()),
                       min(max(point.y(), self._bounds.top()), self._bounds.bottom()))

    # -------------------------------------------------------------- resizing
    def begin_resize(self, corner: str) -> None:
        self._anchor = _corner_point(self.rect(), OPPOSITE[corner])

    def resize_to(self, scene_pos: QPointF) -> None:
        if self._anchor is not None:
            self.set_box(QRectF(self._anchor, self._clamp(scene_pos)))

    def finish_interaction(self) -> None:
        self._anchor = self._press_pos = self._press_rect = None
        if self._on_changed:
            self._on_changed()

    # ---------------------------------------------------------------- moving
    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            event.ignore()
            return
        self._press_pos = event.scenePos()
        self._press_rect = QRectF(self.rect())
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._press_pos is None:
            return
        delta = event.scenePos() - self._press_pos
        moved = self._press_rect.translated(delta)
        # keep the whole box inside the image
        dx = min(0.0, self._bounds.right() - moved.right()) + max(0.0, self._bounds.left() - moved.left())
        dy = min(0.0, self._bounds.bottom() - moved.bottom()) + max(0.0, self._bounds.top() - moved.top())
        self.set_box(moved.translated(dx, dy))

    def mouseReleaseEvent(self, event) -> None:
        self.finish_interaction()
