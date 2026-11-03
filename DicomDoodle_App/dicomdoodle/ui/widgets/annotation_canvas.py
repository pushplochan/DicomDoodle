"""
The main annotator's image view: an :class:`ImageCanvas` plus the tool of
the left mouse button —

* **box** — drag a box for SAM; drag it to move, drag a corner to resize;
  right-click it (or press Enter) to segment
* **pen** / **eraser** — draw or rub out outlines by hand (formerly the
  separate Manual Draw window), painted straight into the slice's stroke
  layer; the eraser first turns AI regions into their outlines, so it
  removes exactly the lines that would be saved
* **pan** — drag to move the image

— and the annotation layers: AI regions as crisp vector outlines (or filled
regions) in each label's color, hand-drawn outlines as pixels exactly as
they are saved, and the last YOLO detection box (green, dashed).
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import QPoint, QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QCursor, QFont, QPainter, QPainterPath, QPen, QPixmap
from PyQt5.QtWidgets import (QGraphicsItem, QGraphicsPathItem, QGraphicsPixmapItem, QGraphicsRectItem,
                             QGraphicsSimpleTextItem, QMenu)

from ...core import mask_ops
from ...core.annotation import SliceAnnotation
from ..icons import icon
from ..qt_utils import numpy_to_pixmap
from .box_roi import BoxROI, _Handle
from .image_canvas import ImageCanvas

DETECTION_COLOR = QColor(64, 220, 120)

# tool -> (name, icon); the order of the tool buttons
TOOLS: Dict[str, Tuple[str, str]] = {
    "box": ("Box", "box"), "pen": ("Pen", "pen"), "eraser": ("Eraser", "eraser"), "pan": ("Pan", "move"),
}


class AnnotationCanvas(ImageCanvas):
    detectRequested = pyqtSignal()        # segment the current box with SAM
    boxChanged = pyqtSignal(bool)         # True when a usable box exists
    toolRequested = pyqtSignal(str)       # a tool was picked in the right-click menu
    strokeStarted = pyqtSignal()          # a pen / eraser stroke began on the current slice
    strokeFinished = pyqtSignal()         # ... and ended

    def __init__(self, parent=None):
        super().__init__(parent)
        self.box: Optional[BoxROI] = None
        self.editing_enabled = True       # False while no series is open or a job runs
        self.tool = "box"
        self.pen_size, self.eraser_size = 1, 5
        self.pen_value = 1                # label value the pen draws with
        # the annotation a stroke paints into (the current slice's), or None
        self.annotation_for_drawing: Callable[[], Optional[SliceAnnotation]] = lambda: None
        self._drawing_from: Optional[QPointF] = None      # corner of a box being dragged out
        self._stroke: Optional[SliceAnnotation] = None    # annotation of the stroke in progress
        self._last: Optional[Tuple[int, int]] = None
        self._contour_items: List[QGraphicsPathItem] = []
        self._detection_items: List[QGraphicsItem] = []

        self._strokes: Optional[np.ndarray] = None        # hand-drawn layer on show
        self._lut = np.zeros((256, 3), dtype=np.uint8)
        self._fill_alpha = 0.0
        self._grow = 1
        self._fill_item = QGraphicsPixmapItem()           # inside of closed hand-drawn outlines
        self._stroke_item = QGraphicsPixmapItem()         # the hand-drawn outlines
        for item, z in ((self._fill_item, 11), (self._stroke_item, 12)):
            item.setZValue(z)
            item.setAcceptedMouseButtons(Qt.NoButton)
            self.scene().addItem(item)
        self.zoomChanged.connect(self._zoom_changed)

    # ----------------------------------------------------------------- tools
    @property
    def is_drawing(self) -> bool:
        """A pen or eraser stroke is in progress (the mouse button is still down)."""
        return self._stroke is not None

    def set_tool(self, tool: str) -> None:
        if tool not in TOOLS:
            return
        if tool != "box":
            self.remove_box()
        self.tool = tool
        self.update_cursor()

    def update_cursor(self) -> None:
        """The current tool's cursor: a solid dot in the label's color for the pen, a
        circle as big as the eraser, a hand to pan."""
        if self.tool == "pan":
            self.viewport().setCursor(Qt.OpenHandCursor)
            return
        if self.tool == "box":
            self.viewport().setCursor(Qt.ArrowCursor)
            return
        if self.tool == "pen":
            self.viewport().setCursor(self._pen_cursor())
            return
        diameter = max(9, int(self.eraser_size * self.zoom) + 2)
        pix = QPixmap(diameter + 2, diameter + 2)
        pix.fill(Qt.transparent)
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(0, 0, 0, 200), 3))
        painter.drawEllipse(1, 1, diameter - 1, diameter - 1)
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.drawEllipse(1, 1, diameter - 1, diameter - 1)
        painter.end()
        self.viewport().setCursor(QCursor(pix))

    def _pen_cursor(self) -> QCursor:
        """A solid dot in the pen's label color, as big as the pen on screen (at least 7 px);
        a dark edge and a light rim keep it visible on bright and on black areas."""
        diameter = max(7, round(self.pen_size * self.zoom))
        diameter += 1 - diameter % 2              # odd, so the dot is centred on the hot spot
        pix = QPixmap(diameter + 4, diameter + 4)
        pix.fill(Qt.transparent)
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(255, 255, 255, 170), 1))
        painter.drawEllipse(QRectF(1.5, 1.5, diameter + 1, diameter + 1))
        painter.setPen(QPen(QColor(0, 0, 0, 220), 1))
        painter.setBrush(QColor(*(int(c) for c in self._lut[self.pen_value][:3])))
        painter.drawEllipse(QRectF(2.5, 2.5, diameter - 1, diameter - 1))
        painter.end()
        return QCursor(pix)

    def restore_cursor(self) -> None:
        if self._space_held:
            self.viewport().setCursor(Qt.OpenHandCursor)
        else:
            self.update_cursor()

    def _zoom_changed(self, _zoom: float) -> None:
        self.restore_cursor()
        if self._grow != self._grow_for_zoom():
            self._render_strokes(fill=False)

    # ------------------------------------------------------------------ box
    def remove_box(self) -> None:
        if self.box is not None:
            self.scene().removeItem(self.box)
            self.box = None
            self.boxChanged.emit(False)

    def box_coords(self):
        return self.box.box() if self.box is not None and self.box.is_usable() else None

    def set_box(self, x1: float, y1: float, x2: float, y2: float) -> None:
        """Place the box in image coordinates, as if it had been drawn with the mouse."""
        if not self.has_image:
            return
        self.remove_box()
        self.box = BoxROI(QRectF(QPointF(x1, y1), QPointF(x2, y2)), self.image_rect(), self._box_changed)
        self.scene().addItem(self.box)
        self._box_changed()

    def _box_changed(self) -> None:
        self.boxChanged.emit(self.box is not None and self.box.is_usable())

    # ------------------------------------------------------------- contours
    def set_contours(self, contours: Dict[int, List[np.ndarray]], colors: Dict[int, Sequence[int]],
                     line_width: float = 2.0, fill_alpha: float = 0.0) -> None:
        """Draw label outlines. contours: {value: [(N, 2) x/y arrays]}."""
        for item in self._contour_items:
            self.scene().removeItem(item)
        self._contour_items = []
        for value, polys in contours.items():
            path = QPainterPath()
            path.setFillRule(Qt.OddEvenFill)
            for poly in polys:
                path.moveTo(poly[0][0] + 0.5, poly[0][1] + 0.5)
                for x, y in poly[1:]:
                    path.lineTo(x + 0.5, y + 0.5)
                path.closeSubpath()
            rgb = colors.get(value, (255, 255, 255))
            item = QGraphicsPathItem(path)
            pen = QPen(QColor(*rgb[:3]), line_width)
            pen.setCosmetic(True)
            pen.setJoinStyle(Qt.RoundJoin)
            item.setPen(pen)
            if fill_alpha > 0:
                item.setBrush(QBrush(QColor(rgb[0], rgb[1], rgb[2], int(255 * fill_alpha))))
            item.setZValue(10)
            item.setAcceptedMouseButtons(Qt.NoButton)
            self.scene().addItem(item)
            self._contour_items.append(item)

    def show_detection(self, box=None, text: str = "") -> None:
        """Green dashed box showing where YOLO found the lesion (None clears)."""
        for item in self._detection_items:
            self.scene().removeItem(item)
        self._detection_items = []
        if box is None:
            return
        x1, y1, x2, y2 = box
        rect_item = QGraphicsRectItem(QRectF(QPointF(x1, y1), QPointF(x2, y2)))
        pen = QPen(DETECTION_COLOR, 2, Qt.DashLine)
        pen.setCosmetic(True)
        rect_item.setPen(pen)
        rect_item.setZValue(15)
        rect_item.setAcceptedMouseButtons(Qt.NoButton)
        self.scene().addItem(rect_item)
        self._detection_items.append(rect_item)
        if text:
            label = QGraphicsSimpleTextItem(text)
            font = QFont(self.font())
            font.setBold(True)
            label.setFont(font)
            label.setBrush(DETECTION_COLOR)
            label.setFlag(QGraphicsItem.ItemIgnoresTransformations)
            label.setPos(x1, y1)
            label.setZValue(16)
            label.setAcceptedMouseButtons(Qt.NoButton)
            self.scene().addItem(label)
            self._detection_items.append(label)

    # ----------------------------------------------------- hand-drawn layer
    def set_stroke_style(self, lut: np.ndarray, fill_alpha: float = 0.0) -> None:
        """Label colors (``LabelRegistry.color_lut()``) and the opacity of the fill
        inside closed hand-drawn outlines (0: outlines only)."""
        recolor = not np.array_equal(lut, self._lut)
        self._lut = lut
        self._fill_alpha = fill_alpha
        if recolor:
            self.restore_cursor()         # the pen's dot has the label's color

    def set_strokes(self, strokes: Optional[np.ndarray]) -> None:
        """Show the hand-drawn outlines of the current slice (None: there are none)."""
        self._strokes = strokes
        self._render_strokes(fill=True)

    def _grow_for_zoom(self) -> int:
        """Line thickness on screen: grown when zoomed out, so one-pixel lines stay visible."""
        return max(1, int(math.ceil(1.0 / max(self.zoom, 1e-3)))) if self.zoom < 1 else 1

    def _render_strokes(self, fill: bool) -> None:
        strokes = self._strokes
        if strokes is None or not strokes.any():
            self._stroke_item.setPixmap(QPixmap())
            self._fill_item.setPixmap(QPixmap())
            return
        self._grow = self._grow_for_zoom()
        self._stroke_item.setPixmap(numpy_to_pixmap(
            mask_ops.rgba_overlay(mask_ops.thicken(strokes, self._grow), self._lut)))
        if fill:          # the fill only changes when a stroke ends
            if self._fill_alpha > 0:
                inside = np.where(strokes > 0, 0, mask_ops.fill_outlines(strokes)).astype(np.uint8)
                self._fill_item.setPixmap(numpy_to_pixmap(mask_ops.rgba_overlay(inside, self._lut,
                                                                                self._fill_alpha)))
            else:
                self._fill_item.setPixmap(QPixmap())

    def _image_point(self, view_pos: QPoint) -> Optional[Tuple[int, int]]:
        pos = self.mapToScene(view_pos)
        rect = self.image_rect()
        x, y = int(math.floor(pos.x())), int(math.floor(pos.y()))
        return (x, y) if 0 <= x < rect.width() and 0 <= y < rect.height() else None

    def _paint_segment(self, start: Tuple[int, int], end: Tuple[int, int]) -> None:
        erase = self.tool == "eraser"
        value = 0 if erase else int(self.pen_value)
        size = self.eraser_size if erase else self.pen_size
        strokes = self._stroke.strokes
        if start == end:
            if size <= 1:
                strokes[start[1], start[0]] = value
            else:
                cv2.circle(strokes, start, size // 2, value, -1)
        else:
            cv2.line(strokes, start, end, value, size)

    def _begin_stroke(self, point: Tuple[int, int]) -> None:
        ann = self.annotation_for_drawing()
        if ann is None:
            return
        ann.begin_edit()
        if self.tool == "eraser":
            ann.regions_to_outlines()
        self._stroke, self._last = ann, point
        self.strokeStarted.emit()          # the owner shows the slice's layers again
        self._strokes = ann.strokes
        self._paint_segment(point, point)
        self._render_strokes(fill=False)

    # --------------------------------------------------------------- events
    def wheelEvent(self, event) -> None:
        if self.is_drawing:               # finish the stroke before changing slice or zoom
            event.accept()
            return
        super().wheelEvent(event)

    def _on_box_item(self, view_pos) -> bool:
        item = self.itemAt(view_pos)
        return isinstance(item, (BoxROI, _Handle))

    def mousePressEvent(self, event) -> None:
        left = (event.button() == Qt.LeftButton and self.has_image and self.editing_enabled
                and not self._space_held)
        if left and self.tool in ("pen", "eraser"):
            point = self._image_point(event.pos())
            if point is not None:
                self._begin_stroke(point)
            event.accept()
            return
        if left and self.tool == "pan":
            self._pan_origin = event.pos()
            self.viewport().setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        if left and self.tool == "box" and not self._on_box_item(event.pos()):
            pos = self.mapToScene(event.pos())
            if self.image_rect().contains(pos):
                self.remove_box()
                self._drawing_from = pos
                self.box = BoxROI(QRectF(pos, pos), self.image_rect(), self._box_changed)
                self.scene().addItem(self.box)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._stroke is not None:
            point = self._image_point(event.pos())
            if point is not None:
                self._paint_segment(self._last, point)
                self._last = point
                self._render_strokes(fill=False)
                self.cursorMoved.emit(point[0], point[1])
            event.accept()
            return
        if self._drawing_from is not None and self.box is not None:
            pos = self.mapToScene(event.pos())
            self.box.set_box(QRectF(self._drawing_from, pos))
            pos_i = self.image_rect()
            if pos_i.contains(pos):
                self.cursorMoved.emit(int(pos.x()), int(pos.y()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._stroke is not None and event.button() == Qt.LeftButton:
            self._stroke = self._last = None
            self._render_strokes(fill=True)
            self.strokeFinished.emit()
            event.accept()
            return
        if self._drawing_from is not None and event.button() == Qt.LeftButton:
            self._drawing_from = None
            if self.box is not None and not self.box.is_usable():
                self.remove_box()          # a click, not a drag
            else:
                self._box_changed()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event) -> None:
        if not self.has_image or self._stroke is not None:
            return
        menu = QMenu(self)
        on_box = self.box is not None and self.box.isUnderMouse() or self._on_box_item(event.pos())
        if self.box is not None and self.box.is_usable():
            detect = menu.addAction(icon("box-detect"), "Detect  (segment box with SAM)")
            detect.setShortcutVisibleInContextMenu(True)
            detect.triggered.connect(self.detectRequested.emit)
            remove = menu.addAction(icon("x"), "Remove box")
            remove.triggered.connect(self.remove_box)
            if on_box:
                menu.setDefaultAction(detect)
            menu.addSeparator()
        if self.editing_enabled:
            for tool, (name, icon_name) in TOOLS.items():
                action = menu.addAction(icon(icon_name), f"{name} tool")
                action.setCheckable(True)
                action.setChecked(tool == self.tool)
                action.triggered.connect(lambda _c=False, t=tool: self.toolRequested.emit(t))
            menu.addSeparator()
        menu.addAction(icon("fit"), "Fit to window", self.fit_to_window)
        menu.addAction(icon("zoom-in"), "Actual size (100%)", self.zoom_actual)
        menu.exec_(event.globalPos())

    def keyPressEvent(self, event) -> None:
        if self.box is not None and event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.detectRequested.emit()
            event.accept()
            return
        if self.box is not None and event.key() in (Qt.Key_Delete, Qt.Key_Backspace, Qt.Key_Escape):
            self.remove_box()
            event.accept()
            return
        super().keyPressEvent(event)
