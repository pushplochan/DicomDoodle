"""
Slice navigation bar.

:class:`SliceTimeline` replaces the original horizontal scrollbar: besides
the handle it shows colored *markers* — ticks (e.g. slices saved this
session, annotation files on disk, unsaved edits) or small dots (blank
masks) — and *bands* (the auto-detect range), so the state of the whole
series is visible at a glance.

:class:`SliceNavigator` wraps it with first/previous/play/next/last buttons
and a slice number box.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from PyQt5.QtCore import QPointF, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFontMetrics, QPainter, QPen
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QSpinBox, QToolTip, QWidget

from ..qt_utils import make_tool_button
from ..theme import theme_color, theme_manager
from ..icons import bind_icon

Band = Tuple[int, int, str, str]     # start, end, color (hex), tooltip


class SliceTimeline(QWidget):
    valueChanged = pyqtSignal(int)

    MARGIN = 10

    def __init__(self, parent=None):
        super().__init__(parent)
        self._count = 0
        self._value = 0
        self._markers: Dict[str, Tuple[List[int], str, str, str]] = {}   # name -> (indices, token, label, style)
        self._bands: List[Band] = []
        self._dragging = False
        self._hover_x: Optional[float] = None
        self.setMinimumHeight(38)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setCursor(Qt.PointingHandCursor)
        # bound method (not a lambda) so Qt disconnects it when this widget is destroyed
        theme_manager().themeChanged.connect(self._theme_changed)

    def _theme_changed(self, *_args) -> None:
        self.update()

    # ------------------------------------------------------------ public API
    def set_count(self, count: int) -> None:
        self._count = max(0, count)
        self._value = min(self._value, max(0, self._count - 1))
        self.update()

    def count(self) -> int:
        return self._count

    def value(self) -> int:
        return self._value

    def set_value(self, value: int, emit: bool = True) -> None:
        if self._count == 0:
            return
        value = max(0, min(self._count - 1, int(value)))
        if value == self._value:
            return
        self._value = value
        self.update()
        if emit:
            self.valueChanged.emit(value)

    def set_markers(self, name: str, indices: Iterable[int], token: str, label: str, style: str = "tick") -> None:
        """Mark slices below the track; ``style`` is "tick" or "dot"."""
        self._markers[name] = (sorted(set(indices)), token, label, style)
        self.update()

    def set_bands(self, bands: List[Band]) -> None:
        self._bands = bands
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(400, 38)

    # ------------------------------------------------------------- geometry
    def _track(self) -> QRectF:
        return QRectF(self.MARGIN, self.height() / 2 - 3, max(1, self.width() - 2 * self.MARGIN), 6)

    def x_of(self, index: int) -> float:
        """Horizontal position of slice ``index`` on the track (widget coordinates)."""
        return self._x_for(index)

    def _x_for(self, index: int) -> float:
        track = self._track()
        if self._count <= 1:
            return track.left()
        return track.left() + track.width() * index / (self._count - 1)

    def _index_at(self, x: float) -> int:
        track = self._track()
        if self._count <= 1:
            return 0
        ratio = (x - track.left()) / track.width()
        return int(round(max(0.0, min(1.0, ratio)) * (self._count - 1)))

    # --------------------------------------------------------------- paint
    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        track = self._track()
        enabled = self.isEnabled() and self._count > 0

        painter.setPen(Qt.NoPen)
        painter.setBrush(theme_color("border_strong"))
        painter.drawRoundedRect(track, 3, 3)

        # range bands
        for start, end, color, _tip in self._bands:
            lo, hi = sorted((start, end))
            x1, x2 = self._x_for(lo) - 3, self._x_for(hi) + 3
            band = QColor(color)
            band.setAlpha(110)
            painter.setBrush(band)
            painter.drawRoundedRect(QRectF(x1, track.top() - 4, max(6.0, x2 - x1), track.height() + 8), 4, 4)

        # markers: short ticks or small dots below the track (later kinds draw on top)
        row_y = track.bottom() + 5
        for _name, (indices, token, _label, style) in self._markers.items():
            if not indices or not self._count:
                continue
            if style == "dot":
                painter.setPen(Qt.NoPen)
                painter.setBrush(theme_color(token))
                for i in indices:
                    painter.drawEllipse(QPointF(self._x_for(i), row_y + 2.5), 1.7, 1.7)
                continue
            painter.setPen(QPen(theme_color(token), 2))
            for i in indices:
                x = self._x_for(i)
                painter.drawLine(QPointF(x, row_y), QPointF(x, row_y + 5))

        if not enabled:
            return
        # handle
        x = self._x_for(self._value)
        accent = theme_color("accent")
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.setBrush(accent)
        painter.drawEllipse(QPointF(x, track.center().y()), 8, 8)

        # hover / drag bubble
        if self._hover_x is not None or self._dragging:
            idx = self._value if self._dragging else self._index_at(self._hover_x)
            text = f"{idx + 1}"
            fm = QFontMetrics(self.font())
            w = fm.horizontalAdvance(text) + 12
            bx = min(max(self._x_for(idx) - w / 2, 0), self.width() - w)
            bubble = QRectF(bx, 0, w, fm.height() + 2)
            painter.setPen(Qt.NoPen)
            painter.setBrush(theme_color("surface_alt"))
            painter.drawRoundedRect(bubble, 5, 5)
            painter.setPen(theme_color("text"))
            painter.drawText(bubble, Qt.AlignCenter, text)

    # --------------------------------------------------------------- input
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._count:
            self._dragging = True
            self.set_value(self._index_at(event.x()))
            self.update()

    def mouseMoveEvent(self, event) -> None:
        self._hover_x = event.x()
        if self._dragging:
            self.set_value(self._index_at(event.x()))
        elif self._count:
            idx = self._index_at(event.x())
            notes = [label for _, (indices, _token, label, _style) in self._markers.items() if idx in indices]
            notes += [tip for s, e, _c, tip in self._bands if min(s, e) <= idx <= max(s, e)]
            QToolTip.showText(event.globalPos(), f"Slice {idx + 1}" + ("\n" + "\n".join(notes) if notes else ""), self)
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        self._dragging = False
        self.update()

    def leaveEvent(self, event) -> None:
        self._hover_x = None
        self.update()

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta:
            self.set_value(self._value + (-1 if delta > 0 else 1))

    def keyPressEvent(self, event) -> None:
        key = event.key()
        steps = {Qt.Key_Left: -1, Qt.Key_Down: -1, Qt.Key_Right: 1, Qt.Key_Up: 1,
                 Qt.Key_PageUp: -10, Qt.Key_PageDown: 10}
        if key in steps:
            self.set_value(self._value + steps[key])
        elif key == Qt.Key_Home:
            self.set_value(0)
        elif key == Qt.Key_End:
            self.set_value(self._count - 1)
        else:
            super().keyPressEvent(event)


class SliceNavigator(QWidget):
    """Timeline + buttons + slice number box."""

    valueChanged = pyqtSignal(int)

    def __init__(self, parent=None, show_play: bool = True):
        super().__init__(parent)
        self.timeline = SliceTimeline(self)
        self.first_btn = make_tool_button("first", "First slice (Home)")
        self.prev_btn = make_tool_button("chevron-left", "Previous slice (← / wheel up)")
        self.play_btn = make_tool_button("play", "Play through the slices (cine)", checkable=True)
        self.next_btn = make_tool_button("chevron-right", "Next slice (→ / wheel down)")
        self.last_btn = make_tool_button("last", "Last slice (End)")
        self.spin = QSpinBox(self)
        self.spin.setMinimumWidth(78)
        self.spin.setAlignment(Qt.AlignRight)
        self.spin.setKeyboardTracking(False)
        self.spin.setToolTip("Go to slice number")
        self.total_label = QLabel("/ 0", self)
        self.total_label.setObjectName("Muted")
        self.total_label.setMinimumWidth(44)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 10, 4)
        layout.setSpacing(4)
        for widget in (self.first_btn, self.prev_btn):
            layout.addWidget(widget)
        if show_play:
            layout.addWidget(self.play_btn)
        else:
            self.play_btn.hide()
        layout.addWidget(self.timeline, 1)
        for widget in (self.next_btn, self.last_btn):
            layout.addWidget(widget)
        layout.addSpacing(6)
        layout.addWidget(self.spin)
        layout.addWidget(self.total_label)

        self._timer = QTimer(self)
        self._timer.setInterval(125)
        self._timer.timeout.connect(self._advance)

        self.timeline.valueChanged.connect(self._timeline_changed)
        self.spin.valueChanged.connect(lambda v: self.set_value(v - 1))
        self.first_btn.clicked.connect(lambda: self.set_value(0))
        self.last_btn.clicked.connect(lambda: self.set_value(self.timeline.count() - 1))
        self.prev_btn.clicked.connect(lambda: self.step(-1))
        self.next_btn.clicked.connect(lambda: self.step(1))
        self.play_btn.toggled.connect(self._toggle_play)
        self.set_count(0)

    # ------------------------------------------------------------ public API
    def set_count(self, count: int) -> None:
        self.timeline.set_count(count)
        self.spin.blockSignals(True)
        self.spin.setRange(1 if count else 0, max(count, 0))
        self.spin.blockSignals(False)
        self.total_label.setText(f"/ {count}")
        has = count > 0
        for widget in (self.first_btn, self.prev_btn, self.play_btn, self.next_btn,
                       self.last_btn, self.spin, self.timeline):
            widget.setEnabled(has)
        if not has:
            self.play_btn.setChecked(False)

    def value(self) -> int:
        return self.timeline.value()

    def set_value(self, value: int, emit: bool = True) -> None:
        before = self.timeline.value()
        self.timeline.set_value(value, emit=False)
        self._sync_spin()
        if emit and self.timeline.value() != before:
            self.valueChanged.emit(self.timeline.value())

    def step(self, delta: int) -> None:
        self.set_value(self.timeline.value() + delta)

    def set_markers(self, name: str, indices: Iterable[int], token: str, label: str, style: str = "tick") -> None:
        self.timeline.set_markers(name, indices, token, label, style)

    def set_bands(self, bands: List[Band]) -> None:
        self.timeline.set_bands(bands)

    def stop_playback(self) -> None:
        self.play_btn.setChecked(False)

    # -------------------------------------------------------------- internal
    def _timeline_changed(self, value: int) -> None:
        self._sync_spin()
        self.valueChanged.emit(value)

    def _sync_spin(self) -> None:
        self.spin.blockSignals(True)
        self.spin.setValue(self.timeline.value() + 1)
        self.spin.blockSignals(False)

    def _toggle_play(self, playing: bool) -> None:
        bind_icon(self.play_btn, "pause" if playing else "play")
        if playing:
            self._timer.start()
        else:
            self._timer.stop()

    def _advance(self) -> None:
        count = self.timeline.count()
        if count:
            self.set_value((self.timeline.value() + 1) % count)
