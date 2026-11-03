"""
Batch-processing panel: the auto-detect range and the automatic blank masks.

The range works like the original "Set Auto Detect Range Start/End"
buttons, but the chosen slices are always visible (the original only
printed them to the console), it can be cleared, and it is drawn as a
colored band on the slice timeline. "Find tumour range" sets it from one
slice of the tumour; a small preview then shows the two other views through
the tumour with the detector's boxes and the range.

The original's two blank-mask ranges are gone: every save now gives each
slice without an annotation file a blank mask automatically. The panel
shows how many slices are annotated, blank or still without a file, and
lets the user switch the automatic blank masks off.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

from PyQt5.QtCore import QRectF, QSize, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import QCheckBox, QGridLayout, QHBoxLayout, QVBoxLayout, QWidget

from ...core.annotation import SliceRange
from ..guide import help_button
from ..qt_utils import label, make_button, make_tool_button, numpy_to_qimage
from ..theme import theme_color
from .annotation_canvas import DETECTION_COLOR
from .cards import Card

if TYPE_CHECKING:
    from ...core.tumour_range import RangeView, TumourRange

AUTO_RANGE_COLOR = "#a78bfa"


class _Dot(QWidget):
    def __init__(self, color: str, parent=None):
        super().__init__(parent)
        self._color = QColor(color)
        self.setFixedSize(10, 10)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(self._color)
        painter.drawEllipse(0, 0, 10, 10)


class RangeRow(QWidget):
    """'● Auto-detect range   [Start · 12] [End · 20]  9 slices  [×]'"""

    setStart = pyqtSignal()
    setEnd = pyqtSignal()
    cleared = pyqtSignal()

    def __init__(self, title: str, color: str, parent=None):
        super().__init__(parent)
        self.start_btn = make_button("Start", tooltip="Use the current slice as the start of this range")
        self.end_btn = make_button("End", tooltip="Use the current slice as the end of this range")
        for button in (self.start_btn, self.end_btn):
            button.setMinimumWidth(78)
        self.clear_btn = make_tool_button("x", "Clear this range")
        self.count_label = label("", "Faint")
        title_label = label(title, "Value")

        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)
        head = QHBoxLayout()
        head.setSpacing(6)
        head.addWidget(_Dot(color))
        head.addWidget(title_label)
        head.addStretch(1)
        head.addWidget(self.count_label)
        head.addWidget(self.clear_btn)
        grid.addLayout(head, 0, 0, 1, 2)
        grid.addWidget(self.start_btn, 1, 0)
        grid.addWidget(self.end_btn, 1, 1)

        self.start_btn.clicked.connect(self.setStart.emit)
        self.end_btn.clicked.connect(self.setEnd.emit)
        self.clear_btn.clicked.connect(self.cleared.emit)

    def show_range(self, rng: SliceRange) -> None:
        fmt = lambda v: "—" if v is None else str(v + 1)  # noqa: E731
        self.start_btn.setText(f"Start · {fmt(rng.start)}")
        self.end_btn.setText(f"End · {fmt(rng.end)}")
        n = len(rng.indices())
        self.count_label.setText(f"{n} slice{'s' if n != 1 else ''}" if n else "")
        self.clear_btn.setEnabled(rng.start is not None or rng.end is not None)


class RangePreview(QWidget):
    """The two other views through the tumour after "Find tumour range": the
    detector's box on each (green, dashed, as on the image; dotted grey where
    it found none and the box was guessed) and the range (the purple band of
    the timeline). As tall as the views need at the panel's width."""

    GAP = 8
    MAX_IMAGE_HEIGHT = 220

    def __init__(self, parent=None):
        super().__init__(parent)
        self._views = []
        policy = self.sizePolicy()
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self.hide()

    def show_views(self, views: List["RangeView"]) -> None:
        self._views = [(numpy_to_qimage(v.image), v) for v in views]
        self.setToolTip("<br>".join(f"<b>{v.plane.capitalize()}</b>: {v.how}" for v in views))
        self.setVisible(bool(views))
        self.updateGeometry()
        self.update()

    def _caption_height(self) -> int:
        return self.fontMetrics().height() + 4

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        if not self._views:
            return 0
        cell_w = (width - self.GAP * (len(self._views) - 1)) / len(self._views)
        image_h = max(cell_w * image.height() / image.width() for image, _ in self._views)
        return int(min(image_h, self.MAX_IMAGE_HEIGHT)) + self._caption_height()

    def sizeHint(self) -> QSize:
        return QSize(260, self.heightForWidth(260))

    def paintEvent(self, event) -> None:
        if not self._views:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        caption_h = self._caption_height()
        cell_w = (self.width() - self.GAP * (len(self._views) - 1)) / len(self._views)
        cell_h = self.height() - caption_h
        band_color = QColor(AUTO_RANGE_COLOR)
        fill = QColor(band_color)
        fill.setAlpha(60)
        for k, (image, view) in enumerate(self._views):
            left = k * (cell_w + self.GAP)
            s = min(cell_w / image.width(), cell_h / image.height())
            w, h = image.width() * s, image.height() * s
            r = QRectF(left + (cell_w - w) / 2, (cell_h - h) / 2, w, h)
            p.drawImage(r, image)
            p.save()
            p.setClipRect(r)
            lo, hi = (v * s for v in view.band)
            if view.along_rows:
                band = QRectF(r.left(), r.top() + lo, w, hi - lo)
                edges = ((band.topLeft(), band.topRight()), (band.bottomLeft(), band.bottomRight()))
            else:
                band = QRectF(r.left() + lo, r.top(), hi - lo, h)
                edges = ((band.topLeft(), band.bottomLeft()), (band.topRight(), band.bottomRight()))
            p.fillRect(band, fill)
            p.setPen(QPen(band_color, 1.5))
            for a, b in edges:
                p.drawLine(a, b)
            x0, y0, x1, y1 = (v * s for v in view.box)
            p.setPen(QPen(QColor(170, 170, 170), 1.2, Qt.DotLine) if view.estimated
                     else QPen(DETECTION_COLOR, 1.5, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawRect(QRectF(r.left() + x0, r.top() + y0, x1 - x0, y1 - y0))
            p.restore()
            caption = view.plane.capitalize()
            if view.estimated:
                caption += " · no box"
            elif view.slices:
                caption += f" · {view.slices[0] + 1}–{view.slices[1] + 1}"
            p.setPen(theme_color("text_muted"))
            p.drawText(QRectF(left, cell_h, cell_w, caption_h), Qt.AlignCenter, caption)


class _Stat(QWidget):
    """A number with a caption under it, e.g. '27 / Blank masks'."""

    def __init__(self, caption: str, parent=None):
        super().__init__(parent)
        self.value = label("—", "Value")
        font = self.value.font()
        font.setPointSizeF(font.pointSizeF() + 2)
        self.value.setFont(font)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.value)
        layout.addWidget(label(caption, "Faint"))


class BatchPanel(QWidget):
    """Emits requests; the main window owns the session and does the work."""

    rangeAction = pyqtSignal(str)           # auto-detect range: "start" / "end" / "clear"
    findRangeRequested = pyqtSignal()
    runAutoRequested = pyqtSignal()
    autoBlankToggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.auto_row = RangeRow("Auto-detect range", AUTO_RANGE_COLOR)
        self.auto_row.setStart.connect(lambda: self.rangeAction.emit("start"))
        self.auto_row.setEnd.connect(lambda: self.rangeAction.emit("end"))
        self.auto_row.cleared.connect(lambda: self.rangeAction.emit("clear"))
        self.find_range_btn = make_button(
            "Find tumour range", "search",
            tooltip="From the slice on screen: the tumour detector boxes the tumour here and on the two other "
                    "views through it (coronal and sagittal for an axial series). Where those boxes start and "
                    "end is the range. Draw a box around the tumour first to choose it.")
        self.found_label = label("", "Faint", wrap=True)
        self.found_label.hide()
        self.preview = RangePreview()
        self.run_auto_btn = make_button("Run auto-detect && save", "sparkles", role="primary",
                                        tooltip="YOLO → SAM on every slice of the range, saving each result "
                                                "(a blank mask where nothing is found)")
        self.auto_label_hint = label("", "Faint", wrap=True)

        self.auto_blank_check = QCheckBox("Save blank masks automatically")
        self.auto_blank_check.setToolTip("When you save, every slice that has no annotation file yet "
                                         "gets a blank mask (no lesion)")
        self.stats = {key: _Stat(caption) for key, caption in
                      (("annotated", "Annotated"), ("blank", "Blank masks"), ("missing", "No file yet"))}
        self.blank_hint = label("", "Faint", wrap=True)

        auto = Card("Auto-detect range")
        auto.add_header_widget(help_button("batch"))
        auto.add(label("YOLO finds the lesion, SAM outlines it, every slice is saved.", "Faint", wrap=True))
        auto.add(self.auto_row)
        auto.add(self.find_range_btn)
        auto.add(self.found_label)
        auto.add(self.preview)
        auto.add(self.auto_label_hint)
        auto.add(self.run_auto_btn)

        blank = Card("Blank masks")
        blank.add_header_widget(help_button("saving"))
        blank.add(label("Annotate only the slices with a lesion. When you save, every slice without an "
                        "annotation file gets a blank mask (“no lesion”).", "Faint", wrap=True))
        blank.add(self.auto_blank_check)
        stats_row = QHBoxLayout()
        stats_row.setSpacing(12)
        for stat in self.stats.values():
            stats_row.addWidget(stat, 1)
        blank.add_layout(stats_row)
        blank.add(self.blank_hint)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(auto)
        layout.addWidget(blank)

        self.find_range_btn.clicked.connect(self.findRangeRequested.emit)
        self.run_auto_btn.clicked.connect(self.runAutoRequested.emit)
        self.auto_blank_check.toggled.connect(self.autoBlankToggled.emit)

    def show_auto_range(self, auto: SliceRange, auto_count: int, active_label_name: Optional[str]) -> None:
        self.auto_row.show_range(auto)
        self.run_auto_btn.setText(f"Run auto-detect && save ({auto_count})" if auto_count
                                  else "Run auto-detect && save")
        self.run_auto_btn.setEnabled(auto_count > 0)
        self.auto_label_hint.setText(f"Results use the active label: {active_label_name}" if active_label_name else "")

    def show_found_range(self, found: Optional["TumourRange"]) -> None:
        """The last "Find tumour range" result under its button (None hides it)."""
        if found is not None and found.estimated:
            self.found_label.setText(f"Estimated from the tumour's size on slice {found.seed + 1}: the detector "
                                     "found no box on the other views — check the range")
        elif found is not None:
            self.found_label.setText(f"From slice {found.seed + 1}: slices {found.first + 1}–{found.last + 1}")
        self.found_label.setVisible(found is not None)
        self.preview.show_views(found.views if found is not None else [])

    def show_blank_masks(self, enabled: bool, counts: Optional[Tuple[int, int, int]]) -> None:
        """``counts``: slices (annotated, with a blank mask, without a file); None without a series."""
        self.auto_blank_check.blockSignals(True)
        self.auto_blank_check.setChecked(enabled)
        self.auto_blank_check.blockSignals(False)
        for stat, value in zip(self.stats.values(), counts or (None, None, None)):
            stat.value.setText("—" if value is None else str(value))
        missing = counts[2] if counts else 0
        if not missing:
            hint = ""
        elif enabled:
            hint = "Slices with no file and no outline get a blank mask the next time you save."
        else:
            hint = "Automatic blank masks are off: slices with no file keep none."
        self.blank_hint.setText(hint)
        self.blank_hint.setVisible(bool(hint))
