"""
Concept diagrams of the User Guide, drawn with QPainter in the current
theme's colors (so they suit both light and dark mode and stay sharp).

    render(name, width, theme, ratio) -> QPixmap
    DESIGN_SIZES[name]                -> (width, height) in design units
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QColor, QPainter, QPen, QPixmap

from ..icons import pixmap
from ..theme import Theme
from .drawing import LESION_RED, arrow, box, card, chip, mri_slice, number, text

RED, YELLOW, GREEN = (255, 0, 0), (255, 255, 0), (0, 255, 0)
CUSTOM = (148, 103, 189)


def _workflow(p: QPainter, t: Theme) -> None:
    card(p, QRectF(0, 0, 760, 190), t)
    steps = [("folder-open", "Open scans", "a folder of DICOM slices"),
             ("tag", "Pick a label", "what you will outline"),
             ("sparkles", "Segment", "Auto Detect, a box, or draw"),
             ("save", "Check & save", "one _seg.dcm per slice"),
             ("layers", "Batch work", "auto-detect a range"),
             ("bar-chart", "Review", "viewer, volume and 3D")]
    x, w, gap = 14, 104, 20
    for i, (icon, title, sub) in enumerate(steps):
        rect = QRectF(x + i * (w + gap), 26, w, 128)
        box(p, rect, t, icon, title, sub)
        number(p, QPointF(rect.left() + 12, rect.top() + 12), i + 1, t.accent)
        if i < len(steps) - 1:
            arrow(p, rect.right() + 3, rect.center().y(), rect.right() + gap - 3, rect.center().y(), t.text_faint)
    text(p, QRectF(0, 160, 760, 22), "Repeat steps 2–4 for each slice with a lesion — every other slice "
         "gets a blank mask automatically.", t.text_muted, 8.5)


def _series(p: QPainter, t: Theme) -> None:
    card(p, QRectF(0, 0, 760, 230), t)
    for k in range(5):                                   # a stack of slices
        r = QRectF(40 + k * 14, 24 + k * 10, 120, 120)
        mri_slice(p, r, lesion=k in (2, 3), faint=k < 4)
    text(p, QRectF(20, 196, 230, 22), "A series = one folder of slices", t.text, 9.5, True)
    arrow(p, 260, 120, 320, 120, t.text_faint)
    mri_slice(p, QRectF(330, 50, 150, 150))
    text(p, QRectF(330, 30, 150, 18), "Slice 133 of 166", t.text_muted, 8.5)
    text(p, QRectF(295, 202, 220, 22), "Annotate one slice at a time", t.text, 9.5, True)
    arrow(p, 492, 120, 540, 120, t.text_faint)
    files = [("image", "IMG_0133.dcm", "the image (never changed)", t.text_muted),
             ("tag", "IMG_0133_seg.dcm", "your annotation file", t.accent)]
    for k, (icon_name, name, note, color) in enumerate(files):
        y = 70 + k * 62
        p.setPen(QPen(QColor(t.border_strong), 1))
        p.setBrush(QColor(t.raised))
        p.drawRoundedRect(QRectF(552, y, 196, 50), 8, 8)
        p.drawPixmap(564, y + 15, pixmap(icon_name, color, 20))
        text(p, QRectF(592, y + 5, 152, 20), name, t.text, 8.2, True, Qt.AlignLeft | Qt.AlignVCenter)
        text(p, QRectF(592, y + 25, 152, 18), note, t.text_muted, 7.8, False, Qt.AlignLeft | Qt.AlignVCenter)


def _labels(p: QPainter, t: Theme) -> None:
    card(p, QRectF(0, 0, 760, 200), t)
    text(p, QRectF(30, 16, 220, 20), "In the Labels panel", t.text_muted, 8.5, True, Qt.AlignLeft | Qt.AlignVCenter)
    for k, (color, name) in enumerate([(RED, "Tumor | Brain"), (YELLOW, "Edema | Brain"), (CUSTOM, "My new label")]):
        chip(p, 30, 46 + k * 40, color, name, t, 190)
    arrow(p, 240, 105, 330, 105, t.accent, "saved as", t)
    p.setPen(QPen(QColor(t.border_strong), 1))
    p.setBrush(QColor(t.raised))
    p.drawRoundedRect(QRectF(345, 34, 390, 140), 10, 10)
    text(p, QRectF(360, 42, 360, 20), "Inside IMG_0133_seg.dcm (DICOM Segmentation)", t.text_muted, 8.5, True,
         Qt.AlignLeft | Qt.AlignVCenter)
    for k, (color, name) in enumerate([(RED, "Segment 1:  Tumor & Brain Boundary"),
                                       (YELLOW, "Segment 2:  Edema & Brain Boundary"),
                                       (CUSTOM, "Segment 3:  My New Label (any name you choose)")]):
        y = 72 + k * 32
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(*color))
        p.drawRoundedRect(QRectF(362, y + 6, 12, 12), 3, 3)
        text(p, QRectF(384, y, 340, 24), name, t.text, 8.8, False, Qt.AlignLeft | Qt.AlignVCenter)


def _three_step(p: QPainter, t: Theme, frames, captions, arrows) -> None:
    card(p, QRectF(0, 0, 760, 230), t)
    xs = [30, 300, 570]
    for k, (x, frame) in enumerate(zip(xs, frames)):
        mri_slice(p, QRectF(x, 22, 160, 160), **frame)
        number(p, QPointF(x + 12, 34), k + 1, t.accent)
        text(p, QRectF(x - 30, 188, 220, 36), captions[k], t.text, 8.6, False, Qt.AlignHCenter | Qt.AlignTop)
    for k, label in enumerate(arrows):
        arrow(p, xs[k] + 172, 102, xs[k + 1] - 12, 102, t.accent, label, t)


def _sam(p: QPainter, t: Theme) -> None:
    _three_step(p, t, [dict(sam_box=True), dict(sam_box=True, filled=RED), dict(outline=RED)],
                ["Left-drag a box around the lesion", "Press Enter (or right-click ▸ Detect)",
                 "The outline appears in your label's color"],
                ["SAM", "done"])


def _autodetect(p: QPainter, t: Theme) -> None:
    _three_step(p, t, [dict(), dict(detect_box=True), dict(detect_box=True, outline=RED)],
                ["Click Auto Detect on any slice", "YOLO finds where the lesion is (green box)",
                 "SAM draws the exact outline"],
                ["YOLO", "SAM"])


def _saved(p: QPainter, t: Theme) -> None:
    card(p, QRectF(0, 0, 760, 220), t)
    mri_slice(p, QRectF(40, 20, 150, 150), filled=RED)
    text(p, QRectF(10, 176, 210, 36), "What you segment: a filled region", t.text, 8.8, True)
    arrow(p, 205, 95, 285, 95, t.accent, "Save", t)
    mri_slice(p, QRectF(300, 20, 150, 150), outline=RED)
    text(p, QRectF(270, 176, 210, 36), "What is saved: its outline, in the label's color", t.text, 8.8, True)
    p.setPen(QPen(QColor(t.border), 1, Qt.DashLine))
    p.drawLine(QPointF(500, 24), QPointF(500, 196))
    mri_slice(p, QRectF(560, 20, 150, 150), lesion=False)
    text(p, QRectF(515, 176, 240, 36), "Blank mask — saved automatically for every slice you leave empty",
         t.text, 8.8, True)


def _timeline(p: QPainter, t: Theme) -> None:
    card(p, QRectF(0, 0, 760, 165), t)
    left, right, y = 40, 720, 52
    count = 40
    x_of = lambda i: left + (right - left) * i / (count - 1)   # noqa: E731
    band = QColor("#a78bfa")                                    # the auto-detect range
    band.setAlpha(110)
    p.setPen(Qt.NoPen)
    p.setBrush(band)
    p.drawRoundedRect(QRectF(x_of(24) - 3, y - 7, x_of(27) - x_of(24) + 6, 14), 4, 4)
    p.setBrush(QColor(t.border_strong))
    p.drawRoundedRect(QRectF(left, y - 3, right - left, 6), 3, 3)
    p.setBrush(QColor(t.text_muted))                            # blank masks: small dots
    for i in list(range(0, 15)) + list(range(24, count)):
        p.drawEllipse(QPointF(x_of(i), y + 11), 1.9, 1.9)
    for indices, token in (((18, 19, 20), "text_faint"), ((15, 16, 17, 21), "success"), ((22, 23), "warning")):
        p.setPen(QPen(QColor(getattr(t, token)), 2))
        for i in indices:
            p.drawLine(QPointF(x_of(i), y + 8), QPointF(x_of(i), y + 14))
    p.setPen(QPen(QColor("#ffffff"), 2))
    p.setBrush(QColor(t.accent))
    p.drawEllipse(QPointF(x_of(21), y), 8, 8)
    legend = [("#a78bfa", "Auto-detect range", "band"), (t.text_muted, "Blank mask (no lesion)", "dot"),
              (t.text_faint, "Annotation file on disk", "tick"), (t.success, "Saved this session", "tick"),
              (t.warning, "Unsaved changes", "tick")]
    for k, (color, name, kind) in enumerate(legend):
        x = 40 + (k % 3) * 240
        yy = 92 + (k // 3) * 30
        p.setPen(Qt.NoPen)
        c = QColor(color)
        if kind == "band":
            c.setAlpha(150)
            p.setBrush(c)
            p.drawRoundedRect(QRectF(x, yy + 5, 22, 10), 3, 3)
        elif kind == "dot":
            p.setBrush(c)
            p.drawEllipse(QPointF(x + 11, yy + 10), 3, 3)
        else:
            p.setPen(QPen(c, 3))
            p.drawLine(QPointF(x + 11, yy + 3), QPointF(x + 11, yy + 17))
        text(p, QRectF(x + 30, yy, 200, 20), name, t.text, 8.6, False, Qt.AlignLeft | Qt.AlignVCenter)


def _volume(p: QPainter, t: Theme) -> None:
    card(p, QRectF(0, 0, 760, 220), t)
    for k in range(4):
        r = QRectF(40 + k * 16, 22 + k * 12, 120, 120)
        mri_slice(p, r, filled=RED if k in (1, 2, 3) else None, faint=k < 3)
    text(p, QRectF(10, 186, 270, 24), "Filled outlines on annotated slices", t.text, 8.8, True)
    arrow(p, 262, 105, 320, 105, t.accent)
    p.setPen(QPen(QColor(t.border_strong), 1))
    p.setBrush(QColor(t.raised))
    p.drawRoundedRect(QRectF(335, 40, 400, 132), 10, 10)
    text(p, QRectF(350, 50, 370, 22), "Lesion volume (cm³)", t.text, 10.5, True, Qt.AlignLeft | Qt.AlignVCenter)
    text(p, QRectF(350, 78, 370, 50), "= lesion pixels  ×  pixel width  ×  pixel height\n    ×  slice thickness  ÷  1000",
         t.accent, 9.2, True, Qt.AlignLeft | Qt.AlignVCenter)
    text(p, QRectF(350, 128, 370, 36), "Spacing and thickness come from the DICOM header (mm). With several "
         "annotated series, their average is shown too.", t.text_muted, 8.2, False, Qt.AlignLeft | Qt.AlignTop)


def _planes(p: QPainter, t: Theme) -> None:
    card(p, QRectF(0, 0, 760, 225), t)
    names = [("Axial", "looking from the feet up", 0), ("Coronal", "looking from the front", 1),
             ("Sagittal", "looking from the side", 2)]
    for k, (name, note, kind) in enumerate(names):
        x = 40 + k * 245
        r = QRectF(x, 22, 150, 150)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(10, 11, 14))
        p.drawRoundedRect(r, 8, 8)
        p.setBrush(QColor(140, 142, 148))
        if kind == 0:
            p.drawEllipse(r.adjusted(28, 18, -28, -18))
        elif kind == 1:
            p.drawEllipse(r.adjusted(24, 14, -24, -40))
            p.drawRect(r.adjusted(64, 104, -64, -12))
        else:
            p.drawEllipse(r.adjusted(16, 14, -32, -40))
            p.drawRect(r.adjusted(62, 104, -60, -12))
        p.setBrush(QColor(250, 250, 250))
        p.drawEllipse(QPointF(r.center().x() + 18, r.center().y() - 22), 9, 9)
        pen = QPen(QColor("#00ff66"), 1)
        p.setPen(pen)
        p.drawLine(QPointF(r.left() + 6, r.center().y() - 22), QPointF(r.right() - 6, r.center().y() - 22))
        p.drawLine(QPointF(r.center().x() + 18, r.top() + 6), QPointF(r.center().x() + 18, r.bottom() - 6))
        text(p, QRectF(x - 20, 178, 190, 20), name, t.text, 9.5, True)
        text(p, QRectF(x - 20, 198, 190, 18), note, t.text_muted, 8.2)
    text(p, QRectF(0, 4, 760, 16), "Green crosshair = the same point in all three views", t.text_muted, 8)


def _overlap(p: QPainter, t: Theme) -> None:
    card(p, QRectF(0, 0, 760, 200), t)
    a, b = QRectF(60, 30, 140, 140), QRectF(140, 30, 140, 140)
    for rect, color in ((a, LESION_RED), (b, QColor("#4c8dff"))):
        fill = QColor(color)
        fill.setAlpha(60)
        p.setPen(QPen(color, 2))
        p.setBrush(fill)
        p.drawEllipse(rect)
    text(p, QRectF(60, 172, 120, 20), "A: actual", "#ff4848", 8.8, True)
    text(p, QRectF(170, 172, 120, 20), "B: transferred", "#4c8dff", 8.8, True)
    lines = [("Dice = 2 × overlap ÷ (A + B)", t.accent), ("IoU = overlap ÷ (A and B together)", t.accent),
             ("1.0 = perfect match   ·   0 = no overlap at all", t.text_muted)]
    for k, (line, color) in enumerate(lines):
        text(p, QRectF(330, 52 + k * 36, 410, 28), line, color, 10 if k < 2 else 8.8, k < 2,
             Qt.AlignLeft | Qt.AlignVCenter)


DIAGRAMS: Dict[str, Tuple[Callable[[QPainter, Theme], None], Tuple[int, int]]] = {
    "workflow": (_workflow, (760, 190)),
    "series": (_series, (760, 230)),
    "labels": (_labels, (760, 200)),
    "sam": (_sam, (760, 230)),
    "autodetect": (_autodetect, (760, 230)),
    "saved": (_saved, (760, 220)),
    "timeline": (_timeline, (760, 165)),
    "volume": (_volume, (760, 220)),
    "planes": (_planes, (760, 225)),
    "overlap": (_overlap, (760, 200)),
}
DESIGN_SIZES = {name: size for name, (_fn, size) in DIAGRAMS.items()}


def render(name: str, width: int, theme: Theme, ratio: float = 1.0) -> QPixmap:
    draw, (dw, dh) = DIAGRAMS[name]
    scale = width / dw
    out = QPixmap(int(round(width * ratio)), int(round(dh * scale * ratio)))
    out.fill(Qt.transparent)
    painter = QPainter(out)
    painter.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing | QPainter.SmoothPixmapTransform)
    painter.scale(scale * ratio, scale * ratio)
    draw(painter, theme)
    painter.end()
    out.setDevicePixelRatio(ratio)
    return out
