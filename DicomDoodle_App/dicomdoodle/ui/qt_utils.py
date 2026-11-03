"""
Small Qt helpers shared by many widgets: numpy <-> QImage conversion,
styled buttons, color swatches and consistent message boxes.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QColor, QIcon, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (QAbstractButton, QFrame, QLabel, QMessageBox, QPushButton,
                             QSizePolicy, QToolButton, QWidget)

from .icons import bind_icon


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #

def numpy_to_qimage(array: np.ndarray) -> QImage:
    """uint8 gray (H, W), RGB (H, W, 3) or RGBA (H, W, 4) -> QImage (owns its data)."""
    array = np.ascontiguousarray(array)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    h, w = array.shape[:2]
    if array.ndim == 2:
        image = QImage(array.data, w, h, w, QImage.Format_Grayscale8)
    elif array.shape[2] == 3:
        image = QImage(array.data, w, h, 3 * w, QImage.Format_RGB888)
    elif array.shape[2] == 4:
        image = QImage(array.data, w, h, 4 * w, QImage.Format_RGBA8888)
    else:
        raise ValueError(f"Unsupported image shape {array.shape}")
    return image.copy()


def numpy_to_pixmap(array: np.ndarray) -> QPixmap:
    return QPixmap.fromImage(numpy_to_qimage(array))


def qimage_to_numpy(image: QImage) -> np.ndarray:
    image = image.convertToFormat(QImage.Format_RGB888)
    w, h = image.width(), image.height()
    ptr = image.constBits()
    ptr.setsize(image.byteCount())
    arr = np.frombuffer(ptr, np.uint8).reshape(h, image.bytesPerLine())[:, : w * 3]
    return arr.reshape(h, w, 3).copy()


# --------------------------------------------------------------------------- #
# Colors
# --------------------------------------------------------------------------- #

def qcolor(rgb: Sequence[int], alpha: int = 255) -> QColor:
    return QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]), alpha)


def readable_text_color(rgb: Sequence[int]) -> QColor:
    """Black or white, whichever reads better on ``rgb``."""
    r, g, b = rgb[:3]
    return QColor(Qt.black) if (0.299 * r + 0.587 * g + 0.114 * b) > 150 else QColor(Qt.white)


def swatch_icon(rgb: Sequence[int], size: int = 14) -> QIcon:
    """Rounded colored square, e.g. for label menus and combo boxes."""
    pix = QPixmap(size * 2, size * 2)
    pix.fill(Qt.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(qcolor(rgb))
    painter.setPen(QPen(QColor(255, 255, 255, 90), 2))
    painter.drawRoundedRect(2, 2, size * 2 - 4, size * 2 - 4, 7, 7)
    painter.end()
    pix.setDevicePixelRatio(2.0)
    return QIcon(pix)


# --------------------------------------------------------------------------- #
# Widgets
# --------------------------------------------------------------------------- #

def set_role(button: QWidget, role: Optional[str]) -> None:
    """Style variant from the theme: 'primary', 'danger', 'ghost', 'link' or None."""
    button.setProperty("role", role or "")
    button.style().unpolish(button)
    button.style().polish(button)


def make_button(text: str = "", icon_name: Optional[str] = None, role: Optional[str] = None,
                tooltip: str = "", parent: Optional[QWidget] = None, icon_token: Optional[str] = None) -> QPushButton:
    button = QPushButton(text, parent)
    if icon_name:
        bind_icon(button, icon_name, icon_token or ("accent_text" if role == "primary" else None))
    if role:
        set_role(button, role)
    if tooltip:
        button.setToolTip(tooltip)
    button.setCursor(Qt.PointingHandCursor)
    return button


def make_tool_button(icon_name: str, tooltip: str, parent: Optional[QWidget] = None,
                     checkable: bool = False, size: int = 18) -> QToolButton:
    button = QToolButton(parent)
    bind_icon(button, icon_name)
    button.setIconSize(QSize(size, size))
    button.setToolTip(tooltip)
    button.setCheckable(checkable)
    button.setAutoRaise(True)
    button.setCursor(Qt.PointingHandCursor)
    return button


def label(text: str = "", object_name: Optional[str] = None, wrap: bool = False,
          parent: Optional[QWidget] = None) -> QLabel:
    widget = QLabel(text, parent)
    if object_name:
        widget.setObjectName(object_name)
    widget.setWordWrap(wrap)
    return widget


def divider(parent: Optional[QWidget] = None) -> QFrame:
    line = QFrame(parent)
    line.setObjectName("Divider")
    line.setFrameShape(QFrame.NoFrame)
    line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    return line


def set_badge(widget: QLabel, kind: Optional[str], text: str = "") -> None:
    """Pill-shaped status label: kind in success / warning / info / danger / neutral."""
    widget.setProperty("badge", kind or "")
    widget.setText(text)
    widget.setVisible(bool(kind and text))
    widget.style().unpolish(widget)
    widget.style().polish(widget)


# --------------------------------------------------------------------------- #
# Message boxes
# --------------------------------------------------------------------------- #

def show_error(parent: Optional[QWidget], title: str, message: str, details: str = "") -> None:
    box = QMessageBox(QMessageBox.Critical, title, message, QMessageBox.Ok, parent)
    if details:
        box.setDetailedText(details)
    box.exec_()


def show_info(parent: Optional[QWidget], title: str, message: str) -> None:
    QMessageBox.information(parent, title, message)


def ask_yes_no(parent: Optional[QWidget], title: str, message: str,
               yes_text: str = "Yes", no_text: str = "Cancel", default_yes: bool = True) -> bool:
    box = QMessageBox(QMessageBox.Question, title, message, QMessageBox.NoButton, parent)
    yes = box.addButton(yes_text, QMessageBox.AcceptRole)
    no = box.addButton(no_text, QMessageBox.RejectRole)
    box.setDefaultButton(yes if default_yes else no)
    box.exec_()
    return box.clickedButton() is yes


def ask_choice(parent: Optional[QWidget], title: str, message: str, choices: Sequence[str],
               cancel_text: str = "Cancel", informative: str = "") -> Optional[int]:
    """Show buttons for each choice; returns its index or None if cancelled."""
    box = QMessageBox(QMessageBox.Question, title, message, QMessageBox.NoButton, parent)
    if informative:
        box.setInformativeText(informative)
    buttons = [box.addButton(text, QMessageBox.AcceptRole) for text in choices]
    cancel = box.addButton(cancel_text, QMessageBox.RejectRole)
    box.setDefaultButton(buttons[0] if buttons else cancel)
    box.exec_()
    clicked: QAbstractButton = box.clickedButton()
    return buttons.index(clicked) if clicked in buttons else None
