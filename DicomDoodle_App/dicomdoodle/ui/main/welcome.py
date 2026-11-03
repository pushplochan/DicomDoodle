"""Start page shown until a DICOM folder is opened."""
from __future__ import annotations

import os
from typing import List

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ... import APP_NAME, APP_TAGLINE, resources
from ..icons import bind_icon
from ..qt_utils import label, make_button, set_role


class WelcomePage(QWidget):
    openRequested = pyqtSignal()
    recentRequested = pyqtSignal(str)
    toolRequested = pyqtSignal(str)
    guideRequested = pyqtSignal(str)          # User Guide chapter id

    TOOLS = [("overlay", "layers", "Annotated DICOM Viewer"), ("mpr", "grid", "Multiplanar && 3D Viewer"),
             ("converter", "scissors", "Multi-frame Converter")]
    RECENT_TEXT_WIDTH = 560          # pixels

    def __init__(self, parent=None):
        super().__init__(parent)
        card = QFrame(self)
        card.setObjectName("Card")
        card.setMaximumWidth(760)
        card.setMinimumWidth(560)           # the page scrolls in very small windows

        logo = QLabel()
        logo_path = resources.find_resource(resources.SPLASH_IMAGE) or resources.find_resource(resources.LOGO_IMAGE)
        if logo_path:
            logo.setPixmap(QPixmap(logo_path).scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        logo.setAlignment(Qt.AlignCenter)

        title = label(APP_NAME, "Title")
        title.setAlignment(Qt.AlignCenter)
        tagline = label(APP_TAGLINE, "Subtitle")
        tagline.setAlignment(Qt.AlignCenter)

        open_btn = self.open_btn = make_button("  Open DICOM folder…", "folder-open", role="primary")
        open_btn.setMinimumHeight(40)
        open_btn.clicked.connect(self.openRequested.emit)
        hint = label("or drag a folder of .dcm slices onto this window", "Faint")
        hint.setAlignment(Qt.AlignCenter)

        self.guide_btn = make_button("Read the Quick Start", "book",
                                     tooltip="Your first annotation in five minutes (User Guide, F1)")
        self.guide_btn.clicked.connect(lambda: self.guideRequested.emit("quick-start"))
        learn = QHBoxLayout()
        learn.addStretch(1)
        learn.addWidget(self.guide_btn, 2)
        learn.addStretch(1)

        self.recent_box = QVBoxLayout()
        self.recent_box.setSpacing(2)
        self.recent_title = label("RECENT FOLDERS", "SectionTitle")

        tools = self.tools_grid = QGridLayout()
        tools.setSpacing(8)
        for i, (key, icon_name, text) in enumerate(self.TOOLS):
            button = QPushButton(text)
            bind_icon(button, icon_name)
            set_role(button, "ghost")
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(lambda _c=False, k=key: self.toolRequested.emit(k))
            tools.addWidget(button, i // 2, i % 2)

        inner = QVBoxLayout(card)
        inner.setContentsMargins(40, 32, 40, 32)
        inner.setSpacing(12)
        inner.addWidget(logo)
        inner.addWidget(title)
        inner.addWidget(tagline)
        inner.addSpacing(14)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(open_btn)
        row.addStretch(1)
        inner.addLayout(row)
        inner.addWidget(hint)
        inner.addSpacing(10)
        inner.addWidget(label(f"NEW TO {APP_NAME.upper()}?", "SectionTitle"))
        inner.addWidget(label("Open one of your MRI series and follow the five-minute Quick Start.", "Muted"))
        inner.addLayout(learn)
        inner.addSpacing(10)
        inner.addWidget(self.recent_title)
        inner.addLayout(self.recent_box)
        inner.addSpacing(10)
        inner.addWidget(label("TOOLS", "SectionTitle"))
        inner.addLayout(tools)

        outer = QVBoxLayout(self)
        outer.addStretch(1)
        center = QHBoxLayout()
        center.addStretch(1)
        center.addWidget(card, 4)
        center.addStretch(1)
        outer.addLayout(center)
        outer.addStretch(1)

    def set_recent(self, folders: List[str]) -> None:
        while self.recent_box.count():
            item = self.recent_box.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        existing = [f for f in folders if os.path.isdir(f)][:5]
        self.recent_title.setVisible(bool(existing))
        for folder in existing:
            text = f"{os.path.basename(folder.rstrip(os.sep)) or folder}   —   {folder}"
            button = make_button(text, "folder", role="link", tooltip=folder)
            # a long path is shortened in the middle, so it never makes the window wider
            button.setText(button.fontMetrics().elidedText(text, Qt.ElideMiddle, self.RECENT_TEXT_WIDTH))
            button.clicked.connect(lambda _c=False, f=folder: self.recentRequested.emit(f))
            self.recent_box.addWidget(button)
