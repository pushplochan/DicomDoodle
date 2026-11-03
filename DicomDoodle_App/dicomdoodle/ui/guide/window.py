"""
The User Guide window.

    toolbar   ‹ Back  › Forward | Start  Quick Start  Shortcuts | A−  A+      Open a DICOM folder…
    sidebar   search box + chapters grouped by part
    page      the chapter (GuideBrowser), with screenshots, diagrams and links

Links inside chapters: ``chapter:<id>`` opens a chapter, ``action:<name>``
asks the main window to do something (open a folder, a tool ...),
``zoom:<shot>`` shows a screenshot full size and http(s) opens the browser.
"""
from __future__ import annotations

import html
import os
import re
from typing import Dict, List, Optional, Tuple

from PyQt5.QtCore import QRectF, QSize, Qt, QTimer, QUrl, pyqtSignal
from PyQt5.QtGui import (QColor, QDesktopServices, QFont, QImageReader, QKeySequence, QPainter, QPainterPath,
                         QPen, QPixmap, QTextDocument)
from PyQt5.QtWidgets import (QAction, QCheckBox, QDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
                             QScrollArea, QShortcut, QSizePolicy, QSplitter, QTextBrowser, QToolBar, QTreeWidget,
                             QTreeWidgetItem, QVBoxLayout, QWidget)

from ... import APP_NAME, APP_VERSION, resources
from ...help import contents
from ..icons import bind_icon, icon, pixmap
from ..qt_utils import label, make_button
from ..theme import current_theme, rgba, theme_manager
from . import diagrams
from .drawing import marker_pixmap, swatch_pixmap
from .markup import Renderer, plain_text, split_resource, stylesheet

COLUMN = 880                     # widest text column (px)
DOC_MARGIN = 30                  # space around the page inside the view
FONT_SIZES = (9.0, 10.0, 10.5, 11.5, 12.5, 14.0)
ID_ROLE = Qt.UserRole + 1


# --------------------------------------------------------------------------- #
# Screenshots
# --------------------------------------------------------------------------- #

def shot_path(name: str) -> str:
    return os.path.join(resources.GUIDE_DIR, f"{name}.png")


def shot_size(name: str) -> Tuple[int, int]:
    size = QImageReader(shot_path(name)).size()
    return (size.width(), size.height()) if size.isValid() else (760, 300)


def _framed(image: QPixmap, width: int, ratio: float, border: str) -> QPixmap:
    """The screenshot at ``width`` logical px, with rounded corners and a hairline border."""
    target_w = max(1, int(round(width * ratio)))
    if image.width() > target_w:
        image = image.scaledToWidth(target_w, Qt.SmoothTransformation)
    scale = image.width() / width                     # device px per logical px
    out = QPixmap(image.size())
    out.fill(Qt.transparent)
    p = QPainter(out)
    p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
    path = QPainterPath()
    rect = QRectF(0.5, 0.5, image.width() - 1, image.height() - 1)
    path.addRoundedRect(rect, 8 * scale, 8 * scale)
    p.setClipPath(path)
    p.drawPixmap(0, 0, image)
    p.setClipping(False)
    p.setPen(QPen(QColor(border), max(1.0, scale)))
    p.drawPath(path)
    p.end()
    out.setDevicePixelRatio(scale)
    return out


def _placeholder(width: int, height: int, ratio: float) -> QPixmap:
    t = current_theme()
    out = QPixmap(int(width * ratio), int(height * ratio))
    out.fill(QColor(t.surface_alt))
    p = QPainter(out)
    p.scale(ratio, ratio)
    p.setPen(QColor(t.text_faint))
    p.drawText(QRectF(0, 0, width, height), Qt.AlignCenter,
               "Screenshot not found — run scripts/make_guide_images.py")
    p.end()
    out.setDevicePixelRatio(ratio)
    return out


# --------------------------------------------------------------------------- #
# The page view
# --------------------------------------------------------------------------- #

class GuideBrowser(QTextBrowser):
    """QTextBrowser that draws the guide's images itself (``loadResource``)
    and keeps the text in a centred column of at most COLUMN px."""

    columnChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("GuideBrowser")
        self.setOpenLinks(False)
        self.setFrameShape(QFrame.NoFrame)
        self.column = COLUMN
        self._images: Dict[str, QPixmap] = {}

    # ------------------------------------------------------------ layout
    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        available = self.width() - self.verticalScrollBar().sizeHint().width()
        side = max(0, (available - COLUMN - 2 * DOC_MARGIN) // 2)
        self.setViewportMargins(side, 0, side, 0)
        column = max(280, available - 2 * side - 2 * DOC_MARGIN)
        if column != self.column:
            self.column = column
            self.columnChanged.emit()

    def image_width(self) -> int:
        return self.column - 6

    # --------------------------------------------------------- resources
    @staticmethod
    def size_of(source: str) -> Tuple[int, int]:
        kind, _, name = source.partition(":")
        if kind == "shot":
            return shot_size(name)
        return diagrams.DESIGN_SIZES.get(name, (760, 200))

    def clear_images(self) -> None:
        self._images.clear()

    def loadResource(self, kind: int, url: QUrl):
        name = url.toString()
        if kind != QTextDocument.ImageResource:
            return super().loadResource(kind, url)
        image = self._images.get(name)
        if image is None:
            image = self._make_image(name)
            if image is None:
                return super().loadResource(kind, url)
            self._images[name] = image
        return image

    def _make_image(self, url: str) -> Optional[QPixmap]:
        base, _theme = split_resource(url)
        scheme, _, rest = base.partition(":")
        t = current_theme()
        ratio = max(1.0, self.devicePixelRatioF())
        if scheme == "shot":
            name, _, width = rest.partition("@")
            width = int(width or COLUMN)
            source = QPixmap(shot_path(name))
            if source.isNull():
                w, h = shot_size(name)
                return _placeholder(width, int(h * width / max(w, 1)), ratio)
            return _framed(source, width, ratio, t.border_strong)
        if scheme == "diagram":
            name, _, width = rest.partition("@")
            if name not in diagrams.DIAGRAMS:
                return None
            return diagrams.render(name, int(width or COLUMN), t, ratio)
        if scheme == "icon":
            name, _, token = rest.partition(":")
            return pixmap(name, getattr(t, token, t.text) if token else t.text, 16)
        if scheme == "step":
            return marker_pixmap(rest, 24, ratio, t.accent, ring=False)
        if scheme == "badge":
            return marker_pixmap(rest, 20, ratio)
        if scheme == "swatch":
            return swatch_pixmap(rest, 16, ratio, rgba(t.text, 0.35))
        return None


class ScreenshotDialog(QDialog):
    """A guide screenshot at full size (fit to the window, or 100 %)."""

    def __init__(self, name: str, caption: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(caption or "Screenshot")
        self.original = QPixmap(shot_path(name))
        self.image = QLabel()
        self.image.setAlignment(Qt.AlignCenter)
        self.area = QScrollArea()
        self.area.setWidget(self.image)
        self.area.setWidgetResizable(True)
        self.area.setAlignment(Qt.AlignCenter)
        self.fit = QCheckBox("Fit to window")
        self.fit.setChecked(True)
        self.fit.toggled.connect(self._show)
        close = make_button("Close", role="primary")
        close.clicked.connect(self.accept)
        bottom = QHBoxLayout()
        bottom.addWidget(label(caption, "Muted", wrap=True), 1)
        bottom.addWidget(self.fit)
        bottom.addWidget(close)
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.addWidget(self.area, 1)
        root.addLayout(bottom)
        screen = self.screen().availableGeometry() if self.screen() else None
        w, h = self.original.width() + 40, self.original.height() + 90
        if screen is not None:
            w, h = min(w, int(screen.width() * 0.92)), min(h, int(screen.height() * 0.9))
        self.resize(max(w, 420), max(h, 300))

    def _show(self, *_args) -> None:
        if self.original.isNull():
            self.image.setText("Screenshot not found.")
            return
        viewport = self.area.viewport().size()
        if self.fit.isChecked() and (self.original.width() > viewport.width()
                                     or self.original.height() > viewport.height()):
            self.image.setPixmap(self.original.scaled(viewport, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.image.setPixmap(self.original)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._show()


# --------------------------------------------------------------------------- #
# The window
# --------------------------------------------------------------------------- #

class GuideWindow(QMainWindow):
    actionRequested = pyqtSignal(str)      # "open", "overlay", "preferences" ...

    def __init__(self, settings=None, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle(f"{APP_NAME} User Guide")
        self.setAttribute(Qt.WA_QuitOnClose, False)       # never keeps the app running on its own
        self.resize(1240, 860)
        self.history: List[str] = []
        self.position = -1
        self.current = contents.HOME
        self.query = ""
        self.matches = []
        size = float(settings.get("guide_font_size", 10.5)) if settings is not None else 10.5
        self.font_index = min(range(len(FONT_SIZES)), key=lambda i: abs(FONT_SIZES[i] - size))
        self.index = {c.id: " ".join((c.title, c.summary, c.keywords, plain_text(c.body))).lower()
                      for c in contents.all_chapters()}

        self.browser = GuideBrowser()
        self.browser.anchorClicked.connect(self._link_clicked)
        self.browser.columnChanged.connect(self._render_soon)

        self._build_sidebar()
        splitter = QSplitter()
        splitter.addWidget(self.sidebar)
        splitter.addWidget(self.browser)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([280, 960])
        splitter.setChildrenCollapsible(False)
        self.setCentralWidget(splitter)
        self._build_toolbar()

        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(120)
        self._render_timer.timeout.connect(lambda: self.render(keep_position=True))
        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(180)
        self._search_timer.timeout.connect(self._apply_search)

        QShortcut(QKeySequence.Find, self, activated=self._focus_search)
        QShortcut(QKeySequence("Ctrl+="), self, activated=lambda: self.change_font(1))
        QShortcut(QKeySequence("Ctrl++"), self, activated=lambda: self.change_font(1))
        QShortcut(QKeySequence("Ctrl+-"), self, activated=lambda: self.change_font(-1))
        theme_manager().themeChanged.connect(self._theme_changed)   # bound: disconnects on delete
        self.show_chapter(contents.HOME)

    # ---------------------------------------------------------------- layout
    def _build_sidebar(self) -> None:
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search the guide   (Ctrl+F)")
        self.search.setClearButtonEnabled(True)
        self.search_icon = self.search.addAction(icon("search"), QLineEdit.LeadingPosition)
        self.search.textChanged.connect(lambda _text: self._search_timer.start())
        self.search.returnPressed.connect(self._open_first_match)

        self.nav = QTreeWidget()
        self.nav.setObjectName("GuideNav")
        self.nav.setHeaderHidden(True)
        self.nav.setRootIsDecorated(False)
        self.nav.setItemsExpandable(False)
        self.nav.setIndentation(10)
        self.nav.setIconSize(QSize(16, 16))
        self.nav.setFocusPolicy(Qt.NoFocus)
        self.groups: List[QTreeWidgetItem] = []
        self.items: Dict[str, QTreeWidgetItem] = {}
        for group, chapters in contents.GROUPS:
            head = QTreeWidgetItem([group.upper()])
            head.setFlags(Qt.ItemIsEnabled)
            head.setFirstColumnSpanned(True)
            head.setSizeHint(0, QSize(0, 34))
            font = QFont(self.nav.font())
            font.setBold(True)
            font.setPointSizeF(max(7.5, font.pointSizeF() - 1.5))
            head.setFont(0, font)
            self.nav.addTopLevelItem(head)
            self.groups.append(head)
            for chapter in chapters:
                item = QTreeWidgetItem([chapter.nav_title])
                item.setData(0, ID_ROLE, chapter.id)
                summary = " ".join(plain_text(chapter.summary).split())
                item.setToolTip(0, f"<b>{html.escape(chapter.title)}</b><br>{html.escape(summary)}")
                item.setSizeHint(0, QSize(0, 30))
                head.addChild(item)
                self.items[chapter.id] = item
        self.nav.expandAll()
        self.nav.itemClicked.connect(self._nav_clicked)
        self._style_nav()

        footer = label(f"{APP_NAME} {APP_VERSION}  ·  User Guide", "Faint")
        footer.setAlignment(Qt.AlignCenter)
        self.sidebar = QWidget()
        self.sidebar.setObjectName("GuideSidebar")
        layout = QVBoxLayout(self.sidebar)
        layout.setContentsMargins(12, 12, 8, 10)
        layout.setSpacing(10)
        layout.addWidget(self.search)
        layout.addWidget(self.nav, 1)
        layout.addWidget(footer)
        self.sidebar.setMinimumWidth(240)

    def _build_toolbar(self) -> None:
        bar = QToolBar("Guide", self)
        bar.setMovable(False)
        bar.setIconSize(QSize(18, 18))
        bar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(bar)

        def add(text: str, icon_name: Optional[str], slot, tip: str, keys=None) -> QAction:
            action = QAction(text, self)
            if icon_name:
                bind_icon(action, icon_name)
            if keys:
                action.setShortcuts(keys)
            action.setToolTip(tip)
            action.triggered.connect(lambda _c=False: slot())
            bar.addAction(action)
            return action

        self.back_action = add("Back", "chevron-left", self.back, "Back (Alt+←)", QKeySequence.keyBindings(QKeySequence.Back))
        self.forward_action = add("Forward", "chevron-right", self.forward, "Forward (Alt+→)",
                                  QKeySequence.keyBindings(QKeySequence.Forward))
        bar.addSeparator()
        add("Start", "book", lambda: self.show_chapter(contents.HOME), "The first page of the guide")
        add("Quick Start", "zap", lambda: self.show_chapter("quick-start"), "Your first annotation in 5 minutes")
        add("Shortcuts", "keyboard", lambda: self.show_chapter("shortcuts"), "All keyboard and mouse shortcuts")
        bar.addSeparator()
        self.smaller_action = add("A−", None, lambda: self.change_font(-1), "Smaller text (Ctrl+−)")
        self.larger_action = add("A+", None, lambda: self.change_font(1), "Larger text (Ctrl+=)")
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        bar.addWidget(spacer)
        add("Open a DICOM folder…", "folder-open", lambda: self.actionRequested.emit("open"),
            "Open one of your series in the main window and follow along")

    def _style_nav(self) -> None:
        t = current_theme()
        for head in self.groups:
            head.setForeground(0, QColor(t.text_faint))
        for chapter in contents.all_chapters():
            self.items[chapter.id].setIcon(0, icon(chapter.icon, t.text_muted))

    # ------------------------------------------------------------ navigation
    def show_chapter(self, chapter_id: str, record: bool = True) -> None:
        chapter = contents.find(chapter_id) or contents.find(contents.HOME)
        if self.query:
            self.search.blockSignals(True)
            self.search.clear()
            self.search.blockSignals(False)
            self.query, self.matches = "", []
            self._filter_nav(None)
        self.current = chapter.id
        if record:
            del self.history[self.position + 1:]
            if not self.history or self.history[-1] != chapter.id:
                self.history.append(chapter.id)
            self.position = len(self.history) - 1
        self._select_nav(chapter.id)
        self.render()
        self._update_actions()

    def back(self) -> None:
        if self.query:
            self.show_chapter(self.current, record=False)
        elif self.position > 0:
            self.position -= 1
            self.show_chapter(self.history[self.position], record=False)

    def forward(self) -> None:
        if self.position + 1 < len(self.history):
            self.position += 1
            self.show_chapter(self.history[self.position], record=False)

    def change_font(self, step: int) -> None:
        self.font_index = max(0, min(len(FONT_SIZES) - 1, self.font_index + step))
        if self.settings is not None:
            self.settings.set("guide_font_size", FONT_SIZES[self.font_index])
        self.render(keep_position=True)
        self._update_actions()

    def _update_actions(self) -> None:
        self.back_action.setEnabled(self.position > 0 or bool(self.query))
        self.forward_action.setEnabled(self.position + 1 < len(self.history))
        self.smaller_action.setEnabled(self.font_index > 0)
        self.larger_action.setEnabled(self.font_index < len(FONT_SIZES) - 1)

    def _select_nav(self, chapter_id: str) -> None:
        item = self.items.get(chapter_id)
        self.nav.setCurrentItem(item)
        if item is not None:
            self.nav.scrollToItem(item)

    def _nav_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        chapter_id = item.data(0, ID_ROLE)
        if chapter_id:
            self.show_chapter(chapter_id)

    def _link_clicked(self, url: QUrl) -> None:
        scheme, _, target = url.toString().partition(":")
        if scheme == "chapter":
            self.show_chapter(target)
        elif scheme == "action":
            self.actionRequested.emit(target)
        elif scheme == "zoom":
            ScreenshotDialog(target, self._caption_of(target), self).exec_()
        elif scheme in ("http", "https", "mailto"):
            QDesktopServices.openUrl(url)

    def _caption_of(self, shot: str) -> str:
        chapter = contents.find(self.current)
        match = re.search(r"!\[(.*?)\]\(shot:" + re.escape(shot) + r"\)", chapter.body if chapter else "")
        return match.group(1) if match else ""

    # ------------------------------------------------------------- rendering
    def render(self, keep_position: bool = False) -> None:
        t = current_theme()
        document = self.browser.document()
        document.setDefaultStyleSheet(stylesheet(t))
        document.setDocumentMargin(DOC_MARGIN)
        font = QFont(self.browser.font())
        font.setPointSizeF(FONT_SIZES[self.font_index])
        document.setDefaultFont(font)
        renderer = Renderer(t, self.browser.image_width(), self.browser.size_of)
        if self.query:
            page = self._search_page(renderer)
        else:
            chapter = contents.find(self.current)
            previous, following = contents.neighbours(chapter.id)
            page = renderer.chapter(contents.group_of(chapter.id), chapter.title, chapter.summary, chapter.body,
                                    (previous.id, previous.title) if previous else None,
                                    (following.id, following.title) if following else None)
        bar = self.browser.verticalScrollBar()
        fraction = bar.value() / bar.maximum() if keep_position and bar.maximum() else 0.0
        self.browser.setHtml(f"<html><body>{page}</body></html>")
        if keep_position:
            QTimer.singleShot(0, lambda: bar.setValue(int(round(fraction * bar.maximum()))))
        else:
            bar.setValue(0)

    def _render_soon(self) -> None:
        self._render_timer.start()

    def _theme_changed(self, *_args) -> None:
        self.browser.clear_images()
        self._style_nav()
        self.render(keep_position=True)

    # ---------------------------------------------------------------- search
    def _focus_search(self) -> None:
        self.search.setFocus()
        self.search.selectAll()

    def _apply_search(self) -> None:
        query = self.search.text().strip()
        words = query.lower().split()
        if not words:
            if self.query:
                self.query, self.matches = "", []
                self._filter_nav(None)
                self._select_nav(self.current)
                self.render()
                self._update_actions()
            return
        self.query = query
        found = [c for c in contents.all_chapters() if all(w in self.index[c.id] for w in words)]

        def relevance(chapter) -> int:       # words in the title count most, then summary and keywords
            title, about = chapter.title.lower(), f"{chapter.summary} {chapter.keywords}".lower()
            return sum(3 if w in title else 2 if w in about else 1 for w in words)
        self.matches = sorted(found, key=relevance, reverse=True)       # stable: ties keep reading order
        self._filter_nav({c.id for c in self.matches})
        self.nav.setCurrentItem(None)
        self.render()
        self._update_actions()

    def _filter_nav(self, visible: Optional[set]) -> None:
        for head in self.groups:
            shown = 0
            for k in range(head.childCount()):
                child = head.child(k)
                hide = visible is not None and child.data(0, ID_ROLE) not in visible
                child.setHidden(hide)
                shown += not hide
            head.setHidden(shown == 0)

    def _open_first_match(self) -> None:
        if self.query and self.matches:
            self.show_chapter(self.matches[0].id)

    def _search_page(self, renderer: Renderer) -> str:
        t = current_theme()
        words = self.query.lower().split()
        out = ['<p class="eyebrow">SEARCH</p>', f"<h1>Results for “{html.escape(self.query)}”</h1>"]
        if not self.matches:
            out.append('<p class="muted" style="font-size:11.5pt;">No chapter contains all of these words. '
                       "Try fewer or different words, or browse the chapters on the left.</p>")
            return "".join(out)
        count = len(self.matches)
        out.append(f'<p class="muted" style="font-size:11.5pt; margin-bottom:14px;">'
                   f'{count} chapter{"s" if count != 1 else ""} · press Enter to open the first one</p>')
        for chapter in self.matches:
            snippet = self._snippet(plain_text(chapter.body), words)
            out.append(
                f'<table width="100%" cellspacing="0" cellpadding="12" style="margin-bottom:8px;"><tr>'
                f'<td style="background-color:{t.surface};">'
                f'<a href="chapter:{chapter.id}" style="font-size:12pt;">{html.escape(chapter.title)}</a>'
                f'&nbsp;&nbsp;<span class="muted">{html.escape(contents.group_of(chapter.id))}</span><br>'
                f'<span class="muted">{renderer.inline(chapter.summary)}</span>'
                + (f'<br><span style="color:{t.text};">{snippet}</span>' if snippet else "")
                + "</td></tr></table>")
        return "".join(out)

    @staticmethod
    def _snippet(text: str, words: List[str]) -> str:
        text = " ".join(text.split())
        low = text.lower()
        positions = [low.find(w) for w in words if low.find(w) >= 0]
        if not positions:
            return ""
        start = max(0, min(positions) - 70)
        end = min(len(text), min(positions) + 150)
        excerpt = ("… " if start else "") + text[start:end].strip() + (" …" if end < len(text) else "")
        escaped = html.escape(excerpt)
        accent = rgba(current_theme().accent, 0.28)
        pattern = re.compile("|".join(re.escape(html.escape(w)) for w in words), re.IGNORECASE)
        return pattern.sub(lambda m: f'<span style="background-color:{accent};">{m.group(0)}</span>', escaped)
