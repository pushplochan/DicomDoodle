"""
Light and dark themes with live switching.

A theme is a set of named color *tokens* (window, surface, text, accent ...).
:meth:`ThemeManager.apply` turns the tokens into a QPalette plus one
application-wide style sheet, so every window — including tool windows
opened later — restyles instantly. Custom-painted widgets read colors with
``theme_color("accent")`` and repaint when ``themeChanged`` fires.

Image viewports stay dark in both themes (the radiology convention).
"""
from __future__ import annotations

import ctypes
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from string import Template
from typing import Optional

from PyQt5.QtCore import QEvent, QObject, pyqtSignal
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QApplication, QWidget


@dataclass(frozen=True)
class Theme:
    name: str
    is_dark: bool
    window: str          # main background
    surface: str         # panels, toolbars, docks
    surface_alt: str     # cards, lists
    raised: str          # buttons
    hover: str           # hover background
    input: str           # text fields
    border: str
    border_strong: str
    text: str
    text_muted: str
    text_faint: str
    accent: str
    accent_hover: str
    accent_pressed: str
    accent_text: str
    success: str
    warning: str
    danger: str
    canvas: str          # image viewport background
    scrollbar: str


DARK = Theme(
    name="dark", is_dark=True,
    window="#111318", surface="#171a20", surface_alt="#1d2027", raised="#252a33",
    hover="#2c323d", input="#14171c", border="#272b34", border_strong="#383e4b",
    text="#e8ebf1", text_muted="#a3abb9", text_faint="#69728a",
    accent="#4c8dff", accent_hover="#6ca3ff", accent_pressed="#3a78e8", accent_text="#ffffff",
    success="#3ccf91", warning="#f5b84b", danger="#ff5d6c",
    canvas="#0a0b0d", scrollbar="#3b414e",
)

LIGHT = Theme(
    name="light", is_dark=False,
    window="#eef1f5", surface="#f8f9fb", surface_alt="#ffffff", raised="#ffffff",
    hover="#e9effa", input="#ffffff", border="#dce1e9", border_strong="#c4cbd7",
    text="#1b2230", text_muted="#556072", text_faint="#97a1b1",
    accent="#2f6fed", accent_hover="#2561d8", accent_pressed="#1e52b7", accent_text="#ffffff",
    success="#16935c", warning="#bf7a0c", danger="#d6344a",
    canvas="#0e1014", scrollbar="#c2c9d4",
)

THEMES = {DARK.name: DARK, LIGHT.name: LIGHT}


def rgba(hex_color: str, alpha: float) -> str:
    c = QColor(hex_color)
    return f"rgba({c.red()}, {c.green()}, {c.blue()}, {alpha:.2f})"


# --------------------------------------------------------------------------- #
# Style sheet
# --------------------------------------------------------------------------- #

_QSS = Template("""
QMainWindow, QDialog { background: $window; }
QMainWindow::separator { background: $border; width: 1px; height: 1px; }
QMainWindow::separator:hover { background: $accent; }
QLabel { background: transparent; }
QLabel#Title { font-size: 15pt; font-weight: 700; }
QLabel#Heading { font-size: 11pt; font-weight: 700; }
QLabel#Subtitle, QLabel#Muted { color: $text_muted; }
QLabel#Faint { color: $text_faint; }
QLabel#SectionTitle { color: $text_faint; font-weight: 700; font-size: 8pt; }
QLabel#Value { font-weight: 600; }
QLabel[badge="success"] { color: $success; background: $success_soft; border-radius: 9px; padding: 2px 9px; font-weight: 600; }
QLabel[badge="warning"] { color: $warning; background: $warning_soft; border-radius: 9px; padding: 2px 9px; font-weight: 600; }
QLabel[badge="info"] { color: $accent; background: $accent_soft; border-radius: 9px; padding: 2px 9px; font-weight: 600; }
QLabel[badge="danger"] { color: $danger; background: $danger_soft; border-radius: 9px; padding: 2px 9px; font-weight: 600; }
QLabel[badge="neutral"] { color: $text_muted; background: $hover; border-radius: 9px; padding: 2px 9px; font-weight: 600; }
QFrame#Card { background: $surface_alt; border: 1px solid $border; border-radius: 10px; }
QFrame#Divider { background: $border; max-height: 1px; min-height: 1px; border: none; }
QWidget#PanelBody, QWidget#DockBody { background: $surface; }
QScrollArea { background: transparent; border: none; }
QScrollArea > QWidget > QWidget { background: transparent; }
QToolTip { background: $surface_alt; color: $text; border: 1px solid $border_strong; border-radius: 6px; padding: 5px 8px; }

QMenuBar { background: $surface; border-bottom: 1px solid $border; padding: 2px 4px; }
QMenuBar::item { background: transparent; padding: 5px 10px; border-radius: 6px; }
QMenuBar::item:selected { background: $hover; }
QMenu { background: $surface_alt; border: 1px solid $border_strong; border-radius: 8px; padding: 5px; }
QMenu::item { padding: 6px 24px 6px 10px; border-radius: 6px; background: transparent; }
QMenu::item:selected { background: $accent_soft; color: $text; }
QMenu::item:disabled { color: $text_faint; }
QMenu::separator { height: 1px; background: $border; margin: 5px 6px; }
QMenu::icon { padding-left: 6px; }

QToolBar { background: $surface; border: none; border-bottom: 1px solid $border; padding: 4px 6px; spacing: 2px; }
QToolBar::separator { background: $border; width: 1px; margin: 6px 5px; }
QToolButton { background: transparent; color: $text; border: 1px solid transparent; border-radius: 7px; padding: 5px 7px; }
QToolButton:hover { background: $hover; border-color: $border; }
QToolButton:pressed { background: $accent_soft; }
QToolButton:checked { background: $accent_soft; border-color: $accent_border; }
QToolButton:disabled { color: $text_faint; }
QToolButton::menu-indicator { image: none; width: 0px; }
QToolBarExtension { qproperty-icon: url("$chevrons_right"); padding: 0px; }
QToolBar#ToolStrip { padding: 3px 6px; }
QToolBar#ToolStrip QToolButton { padding: 4px 6px; }
QToolBar#ToolStrip QToolBarExtension { padding: 0px; }

QPushButton { background: $raised; color: $text; border: 1px solid $border_strong; border-radius: 7px; padding: 6px 14px; }
QPushButton:hover { background: $hover; border-color: $accent; }
QPushButton:pressed { background: $accent_soft; }
QPushButton:checked { background: $accent_soft; border-color: $accent; }
QPushButton:disabled { background: $surface; color: $text_faint; border-color: $border; }
QPushButton[role="primary"] { background: $accent; border-color: $accent; color: $accent_text; font-weight: 600; }
QPushButton[role="primary"]:hover { background: $accent_hover; border-color: $accent_hover; }
QPushButton[role="primary"]:pressed { background: $accent_pressed; }
QPushButton[role="primary"]:disabled { background: $raised; border-color: $border; color: $text_faint; }
QPushButton[role="danger"] { color: $danger; }
QPushButton[role="danger"]:hover { background: $danger_soft; border-color: $danger; }
QPushButton[role="ghost"] { background: transparent; border-color: transparent; }
QPushButton[role="ghost"]:hover { background: $hover; border-color: $border; }
QPushButton[role="link"] { background: transparent; border: none; color: $accent; padding: 2px 2px; text-align: left; }
QPushButton[role="link"]:hover { color: $accent_hover; }

QLineEdit, QAbstractSpinBox, QComboBox { background: $input; color: $text; border: 1px solid $border_strong; border-radius: 6px; padding: 4px 8px; min-height: 20px; selection-background-color: $accent; selection-color: $accent_text; }
QLineEdit:hover, QAbstractSpinBox:hover, QComboBox:hover { border-color: $text_faint; }
QLineEdit:focus, QAbstractSpinBox:focus, QComboBox:focus { border-color: $accent; }
QLineEdit:disabled, QAbstractSpinBox:disabled, QComboBox:disabled { color: $text_faint; background: $surface; border-color: $border; }
QLineEdit[readOnly="true"] { background: $surface; }
QComboBox { padding-right: 26px; }
QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: center right; width: 24px; border: none; }
QComboBox::down-arrow { image: url("$chevron_down"); width: 12px; height: 12px; }
QComboBox::down-arrow:on { image: url("$chevron_up"); }
QComboBox QAbstractItemView { background: $surface_alt; color: $text; border: 1px solid $border_strong; selection-background-color: $accent_soft; selection-color: $text; outline: 0; padding: 4px; }
QAbstractSpinBox { padding-right: 22px; }
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button { subcontrol-origin: border; width: 20px; border: none; background: transparent; }
QAbstractSpinBox::up-button { subcontrol-position: top right; margin: 2px 2px 0 0; }
QAbstractSpinBox::down-button { subcontrol-position: bottom right; margin: 0 2px 2px 0; }
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover { background: $hover; border-radius: 4px; }
QAbstractSpinBox::up-arrow { image: url("$chevron_up"); width: 10px; height: 10px; }
QAbstractSpinBox::down-arrow { image: url("$chevron_down"); width: 10px; height: 10px; }
QPlainTextEdit, QTextEdit, QTextBrowser { background: $input; color: $text; border: 1px solid $border; border-radius: 8px; padding: 4px; selection-background-color: $accent; selection-color: $accent_text; }

QCheckBox, QRadioButton { spacing: 8px; background: transparent; }
QCheckBox::indicator, QGroupBox::indicator { width: 16px; height: 16px; border-radius: 4px; border: 1px solid $border_strong; background: $input; }
QCheckBox::indicator:hover, QGroupBox::indicator:hover { border-color: $accent; }
QCheckBox::indicator:checked, QGroupBox::indicator:checked { background: $accent; border-color: $accent; image: url("$check"); }
QCheckBox::indicator:disabled { background: $surface; border-color: $border; }
QCheckBox:disabled, QRadioButton:disabled { color: $text_faint; }
QRadioButton::indicator { width: 16px; height: 16px; border-radius: 9px; border: 1px solid $border_strong; background: $input; }
QRadioButton::indicator:hover { border-color: $accent; }
QRadioButton::indicator:checked { width: 8px; height: 8px; border: 5px solid $accent; background: $accent_text; }

QSlider { background: transparent; min-height: 18px; }
QSlider::groove:horizontal { height: 4px; background: $border_strong; border-radius: 2px; }
QSlider::sub-page:horizontal { background: $accent; border-radius: 2px; }
QSlider::add-page:horizontal { background: $border_strong; border-radius: 2px; }
QSlider::handle:horizontal { background: $slider_handle; border: 2px solid $accent; width: 12px; height: 12px; margin: -6px 0; border-radius: 8px; }
QSlider::handle:horizontal:hover { background: $accent; }
QSlider::groove:vertical { width: 4px; background: $border_strong; border-radius: 2px; }
QSlider::add-page:vertical { background: $accent; border-radius: 2px; }
QSlider::sub-page:vertical { background: $border_strong; border-radius: 2px; }
QSlider::handle:vertical { background: $slider_handle; border: 2px solid $accent; width: 12px; height: 12px; margin: 0 -6px; border-radius: 8px; }
QSlider::handle:disabled { border-color: $border_strong; }
QSlider::sub-page:horizontal:disabled { background: $border_strong; }

QProgressBar { background: $raised; color: $text; border: 1px solid $border; border-radius: 6px; text-align: center; min-height: 14px; }
QProgressBar::chunk { background: $accent; border-radius: 5px; }

QScrollBar:vertical { background: transparent; width: 11px; margin: 0; }
QScrollBar::handle:vertical { background: $scrollbar; border-radius: 3px; min-height: 30px; margin: 2px 3px; }
QScrollBar:horizontal { background: transparent; height: 11px; margin: 0; }
QScrollBar::handle:horizontal { background: $scrollbar; border-radius: 3px; min-width: 30px; margin: 3px 2px; }
QScrollBar::handle:hover { background: $text_faint; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0px; height: 0px; border: none; background: none; }
QScrollBar::add-page, QScrollBar::sub-page { background: none; }

QTabWidget::pane { border: 1px solid $border; border-radius: 8px; background: $surface; top: -1px; }
QTabBar { background: transparent; }
QTabBar::tab { background: transparent; color: $text_muted; padding: 7px 14px; border: none; border-bottom: 2px solid transparent; }
QTabBar::tab:hover { color: $text; }
QTabBar::tab:selected { color: $text; border-bottom: 2px solid $accent; }

QGroupBox { background: $surface_alt; border: 1px solid $border; border-radius: 9px; margin-top: 16px; padding: 12px 10px 10px 10px; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 12px; padding: 0 4px; color: $text_muted; font-weight: 600; }

QAbstractItemView { background: $surface_alt; color: $text; border: 1px solid $border; border-radius: 8px; alternate-background-color: $surface; selection-background-color: $accent_soft; selection-color: $text; outline: 0; }
QAbstractItemView::item:hover { background: $hover; }
QAbstractItemView::item:selected { background: $accent_soft; color: $text; }
QTableView { gridline-color: $border; }
QHeaderView { background: transparent; border: none; }
QHeaderView::section { background: $surface; color: $text_muted; border: none; border-bottom: 1px solid $border; padding: 6px 8px; font-weight: 600; }
QTableCornerButton::section { background: $surface; border: none; }

QDockWidget { color: $text; }
QDockWidget::title { background: $surface; padding: 8px 10px; border-bottom: 1px solid $border; text-align: left; }
QDockWidget::close-button, QDockWidget::float-button { border: none; background: transparent; padding: 1px; }
QDockWidget::close-button:hover, QDockWidget::float-button:hover { background: $hover; border-radius: 4px; }

QStatusBar { background: $surface; color: $text_muted; border-top: 1px solid $border; }
QStatusBar::item { border: none; }
QStatusBar QLabel { color: $text_muted; padding: 0 4px; }
QSplitter::handle { background: $border; }
QSplitter::handle:hover { background: $accent; }
QColorDialog { background: $window; }
QDialogButtonBox { dialogbuttonbox-buttons-have-icons: 0; }

QWidget#GuideSidebar { background: $surface; }
QTreeWidget#GuideNav { background: transparent; border: none; }
QTreeWidget#GuideNav::item { padding: 2px 6px; border-radius: 6px; }
QTreeWidget#GuideNav::item:hover { background: $hover; }
QTreeWidget#GuideNav::item:selected { background: $accent_soft; }
QTextBrowser#GuideBrowser { background: $window; border: none; border-radius: 0px; padding: 0px; }
QFrame#RefCard { background: $surface_alt; border: 1px solid $border; border-radius: 10px; }
QFrame#RefCard:hover { border-color: $border_strong; }
QFrame#CiteCard { background: $accent_soft; border: 1px solid $accent_border; border-radius: 12px; }
""")


def build_stylesheet(theme: Theme, asset_dir: str) -> str:
    tokens = asdict(theme)
    tokens.update(
        accent_soft=rgba(theme.accent, 0.20 if theme.is_dark else 0.12),
        accent_border=rgba(theme.accent, 0.55),
        success_soft=rgba(theme.success, 0.16),
        warning_soft=rgba(theme.warning, 0.16),
        danger_soft=rgba(theme.danger, 0.14),
        slider_handle=theme.surface_alt if theme.is_dark else "#ffffff",
        chevron_down=_asset_url(asset_dir, "chevron_down.svg"),
        chevron_up=_asset_url(asset_dir, "chevron_up.svg"),
        check=_asset_url(asset_dir, "check.svg"),
        chevrons_right=_asset_url(asset_dir, "chevrons_right.svg"),
    )
    return _QSS.substitute(tokens)


def _asset_url(asset_dir: str, name: str) -> str:
    return os.path.join(asset_dir, name).replace("\\", "/")


def _write_assets(theme: Theme) -> str:
    """Small SVG images the style sheet needs (arrows, check mark)."""
    folder = os.path.join(tempfile.gettempdir(), f"dicomdoodle_theme_{os.getpid()}", theme.name)
    os.makedirs(folder, exist_ok=True)
    stroke = theme.text_muted
    svgs = {
        "chevron_down.svg": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
                            f'stroke="{stroke}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">'
                            f'<polyline points="6 9 12 15 18 9"/></svg>',
        "chevron_up.svg": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
                          f'stroke="{stroke}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">'
                          f'<polyline points="18 15 12 9 6 15"/></svg>',
        "chevrons_right.svg": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
                              f'stroke="{theme.text}" stroke-width="2.2" stroke-linecap="round" '
                              f'stroke-linejoin="round"><polyline points="13 17 18 12 13 7"/>'
                              f'<polyline points="6 17 11 12 6 7"/></svg>',
        "check.svg": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
                     f'stroke="{theme.accent_text}" stroke-width="3.2" stroke-linecap="round" '
                     f'stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>',
    }
    for name, content in svgs.items():
        with open(os.path.join(folder, name), "w", encoding="utf-8") as fh:
            fh.write(content)
    return folder


# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #

def build_palette(theme: Theme) -> QPalette:
    c = lambda value: QColor(value)  # noqa: E731
    pal = QPalette()
    pal.setColor(QPalette.Window, c(theme.window))
    pal.setColor(QPalette.WindowText, c(theme.text))
    pal.setColor(QPalette.Base, c(theme.input))
    pal.setColor(QPalette.AlternateBase, c(theme.surface))
    pal.setColor(QPalette.ToolTipBase, c(theme.surface_alt))
    pal.setColor(QPalette.ToolTipText, c(theme.text))
    pal.setColor(QPalette.PlaceholderText, c(theme.text_faint))
    pal.setColor(QPalette.Text, c(theme.text))
    pal.setColor(QPalette.Button, c(theme.raised))
    pal.setColor(QPalette.ButtonText, c(theme.text))
    pal.setColor(QPalette.BrightText, c(theme.danger))
    pal.setColor(QPalette.Light, c(theme.hover))
    pal.setColor(QPalette.Midlight, c(theme.border_strong))
    pal.setColor(QPalette.Mid, c(theme.border))
    pal.setColor(QPalette.Dark, c(theme.border_strong))
    pal.setColor(QPalette.Shadow, QColor(0, 0, 0, 120))
    pal.setColor(QPalette.Highlight, c(theme.accent))
    pal.setColor(QPalette.HighlightedText, c(theme.accent_text))
    pal.setColor(QPalette.Link, c(theme.accent))
    pal.setColor(QPalette.LinkVisited, c(theme.accent_pressed))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        pal.setColor(QPalette.Disabled, role, c(theme.text_faint))
    return pal


# --------------------------------------------------------------------------- #
# Manager
# --------------------------------------------------------------------------- #

class ThemeManager(QObject):
    """Owns the active theme and re-applies it application-wide."""

    themeChanged = pyqtSignal(object)   # the new Theme

    def __init__(self, app: QApplication, name: str = "dark"):
        super().__init__(app)
        self._app = app
        self._theme = THEMES.get(name, DARK)
        app.setStyle("Fusion")
        app.installEventFilter(self)

    @property
    def theme(self) -> Theme:
        return self._theme

    @property
    def is_dark(self) -> bool:
        return self._theme.is_dark

    def apply(self, name: Optional[str] = None) -> Theme:
        if name is not None:
            self._theme = THEMES.get(name, DARK)
        theme = self._theme
        self._app.setPalette(build_palette(theme))
        self._app.setStyleSheet(build_stylesheet(theme, _write_assets(theme)))
        for widget in self._app.topLevelWidgets():
            if widget.isWindow() and widget.isVisible():
                apply_native_title_bar(widget, theme.is_dark)
        self.themeChanged.emit(theme)
        return theme

    def toggle(self) -> Theme:
        return self.apply(LIGHT.name if self._theme.is_dark else DARK.name)

    def color(self, token: str) -> QColor:
        return QColor(getattr(self._theme, token))

    def eventFilter(self, obj, event):  # dark title bars for windows shown later
        if event.type() == QEvent.Show and isinstance(obj, QWidget) and obj.isWindow():
            apply_native_title_bar(obj, self._theme.is_dark)
        return False


_MANAGER: Optional[ThemeManager] = None


def install_theme_manager(app: QApplication, name: str) -> ThemeManager:
    global _MANAGER
    _MANAGER = ThemeManager(app, name)
    _MANAGER.apply()
    return _MANAGER


def theme_manager() -> ThemeManager:
    if _MANAGER is None:
        raise RuntimeError("install_theme_manager() has not been called")
    return _MANAGER


def current_theme() -> Theme:
    return _MANAGER.theme if _MANAGER is not None else DARK


def theme_color(token: str) -> QColor:
    return QColor(getattr(current_theme(), token))


def apply_native_title_bar(widget: QWidget, dark: bool) -> None:
    """Dark window title bar on Windows 10/11 (no-op elsewhere)."""
    if not sys.platform.startswith("win"):
        return
    try:
        hwnd = int(widget.winId())
        value = ctypes.c_int(1 if dark else 0)
        for attribute in (20, 19):   # DWMWA_USE_IMMERSIVE_DARK_MODE (new, old builds)
            if ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attribute, ctypes.byref(value),
                                                          ctypes.sizeof(value)) == 0:
                break
    except Exception:
        pass
