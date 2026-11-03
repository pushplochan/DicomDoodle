"""
The built-in, illustrated User Guide (it replaces the PDF manual).

    configure(settings, on_action)   once, by the main window
    show_guide("labels")             open the guide at a chapter (one window, reused)
    help_button("batch")             small (?) button that opens a chapter
    help_action(window, "mpr")       toolbar action "Help" that opens a chapter
    close_guide()                    when the app closes

The chapters themselves are text in ``dicomdoodle/help/``; this package
renders them: markup.py (text -> HTML), diagrams.py + drawing.py (pictures
drawn in the theme's colors) and window.py (the guide window).
"""
from __future__ import annotations

from typing import Callable, Optional

from PyQt5.QtWidgets import QAction, QToolButton

from ..icons import bind_icon
from ..qt_utils import make_tool_button

_window = None
_settings = None
_on_action: Optional[Callable[[str], None]] = None


def configure(settings=None, on_action: Optional[Callable[[str], None]] = None) -> None:
    """``on_action(name)`` runs links such as "Open a DICOM folder" in the main window."""
    global _settings, _on_action
    _settings, _on_action = settings, on_action


def _dispatch(name: str) -> None:
    if _on_action is not None:
        _on_action(name)


def show_guide(topic: str = ""):
    """Show the guide; ``topic`` is a chapter id (empty: stay where the reader was)."""
    global _window
    from .window import GuideWindow
    if _window is None:
        _window = GuideWindow(_settings)
        _window.actionRequested.connect(_dispatch)
    if topic:
        _window.show_chapter(topic)
    _window.showNormal()
    _window.raise_()
    _window.activateWindow()
    return _window


def close_guide() -> None:
    if _window is not None:
        _window.close()


def help_button(topic: str, tooltip: str = "Help on this — opens the User Guide") -> QToolButton:
    button = make_tool_button("help", tooltip, size=16)
    button.clicked.connect(lambda: show_guide(topic))
    return button


def help_action(parent, topic: str, text: str = "Help") -> QAction:
    action = QAction(text, parent)
    bind_icon(action, "help")
    action.setToolTip("How to use this tool — opens the User Guide")
    action.triggered.connect(lambda _checked=False: show_guide(topic))
    return action
