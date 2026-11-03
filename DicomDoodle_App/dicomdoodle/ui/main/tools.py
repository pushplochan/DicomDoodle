"""
Opens the tool windows and dialogs from the main window.

Tool modules are imported only when first opened, so a missing optional
package (VTK, SimpleITK, matplotlib ...) disables just that tool with a
clear message instead of stopping the whole app. Each tool window exists
once; opening it again brings it to the front. (The original kept every
tool in one variable, so opening a second tool could destroy the first.)
"""
from __future__ import annotations

import logging
import traceback
from typing import TYPE_CHECKING, Callable, Dict

from PyQt5.QtCore import QObject, Qt
from PyQt5.QtWidgets import QWidget

from ..qt_utils import show_error

if TYPE_CHECKING:
    from .window import MainWindow

log = logging.getLogger("dicomdoodle")


class ToolLauncher(QObject):
    def __init__(self, window: "MainWindow"):
        super().__init__(window)
        self.w = window
        self._windows: Dict[str, QWidget] = {}

    def _current_folder(self) -> str:
        return self.w.session.series.folder if self.w.session is not None else ""

    def _show(self, key: str, factory: Callable[[], QWidget], requirement: str = "") -> None:
        window = self._windows.get(key)
        if window is not None:
            try:
                window.showNormal()
                window.raise_()
                window.activateWindow()
                return
            except RuntimeError:          # was deleted
                self._windows.pop(key, None)
        try:
            window = factory()
        except ImportError as exc:
            show_error(self.w, "Tool unavailable",
                       f"This tool needs an optional package that is not installed: {exc.name or exc}.\n\n"
                       f"{requirement or 'Install the requirements listed in requirements.txt.'}",
                       traceback.format_exc())
            return
        except Exception as exc:
            show_error(self.w, "Could not open tool", str(exc), traceback.format_exc())
            return
        window.setAttribute(Qt.WA_DeleteOnClose, True)
        window.destroyed.connect(lambda *_: self._windows.pop(key, None))
        self._windows[key] = window
        window.show()

    def open_by_key(self, key: str) -> None:
        {"overlay": self.overlay_viewer, "viewer3d": self.viewer_3d, "mpr": self.mpr_viewer,
         "converter": self.multiframe_converter}[key]()

    def close_all(self) -> None:
        for window in list(self._windows.values()):
            try:
                window.close()
            except RuntimeError:
                pass

    # ----------------------------------------------------------------- tools
    def overlay_viewer(self) -> None:
        def make():
            from ..tools.overlay_viewer import OverlayViewerWindow
            return OverlayViewerWindow(self.w.labels, self.w.settings, initial_folder=self._current_folder())
        self._show("overlay", make)

    def mpr_viewer(self, tab: str = "multiplanar") -> None:
        """The Multiplanar & 3D Viewer; ``tab``: "multiplanar" or "3d", the controls to show."""
        def make():
            from ..tools.mpr.window import MPRWindow
            return MPRWindow(self.w.labels, initial_folder=self._current_folder(), models=self.w.models, tab=tab)
        self._show("mpr", make, "The Multiplanar & 3D Viewer needs 'SimpleITK', 'matplotlib' and 'vtk'.")
        window = self._windows.get("mpr")
        if window is not None and tab == "3d":
            window.control_tabs.setCurrentIndex(1)

    def viewer_3d(self) -> None:
        """The former 3D Model Viewer: the Multiplanar & 3D Viewer at its 3D view settings."""
        self.mpr_viewer("3d")

    def multiframe_converter(self) -> None:
        def make():
            from ..tools.multiframe_converter import MultiFrameConverterWindow
            return MultiFrameConverterWindow(start_dir=self._current_folder() or self.w.settings.get("last_dir"))
        self._show("converter", make)

    # ------------------------------------------------------------ user guide
    def help(self, topic: str = "") -> None:
        """Open the User Guide, optionally at a chapter (see help/contents.py)."""
        from ..guide import show_guide
        show_guide(topic)

    def guide_action(self, name: str) -> None:
        """Run a link of the User Guide, e.g. “Open a DICOM folder” (action:open)."""
        w = self.w
        handlers = {
            "open": w.open_folder_dialog,
            "overlay": self.overlay_viewer, "viewer3d": self.viewer_3d, "mpr": self.mpr_viewer,
            "converter": self.multiframe_converter, "preferences": self.preferences, "shortcuts": self.shortcuts,
            "references": self.citations, "about": self.about, "add_label": w.label_panel.add_label,
            "theme": w.toggle_theme,
        }
        handler = handlers.get(name)
        if handler is None:
            log.warning("Unknown User Guide action: %s", name)
            return
        w.showNormal()
        w.raise_()
        w.activateWindow()
        handler()

    # --------------------------------------------------------------- dialogs

    def shortcuts(self) -> None:
        from ..dialogs.shortcuts_dialog import ShortcutsDialog
        ShortcutsDialog(self.w).exec_()

    def about(self) -> None:
        from ..dialogs.about_dialog import AboutDialog
        AboutDialog(self.w).exec_()

    def citations(self) -> None:
        from ..dialogs.citations_dialog import CitationsDialog
        CitationsDialog(self.w).exec_()

    def preferences(self) -> None:
        from ..dialogs.settings_dialog import SettingsDialog
        if SettingsDialog(self.w.settings, self.w.models, self.w).exec_():
            self.w.apply_settings()
            log.info("Preferences saved")
