"""
Application bootstrap.

Creates, in order: the QApplication, the theme, the splash screen, logging,
the shared services (settings, labels, AI models) and the main window, then
starts loading SAM in the background so the window appears immediately (the
original loaded SAM before showing any window).
"""
from __future__ import annotations

import logging
import os
import sys
import traceback

# Windows: torch must be imported before PyQt5, otherwise loading torch's DLLs
# later fails with "WinError 1114" (same workaround as the original tool).
if sys.platform.startswith("win"):
    try:
        import torch  # noqa: F401
    except Exception:
        pass

from PyQt5.QtCore import QCoreApplication, Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication

from . import APP_NAME, APP_VERSION, resources
from .settings import Settings


def _setup_logging():
    from .ui.widgets.log_panel import QtLogHandler

    logger = logging.getLogger("dicomdoodle")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s", "%H:%M:%S"))
    gui = QtLogHandler()
    logger.addHandler(console)
    logger.addHandler(gui)
    return gui


def _install_exception_hook() -> None:
    """Log and report unexpected errors instead of letting PyQt abort the app."""
    log = logging.getLogger("dicomdoodle")
    state = {"showing": False}

    def hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        details = "".join(traceback.format_exception(exc_type, exc, tb))
        if issubclass(exc_type, RuntimeError) and "has been deleted" in str(exc):
            log.debug("Ignored a callback into a closed window: %s", exc)
            return
        sys.stderr.write(details)
        log.error("Unexpected error: %s: %s", exc_type.__name__, exc)
        if state["showing"]:
            return
        state["showing"] = True
        try:
            from .ui.qt_utils import show_error
            show_error(QApplication.activeWindow(), "Unexpected error",
                       f"{exc_type.__name__}: {exc}\n\nYour work is still open; please save it.", details)
        except Exception:
            pass
        finally:
            state["showing"] = False

    sys.excepthook = hook


def _restore_qt_plugin_path() -> None:
    """opencv-python may point Qt at its own plugins; undo that once cv2 is loaded."""
    path = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH", "")
    if "cv2" in path:
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)


def _warm_up_models(window, models) -> None:
    from .workers import run_task

    window.set_ai_status("loading", "Loading SAM…")
    log = logging.getLogger("dicomdoodle")

    def ready(device: str) -> None:
        window.set_ai_status("ready", f"AI ready · {device}")
        log.info("SAM model loaded (%s)", device)

    def failed(message: str, _details: str) -> None:
        window.set_ai_status("error", "AI models unavailable")
        log.warning("Could not preload SAM: %s", message.splitlines()[0])

    run_task(lambda _ctx: models.warm_up(), on_result=ready, on_error=failed)


def run(argv) -> int:
    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)

    from .ui.theme import install_theme_manager
    settings = Settings()
    install_theme_manager(app, settings.get("theme"))
    icon_path = resources.find_resource(resources.WINDOW_ICON) or resources.find_resource(resources.LOGO_IMAGE)
    if icon_path:
        app.setWindowIcon(QIcon(icon_path))

    from .ui.splash import SplashScreen
    splash = SplashScreen()
    splash.show()
    splash.set_status("Loading modules…")

    log_handler = _setup_logging()
    _install_exception_hook()
    from .core.ai_models import ModelManager
    from .ui.label_manager import LabelManager
    from .ui.main import MainWindow
    _restore_qt_plugin_path()

    splash.set_status("Preparing workspace…")
    models = ModelManager(settings.get("models_dir"), settings.get("device"))
    labels = LabelManager(settings)
    window = MainWindow(settings, labels, models, log_handler)
    window.show()
    splash.finish(window)

    if settings.get("preload_sam"):
        _warm_up_models(window, models)
    folder = next((a for a in argv[1:] if os.path.isdir(a)), None)
    if folder:
        window.open_folder(folder)
    return app.exec_()
