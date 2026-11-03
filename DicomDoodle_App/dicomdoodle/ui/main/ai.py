"""
The AI side of the main window: the model card, the processor or graphics
card, and the AI button at the right of the status bar.

At start, in the background: find the graphics card, find the model card
(installing it the first time, see core/ai/model_card.py) and load its models
when "Load the AI models when DicomDoodle starts" is on. A notification at the
bottom right shows the progress and disappears when the AI is ready. Without
a model card the AI actions stay off: annotation by hand only.

The AI button shows the state; its menu switches between the processor (the
default) and the graphics card, loads a model card and opens the AI
preferences.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

from PyQt5.QtCore import QObject, QRectF, QSize, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt5.QtWidgets import QAction, QActionGroup, QFileDialog, QMenu, QToolButton

from ...core.ai import devices, model_card
from ...core.ai_models import LOADING
from ...workers import run_task
from ..icons import bind_icon
from ..qt_utils import ask_yes_no, show_error
from ..theme import theme_manager

if TYPE_CHECKING:
    from ...ui.widgets.toast import Toast
    from .window import MainWindow

log = logging.getLogger("dicomdoodle")

NO_CARD_TIP = ("No model card: the AI is off, annotate by hand with the pen.\n"
               "Get the DicomDoodle model card (a separate download), put it in your Downloads folder and restart "
               "DicomDoodle — or open it with File ▸ Load Model Card….")
FEATURE_NEEDS = {"detect": "Auto Detect", "segment": "Detect in Box", "range": "Find tumour range"}


def _dot(color: str) -> QIcon:
    """The AI state's colored dot."""
    pixmap = QPixmap(24, 24)
    pixmap.setDevicePixelRatio(2.0)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(color))
    painter.drawEllipse(QRectF(2.0, 2.0, 8.0, 8.0))
    painter.end()
    return QIcon(pixmap)


class AIController(QObject):
    changed = pyqtSignal()                 # the model card, the device or the loading state changed

    def __init__(self, window: "MainWindow"):
        super().__init__(window)
        self.w = window
        self.models = window.models
        self.state = "starting"            # starting · installing · loading · ready · idle · none · error
        self._job_running = False
        self._queued: Optional[tuple] = None
        self._toast: Optional["Toast"] = None

        self.button = QToolButton()
        self.button.setAutoRaise(True)
        self.button.setPopupMode(QToolButton.InstantPopup)
        self.button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.button.setIconSize(QSize(12, 12))
        self.button.setCursor(Qt.PointingHandCursor)
        self.menu = QMenu(self.button)
        self.card_header = self.menu.addAction("")
        self.card_header.setEnabled(False)
        self.menu.addSeparator()
        group = QActionGroup(self.menu)
        self.cpu_action = QAction("Run the AI on the processor (CPU)", self.menu, checkable=True)
        self.gpu_action = QAction("Run the AI on the graphics card", self.menu, checkable=True)
        for action, device in ((self.cpu_action, devices.CPU), (self.gpu_action, devices.GPU)):
            group.addAction(action)
            self.menu.addAction(action)
            action.triggered.connect(lambda _c=False, d=device: self.set_device(d))
        bind_icon(self.gpu_action, "cpu")
        self.menu.addSeparator()
        self.load_action = self.menu.addAction("Load Model Card…")
        bind_icon(self.load_action, "download")
        self.load_action.triggered.connect(self.choose_card)
        self.prefs_action = self.menu.addAction("AI Preferences…")
        bind_icon(self.prefs_action, "settings")
        self.prefs_action.triggered.connect(lambda: self.w.tools.preferences("ai"))
        self.menu.aboutToShow.connect(self._refresh_menu)
        self.button.setMenu(self.menu)
        theme_manager().themeChanged.connect(self._theme_changed)
        self._status = ("idle", "")

    # ----------------------------------------------------------------- status
    def show_status(self, state: str, text: str) -> None:
        """The AI button: a colored dot and ``text`` (state: ready, loading, error, idle)."""
        self._status = (state, text)
        colors = {"ready": "success", "loading": "warning", "error": "danger", "idle": "text_faint"}
        self.button.setIcon(_dot(getattr(theme_manager().theme, colors.get(state, "text_faint"))))
        self.button.setText(f"{text}  ▾")            # ▾: a menu opens on click
        self.button.setToolTip(self._tooltip())

    def _theme_changed(self, *_args) -> None:
        self.show_status(*self._status)

    def _tooltip(self) -> str:
        m = self.models
        if self.state == "none":
            return NO_CARD_TIP
        lines = [f"Model card: {m.card.label}" if m.card is not None else "Model card: none"]
        for row in m.info():
            if row.title:
                where = f", {row.device}" if row.loaded else ""
                lines.append(f"{row.purpose}: {row.title} ({row.engine or '—'}{where})")
        lines.append("Click to switch between the processor and the graphics card.")
        return "\n".join(lines)

    def refresh(self) -> None:
        """Show the state after a change (card, device, models loaded or not)."""
        m = self.models
        if self._job_running:              # the job reports when it is done
            return
        if not m.ready:
            self.state = "none" if m.card is None else "error"
            self.show_status("idle" if m.card is None else "error",
                             "Manual annotation only" if m.card is None else "AI models unusable")
        elif m.loaded:
            self.state = "ready"
            self.show_status("ready", f"AI ready · {self._where()}")
        else:
            self.state = "idle"
            self.show_status("idle", "AI models load on first use")
        self.changed.emit()
        self.w.update_ui_state()

    def _where(self) -> str:
        return "GPU" if not self.models.runs_on_cpu() else "CPU"

    def _refresh_menu(self) -> None:
        m = self.models
        self.card_header.setText(f"Model card: {m.card.label}" if m.card is not None
                                 else "No model card: manual annotation only")
        gpu = devices.known_gpu()
        on_gpu = m.device == devices.GPU
        self.cpu_action.setChecked(not on_gpu)
        self.gpu_action.setChecked(on_gpu)
        if gpu is None:
            self.gpu_action.setText("Run the AI on the graphics card (looking for one…)")
            self.gpu_action.setEnabled(on_gpu)
        elif gpu.available:
            self.gpu_action.setText(f"Run the AI on the graphics card: {gpu.describe()}")
            self.gpu_action.setEnabled(True)
        else:
            self.gpu_action.setText("Run the AI on the graphics card (none that can be used)")
            self.gpu_action.setEnabled(on_gpu)
            self.gpu_action.setToolTip(gpu.problem)
        self.menu.setToolTipsVisible(True)
        busy = self._job_running or self.w.busy
        self.cpu_action.setEnabled(not busy)
        self.gpu_action.setEnabled(self.gpu_action.isEnabled() and not busy)
        self.load_action.setEnabled(not busy)

    def explain(self, feature: str) -> bool:
        """True when ``feature`` can run; otherwise tells the user why not (no card, still loading)."""
        if self.models.can(feature):
            return True
        if self.state in ("starting", "installing"):
            self.w.toasts.show("The AI models are still being prepared — a moment, please.", "info")
        else:
            self.w.toasts.show(f"{FEATURE_NEEDS.get(feature, 'The AI')} needs the DicomDoodle model card: "
                               "File ▸ Load Model Card…", "warning", 6000)
        return False

    # ------------------------------------------------------------- start / jobs
    def start(self, card_path: str = "") -> None:
        """Find (or install) the model card and load the models, in the background."""
        self._prepare(card_path, load=bool(self.w.settings.get("preload_sam")) or bool(card_path))

    def _prepare(self, chosen: str = "", load: bool = True) -> None:
        if self._job_running:
            self._queued = (chosen, load)
            return
        self._job_running = True
        self.state = "starting"
        self.show_status("loading", "Looking for the AI models…")
        search = bool(self.w.settings.get("find_model_cards"))
        models = self.models

        def job(ctx):
            event = models.prepare(ctx, chosen=chosen, search=search)
            where = models.load_all(ctx) if load and models.ready else ""
            return event, where

        run_task(job, on_progress=self._progress, on_result=self._prepared,
                 on_error=lambda message, details: self._failed(message, details, bool(chosen)),
                 on_finished=self._job_done)

    def _load(self, title: str) -> None:
        """Load (or reload) the models in the background, e.g. after switching the device."""
        if self._job_running:
            self._queued = ("", True)
            return
        self._job_running = True
        self.state = "loading"
        self.show_status("loading", "Loading the AI models…")
        self._toast = self.w.toasts.progress(title, "Loading the AI models…")
        run_task(lambda ctx: ("ready", self.models.load_all(ctx)), on_progress=self._progress,
                 on_result=self._prepared, on_error=lambda m, d: self._failed(m, d, False), on_finished=self._job_done)

    def _progress(self, done: int, total: int, message: str) -> None:
        if message.startswith(model_card.INSTALLING):
            self.state = "installing"
            self.show_status("loading", f"Installing the model card… {100 * done // max(total, 1)} %")
            title = "Installing the AI models"
        elif message.startswith(LOADING):
            self.state = "loading"
            self.show_status("loading", "Loading the AI models…")
            title = "Loading the AI models"
            detail = message.split("… ", 1)[-1]
            message = detail[:1].upper() + detail[1:] + "…"            # "Lesion detector (1 of 3)…"
        else:
            return
        if self._toast is None:
            self._toast = self.w.toasts.progress(title, message)
        try:
            self._toast.set_title(title)
            self._toast.set_message(message)
            self._toast.set_progress(done, total)
        except RuntimeError:              # dismissed by a click
            self._toast = None

    def _close_toast(self) -> None:
        if self._toast is not None:
            try:
                self._toast.dismiss()
            except RuntimeError:
                pass
            self._toast = None

    def _finish_toast(self, message: str, kind: str = "success", duration_ms: int = 2500) -> None:
        if self._toast is not None:
            try:
                self._toast.finish(message, kind, duration_ms)
            except RuntimeError:
                self.w.toasts.show(message, kind, duration_ms)
            self._toast = None
        elif kind != "success":
            self.w.toasts.show(message, kind, duration_ms)

    def _prepared(self, result) -> None:
        event, where = result
        m = self.models
        self.state, self._job_running = "", False       # (the task's finished signal comes after this one)
        if m.card is None:
            self._close_toast()
            self.w.toasts.show("No model card found: manual annotation only. To use the AI, put the DicomDoodle "
                               "model card in your Downloads folder and restart, or use File ▸ Load Model Card….",
                               "info", 7000)
            log.info("No model card: the AI is off (manual annotation only)")
        elif not m.ready:
            self._finish_toast(f"The model card “{m.card.label}” has no model this computer can run.", "error", 8000)
        elif where:
            gpu_note = " (the graphics card failed: the processor runs it)" if m.gpu_failed() else ""
            card_note = f"Model card {m.card.label} installed. " if event in ("installed", "updated") else ""
            self._finish_toast(f"{card_note}AI ready on the {'graphics card' if where != 'CPU' else 'processor'}"
                               f"{gpu_note}.", "success", 3500 if card_note or gpu_note else 2500)
        elif event in ("installed", "updated"):
            self._finish_toast(f"Model card {m.card.label} installed: the models load when you first use them.")
        self.refresh()
        self._gpu_notes()

    def _failed(self, message: str, details: str, chosen: bool) -> None:
        self.state, self._job_running = "", False
        log.warning("AI models: %s", message.splitlines()[0] if message else "failed")
        if chosen:
            self._finish_toast("The model card was not installed.", "error", 4000)
            show_error(self.w, "Model card", message, details)
        else:
            self._finish_toast(f"AI models unavailable: {message.splitlines()[0]}", "error", 9000)
        self.refresh()
        if not self.models.ready:
            self.show_status("error" if self.models.card is not None else "idle",
                             "AI error" if self.models.card is not None else "Manual annotation only")

    def _job_done(self) -> None:
        self._job_running = False
        if self._queued is not None:
            chosen, load = self._queued
            self._queued = None
            self._prepare(chosen, load)

    def _gpu_notes(self) -> None:
        """Once: tell about a graphics card the AI could use; always: a chosen card that cannot be used."""
        gpu, settings = devices.known_gpu(), self.w.settings
        if gpu is None or not self.models.ready:
            return
        if self.models.device == devices.GPU and not gpu.available:
            self.w.toasts.show(f"The AI runs on the processor: {gpu.problem}", "warning", 8000)
        elif self.models.device == devices.CPU and gpu.available and not settings.get("gpu_hint_shown"):
            settings.set("gpu_hint_shown", True)
            self.w.toasts.show(f"Graphics card found: {gpu.describe()}. The AI runs on the processor; to use the "
                               "graphics card, click the AI status at the bottom right.", "info", 10000)

    # ---------------------------------------------------------------- actions
    def set_device(self, device: str) -> None:
        device = devices.normalize(device)
        if self.w.busy or self._job_running:
            self.w.toasts.show("Please wait until the current job finishes.", "warning")
            self._refresh_menu()
            return
        self.w.settings.set("device", device)
        if device == self.models.device:
            return
        was_loaded = self.models.loaded
        self.models.configure(device=device)
        log.info("AI device: %s", "graphics card" if device == devices.GPU else "processor (CPU)")
        if self.models.ready and (was_loaded or self.w.settings.get("preload_sam")):
            self._load("Moving the AI to the " + ("graphics card" if device == devices.GPU else "processor"))
        else:
            self.refresh()

    def apply_settings(self) -> None:
        """After the Preferences dialog: the device may have changed."""
        device = devices.normalize(self.w.settings.get("device"))
        if device != self.models.device:
            self.w.settings.set("device", self.models.device)        # set_device() saves it again
            self.set_device(device)
        self.models.configure(models_dir=self.w.settings.get("models_dir"))

    def choose_card(self, parent=None) -> None:
        downloads = next((f for f in model_card.user_folders() if os.path.isdir(f)), os.path.expanduser("~"))
        path, _filter = QFileDialog.getOpenFileName(
            parent or self.w, "Load Model Card", downloads,
            f"DicomDoodle model card (*{model_card.CARD_EXTENSION} *.zip);;All files (*)")
        if path:
            self.load_card(path)

    def load_card(self, path: str) -> None:
        """Install the card at ``path`` (File ▸ Load Model Card…, a dropped file, the command line)."""
        if self.w.busy:
            self.w.toasts.show("Please wait until the current job finishes.", "warning")
            return
        try:
            card = model_card.read_card(path)
        except model_card.CardError as exc:
            show_error(self.w, "Model card", str(exc))
            return
        self.w.settings.set("find_model_cards", True)
        log.info("Loading the model card %s (%s)", card.label, path)
        self._prepare(path, load=True)

    def remove_card(self, parent=None) -> None:
        m = self.models
        if m.card is None:
            return
        if self.w.busy or self._job_running:
            self.w.toasts.show("Please wait until the current job finishes.", "warning")
            return
        if not ask_yes_no(parent or self.w, "Remove the model card",
                          f"Remove the installed model card “{m.card.label}”?\n\nDicomDoodle then works without AI "
                          "(manual annotation only) and no longer looks for model cards, until you load one "
                          "with File ▸ Load Model Card…. A downloaded card file is not deleted.",
                          yes_text="Remove", default_yes=False):
            return
        m.remove_card()
        self.w.settings.set("find_model_cards", False)
        self.w.toasts.show("Model card removed: manual annotation only.", "info")
        self.state = ""
        self.refresh()
