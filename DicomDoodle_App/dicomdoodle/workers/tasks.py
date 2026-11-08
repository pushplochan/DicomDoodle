"""
Run any function in a worker thread and get Qt signals back on the GUI thread.

    task = run_task(load_series, folder,
                    on_result=self._series_loaded,
                    on_error=self._show_error,
                    on_progress=dialog.set_progress)
    ...
    task.cancel()

The function receives a :class:`TaskContext` as its first argument. The
context is a ``core.progress.Progress``, so it can be passed straight to any
core function that reports progress or should stop on Cancel.
"""
from __future__ import annotations

import logging
import threading
import traceback
from typing import Any, Callable, Optional, Set

from PyQt5.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal

from ..core.progress import OperationCancelled, Progress

log = logging.getLogger(__name__)

_RUNNING: Set["Task"] = set()      # keeps Python wrappers alive while running


class TaskSignals(QObject):
    progress = pyqtSignal(int, int, str)    # done, total, message
    succeeded = pyqtSignal(object)          # return value
    failed = pyqtSignal(str, str)           # message, traceback
    cancelled = pyqtSignal()
    finished = pyqtSignal()                 # always emitted last


class TaskContext(Progress):
    """Progress reporter + cancel flag handed to the task function."""

    def __init__(self, signals: TaskSignals, cancel_event: threading.Event):
        super().__init__(report=lambda d, t, m: signals.progress.emit(int(d), int(t), str(m)),
                         is_cancelled=cancel_event.is_set)


class Task(QRunnable):
    """A function call executed on Qt's global thread pool."""

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any):
        super().__init__()
        self.setAutoDelete(False)
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self.signals = TaskSignals()
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancel.is_set()

    def run(self) -> None:  # executed in the worker thread
        context = TaskContext(self.signals, self._cancel)
        try:
            result = self.fn(context, *self.args, **self.kwargs)
        except OperationCancelled:
            self.signals.cancelled.emit()
        except Exception as exc:  # report every failure to the GUI
            log.exception("Background task failed")
            self.signals.failed.emit(str(exc) or exc.__class__.__name__, traceback.format_exc())
        else:
            # A function that returns normally after Cancel (e.g. a batch job
            # handing back the slices it finished) still delivers its result.
            self.signals.succeeded.emit(result)
        finally:
            self.signals.finished.emit()


def run_task(fn: Callable[..., Any], *args: Any,
             on_result: Optional[Callable[[Any], None]] = None,
             on_error: Optional[Callable[[str, str], None]] = None,
             on_progress: Optional[Callable[[int, int, str], None]] = None,
             on_cancelled: Optional[Callable[[], None]] = None,
             on_finished: Optional[Callable[[], None]] = None,
             **kwargs: Any) -> Task:
    """Start ``fn(context, *args, **kwargs)`` in the background and return the Task."""
    task = Task(fn, *args, **kwargs)
    if on_progress:
        task.signals.progress.connect(on_progress)
    if on_result:
        task.signals.succeeded.connect(on_result)
    if on_error:
        task.signals.failed.connect(on_error)
    if on_cancelled:
        task.signals.cancelled.connect(on_cancelled)
    if on_finished:
        task.signals.finished.connect(on_finished)
    task.signals.finished.connect(lambda: _RUNNING.discard(task))
    _RUNNING.add(task)
    QThreadPool.globalInstance().start(task)
    return task
