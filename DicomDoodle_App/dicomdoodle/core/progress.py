"""
Shared vocabulary for long-running operations.

Core functions that may take a while (loading a series, batch saving,
volume calculation ...) accept an optional :class:`Progress`. The GUI passes
one that forwards to a progress bar and knows whether Cancel was pressed;
scripts can simply omit it.
"""
from typing import Callable, Optional

ReportFn = Callable[[int, int, str], None]
CancelFn = Callable[[], bool]


class OperationCancelled(Exception):
    """Raised inside a long operation once the user has pressed Cancel."""


class Progress:
    """Bundles a progress reporter and a cancellation check."""

    def __init__(self, report: Optional[ReportFn] = None, is_cancelled: Optional[CancelFn] = None):
        self._report = report
        self._is_cancelled = is_cancelled

    def __call__(self, done: int, total: int, message: str = "") -> None:
        """Report progress, raising :class:`OperationCancelled` if cancelled."""
        self.check()
        if self._report is not None:
            self._report(done, total, message)

    @property
    def cancelled(self) -> bool:
        return bool(self._is_cancelled and self._is_cancelled())

    def check(self) -> None:
        if self.cancelled:
            raise OperationCancelled()


def ensure_progress(progress: Optional[Progress]) -> Progress:
    """Return ``progress`` or a do-nothing instance, so callers never test for None."""
    return progress if progress is not None else Progress()
