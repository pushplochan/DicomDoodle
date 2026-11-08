"""
Background execution so the GUI never freezes.

    tasks.py  generic "run this function in a thread" helper with progress,
              cancellation and result/error signals delivered on the GUI thread
    jobs.py   the concrete long-running jobs of the annotator (load series,
              detect, segment, save, batch processing)
"""
from .tasks import Task, TaskContext, run_task  # noqa: F401
