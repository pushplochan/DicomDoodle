#!/usr/bin/env python3
"""
DicomDoodle — application entry point.

Run it from anywhere:

    python DicomDoodle_App/main.py

This file is deliberately tiny. It makes the ``dicomdoodle`` package
importable and hands over to :func:`dicomdoodle.app.run`, which builds the
Qt application, the splash screen and the main window.

It is also the entry point of the Windows build (DicomDoodle.exe, made by
packaging/DicomDoodle.spec), which has three more options:

    --hdbet-runner IMAGE MASK   skull stripping in its own process (core/skull_strip.py
                                starts the exe with it: there is no python.exe to run
                                core/hdbet_runner.py with)
    --self-test [REPORT]        checks that every module and AI model loads (dicomdoodle/selftest.py)
    --probe-gpu RESULT          tests in its own process whether the AI can run on the graphics card
                                (core/ai/devices.py starts it, so a failing driver cannot stop the app)
"""
import multiprocessing
import os
import sys

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)


def _prepare_frozen(helper: bool) -> None:
    """DicomDoodle.exe has no console: print() and progress bars go to a log
    file instead of nowhere (some libraries fail when sys.stdout is None),
    and Ultralytics must never try to pip-install packages. A helper process
    (``helper``: --hdbet-runner, --probe-gpu) never touches the app's log."""
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    if sys.stdout is not None and sys.stderr is not None:
        return
    from dicomdoodle.settings import config_dir
    try:
        if helper:
            raise OSError("a helper process")
        os.makedirs(config_dir(), exist_ok=True)
        stream = open(os.path.join(config_dir(), "dicomdoodle.log"), "w", encoding="utf-8", buffering=1)
    except OSError:
        stream = open(os.devnull, "w", encoding="utf-8")
    sys.stdout = sys.stdout or stream
    sys.stderr = sys.stderr or stream


def main() -> int:
    from dicomdoodle.core import hdbet_runner
    from dicomdoodle.core.ai import devices
    if getattr(sys, "frozen", False):
        _prepare_frozen(helper=sys.argv[1:2] in ([hdbet_runner.FLAG], [devices.PROBE_FLAG]))
    if sys.argv[1:2] == [hdbet_runner.FLAG]:
        return hdbet_runner.main(sys.argv[1:])
    if sys.argv[1:2] == [devices.PROBE_FLAG]:
        return devices.probe_main(sys.argv[2:])
    if sys.argv[1:2] == ["--self-test"]:
        from dicomdoodle.selftest import main as self_test
        return self_test(sys.argv[2:])
    from dicomdoodle.app import run
    return run(sys.argv)


if __name__ == "__main__":
    multiprocessing.freeze_support()      # DicomDoodle.exe: a library's worker process must not open the app again
    sys.exit(main())
