"""
Skull stripping with HD-BET (Isensee et al. 2019) — the one volume-processing
step of the Multiplanar & 3D Viewer (it replaced the classical skull strip,
z-score and connected components of the original volume processing).

:func:`brain_mask` gives the brain mask of an image. The viewer computes it
once per scan and applies it (:func:`apply_mask`) to every grid in the same
space: the 2D views after resampling, the registered second series, the 3D
model.

HD-BET runs in a separate Python process (hdbet_runner.py), so Cancel stops
it at once and its PyTorch settings never reach the app. Without HD-BET's
Python package, the ``hd-bet`` command is run instead. About 40 s per series
on a CPU (166 slices of 512 x 512); much faster with a graphics card, which it
uses when the preferences run the AI on the graphics card and PyTorch can use
CUDA (core/ai/devices.py: torch_device) — never in the Windows build, whose
PyTorch is CPU-only.
"""
from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Optional

import numpy as np

from . import hdbet_runner
from .ai import devices
from .progress import OperationCancelled, Progress, ensure_progress

HAVE_HDBET_PACKAGE = importlib.util.find_spec("HD_BET") is not None
HDBET_COMMAND = shutil.which("hd-bet")
HAVE_HDBET = HAVE_HDBET_PACKAGE or HDBET_COMMAND is not None
INSTALL_HINT = "pip install hd-bet"
RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hdbet_runner.py")

_TILES = re.compile(r"(\d+)/(\d+) \[")        # nnU-Net's progress bar: one step per image tile


class SkullStripError(RuntimeError):
    """HD-BET is not installed or failed."""


def runner_command(source: str, target: str) -> list:
    """The command that runs hdbet_runner.py on ``source``. The Windows build
    (DicomDoodle.exe, made by PyInstaller) has no python.exe to run the file
    with, so it starts itself with main.py's --hdbet-runner option."""
    if getattr(sys, "frozen", False):
        return [sys.executable, hdbet_runner.FLAG, source, target]
    return [sys.executable, RUNNER, source, target]


def brain_mask(image, progress: Optional[Progress] = None):
    """HD-BET's brain mask of ``image`` (SimpleITK): uint8, 1 = brain, on the
    same grid. Raises SkullStripError, or OperationCancelled once Cancel was
    pressed (HD-BET is stopped)."""
    import SimpleITK as sitk

    if not HAVE_HDBET:
        raise SkullStripError(f"HD-BET is not installed. Install it with: {INSTALL_HINT}")
    step = ensure_progress(progress)
    with tempfile.TemporaryDirectory(prefix="dicomdoodle_hdbet_") as work:
        source, log_path = os.path.join(work, "image.nii.gz"), os.path.join(work, "hdbet.log")
        step(0, 0, "Preparing the scan for HD-BET…")
        sitk.WriteImage(image, source)
        if HAVE_HDBET_PACKAGE:
            target = os.path.join(work, "mask.nii.gz")
            command = runner_command(source, target)
        else:                                 # the command writes the mask as <output>_bet.nii.gz
            target = os.path.join(work, "brain_bet.nii.gz")
            device = devices.torch_device()
            command = [HDBET_COMMAND, "-i", source, "-o", os.path.join(work, "brain.nii.gz"), "-device", device,
                       "--save_bet_mask", "--no_bet_image"] + (["--disable_tta"] if device == "cpu" else [])
        started = time.time()
        env = dict(os.environ, **{devices.TORCH_DEVICE_ENV: devices.torch_device()})   # CPU unless chosen
        with open(log_path, "w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=work, env=env)
        try:
            while process.poll() is None:
                done, total = _tiles(log_path)
                step(done, total, f"HD-BET is removing the skull… {time.time() - started:.0f} s")
                time.sleep(0.5)
        except OperationCancelled:
            _stop(process)
            raise
        if process.returncode != 0 or not os.path.isfile(target):
            with open(log_path, encoding="utf-8", errors="replace") as log:
                tail = log.read()[-1500:]
            raise SkullStripError(f"HD-BET stopped with exit code {process.returncode}.\n\n{tail}")
        mask = sitk.Cast(sitk.ReadImage(target) > 0, sitk.sitkUInt8)
    if mask.GetSize() == image.GetSize():
        mask.CopyInformation(image)           # exactly the input grid (the file round trip rounds it)
    return mask


def apply_mask(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The image inside the brain mask, 0 outside (as HD-BET's own output)."""
    return np.where(mask > 0, array, 0).astype(array.dtype, copy=False)


def _tiles(log_path: str):
    """(done, total) of nnU-Net's progress bar in HD-BET's output so far; (0, 0) before it starts."""
    try:
        with open(log_path, encoding="utf-8", errors="replace") as log:
            found = _TILES.findall(log.read()[-400:])
    except OSError:
        return 0, 0
    return tuple(int(v) for v in found[-1]) if found else (0, 0)


def _stop(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
