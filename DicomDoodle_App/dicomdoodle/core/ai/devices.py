"""
Where the AI runs: the processor (CPU, the default) or a graphics card (GPU).

The user chooses (Preferences ▸ AI Models, or the AI button of the status
bar); this module finds out which graphics card can be used, and how:

  * ONNX Runtime + DirectML  Windows 10/11, any DirectX 12 card (NVIDIA, AMD, Intel):
                             the Windows build's onnxruntime-directml package
  * ONNX Runtime + CUDA      NVIDIA, with the onnxruntime-gpu package
  * PyTorch + CUDA           NVIDIA, with a CUDA build of PyTorch (also used by HD-BET)

A runtime counts only after a tiny test network really ran on it: ONNX
Runtime lists CUDA even where its libraries are missing, and then quietly
runs on the CPU. That test runs in a separate process (``--probe-gpu``), so a
graphics driver that crashes cannot take the app down — the processor, the
default, never depends on the graphics card. Finding out takes a few seconds,
so detect_gpu() runs in the background and remembers its answer.

HD-BET (core/skull_strip.py) follows the same choice through torch_device():
it can use the graphics card only through PyTorch's CUDA — never in the
Windows build, whose PyTorch is CPU-only.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)

CPU, GPU = "cpu", "gpu"
DML, CUDA, CPU_PROVIDER = "DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"
PROVIDER_LABEL = {DML: "DirectML", CUDA: "CUDA"}
TORCH_DEVICE_ENV = "DICOMDOODLE_TORCH_DEVICE"     # tells hdbet_runner.py where to run ("cpu" / "cuda")
PROBE_FLAG = "--probe-gpu"                        # main.py's option that runs probe_main() (DicomDoodle.exe)
PROBE_TIMEOUT = 120                               # seconds: a first start of a graphics driver can be slow

# y = x + x, the smallest ONNX network: tests whether a runtime really works (made with the onnx package)
_PROBE = base64.b64decode("CAcSC2RpY29tZG9vZGxlOkEKDgoBeAoBeBIBeSIDQWRkEgVwcm9iZVoTCgF4Eg4KDAgBEggKAggBCgIIBGIT"
                          "CgF5Eg4KDAgBEggKAggBCgIIBEIECgAQDQ==")

# PCI vendors of real graphics cards, most powerful kind first (virtual and remote-desktop adapters have none)
_VENDORS = {"10DE": 0, "1002": 1, "1022": 1, "8086": 2, "5143": 3}


def normalize(device: Optional[str]) -> str:
    """The stored preference as CPU or GPU ("auto" of earlier versions means the CPU, the new default)."""
    return GPU if device in (GPU, "cuda") else CPU


@dataclass
class GpuInfo:
    names: List[str] = field(default_factory=list)   # graphics cards found (for display only)
    onnx_provider: str = ""                          # ONNX Runtime provider that really ran on a card
    torch_cuda: bool = False                         # PyTorch can use CUDA
    problem: str = ""                                # why no card can be used, in the user's words

    @property
    def available(self) -> bool:
        return bool(self.onnx_provider or self.torch_cuda)

    @property
    def name(self) -> str:
        return self.names[0] if self.names else "Graphics card"

    def describe(self) -> str:
        """e.g. 'NVIDIA GeForce RTX 3060 (DirectML)'."""
        via = PROVIDER_LABEL.get(self.onnx_provider) or ("CUDA" if self.torch_cuda else "")
        return f"{self.name} ({via})" if via else self.name


_lock = threading.Lock()
_found: Optional[GpuInfo] = None
_preferred = CPU
_background: Optional[threading.Thread] = None


def detect_gpu(refresh: bool = False) -> GpuInfo:
    """Which graphics card the AI can use (tests it the first time; the answer is kept)."""
    global _found
    with _lock:
        if _found is None or refresh:
            _found = _detect()
            log.info("Graphics card: %s", _found.describe() if _found.available else (_found.problem or "none"))
        return _found


def known_gpu() -> Optional[GpuInfo]:
    """detect_gpu()'s answer when it has run already; None otherwise (never waits)."""
    return _found


def detect_gpu_soon() -> None:
    """Start detect_gpu() in a background thread (for the menus), unless it ran or runs already."""
    global _background
    with _lock:
        if _found is not None or (_background is not None and _background.is_alive()):
            return
        _background = threading.Thread(target=detect_gpu, name="detect-gpu", daemon=True)
        _background.start()


def _detect() -> GpuInfo:
    info = GpuInfo(names=list(dict.fromkeys(gpu_names())))    # two cards of one kind: one name
    info.torch_cuda = _torch_cuda()
    info.onnx_provider = _onnx_gpu_provider()
    if info.torch_cuda and not info.names:
        try:
            import torch
            info.names = [torch.cuda.get_device_name(0)]
        except Exception:
            pass
    if not info.available:
        if not info.names:
            info.problem = "No graphics card was found."
        elif not _have_gpu_runtime():
            info.problem = (f"{info.names[0]} was found, but no graphics card runtime is installed "
                            "(onnxruntime-directml on Windows, onnxruntime-gpu or a CUDA PyTorch elsewhere).")
        else:
            info.problem = f"{info.names[0]} was found, but it could not run the AI (is its driver up to date?)."
    return info


def _torch_cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _have_gpu_runtime() -> bool:
    try:
        import onnxruntime as ort
        return bool({DML, CUDA} & set(ort.get_available_providers()))
    except Exception:
        return False


def _onnx_gpu_provider() -> str:
    """The ONNX Runtime provider that runs on a graphics card, tested in its own process."""
    try:
        import onnxruntime as ort
        if not {DML, CUDA} & set(ort.get_available_providers()):
            return ""                                  # a CPU-only ONNX Runtime: nothing to test
    except Exception:
        return ""
    handle, result = tempfile.mkstemp(prefix="dicomdoodle_gpu_", suffix=".json")
    os.close(handle)
    if getattr(sys, "frozen", False):                  # DicomDoodle.exe runs probe_main() with main.py's option
        command = [sys.executable, PROBE_FLAG, result]
    else:
        command = [sys.executable, os.path.abspath(__file__), PROBE_FLAG, result]
    try:
        subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=PROBE_TIMEOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        with open(result, encoding="utf-8") as fh:
            found = json.load(fh)
        for provider, problem in found.get("problems", {}).items():
            log.info("%s cannot be used: %s", PROVIDER_LABEL.get(provider, provider), problem)
        return str(found.get("provider") or "")
    except Exception as exc:                           # a crash, a timeout: no graphics card
        log.info("The graphics card test did not finish: %s", exc)
        return ""
    finally:
        try:
            os.remove(result)
        except OSError:
            pass


def probe_main(argv=None) -> int:
    """In its own process: which ONNX Runtime provider really runs on a graphics card; writes
    {"provider": ..., "problems": {...}} as JSON to the file given."""
    argv = sys.argv[1:] if argv is None else argv
    target = argv[-1]
    found = {"provider": "", "problems": {}}
    try:
        import numpy as np
        import onnxruntime as ort
        for provider in (DML, CUDA):
            if provider not in ort.get_available_providers():
                continue
            try:
                session, used = create_session(_PROBE, provider)
                if used == provider and np.allclose(session.run(None, {"x": np.ones((1, 4), np.float32)})[0], 2.0):
                    found["provider"] = provider
                    break
                found["problems"][provider] = "the test ran on the processor instead"
            except Exception as exc:
                found["problems"][provider] = str(exc).splitlines()[0][:300]
    except Exception as exc:
        found["problems"]["onnxruntime"] = str(exc)[:300]
    with open(target, "w", encoding="utf-8") as fh:
        json.dump(found, fh)
    return 0


# --------------------------------------------------------------------------- #
# ONNX Runtime sessions
# --------------------------------------------------------------------------- #

def onnx_provider(device: str) -> str:
    """The ONNX Runtime provider for ``device``: DirectML / CUDA for the GPU when one works, '' for
    the CPU (also while detect_gpu() has not run: ModelManager.prepare() runs it first)."""
    info = known_gpu()
    return info.onnx_provider if normalize(device) == GPU and info is not None else ""


def session_options(provider: str = ""):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3                    # errors only
    if provider == DML:                               # what DirectML requires
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if (os.cpu_count() or 1) > 64:                    # big servers: more threads only add overhead
        options.intra_op_num_threads = 32
    return options


def _providers(provider: str, with_options: bool) -> list:
    if provider == DML:                               # the most powerful card when a laptop has two
        return [(DML, {"performance_preference": "high_performance", "device_filter": "gpu"})
                if with_options else DML, CPU_PROVIDER]
    if provider == CUDA:                              # full float32 precision, as on the CPU
        return [(CUDA, {"use_tf32": "0"}) if with_options else CUDA, CPU_PROVIDER]
    return [CPU_PROVIDER]


def create_session(model, provider: str = "") -> Tuple[object, str]:
    """An ONNX Runtime session for ``model`` (path or bytes) on ``provider`` ('' = CPU), and the
    provider it really runs on. Provider options an older ONNX Runtime does not know are left out."""
    import onnxruntime as ort

    if not provider:
        return ort.InferenceSession(model, session_options(), providers=_providers("", False)), ""
    try:
        session = ort.InferenceSession(model, session_options(provider), providers=_providers(provider, True))
    except Exception:
        session = ort.InferenceSession(model, session_options(provider), providers=_providers(provider, False))
    used = session.get_providers()[0]
    return session, used if used != CPU_PROVIDER else ""


# --------------------------------------------------------------------------- #
# The device chosen in the preferences (HD-BET follows it)
# --------------------------------------------------------------------------- #

def set_preferred(device: str) -> None:
    global _preferred
    _preferred = normalize(device)


def preferred() -> str:
    return _preferred


def torch_device(device: Optional[str] = None) -> str:
    """'cuda' when the graphics card is chosen and PyTorch can use CUDA, else 'cpu'."""
    if normalize(_preferred if device is None else device) != GPU:
        return "cpu"
    return "cuda" if (known_gpu() or detect_gpu()).torch_cuda else "cpu"


# --------------------------------------------------------------------------- #
# Names of the graphics cards (display only)
# --------------------------------------------------------------------------- #

def gpu_names() -> List[str]:
    """The graphics cards of this computer, the most powerful kind first."""
    try:
        if sys.platform.startswith("win"):
            return _windows_gpu_names()
        return _nvidia_names()
    except Exception as exc:                          # names are a nicety: never fail over them
        log.debug("Could not list the graphics cards: %s", exc)
        return []


def _vendor(device_id: str) -> Optional[str]:
    text = device_id.upper()
    start = text.find("VEN_")
    return text[start + 4:start + 8] if start >= 0 else None


def _ranked(found: List[Tuple[str, str]]) -> List[str]:
    """(name, PCI id) pairs -> names of real cards, NVIDIA / AMD before Intel, without repeats."""
    seen, cards = set(), []
    for name, device_id in found:
        vendor = _vendor(device_id)
        name = name.strip()
        if name and vendor in _VENDORS and name.lower() not in seen:
            seen.add(name.lower())
            cards.append((_VENDORS[vendor], len(cards), name))
    return [name for _rank, _order, name in sorted(cards)]


def _windows_gpu_names() -> List[str]:
    import ctypes
    import winreg
    from ctypes import wintypes

    class DisplayDevice(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("DeviceName", wintypes.WCHAR * 32),
                    ("DeviceString", wintypes.WCHAR * 128), ("StateFlags", wintypes.DWORD),
                    ("DeviceID", wintypes.WCHAR * 128), ("DeviceKey", wintypes.WCHAR * 128)]

    found = []
    for index in range(32):                           # the adapters driving a screen
        device = DisplayDevice()
        device.cb = ctypes.sizeof(device)
        if not ctypes.windll.user32.EnumDisplayDevicesW(None, index, ctypes.byref(device), 0):
            break
        found.append((device.DeviceString, device.DeviceID))
    # every installed display adapter, also a laptop's second card that drives no screen
    path = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as display_class:
            for index in range(64):
                try:
                    with winreg.OpenKey(display_class, f"{index:04d}") as adapter:
                        name = winreg.QueryValueEx(adapter, "DriverDesc")[0]
                        device_id = winreg.QueryValueEx(adapter, "MatchingDeviceId")[0]
                except OSError:
                    continue
                found.append((str(name), str(device_id)))
    except OSError:
        pass
    return _ranked(found)


def _nvidia_names() -> List[str]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    result = subprocess.run([exe, "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                            text=True, timeout=10)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


if __name__ == "__main__":                            # the graphics card test, run by _onnx_gpu_provider()
    sys.exit(probe_main())
