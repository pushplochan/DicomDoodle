"""
Self-test of an installed DicomDoodle, above all the Windows build: every
module imports, a DICOM SEG file is written and read back, and the DICOM code
dictionary, HTTPS, the model card's AI models and HD-BET's network code work.

    DicomDoodle.exe --self-test [REPORT]          (or: python main.py --self-test [REPORT])

The AI models come from the model card in the environment variable
DICOMDOODLE_MODEL_CARD (installed into a temporary folder, nothing of the
user's is touched), else from the installed card; without either that check
is skipped. The result goes to REPORT (default: printed); the exit code is 0
when no check failed. The Windows build workflow
(.github/workflows/windows-installer.yml) runs it on DicomDoodle.exe, with the
model card of the release model-card-1.0, before it makes the installer.
"""
from __future__ import annotations

import importlib
import os
import pkgutil
import platform
import shutil
import sys
import tempfile
import time
import traceback

import numpy as np

from . import APP_VERSION

MODEL_CARD_ENV = "DICOMDOODLE_MODEL_CARD"


class Skipped(Exception):
    """A check that cannot run here (reported, not counted as a failure)."""


def _modules() -> str:
    import dicomdoodle
    names = [info.name for info in pkgutil.walk_packages(dicomdoodle.__path__, "dicomdoodle.")
             if not info.name.endswith("__main__")]          # importing __main__ would start the app
    for name in names:
        importlib.import_module(name)
    return f"{len(names)} modules"


def _dicom_libraries() -> str:
    import gdcm  # noqa: F401  (compressed DICOM files: JPEG lossless and JPEG-LS)
    import highdicom
    import pydicom
    import SimpleITK  # noqa: F401
    import vtk
    from pydicom.pixels.decoders import JPEG2000LosslessDecoder, JPEGLosslessSV1Decoder, JPEGLSLosslessDecoder

    missing = [decoder.UID.name for decoder in (JPEGLosslessSV1Decoder, JPEGLSLosslessDecoder, JPEG2000LosslessDecoder)
               if not decoder.is_available]
    if missing:
        raise RuntimeError("no decoder for " + ", ".join(missing))
    return f"pydicom {pydicom.__version__}, highdicom {highdicom.__version__}, VTK {vtk.vtkVersion.GetVTKVersion()}"


def _synthetic_slice(path: str, size: int = 64) -> None:
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID = MRImageStorage, generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = MRImageStorage, meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = generate_uid(), generate_uid(), generate_uid()
    ds.Modality, ds.PatientName, ds.PatientID = "MR", "Self^Test", "SELFTEST"
    ds.InstanceNumber, ds.ImagePositionPatient, ds.ImageOrientationPatient = 1, [0.0, 0.0, 0.0], [1, 0, 0, 0, 1, 0]
    ds.PixelSpacing, ds.SliceThickness = [0.5, 0.5], 2.0
    ds.Rows = ds.Columns = size
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    ds.PixelData = np.arange(size * size, dtype=np.uint16).tobytes()
    ds.save_as(path, enforce_file_format=True)


def _seg_files() -> str:
    from .core.labels import LabelRegistry
    from .core.seg_io import read_segmentation, write_segmentation

    folder = tempfile.mkdtemp(prefix="dicomdoodle_selftest_")
    try:
        source, target = os.path.join(folder, "slice.dcm"), os.path.join(folder, "slice_seg.dcm")
        _synthetic_slice(source)
        label_map = np.zeros((64, 64), np.uint8)
        label_map[20:40, 20:40] = 1
        registry = LabelRegistry()
        write_segmentation(label_map, source, target, registry=registry)
        pixels = int(np.count_nonzero(read_segmentation(target, registry).label_map))
        if pixels == 0:
            raise RuntimeError("the SEG file read back empty")
        return f"wrote and read back a SEG file ({pixels} labelled pixels)"
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def _dicom_codes() -> str:
    from .core.coding import search_dicom

    terms = search_dicom("tumour")
    if not any("neoplasm" in term.label.lower() for term in terms):
        raise RuntimeError("'tumour' did not find Neoplasm")
    return f"'tumour' -> {len(terms)} DICOM terms"


def _https() -> str:
    import ssl

    import certifi

    context = ssl.create_default_context()         # BioPortal search (urllib): the system's certificates
    if not os.path.isfile(certifi.where()):         # HD-BET's download (requests): certifi's
        raise RuntimeError(f"certificate file missing: {certifi.where()}")
    return f"{ssl.OPENSSL_VERSION}; {len(context.get_ca_certs())} system certificates loaded"


def _ai_models() -> str:
    from .core.ai import model_card
    from .core.ai_models import ModelManager

    card = os.environ.get(MODEL_CARD_ENV, "")
    saved = os.environ.get(model_card.DATA_DIR_ENV)
    data = tempfile.mkdtemp(prefix="dicomdoodle_selftest_")
    if card:
        os.environ[model_card.DATA_DIR_ENV] = data           # install it there, not in the user's folder
    elif model_card.installed() is None:
        shutil.rmtree(data, ignore_errors=True)
        raise Skipped(f"no model card ({MODEL_CARD_ENV} not set, none installed): the AI is off")
    try:
        models = ModelManager("", "cpu")
        models.prepare(chosen=card, search=False)           # no card files sought: nothing is installed
        image = np.zeros((256, 256), np.uint8)
        image[96:160, 96:160] = 200                          # a bright square
        ran = []
        for kind in ("yolo", "yoloe"):
            if models.has(kind):
                models.predict(kind, image)
                ran.append(kind)
        mask = models.segment_box(image, (90, 90, 166, 166))
        parts = [f"{row.purpose} {row.title} ({row.engine})" for row in models.info() if row.loaded]
        if len(parts) < 3:
            raise RuntimeError(f"only {len(parts)} of the 3 models loaded: " + "; ".join(parts))
        return f"{models.card.label}: " + "; ".join(parts) + f" ran on the {models.device_name()} " \
               f"(SAM: {int(mask.sum())} pixels)"
    finally:
        if saved is None:
            os.environ.pop(model_card.DATA_DIR_ENV, None)
        else:
            os.environ[model_card.DATA_DIR_ENV] = saved
        shutil.rmtree(data, ignore_errors=True)


def _ai_runtime() -> str:
    """ONNX Runtime's version and providers, and the graphics card the AI could use (information)."""
    import onnxruntime as ort

    from .core.ai import devices

    gpu = devices.detect_gpu()
    card = gpu.describe() if gpu.available else (gpu.problem or "none")
    return (f"ONNX Runtime {ort.__version__} ({', '.join(p.replace('ExecutionProvider', '') for p in ort.get_available_providers())}); "
            f"graphics card: {card}")


def _hdbet() -> str:
    import pydoc

    import nnunetv2
    from HD_BET.checkpoint_download import maybe_download_parameters  # noqa: F401
    from HD_BET.hd_bet_prediction import get_hdbet_predictor  # noqa: F401
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class

    from .core import skull_strip

    if not skull_strip.HAVE_HDBET_PACKAGE:
        raise RuntimeError("HD_BET package not found")
    # nnU-Net finds these classes by name at run time (HD-BET 2.0.1's settings)
    root = nnunetv2.__path__[0]
    found = {
        "trainer": recursive_find_python_class(os.path.join(root, "training", "nnUNetTrainer"), "nnUNetTrainerDA5",
                                               "nnunetv2.training.nnUNetTrainer"),
        "preprocessor": recursive_find_python_class(os.path.join(root, "preprocessing"), "DefaultPreprocessor",
                                                    "nnunetv2.preprocessing"),
        "network": pydoc.locate("dynamic_network_architectures.architectures.unet.PlainConvUNet"),
    }
    missing = [name for name, cls in found.items() if cls is None]
    if missing:
        raise RuntimeError("nnU-Net classes not found: " + ", ".join(missing))
    return "HD-BET's network code is complete (its weights are downloaded on first use)"


def _cpp_runtime() -> str:
    """Windows: version and file of the C++ runtime (msvcp140.dll) the app loaded; PyTorch needs 14.40 or newer."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetModuleHandleW.restype = wintypes.HMODULE
    kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
    kernel32.GetModuleFileNameW.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    handle = kernel32.GetModuleHandleW("msvcp140.dll")
    if not handle:
        return "msvcp140.dll is not loaded"
    path = ctypes.create_unicode_buffer(1024)
    kernel32.GetModuleFileNameW(handle, path, 1024)
    version = ctypes.WinDLL("version")
    size = version.GetFileVersionInfoSizeW(path.value, None)
    data = ctypes.create_string_buffer(size)
    version.GetFileVersionInfoW(path.value, 0, size, data)
    fixed, length = ctypes.c_void_p(), wintypes.UINT()
    version.VerQueryValueW(data, "\\", ctypes.byref(fixed), ctypes.byref(length))
    words = ctypes.cast(fixed, ctypes.POINTER(wintypes.DWORD * 4)).contents   # signature, struct version, file version
    return f"{words[2] >> 16}.{words[2] & 0xFFFF}.{words[3] >> 16}.{words[3] & 0xFFFF} {path.value}"


CHECKS = [
    ("modules", _modules),
    ("DICOM libraries", _dicom_libraries),
    ("DICOM SEG files", _seg_files),
    ("DICOM code dictionary", _dicom_codes),
    ("secure connections (HTTPS)", _https),
    ("AI models", _ai_models),
    ("HD-BET", _hdbet),
]


def main(argv) -> int:
    from . import app  # noqa: F401  (on Windows it loads torch before PyQt5, as the app does)
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication
    qt_app = QApplication.instance() or QApplication(["DicomDoodle"])  # noqa: F841  (UI modules need one)

    lines = [f"DicomDoodle {APP_VERSION} self-test - Python {platform.python_version()}, {platform.platform()}, "
             f"{'Windows build' if getattr(sys, 'frozen', False) else 'source'}"]
    failed = 0
    for name, check in CHECKS:
        started = time.time()
        try:
            detail = check()
            lines.append(f"OK    {name}: {detail} ({time.time() - started:.1f} s)")
        except Skipped as reason:
            lines.append(f"SKIP  {name}: {reason}")
        except Exception:
            failed += 1
            lines.append(f"FAIL  {name}:\n{traceback.format_exc()}")
    try:
        lines.append(f"INFO  AI runtime: {_ai_runtime()}")
    except Exception as exc:                         # information only, never a failure
        lines.append(f"INFO  AI runtime: could not be read ({exc!r})")
    if sys.platform.startswith("win"):
        try:
            lines.append(f"INFO  C++ runtime: {_cpp_runtime()}")
        except Exception as exc:                     # information only, never a failure
            lines.append(f"INFO  C++ runtime: could not be read ({exc!r})")
    lines.append(f"{failed} check(s) FAILED" if failed else "All checks passed.")

    report = "\n".join(lines) + "\n"
    if argv:
        with open(argv[0], "w", encoding="utf-8") as fh:
            fh.write(report)
    else:
        print(report, end="")
    return 1 if failed else 0
