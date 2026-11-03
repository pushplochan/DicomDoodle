# PyInstaller recipe for DicomDoodle.exe, the Windows build. The build workflow
# (.github/workflows/windows-installer.yml) runs, from the repository root:
#
#     pyinstaller --noconfirm --distpath build/dist --workpath build/work DicomDoodle_App/packaging/DicomDoodle.spec
#
# and gets build/dist/DicomDoodle/ (DicomDoodle.exe + _internal/), which
# DicomDoodle.iss packs into the installer. The same recipe builds on Linux, for testing.
#
# The three AI model files are bundled from the folder named by the environment
# variable DICOMDOODLE_MODELS (default: the repository root, where they sit locally).
import os
import re
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

APP_DIR = os.path.dirname(SPECPATH)                    # DicomDoodle_App/  (SPECPATH = this file's folder)
REPO_DIR = os.path.dirname(APP_DIR)
MODELS_DIR = os.environ.get("DICOMDOODLE_MODELS") or REPO_DIR
MODELS = ["best_flair.pt", "yoloe-11l-seg-pf.pt", "sam2.1_l.pt"]

missing = [name for name in MODELS if not os.path.isfile(os.path.join(MODELS_DIR, name))]
if missing:
    raise SystemExit(f"AI model files missing in {MODELS_DIR}: {', '.join(missing)}")
sys.path.insert(0, APP_DIR)                            # lets collect_submodules find the dicomdoodle package

datas = [(os.path.join(APP_DIR, "dicomdoodle", "assets"), os.path.join("dicomdoodle", "assets"))]
datas += [(os.path.join(REPO_DIR, name), ".") for name in ("file.ico", "logo.png", "Start.png")]
datas += [(os.path.join(MODELS_DIR, name), ".") for name in MODELS]
datas += collect_data_files("ultralytics")             # model settings (cfg/*.yaml)
datas += collect_data_files("highdicom")               # tables from the DICOM standard, ICC profile
datas += collect_data_files("pydicom")                 # palettes, charsets; `import pydicom` opens its examples

hiddenimports = (collect_submodules("dicomdoodle")     # every module, so --self-test sees them all
                 + collect_submodules("pydicom")       # pixel-data decoders are imported by name
                 + collect_submodules("nnunetv2")      # HD-BET: nnU-Net finds its classes by name
                 + collect_submodules("dynamic_network_architectures")
                 + ["gdcm"])                           # pydicom's decoder for JPEG lossless / JPEG-LS

CPP_RUNTIME = re.compile(r"(msvcp140(_\w+)?|vcruntime140(_\d)?|concrt140|vcomp140)\.dll", re.IGNORECASE)


def dll_version(path):
    """File version of a Windows DLL, e.g. (14, 44, 35211, 0); (0, 0, 0, 0) without one."""
    import pefile                                      # comes with PyInstaller on Windows
    pe = pefile.PE(path, fast_load=True)
    pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_RESOURCE"]])
    info = getattr(pe, "VS_FIXEDFILEINFO", None)
    pe.close()
    if not info:
        return (0, 0, 0, 0)
    info = info[0] if isinstance(info, list) else info
    return (info.FileVersionMS >> 16, info.FileVersionMS & 0xFFFF, info.FileVersionLS >> 16, info.FileVersionLS & 0xFFFF)


def newest_cpp_runtime(binaries, extra_dirs=()):
    """Microsoft's C++ runtime (msvcp140.dll, ...) comes with several packages: PyQt5's (Qt 5.15.2)
    is 14.26, too old for PyTorch, whose c10.dll then fails to load ("WinError 1114"). Every copy
    becomes the newest version found in the packages or in ``extra_dirs`` (this PC's System32),
    and one copy goes to the top folder, so whichever Windows loads first is the newest."""
    found = {}
    for dest, source, _ in binaries:
        name = os.path.basename(dest).lower()
        if CPP_RUNTIME.fullmatch(name):
            found.setdefault(name, []).append(source)
    newest = {}
    for name, sources in found.items():
        candidates = sources + [os.path.join(d, name) for d in extra_dirs if os.path.isfile(os.path.join(d, name))]
        newest[name] = max(candidates, key=dll_version)
        print(f"C++ runtime: {name} {'.'.join(map(str, dll_version(newest[name])))} from {newest[name]}")
    result = [(dest, newest.get(os.path.basename(dest).lower(), source), kind) for dest, source, kind in binaries]
    top_level = {dest.lower() for dest, _, _ in result}
    return result + [(name, path, "BINARY") for name, path in newest.items() if name not in top_level]


a = Analysis(
    [os.path.join(APP_DIR, "main.py")],
    pathex=[APP_DIR],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "PyQt6", "PySide2", "PySide6", "IPython", "notebook", "tensorboard", "tensorflow"],
)
if sys.platform == "win32":
    a.binaries = newest_cpp_runtime(a.binaries, [os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="DicomDoodle",
          console=False, icon=os.path.join(REPO_DIR, "file.ico"), upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="DicomDoodle", upx=False)
