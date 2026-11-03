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

a = Analysis(
    [os.path.join(APP_DIR, "main.py")],
    pathex=[APP_DIR],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "PyQt6", "PySide2", "PySide6", "IPython", "notebook", "tensorboard", "tensorflow"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="DicomDoodle",
          console=False, icon=os.path.join(REPO_DIR, "file.ico"), upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="DicomDoodle", upx=False)
