"""
Locating files the app needs at runtime: AI model weights, images and the
User Guide's screenshots.

The model weights (~650 MB) live in the original DicomDoodle folder, one
level above this app, so they are *found* rather than copied. Every lookup
searches, in order:

  1. extra folders passed by the caller (e.g. Preferences -> AI models folder)
  2. the PyInstaller bundle folder, when running as a frozen executable
  3. DicomDoodle_App/assets/
  4. DicomDoodle_App/
  5. the parent DicomDoodle/ folder (where the original tool keeps them)
  6. the current working directory
"""
import os
import sys
from typing import Iterable, List, Optional

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(PACKAGE_DIR)        # .../DicomDoodle/DicomDoodle_App
LEGACY_DIR = os.path.dirname(APP_DIR)         # .../DicomDoodle

SAM_WEIGHTS = "sam2.1_l.pt"
YOLO_WEIGHTS = "best_flair.pt"
YOLOE_WEIGHTS = "yoloe-11l-seg-pf.pt"
LOGO_IMAGE = "logo.png"
SPLASH_IMAGE = "Start.png"
WINDOW_ICON = "file.ico"

# Screenshots of the User Guide (made by scripts/make_guide_images.py)
GUIDE_DIR = os.path.join(PACKAGE_DIR, "assets", "guide")


def search_dirs(extra_dirs: Iterable[str] = ()) -> List[str]:
    """All folders searched for resources, most specific first."""
    dirs = [d for d in extra_dirs if d]
    if getattr(sys, "frozen", False):
        dirs.append(getattr(sys, "_MEIPASS", os.path.dirname(sys.executable)))
    dirs += [os.path.join(APP_DIR, "assets"), APP_DIR, LEGACY_DIR, os.getcwd()]

    unique, seen = [], set()
    for folder in dirs:
        key = os.path.normcase(os.path.abspath(folder))
        if key not in seen:
            seen.add(key)
            unique.append(folder)
    return unique


def find_resource(filename: str, extra_dirs: Iterable[str] = ()) -> Optional[str]:
    """Absolute path of ``filename`` in the first search folder that has it."""
    for folder in search_dirs(extra_dirs):
        candidate = os.path.join(folder, filename)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return None
