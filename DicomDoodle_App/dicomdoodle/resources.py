"""
Locating files the app needs at runtime: images and the User Guide's
screenshots. (The AI models come from the model card: core/ai/model_card.py.)

The logo images live in dicomdoodle/assets/. Every lookup searches, in order:

  1. extra folders passed by the caller
  2. the PyInstaller bundle folder, when running as a frozen executable
  3. dicomdoodle/assets/
  4. DicomDoodle_App/
  5. the current working directory
"""
import os
import sys
from typing import Iterable, List, Optional

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(PACKAGE_DIR)        # .../DicomDoodle/DicomDoodle_App
LEGACY_DIR = os.path.dirname(APP_DIR)         # .../DicomDoodle (the repository root)
ASSETS_DIR = os.path.join(PACKAGE_DIR, "assets")

LOGO_IMAGE = "logo.png"
SPLASH_IMAGE = "Start.png"
WINDOW_ICON = "file.ico"

# Screenshots of the User Guide (made by scripts/make_guide_images.py)
GUIDE_DIR = os.path.join(ASSETS_DIR, "guide")


def search_dirs(extra_dirs: Iterable[str] = ()) -> List[str]:
    """All folders searched for resources, most specific first."""
    dirs = [d for d in extra_dirs if d]
    if getattr(sys, "frozen", False):
        dirs.append(getattr(sys, "_MEIPASS", os.path.dirname(sys.executable)))
    dirs += [ASSETS_DIR, APP_DIR, os.getcwd()]

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
