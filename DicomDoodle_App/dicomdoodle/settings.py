"""
Persistent user preferences, stored as a small human-readable JSON file:

    Windows : %APPDATA%\\DicomDoodle\\settings.json
    macOS   : ~/Library/Application Support/DicomDoodle/settings.json
    Linux   : ~/.config/DicomDoodle/settings.json

Only plain JSON types are stored. Missing keys fall back to ``DEFAULTS`` and
a corrupt file is renamed to ``settings.json.bak`` instead of crashing.
"""
import json
import os
import sys
from typing import Any, Dict, Optional

from . import APP_NAME

DEFAULTS: Dict[str, Any] = {
    # Appearance
    "theme": "dark",                     # "dark" | "light"
    # Labels
    "custom_labels": [],                 # user-defined labels, see core.labels
    "active_label": 1,
    # Files
    "recent_folders": [],
    "last_dir": "",
    "confirm_overwrite": True,           # ask before replacing older _seg.dcm files
    "auto_blank_masks": True,            # each save gives every slice without a _seg.dcm a blank mask
    # Annotation display
    "overlay_mode": "boundary",          # "boundary" | "filled"
    "line_width": 2,
    "fill_opacity": 0.35,
    "show_saved_overlay": True,          # show _seg.dcm already on disk under the working annotation
    # Drawing by hand
    "pen_size": 1,                       # pixels
    "eraser_size": 5,
    # AI models: the model card (core/ai/model_card.py)
    "models_dir": "",                    # another folder searched for model cards
    "device": "cpu",                     # "cpu" | "gpu": where the AI runs ("auto"/"cuda" of v2.0 = cpu/gpu)
    "preload_sam": True,                 # load the AI models in the background at startup
    "find_model_cards": True,            # look for model cards at start (off after Remove, until one is loaded)
    "gpu_hint_shown": False,             # the "graphics card found" note was shown once
    # Ontology
    "bioportal_api_key": "f46b5f95-b7f6-423a-a31e-111385f140ba",
    # Window layout (base64-encoded Qt state)
    "window_geometry": "",
    "window_state": "",
}

MAX_RECENT_FOLDERS = 8


def config_dir() -> str:
    """Platform-appropriate folder for DicomDoodle's settings file."""
    if sys.platform.startswith("win"):
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(base, APP_NAME)


class Settings:
    """Dictionary-like access to the preferences file; saves on every change."""

    def __init__(self, path: Optional[str] = None):
        self.path = path or os.path.join(config_dir(), "settings.json")
        self._values: Dict[str, Any] = {}
        self._load()

    # ------------------------------------------------------------------ access
    def get(self, key: str, default: Any = None) -> Any:
        if key in self._values:
            return self._values[key]
        if default is not None:
            return default
        return DEFAULTS.get(key)

    def set(self, key: str, value: Any, save: bool = True) -> None:
        self._values[key] = value
        if save:
            self.save()

    def update(self, **values: Any) -> None:
        self._values.update(values)
        self.save()

    def add_recent_folder(self, folder: str) -> None:
        folder = os.path.abspath(folder)
        recent = [f for f in self.get("recent_folders", []) if os.path.abspath(f) != folder]
        recent.insert(0, folder)
        self._values["recent_folders"] = recent[:MAX_RECENT_FOLDERS]
        self._values["last_dir"] = folder
        self.save()

    # ----------------------------------------------------------- persistence
    def _load(self) -> None:
        if not os.path.isfile(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                self._values = data
        except (OSError, ValueError):
            try:
                os.replace(self.path, self.path + ".bak")
            except OSError:
                pass

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp_path = self.path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(self._values, fh, indent=2)
            os.replace(tmp_path, self.path)
        except OSError:
            pass  # preferences are a convenience; never crash the app over them
