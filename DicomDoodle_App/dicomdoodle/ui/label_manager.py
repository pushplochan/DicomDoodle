"""
The application-wide label list, shared by every window.

Wraps :class:`core.labels.LabelRegistry` with Qt signals and saves user
labels to the settings file, so a label added in one window immediately
shows up in every other window (and survives a restart).
"""
from __future__ import annotations

import json
from typing import List, Optional

from PyQt5.QtCore import QObject, pyqtSignal

from ..core.labels import LabelError, LabelRegistry, SegLabel
from ..settings import Settings


class LabelManager(QObject):
    labelsChanged = pyqtSignal()

    def __init__(self, settings: Settings, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._settings = settings
        self.registry = LabelRegistry(settings.get("custom_labels", []))

    # -------------------------------------------------------------- queries
    def labels(self) -> List[SegLabel]:
        return self.registry.labels()

    def get(self, value: int) -> Optional[SegLabel]:
        return self.registry.get(value)

    def snapshot(self) -> dict:
        """{value: SegLabel} copy (labels are immutable) that background jobs can use safely."""
        return {lab.value: lab for lab in self.registry.labels()}

    # -------------------------------------------------------------- editing
    def add(self, name: str, color, short_name: str = "") -> SegLabel:
        label = self.registry.add(name, color, short_name)
        self._changed()
        return label

    def update(self, value: int, name: str, color, short_name: str = "") -> SegLabel:
        label = self.registry.update(value, name=name, color=color, short_name=short_name)
        self._changed()
        return label

    def remove(self, value: int) -> SegLabel:
        label = self.registry.remove(value)
        self._changed()
        return label

    def promote(self, value: int) -> SegLabel:
        label = self.registry.promote(value)
        self._changed()
        return label

    def notify_transient_added(self) -> None:
        """Call after reading files that may have introduced new label names."""
        self.labelsChanged.emit()

    # --------------------------------------------------------- import/export
    def export_to(self, path: str) -> int:
        labels = self.registry.to_dicts()
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"dicomdoodle_labels": labels}, fh, indent=2)
        return len(labels)

    def import_from(self, path: str) -> int:
        """Add labels from a file exported by another user; returns the count added."""
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        items = data.get("dicomdoodle_labels", data) if isinstance(data, dict) else data
        added = 0
        for item in items:
            try:
                existing = self.registry.find_by_name(item["name"])
                if existing is None:
                    self.registry.add(item["name"], item["color"], item.get("short_name", ""))
                    added += 1
            except (LabelError, KeyError, TypeError):
                continue
        if added:
            self._changed()
        return added

    # ------------------------------------------------------------- internal
    def _changed(self) -> None:
        self._settings.set("custom_labels", self.registry.to_dicts())
        self.labelsChanged.emit()
