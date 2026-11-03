"""
Segmentation labels: the names, colors and pixel values used for annotation.

Every annotation is stored as a *label map*: a uint8 image where 0 is
background and any other number is the ``value`` of a :class:`SegLabel`.
When a slice is saved, each value becomes one segment of the DICOM SEG file,
named after the label (DICOM ``SegmentLabel``).

DicomDoodle ships six built-in labels — values 1-6, identical to the original
tool, so existing ``_seg.dcm`` files keep their meaning. Users can add their
own labels with any color; those get values 7, 8, ... and are stored in the
settings file. Built-in labels cannot be edited or removed.

Each label also carries the coded concepts written with its segments
(DICOM Segmented Property Category and Type, see core/coding.py): the
built-in labels name what they outline (Neoplasm, Necrosis, Edema), other
labels are written as a Lesion until the Ontology Search gives them a term.
"""
from __future__ import annotations

import functools
import re
import threading
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .coding import EDEMA, LESION, MORPHOLOGICALLY_ABNORMAL_STRUCTURE, NECROSIS, NEOPLASM, Code

RGB = Tuple[int, int, int]

BACKGROUND = 0
FIRST_CUSTOM_VALUE = 7
MAX_LABEL_VALUE = 255
MAX_NAME_LENGTH = 64          # DICOM "LO" value-representation limit (SegmentLabel)
SIMILAR_COLOR_DISTANCE = 60   # RGB distance below which two colors look alike

# SegmentLabel texts that mean "nothing annotated" (used for blank masks).
EMPTY_SEGMENT_NAMES = frozenset({"no segmentation", "no boundary"})

# High-contrast colors offered as suggestions and used for labels found in
# files that are not in the registry.
DISTINCT_COLORS: Tuple[RGB, ...] = (
    (255, 140, 0), (148, 103, 189), (23, 190, 207), (188, 189, 34),
    (227, 119, 194), (140, 86, 75), (127, 255, 212), (220, 20, 60),
    (124, 252, 0), (255, 215, 0), (100, 149, 237), (255, 105, 180),
    (0, 128, 128), (244, 164, 96), (154, 205, 50), (186, 85, 211),
    (0, 191, 255), (255, 99, 71), (210, 180, 140), (60, 179, 113),
)


@dataclass(frozen=True)
class SegLabel:
    """One annotation label."""

    value: int                 # pixel value in the label map (1-255)
    name: str                  # written to DICOM as SegmentLabel
    color: RGB                 # display color
    short_name: str = ""       # compact text for buttons/lists (optional)
    builtin: bool = False      # shipped with DicomDoodle, read-only
    transient: bool = False    # found in a file during this session, not saved
    category: Code = MORPHOLOGICALLY_ABNORMAL_STRUCTURE    # DICOM Segmented Property Category
    property_type: Code = LESION                           # DICOM Segmented Property Type

    @property
    def display_name(self) -> str:
        return self.short_name or self.name

    @property
    def hex(self) -> str:
        return "#{:02X}{:02X}{:02X}".format(*self.color)

    @property
    def editable(self) -> bool:
        return not self.builtin

    def to_dict(self) -> dict:
        return {"value": self.value, "name": self.name,
                "short_name": self.short_name, "color": self.hex}


# The original six DicomDoodle labels (utils.create_dicom_segmentation).
DEFAULT_LABELS: Tuple[SegLabel, ...] = (
    SegLabel(1, "Tumor & Brain Boundary", (255, 0, 0), "Tumor | Brain", builtin=True, property_type=NEOPLASM),
    SegLabel(2, "Tumor & Dura Boundary", (0, 255, 0), "Tumor | Dura", builtin=True, property_type=NEOPLASM),
    SegLabel(3, "Tumor & Necrosis Boundary", (0, 0, 255), "Tumor | Necrosis", builtin=True, property_type=NECROSIS),
    SegLabel(4, "Edema & Brain Boundary", (255, 255, 0), "Edema | Brain", builtin=True, property_type=EDEMA),
    SegLabel(5, "Tumor & Edema Boundary", (0, 255, 255), "Tumor | Edema", builtin=True, property_type=NEOPLASM),
    SegLabel(6, "Magenta Boundary", (255, 0, 255), "Magenta", builtin=True),
)


class LabelError(ValueError):
    """Raised when a label cannot be created or edited (bad name, color ...)."""


ColorLike = Union[str, Sequence[int]]


def parse_color(value: ColorLike) -> RGB:
    """Accept '#RRGGBB', 'RRGGBB' or an (r, g, b) sequence."""
    if isinstance(value, str):
        text = value.strip().lstrip("#")
        if not re.fullmatch(r"[0-9a-fA-F]{6}", text):
            raise LabelError(f"Invalid color: {value!r}")
        return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)
    rgb = tuple(int(c) for c in value)
    if len(rgb) != 3 or any(c < 0 or c > 255 for c in rgb):
        raise LabelError(f"Invalid color: {value!r}")
    return rgb  # type: ignore[return-value]


def color_distance(a: RGB, b: RGB) -> float:
    return float(np.sqrt(sum((int(x) - int(y)) ** 2 for x, y in zip(a, b))))


def fallback_color(value: int) -> RGB:
    """Deterministic color for a value that has no label."""
    return DISTINCT_COLORS[value % len(DISTINCT_COLORS)]


def _clean_name(name: str) -> str:
    return " ".join(str(name).split())


def _locked(method):
    """Run a registry method under its lock (the registry is shared with worker threads)."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper


class LabelRegistry:
    """The built-in labels plus the user's own; the single source of truth."""

    def __init__(self, custom: Iterable[Union[dict, SegLabel]] = ()):
        self._lock = threading.RLock()
        self._labels: Dict[int, SegLabel] = {lab.value: lab for lab in DEFAULT_LABELS}
        for item in custom:
            try:
                self._insert_custom(item)
            except (LabelError, KeyError, TypeError, ValueError):
                continue  # skip a corrupt entry rather than refusing to start

    # ------------------------------------------------------------- queries
    @_locked
    def labels(self, include_transient: bool = True) -> List[SegLabel]:
        return [lab for _, lab in sorted(self._labels.items())
                if include_transient or not lab.transient]

    def custom_labels(self) -> List[SegLabel]:
        return [lab for lab in self.labels() if not lab.builtin and not lab.transient]

    @_locked
    def get(self, value: int) -> Optional[SegLabel]:
        return self._labels.get(int(value))

    @_locked
    def __contains__(self, value: int) -> bool:
        return int(value) in self._labels

    @_locked
    def find_by_name(self, name: str) -> Optional[SegLabel]:
        key = _clean_name(name).casefold()
        for lab in self._labels.values():
            if lab.name.casefold() == key:
                return lab
        return None

    def name_for(self, value: int) -> str:
        lab = self.get(value)
        return lab.name if lab else f"Label {int(value)}"

    def color_for(self, value: int) -> RGB:
        lab = self.get(value)
        return lab.color if lab else fallback_color(int(value))

    @_locked
    def value_for_segment(self, segment_label: Optional[str], create: bool = True) -> int:
        """Pixel value for a DICOM ``SegmentLabel`` read from a file.

        Returns 0 for "No Segmentation"-style labels. Unknown names get a
        session-only (transient) label so they are still displayed, unless
        ``create`` is False, in which case 0 is returned.
        """
        if not segment_label or _clean_name(segment_label).casefold() in EMPTY_SEGMENT_NAMES:
            return BACKGROUND
        lab = self.find_by_name(segment_label)
        if lab is not None:
            return lab.value
        if not create:
            return BACKGROUND
        return self.add_transient(segment_label).value

    @_locked
    def next_free_value(self) -> int:
        for value in range(FIRST_CUSTOM_VALUE, MAX_LABEL_VALUE + 1):
            if value not in self._labels:
                return value
        raise LabelError("All 255 label values are in use.")

    @_locked
    def similar_labels(self, color: RGB, exclude_value: Optional[int] = None,
                       threshold: float = SIMILAR_COLOR_DISTANCE) -> List[SegLabel]:
        return [lab for lab in self.labels()
                if lab.value != exclude_value and color_distance(lab.color, color) < threshold]

    def suggested_colors(self, count: int = 12) -> List[RGB]:
        """Distinct colors that are not already (nearly) used by a label."""
        free = [c for c in DISTINCT_COLORS if not self.similar_labels(c)]
        return free[:count]

    @_locked
    def color_lut(self) -> np.ndarray:
        """(256, 3) uint8 lookup table: label value -> RGB (0 stays black)."""
        lut = np.zeros((MAX_LABEL_VALUE + 1, 3), dtype=np.uint8)
        for value in range(1, MAX_LABEL_VALUE + 1):
            lut[value] = fallback_color(value)
        for lab in self._labels.values():
            lut[lab.value] = lab.color
        return lut

    # ----------------------------------------------------------- validation
    @_locked
    def validate(self, name: str, color: ColorLike,
                 exclude_value: Optional[int] = None) -> Tuple[str, RGB]:
        """Check a name/color pair; returns the cleaned values or raises LabelError."""
        clean = _clean_name(name)
        if not clean:
            raise LabelError("Please enter a label name.")
        if len(clean) > MAX_NAME_LENGTH:
            raise LabelError(f"Label names can have at most {MAX_NAME_LENGTH} characters "
                             f"(DICOM limit); this one has {len(clean)}.")
        if "\\" in clean or any(ord(ch) < 32 for ch in clean):
            raise LabelError("Label names cannot contain backslashes or control characters.")
        if clean.casefold() in EMPTY_SEGMENT_NAMES:
            raise LabelError(f"'{clean}' is reserved for blank masks.")
        existing = self.find_by_name(clean)
        if existing is not None and existing.value != exclude_value:
            raise LabelError(f"A label named '{existing.name}' already exists.")

        rgb = parse_color(color)
        if rgb == (0, 0, 0):
            raise LabelError("Black is reserved for the background; please pick another color.")
        for lab in self._labels.values():
            if lab.value != exclude_value and lab.color == rgb:
                raise LabelError(f"'{lab.display_name}' already uses this exact color.")
        return clean, rgb

    # ------------------------------------------------------------- editing
    @_locked
    def add(self, name: str, color: ColorLike, short_name: str = "") -> SegLabel:
        clean, rgb = self.validate(name, color)
        label = SegLabel(self.next_free_value(), clean, rgb, _clean_name(short_name))
        self._labels[label.value] = label
        return label

    @_locked
    def update(self, value: int, name: Optional[str] = None, color: Optional[ColorLike] = None,
               short_name: Optional[str] = None) -> SegLabel:
        current = self._require_editable(value)
        clean, rgb = self.validate(current.name if name is None else name,
                                   current.color if color is None else color,
                                   exclude_value=current.value)
        updated = replace(current, name=clean, color=rgb,
                          short_name=current.short_name if short_name is None else _clean_name(short_name))
        self._labels[value] = updated
        return updated

    @_locked
    def remove(self, value: int) -> SegLabel:
        label = self._require_editable(value)
        del self._labels[label.value]
        return label

    @_locked
    def add_transient(self, name: str) -> SegLabel:
        """Register a label seen in a file so it can be shown (not persisted)."""
        clean = _clean_name(name)[:MAX_NAME_LENGTH] or "Unnamed"
        existing = self.find_by_name(clean)
        if existing is not None:
            return existing
        value = self.next_free_value()
        label = SegLabel(value, clean, fallback_color(value), transient=True)
        self._labels[value] = label
        return label

    @_locked
    def promote(self, value: int) -> SegLabel:
        """Turn a transient label into a regular custom label."""
        label = self._require_editable(value)
        promoted = replace(label, transient=False)
        self._labels[value] = promoted
        return promoted

    # --------------------------------------------------------- persistence
    @_locked
    def to_dicts(self) -> List[dict]:
        return [lab.to_dict() for lab in self.custom_labels()]

    def _insert_custom(self, item: Union[dict, SegLabel]) -> None:
        if isinstance(item, SegLabel):
            item = item.to_dict()
        value = int(item["value"])
        if value < FIRST_CUSTOM_VALUE or value > MAX_LABEL_VALUE or value in self._labels:
            raise LabelError(f"Invalid or duplicate label value {value}")
        clean, rgb = self.validate(item["name"], item["color"])
        self._labels[value] = SegLabel(value, clean, rgb, _clean_name(item.get("short_name", "")))

    def _require_editable(self, value: int) -> SegLabel:
        label = self.get(value)
        if label is None:
            raise LabelError(f"No label with value {value}.")
        if label.builtin:
            raise LabelError(f"'{label.display_name}' is a built-in label and cannot be changed.")
        return label
