"""Drawing label slices over grayscale MPR slices, and overlap scores."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .segmentation import SegmentInfo


def gray_to_rgb(gray_uint8: np.ndarray) -> np.ndarray:
    return np.stack([gray_uint8] * 3, axis=-1)


def blend_label_overlay(rgb_uint8: np.ndarray, label_slice: Optional[np.ndarray],
                        segments: Dict[int, SegmentInfo], alpha: float = 0.45) -> np.ndarray:
    """Translucent fill of every labelled region in its segment color."""
    if label_slice is None or not np.any(label_slice):
        return rgb_uint8
    out = rgb_uint8.astype(np.float32).copy()
    for number, info in segments.items():
        mask = label_slice == number
        if np.any(mask):
            out[mask] = (1 - alpha) * out[mask] + alpha * np.array(info.color_rgb, dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def blend_label_boundary(rgb_uint8: np.ndarray, label_slice: Optional[np.ndarray],
                         segments: Dict[int, SegmentInfo], thickness: int = 2, aspect: float = 1.0) -> np.ndarray:
    """Outline of every labelled region (per segment, so touching labels keep their edges).

    ``aspect`` is the slice's on-screen pixel aspect (row / column spacing,
    see display_aspect). On a reformat of thick slices a row is drawn several
    times taller than a column is wide, so the edge is widened only along the
    finer axis and stays about equally wide on screen all round (before, the
    top and bottom of every region became thick bars).
    """
    if label_slice is None or not np.any(label_slice):
        return rgb_uint8
    from scipy.ndimage import binary_dilation, binary_erosion
    out = rgb_uint8.copy()
    structure = np.ones((3, 3), dtype=bool)
    grow_rows = int(round((thickness - 1) / max(aspect, 1.0)))
    grow_cols = int(round((thickness - 1) * min(aspect, 1.0)))
    for number in np.unique(label_slice):
        if number == 0:
            continue
        mask = label_slice == number
        edge = mask & ~binary_erosion(mask, structure=structure, border_value=0)
        if grow_rows == grow_cols == thickness - 1 > 0:
            edge = binary_dilation(edge, iterations=thickness - 1)
        else:
            if grow_rows > 0:
                edge = binary_dilation(edge, structure=np.ones((3, 1), bool), iterations=grow_rows)
            if grow_cols > 0:
                edge = binary_dilation(edge, structure=np.ones((1, 3), bool), iterations=grow_cols)
        info = segments.get(int(number))
        out[edge] = info.color_rgb if info else (255, 0, 0)
    return out


def dice_coefficient(a: np.ndarray, b: np.ndarray) -> float:
    """2|A∩B| / (|A|+|B|) over all labels (NaN when both are empty)."""
    a, b = a > 0, b > 0
    denominator = a.sum() + b.sum()
    return float("nan") if denominator == 0 else float(2 * np.logical_and(a, b).sum() / denominator)


def iou_coefficient(a: np.ndarray, b: np.ndarray) -> float:
    """|A∩B| / |A∪B| over all labels (NaN when both are empty)."""
    a, b = a > 0, b > 0
    union = np.logical_or(a, b).sum()
    return float("nan") if union == 0 else float(np.logical_and(a, b).sum() / union)
