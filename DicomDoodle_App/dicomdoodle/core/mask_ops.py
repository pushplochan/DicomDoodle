"""
Label-map helpers: boundaries for saving, contours for display, overlays.

A *label map* is a uint8 array (rows, cols) where 0 is background and other
values are label values (see core.labels).
"""
from typing import Dict, List, Optional

import cv2
import numpy as np

# Colors used by the "filled regions" overlay of the Annotated DICOM Viewer
# (identical list to the original "Show Mask" mode).
REGION_COLORS = [
    (255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 0, 0),
    (0, 165, 255), (128, 0, 128), (255, 255, 255), (0, 0, 255), (255, 192, 203),
    (139, 69, 19), (128, 128, 128), (0, 128, 128), (255, 215, 0), (75, 0, 130),
    (173, 216, 230), (255, 165, 0), (127, 255, 212), (220, 20, 60), (124, 252, 0),
]


def label_values(label_map: Optional[np.ndarray]) -> List[int]:
    """Label values present in the map (background excluded), ascending."""
    if label_map is None:
        return []
    return [int(v) for v in np.unique(label_map) if v != 0]


def boundary_map(label_map: np.ndarray) -> np.ndarray:
    """Outline of every filled region, as saved by the main annotator.

    Same as the original save step: for each label (ascending value) take
    the Canny edges (100/200) of its binary mask and paint them with the
    label value.
    """
    out = np.zeros(label_map.shape, dtype=np.uint8)
    for value in label_values(label_map):
        mask = (label_map == value).astype(np.uint8) * 255
        edges = cv2.Canny(mask, 100, 200)
        out[edges != 0] = value
    return out


def outline_map(regions: np.ndarray, strokes: Optional[np.ndarray] = None) -> np.ndarray:
    """What is saved for a slice: the outline of every filled region (see
    :func:`boundary_map`) with the hand-drawn outlines on top, exactly as drawn."""
    out = boundary_map(regions)
    if strokes is not None:
        drawn = strokes > 0
        out[drawn] = strokes[drawn]
    return out


def label_contours(label_map: np.ndarray) -> Dict[int, List[np.ndarray]]:
    """{value: [contour (N, 2) array of x, y points, ...]} for vector display."""
    result: Dict[int, List[np.ndarray]] = {}
    for value in label_values(label_map):
        mask = (label_map == value).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        result[value] = [c.reshape(-1, 2) for c in contours if len(c) >= 2]
    return result


def fill_closed_boundaries(mask: np.ndarray) -> np.ndarray:
    """Fill the inside of closed outlines (used for volume measurement)."""
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(binary)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def fill_outlines(label_map: np.ndarray) -> np.ndarray:
    """Each label's closed outlines filled with the label value; the lines keep their own label."""
    out = np.zeros_like(label_map)
    for value in label_values(label_map):
        out[fill_closed_boundaries(label_map == value) > 0] = value
    lines = label_map > 0
    out[lines] = label_map[lines]
    return out


def to_rgb(gray: np.ndarray) -> np.ndarray:
    if gray.ndim == 3:
        return gray
    return np.repeat(gray[:, :, None], 3, axis=-1)


def adjust_brightness_contrast(gray: np.ndarray, contrast: float = 1.0, brightness: float = 0.0) -> np.ndarray:
    """Display-only adjustment: pixel * contrast + brightness, clipped to 0-255."""
    if contrast == 1.0 and brightness == 0.0:
        return gray
    out = gray.astype(np.float32) * contrast + brightness
    return np.clip(out, 0, 255).astype(np.uint8)


def thicken(label_map: np.ndarray, pixels: int) -> np.ndarray:
    """Grow thin label lines so they stay visible when the view is zoomed out."""
    if pixels <= 1 or not np.any(label_map):
        return label_map
    kernel = np.ones((pixels, pixels), np.uint8)
    return cv2.dilate(label_map, kernel)


def overlay_labels(base: np.ndarray, label_map: np.ndarray, lut: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Paint labelled pixels onto an image with their label colors."""
    out = to_rgb(base).copy()
    mask = label_map > 0
    if not np.any(mask):
        return out
    colors = lut[label_map[mask]].astype(np.float32)
    if alpha >= 1.0:
        out[mask] = colors.astype(np.uint8)
    else:
        out[mask] = ((1 - alpha) * out[mask].astype(np.float32) + alpha * colors).astype(np.uint8)
    return out


def rgba_overlay(label_map: np.ndarray, lut: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """(rows, cols, 4) RGBA image: label colors where labelled, transparent elsewhere."""
    rgba = np.zeros(label_map.shape + (4,), dtype=np.uint8)
    mask = label_map > 0
    rgba[mask, :3] = lut[label_map[mask]]
    rgba[mask, 3] = int(round(255 * alpha))
    return rgba


def region_fill(label_map: np.ndarray) -> np.ndarray:
    """RGB image with each closed region filled in a different color.

    Port of the original "Show Mask" view: contours of the mask, largest
    first, each filled with the next color of REGION_COLORS.
    """
    mask = (label_map > 0).astype(np.uint8) if label_map.dtype != np.uint8 else label_map
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(label_map.shape + (3,), dtype=np.uint8)
    for i, contour in enumerate(sorted(contours, key=cv2.contourArea, reverse=True)):
        cv2.drawContours(out, [contour], -1, REGION_COLORS[i % len(REGION_COLORS)], thickness=cv2.FILLED)
    return out


def blend_filled(base: np.ndarray, label_map: np.ndarray, lut: np.ndarray, alpha: float) -> np.ndarray:
    """Translucent label colors over the image (used by 'Plot Results')."""
    return overlay_labels(base, label_map, lut, alpha)
