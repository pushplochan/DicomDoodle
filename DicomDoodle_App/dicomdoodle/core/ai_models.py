"""
The AI models behind automatic annotation (all through Ultralytics):

  * YOLO   ``best_flair.pt``        finds a tumor bounding box
  * YOLOE  ``yoloe-11l-seg-pf.pt``  prompt-free fallback when YOLO finds
                                    nothing; the box with the brightest
                                    content is taken as the tumor
  * SAM 2.1 ``sam2.1_l.pt``         turns a box into a precise mask

Detection logic is the original DicomDoodle logic. What changed: models are
loaded once and cached (the original reloaded YOLO/YOLOE from disk for every
slice), and a lock lets background threads share them safely.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .. import resources

log = logging.getLogger(__name__)


class ModelUnavailable(RuntimeError):
    """A model's weights file could not be found or loaded."""


@dataclass
class Detection:
    """A bounding box in image pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int
    source: str          # "YOLO" or "YOLOE"

    @property
    def box(self):
        return self.x1, self.y1, self.x2, self.y2


def _to_rgb(image: np.ndarray) -> np.ndarray:
    """Gray uint8 -> 3-channel uint8 (what the original fed the models)."""
    return image if image.ndim == 3 else np.repeat(image[:, :, None], 3, axis=-1)


class ModelManager:
    """Loads models on first use and keeps them in memory."""

    def __init__(self, models_dir: str = "", device: str = "auto"):
        self._lock = threading.RLock()
        self._models = {}
        self.models_dir = models_dir
        self.device = device

    # -------------------------------------------------------- configuration
    def configure(self, models_dir: Optional[str] = None, device: Optional[str] = None) -> None:
        with self._lock:
            if models_dir is not None and models_dir != self.models_dir:
                self.models_dir = models_dir
                self._models.clear()
            if device is not None:
                self.device = device

    def weights_path(self, filename: str) -> str:
        path = resources.find_resource(filename, [self.models_dir])
        if path is None:
            raise ModelUnavailable(
                f"Model weights '{filename}' were not found.\n\n"
                "Place the file in the DicomDoodle folder or choose its folder in "
                "Preferences ▸ AI Models.")
        return path

    def _predict_kwargs(self) -> dict:
        kwargs = {"verbose": False}
        if self.device and self.device != "auto":
            kwargs["device"] = self.device
        return kwargs

    # --------------------------------------------------------------- status
    @property
    def sam_loaded(self) -> bool:
        return "sam" in self._models

    def device_name(self) -> str:
        if self.device == "cpu":
            return "CPU"
        try:
            import torch
            if torch.cuda.is_available():
                return f"GPU ({torch.cuda.get_device_name(0)})"
        except Exception:
            pass
        return "CPU"

    def available(self) -> dict:
        """Which weight files can be found (shown in Preferences)."""
        return {name: resources.find_resource(name, [self.models_dir])
                for name in (resources.SAM_WEIGHTS, resources.YOLO_WEIGHTS, resources.YOLOE_WEIGHTS)}

    # -------------------------------------------------------------- loading
    def _get(self, kind: str):
        with self._lock:
            if kind not in self._models:
                try:
                    from ultralytics import SAM, YOLO, YOLOE
                except Exception as exc:
                    raise ModelUnavailable(f"The 'ultralytics' package could not be imported: {exc}") from exc
                loaders = {
                    "sam": (SAM, resources.SAM_WEIGHTS),
                    "yolo": (YOLO, resources.YOLO_WEIGHTS),
                    "yoloe": (YOLOE, resources.YOLOE_WEIGHTS),
                }
                cls, filename = loaders[kind]
                path = self.weights_path(filename)
                log.info("Loading %s model from %s", kind.upper(), path)
                try:
                    self._models[kind] = cls(path)
                except Exception as exc:
                    raise ModelUnavailable(f"Could not load {filename}: {exc}") from exc
            return self._models[kind]

    def warm_up(self) -> str:
        """Load SAM ahead of time (called in the background at startup)."""
        self._get("sam")
        return self.device_name()

    def runs_on_cpu(self) -> bool:
        return self.device_name() == "CPU"

    def predict(self, kind: str, image: np.ndarray, **kwargs):
        """Raw Ultralytics results of model ``kind`` ("sam", "yolo" or "yoloe")
        on ``image``, on the configured device — for callers with their own
        detection logic (the MPR viewer's Detect button)."""
        rgb = _to_rgb(image)
        with self._lock:
            return self._get(kind).predict(rgb, **{**self._predict_kwargs(), **kwargs})

    # ------------------------------------------------------------ inference
    def detect_box(self, image: np.ndarray) -> Optional[Detection]:
        """YOLO first; if it finds nothing, YOLOE's brightest box (original logic)."""
        rgb = _to_rgb(image)
        with self._lock:
            results = self._get("yolo").predict(rgb, **self._predict_kwargs())
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                x1, y1, x2, y2 = map(int, boxes[0].xyxy[0])
                return Detection(x1, y1, x2, y2, "YOLO")

            results = self._get("yoloe").predict(rgb, **self._predict_kwargs())
            best, best_intensity = None, -1.0
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = rgb[y1:y2, x1:x2]
                if roi.size > 0:
                    intensity = float(np.mean(roi))
                    if intensity > best_intensity:
                        best_intensity = intensity
                        best = Detection(x1, y1, x2, y2, "YOLOE")
            return best

    def segment_box(self, image: np.ndarray, box: Sequence[float]) -> np.ndarray:
        """SAM mask (bool, image-sized) for a box given as (x1, y1, x2, y2)."""
        rgb = _to_rgb(image)
        with self._lock:
            results = self._get("sam").predict(rgb, bboxes=[float(v) for v in box], **self._predict_kwargs())
        masks = results[0].masks
        if masks is None or len(masks.data) == 0:
            return np.zeros(image.shape[:2], dtype=bool)
        mask = masks.data[0].cpu().numpy() > 0
        if mask.shape != image.shape[:2]:
            import cv2
            mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]),
                              interpolation=cv2.INTER_NEAREST) > 0
        return mask
