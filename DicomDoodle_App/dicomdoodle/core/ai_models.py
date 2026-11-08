"""
The AI models behind automatic annotation:

  * lesion detector    (YOLO26x-seg, one class "tumor")  finds a tumor bounding box
  * fallback detector  (YOLOE-11L-seg prompt-free)       used when the lesion detector finds
                                                          nothing; the box with the brightest
                                                          content is taken as the tumor
  * segmentation       (SAM 2.1 large)                   turns a box into a precise mask

They come from the model card (core/ai/model_card.py), a separate download:
without one the AI is off and DicomDoodle is for manual annotation only. Each
model runs in the fastest format the card holds for this computer (ONNX
Runtime, else PyTorch through Ultralytics) on the device chosen in the
preferences — the processor (CPU, the default) or a graphics card.

Detection logic is the original DicomDoodle logic. What changed: models are
loaded once and cached (the original reloaded YOLO/YOLOE from disk for every
slice), and a lock lets background threads share them safely. Every engine
returns Ultralytics Results, so callers with their own logic (the MPR viewer's
Detect button, via predict()) work with all of them.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .ai import devices, model_card
from .ai.model_files import ENGINE_NAME, PURPOSE, SAM, YOLO, YOLOE
from .progress import Progress, ensure_progress

log = logging.getLogger(__name__)

KINDS = (YOLO, YOLOE, SAM)
LOADING = "Loading the AI models"         # load_all()'s progress messages start so
NO_CARD = ("The AI needs the DicomDoodle model card, a separate download.\n\n"
           "Put the model card file (DicomDoodle-ModelCard-….ddcard) in your Downloads folder and restart "
           "DicomDoodle, or open it with File ▸ Load Model Card…. Without it, annotate by hand with the pen.")


class ModelUnavailable(RuntimeError):
    """A model is missing from the model card or could not be loaded."""


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


@dataclass
class ModelInfo:
    """What runs one purpose, for the preferences and the status bar."""

    kind: str
    purpose: str
    title: str = ""          # "" = the card has no such model
    engine: str = ""         # "ONNX Runtime" / "PyTorch"
    device: str = ""         # "CPU" / "GPU" once loaded
    loaded: bool = False
    problem: str = ""


def _to_rgb(image: np.ndarray) -> np.ndarray:
    """Gray uint8 -> 3-channel uint8 (what the original fed the models)."""
    return image if image.ndim == 3 else np.repeat(image[:, :, None], 3, axis=-1)


class ModelManager:
    """Finds the model card, loads models on first use (or ahead) and keeps them in memory."""

    def __init__(self, models_dir: str = "", device: str = devices.CPU):
        self._lock = threading.RLock()
        self._models: Dict[str, object] = {}      # kind -> loaded model (onnx_models / torch_models)
        self._chosen: Dict[str, model_card.Candidate] = {}
        self._available: Dict[str, bool] = {}
        self.models_dir = models_dir              # another folder searched for model cards
        self.device = devices.normalize(device)
        self.card: Optional[model_card.InstalledCard] = None
        self.prepared = False                     # prepare() has run: card is known
        devices.set_preferred(self.device)

    # -------------------------------------------------------- configuration
    def configure(self, models_dir: Optional[str] = None, device: Optional[str] = None) -> bool:
        """Change the extra card folder or the device; True when loaded models had to be dropped
        (a device change waits for a running model call)."""
        if models_dir is not None:
            self.models_dir = models_dir
        if device is None or devices.normalize(device) == self.device:
            return False
        with self._lock:
            self.device = devices.normalize(device)
            devices.set_preferred(self.device)
            dropped = bool(self._models)
            self.unload()
            return dropped

    def unload(self) -> None:
        with self._lock:
            self._models.clear()
            self._chosen.clear()

    # ------------------------------------------------------------ model card
    def prepare(self, progress: Optional[Progress] = None, chosen: str = "", search: bool = True) -> str:
        """Find (and install when needed) the model card — see model_card.prepare(); returns what
        happened: "installed", "updated", "ready" or "none". Raises CardError for a bad ``chosen``."""
        with self._lock:
            if self.device == devices.GPU:
                devices.detect_gpu()              # first: it decides the engines (ranked())
            else:
                devices.detect_gpu_soon()         # for the menus; the processor does not wait for it
            if chosen:
                self.unload()                     # Windows cannot replace files that are in use
            try:
                card, event = model_card.prepare(chosen, [self.models_dir], search, progress)
            finally:
                self.prepared = True
            if card is None or self.card is None or card.identity != self.card.identity:
                self.unload()
            self.card = card
            self._available = {kind: any(not problem for _c, problem in model_card.ranked(card, kind, self.device))
                               for kind in KINDS}
            return event

    def remove_card(self) -> None:
        """Delete the installed card: manual annotation only."""
        with self._lock:
            self.unload()
            model_card.remove_installed()
            self.card, self._available = None, {}

    def has(self, kind: str) -> bool:
        """The card has a model for ``kind`` that this computer can run."""
        return self._available.get(kind, False)

    def can(self, feature: str) -> bool:
        """What the AI can do with this card: "segment" (Detect in Box), "detect" (Auto Detect and
        the auto-detect batch), "range" (Find tumour range) and "tumour" (the MPR viewer's Detect)."""
        segment = self.has(SAM)
        return {"segment": segment,
                "detect": segment and (self.has(YOLO) or self.has(YOLOE)),
                "range": self.has(YOLO),
                "tumour": segment and self.has(YOLO)}.get(feature, False)

    @property
    def ready(self) -> bool:
        """Some AI can run (a usable card is installed)."""
        return self.can("segment") or self.can("range")

    # --------------------------------------------------------------- status
    @property
    def sam_loaded(self) -> bool:
        return SAM in self._models

    @property
    def loaded(self) -> bool:
        """At least one model is in memory."""
        return bool(self._models)

    @property
    def all_loaded(self) -> bool:
        kinds = [kind for kind in KINDS if self.has(kind)]
        return bool(kinds) and all(kind in self._models for kind in kinds)

    def gpu_name(self) -> str:
        info = devices.known_gpu()
        return info.name if info is not None and info.available else "graphics card"

    def device_name(self) -> str:
        """"CPU" or "GPU (<card>)": where the loaded models run (before loading: where they will)."""
        loaded = list(self._models.values())
        on_gpu = any(m.on_gpu for m in loaded) if loaded else (
            self.device == devices.GPU and (devices.known_gpu() or devices.GpuInfo()).available)
        return f"GPU ({self.gpu_name()})" if on_gpu else "CPU"

    def runs_on_cpu(self) -> bool:
        return self.device_name() == "CPU"

    def gpu_failed(self) -> bool:
        """The graphics card was chosen, but a model fell back to the CPU."""
        return self.device == devices.GPU and any(
            getattr(m, "gpu_failed", False) or not m.on_gpu for m in self._models.values())

    def info(self) -> List[ModelInfo]:
        """One entry per purpose: which model runs it, in which engine, where."""
        rows = []
        for kind in KINDS:
            row = ModelInfo(kind, PURPOSE[kind])
            ranked = model_card.ranked(self.card, kind, self.device)
            chosen = self._chosen.get(kind) or next((c for c, problem in ranked if not problem), None)
            if chosen is not None:
                row.title, row.engine = chosen.title, ENGINE_NAME[chosen.engine]
            elif ranked:
                row.title, row.problem = ranked[0][0].title, ranked[0][1]
            model = self._models.get(kind)
            if model is not None:
                row.loaded, row.device = True, ("GPU" if model.on_gpu else "CPU")
            rows.append(row)
        return rows

    # -------------------------------------------------------------- loading
    def _get(self, kind: str):
        with self._lock:
            if kind not in self._models:
                self._models[kind] = self._load(kind)
            return self._models[kind]

    def _load(self, kind: str):
        if not self.prepared:
            self.prepare()
        if self.card is None:
            raise ModelUnavailable(NO_CARD)
        problems = []
        for candidate, problem in model_card.ranked(self.card, kind, self.device):
            label = f"{candidate.title} ({ENGINE_NAME[candidate.engine]})"
            if problem:
                problems.append(f"{label}: {problem}")
                continue
            started = time.time()
            try:
                model = self._open(candidate)
            except Exception as exc:
                log.warning("Could not load %s: %s", label, exc)
                problems.append(f"{label}: {exc}")
                continue
            self._chosen[kind] = candidate
            log.info("%s: %s on the %s, loaded in %.1f s", PURPOSE[kind], label,
                     "GPU" if model.on_gpu else "CPU", time.time() - started)
            return model
        detail = ("\n\n" + "\n".join(problems)) if problems else ""
        raise ModelUnavailable(f"The model card “{self.card.label}” has no {PURPOSE[kind].lower()} "
                               f"that runs on this computer.{detail}")

    def _open(self, candidate: model_card.Candidate):
        if candidate.engine == "onnx":
            from .ai.onnx_models import OnnxSam2, OnnxYolo
            if candidate.kind == SAM:
                return OnnxSam2(candidate.files[0], candidate.files[1], self.device)
            return OnnxYolo(candidate.files[0], self.device)
        from .ai.torch_models import TorchModel
        return TorchModel(candidate.files[0], self.device)

    def load_all(self, progress: Optional[Progress] = None) -> str:
        """Load every model of the card ahead of use (at start, in the background) and run each once
        on a blank image — the first run of a model is slow (memory, the graphics card's programs), so
        the user's first Auto Detect is not. Returns where they run."""
        step = ensure_progress(progress)
        kinds = [kind for kind in KINDS if self.has(kind)]
        blank = np.zeros((512, 512, 3), np.uint8)
        for n, kind in enumerate(kinds):
            step(n, len(kinds), f"{LOADING}… {PURPOSE[kind].lower()} ({n + 1} of {len(kinds)})")
            with self._lock:
                model = self._get(kind)
                prompt = {"bboxes": [192.0, 192.0, 320.0, 320.0]} if kind == SAM else {}
                model.predict(blank, **prompt)
        step(len(kinds), len(kinds), "AI models loaded")
        return self.device_name()

    def warm_up(self) -> str:
        """Load the models ahead of time (kept for scripts; the app uses prepare() + load_all())."""
        if not self.prepared:
            self.prepare()
        return self.load_all()

    def predict(self, kind: str, image: np.ndarray, **kwargs):
        """Raw Ultralytics results of model ``kind`` ("sam", "yolo" or "yoloe")
        on ``image``, on the configured device — for callers with their own
        detection logic (the MPR viewer's Detect button)."""
        rgb = _to_rgb(image)
        with self._lock:
            return self._get(kind).predict(rgb, **kwargs)

    # ------------------------------------------------------------ inference
    def detect_box(self, image: np.ndarray) -> Optional[Detection]:
        """YOLO first; if it finds nothing, YOLOE's brightest box (original logic)."""
        rgb = _to_rgb(image)
        with self._lock:
            if self.has(YOLO) or not self.has(YOLOE):
                results = self._get(YOLO).predict(rgb)
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    x1, y1, x2, y2 = map(int, boxes[0].xyxy[0])
                    return Detection(x1, y1, x2, y2, "YOLO")
                if not self.has(YOLOE):
                    return None

            results = self._get(YOLOE).predict(rgb)
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
            results = self._get(SAM).predict(rgb, bboxes=[float(v) for v in box])
        masks = results[0].masks
        if masks is None or len(masks.data) == 0:
            return np.zeros(image.shape[:2], dtype=bool)
        mask = masks.data[0].cpu().numpy() > 0
        if mask.shape != image.shape[:2]:
            import cv2
            mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]),
                              interpolation=cv2.INTER_NEAREST) > 0
        return mask
