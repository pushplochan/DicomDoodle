"""
The AI models as PyTorch checkpoints (.pt / .pth), run by Ultralytics as
DicomDoodle always did. Used when a model card holds a model only in this
format, or when ONNX Runtime is not installed.

Ultralytics recognizes SAM checkpoints by their file name (sam2.1_l.pt ...);
here the builder comes from the file's content (model_files.py), so any name
works.
"""
from __future__ import annotations

import numpy as np

from . import devices
from .model_files import SAM, YOLOE, ModelFile


class TorchModel:
    engine = "torch"
    gpu_failed = False

    def __init__(self, model: ModelFile, device: str):
        self.device = devices.torch_device(device)          # "cuda" or "cpu"
        self.model = _load(model)

    @property
    def on_gpu(self) -> bool:
        return self.device == "cuda"

    def predict(self, image: np.ndarray, **kwargs):
        return self.model.predict(image, **{"verbose": False, "device": self.device, **kwargs})


def _load(model: ModelFile):
    if model.kind == SAM:
        return _sam(model)
    if model.kind == YOLOE:
        from ultralytics import YOLOE as UltralyticsYOLOE
        return UltralyticsYOLOE(model.path)
    from ultralytics import YOLO
    return YOLO(model.path)


def _sam(model: ModelFile):
    from ultralytics import SAM as UltralyticsSAM
    from ultralytics.models.sam import build

    builder = getattr(build, model.details["builder"])

    class NamedSAM(UltralyticsSAM):
        """Ultralytics' SAM with the builder chosen from the weights instead of the file name."""

        def _load(self, weights: str, task=None):
            self.model = builder(weights)

    sam = NamedSAM(model.path)
    sam.is_sam2 = bool(model.details.get("sam2"))           # picks SAM2Predictor (SAM.__init__ reads the name)
    return sam
