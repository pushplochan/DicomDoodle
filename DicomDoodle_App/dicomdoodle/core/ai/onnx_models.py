"""
The AI models in ONNX format, run with ONNX Runtime: faster than PyTorch on a
processor, and the way to a graphics card in the Windows build (DirectML).

predict() takes the arguments of Ultralytics' predict() that DicomDoodle uses
and returns Ultralytics Results, made the way Ultralytics' own predictors make
them — the same letterbox, normalization, non-maximum suppression and mask
scaling (Ultralytics' functions are called) — so the code that reads the
results does not know which engine ran. On the guide's series the masks of the
ONNX SAM 2.1 equal the PyTorch ones pixel for pixel, and the detectors give the
same boxes (scripts/make_model_card.py checks every model it converts).

  OnnxYolo  a YOLO / YOLOE detector exported by Ultralytics (dynamic input size, so
            the input is letterboxed exactly as for the PyTorch model)
  OnnxSam2  SAM 2 as two files, an image encoder and a prompt decoder; prompts are
            boxes and / or points; the last image encodings are kept, so prompting the
            same slice again (Detect in Box after Auto Detect) skips the encoder.
"""
from __future__ import annotations

import ast
import hashlib
import logging
import os
from collections import OrderedDict
from typing import Optional

import numpy as np

from . import devices
from .model_files import ModelFile

log = logging.getLogger(__name__)

SAM_MEAN = np.array([123.675, 116.28, 103.53], np.float32)   # Ultralytics' SAM normalization (RGB)
SAM_STD = np.array([58.395, 57.12, 57.375], np.float32)
FEATURE_CACHE = 2                                              # image encodings kept


def _nms():
    try:
        from ultralytics.utils.nms import non_max_suppression      # Ultralytics 8.4
    except ImportError:
        from ultralytics.utils.ops import non_max_suppression      # Ultralytics 8.3
    return non_max_suppression


class OnnxModel:
    """An ONNX Runtime session on the CPU or a graphics card. When the graphics card fails — at
    the start or later (out of its memory, a driver reset) — the model moves to the CPU."""

    engine = "onnx"
    gpu_failed = False         # the graphics card was asked for but failed: the CPU runs it

    def __init__(self, path: str, device: str):
        self.path = path
        self.provider = devices.onnx_provider(device)
        try:
            self.session, self.provider = devices.create_session(path, self.provider)
        except Exception as exc:
            if not self.provider:
                raise
            self._to_cpu(exc)

    def _to_cpu(self, error: Exception) -> None:
        log.warning("The graphics card failed for %s (%s); it runs on the processor",
                    os.path.basename(self.path), str(error).splitlines()[0][:200] if str(error) else error)
        self.session, self.provider = devices.create_session(self.path, "")
        self.gpu_failed = True

    def run(self, feeds: dict) -> list:
        try:
            return self.session.run(None, feeds)
        except Exception as exc:
            if not self.provider:
                raise
            self._to_cpu(exc)
            return self.session.run(None, feeds)

    @property
    def on_gpu(self) -> bool:
        return bool(self.provider)


class OnnxYolo(OnnxModel):
    def __init__(self, model: ModelFile, device: str):
        super().__init__(model.path, device)
        meta = self.session.get_modelmeta().custom_metadata_map
        self.names = {int(k): str(v) for k, v in ast.literal_eval(meta["names"]).items()}
        imgsz = ast.literal_eval(meta.get("imgsz", "[640, 640]"))
        self.imgsz = list(imgsz) if isinstance(imgsz, (list, tuple)) else [int(imgsz)] * 2
        self.stride = int(meta.get("stride", 32))
        self.task = meta.get("task", "detect")
        self.end2end = meta.get("end2end", "") == "True"
        first = self.session.get_inputs()[0]
        self.input_name = first.name
        self.dynamic = isinstance(first.shape[2], str) or isinstance(first.shape[3], str)

    def predict(self, image: np.ndarray, conf: float = 0.25, iou: float = 0.7, max_det: int = 300,
                classes=None, agnostic_nms: bool = False, **_ignored):
        """Ultralytics' YOLO predict() on one HWC uint8 image (its defaults: conf 0.25, iou 0.7)."""
        import torch
        from ultralytics.data.augment import LetterBox
        from ultralytics.engine.results import Results
        from ultralytics.utils import ops

        # as BasePredictor.pre_transform / preprocess: letterbox (minimal padding when the
        # input size can vary, as for PyTorch), BGR -> RGB, HWC -> CHW, 0..1
        letterboxed = LetterBox(self.imgsz, auto=self.dynamic, stride=self.stride)(image=image)
        x = np.ascontiguousarray(letterboxed[..., ::-1].transpose(2, 0, 1)[None]).astype(np.float32) / 255.0
        outputs = self.run({self.input_name: x})
        # as DetectionPredictor.postprocess (a segment model's mask coefficients follow the classes)
        found = _nms()(torch.from_numpy(outputs[0]), conf, iou, classes, agnostic_nms, max_det=max_det,
                       nc=0 if self.task == "detect" else len(self.names), end2end=self.end2end)[0]
        masks = None
        if self.task == "segment" and len(found):
            # as SegmentationPredictor.construct_result: masks at the letterboxed size, and only
            # the detections whose mask is not empty are kept
            masks = ops.process_mask(torch.from_numpy(outputs[1])[0], found[:, 6:], found[:, :4], x.shape[2:],
                                     upsample=True)
            found[:, :4] = ops.scale_boxes(x.shape[2:], found[:, :4], image.shape)
            keep = masks.amax((-2, -1)) > 0
            found, masks = found[keep], masks[keep]
        else:
            found[:, :4] = ops.scale_boxes(x.shape[2:], found[:, :4], image.shape)
        return [Results(image, path="image0.jpg", names=self.names, boxes=found[:, :6], masks=masks)]


class OnnxSam2:
    """SAM 2 from its two ONNX halves (the encoder runs once per image, the decoder once per prompt)."""

    engine = "onnx"

    def __init__(self, encoder: ModelFile, decoder: ModelFile, device: str):
        self.encoder = OnnxModel(encoder.path, device)
        self.decoder = OnnxModel(decoder.path, device)
        meta = self.encoder.session.get_modelmeta().custom_metadata_map
        self.size = int(meta.get("image_size", 1024))
        self.threshold = float(meta.get("mask_threshold", 0.0))
        self._features: "OrderedDict[bytes, tuple]" = OrderedDict()

    @property
    def on_gpu(self) -> bool:
        return self.encoder.on_gpu

    @property
    def gpu_failed(self) -> bool:
        return self.encoder.gpu_failed or self.decoder.gpu_failed

    def _encode(self, image: np.ndarray):
        digest = hashlib.blake2b(str((image.shape, image.dtype.str)).encode(), digest_size=16)
        digest.update(np.ascontiguousarray(image).data)
        key = digest.digest()
        if key in self._features:
            self._features.move_to_end(key)
            return self._features[key]
        from ultralytics.data.augment import LetterBox

        # as Ultralytics' SAM Predictor: aspect-preserving resize to 1024, padded at the right and
        # bottom with grey 114, BGR -> RGB, normalized with SAM's mean and std
        letterboxed = LetterBox(self.size, auto=False, center=False)(image=image)
        x = (letterboxed[..., ::-1].astype(np.float32) - SAM_MEAN) / SAM_STD
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
        features = tuple(self.encoder.run({"image": x}))
        self._features[key] = features
        while len(self._features) > FEATURE_CACHE:
            self._features.popitem(last=False)
        return features

    def predict(self, image: np.ndarray, bboxes=None, points=None, labels=None, masks=None, **_ignored):
        """Ultralytics' SAM.predict() with box and / or point prompts (one mask per object)."""
        import torch
        from ultralytics.engine.results import Results
        from ultralytics.models.sam.amg import batched_mask_to_box
        from ultralytics.utils import ops

        if masks is not None:
            raise NotImplementedError("mask prompts are not supported by the ONNX SAM")
        coords, kinds = self._prompts(image.shape[:2], bboxes, points, labels)
        embed, feats_0, feats_1 = self._encode(image)
        logits, scores = [], []
        for c, k in zip(coords, kinds):
            low_res, iou = self.decoder.run({
                "image_embed": embed, "high_res_feats_0": feats_0, "high_res_feats_1": feats_1,
                "point_coords": c[None], "point_labels": k[None]})
            logits.append(low_res[0, 0])
            scores.append(float(iou[0, 0]))
        # as Predictor.postprocess: scale the low-resolution logits to the image, threshold
        pred = torch.from_numpy(np.stack(logits)).float()
        found = ops.scale_masks(pred[None], image.shape[:2], padding=False)[0] > self.threshold
        boxes = batched_mask_to_box(found).float()
        boxes = torch.cat([boxes, torch.tensor(scores)[:, None], torch.arange(len(scores)).float()[:, None]], dim=-1)
        names = {i: str(i) for i in range(len(scores))}
        return [Results(image, path="image0.jpg", names=names, masks=found, boxes=boxes)]

    def _prompts(self, shape, bboxes, points, labels):
        """Prompts as Ultralytics' SAM2Predictor._prepare_prompts makes them: per object, the box
        corners (labels 2 and 3) then the points (1 = foreground, 0 = background), in 1024 space."""
        r = min(self.size / shape[0], self.size / shape[1])
        objects_points: Optional[np.ndarray] = None
        objects_labels: Optional[np.ndarray] = None
        if points is not None:
            p = np.asarray(points, np.float32)
            p = p[None] if p.ndim == 1 else p
            lab = np.ones(p.shape[:-1], np.float32) if labels is None else np.asarray(labels, np.float32)
            if p.ndim == 2:                       # (N, 2): N objects with one point each
                p, lab = p[:, None, :], lab.reshape(-1)[:, None]
            objects_points, objects_labels = p * r, lab.reshape(p.shape[:2])
        if bboxes is not None:
            b = np.asarray(bboxes, np.float32)
            b = (b[None] if b.ndim == 1 else b).reshape(-1, 2, 2) * r
            box_labels = np.tile(np.array([[2, 3]], np.float32), (len(b), 1))
            if objects_points is not None:
                objects_points = np.concatenate([b, objects_points], axis=1)
                objects_labels = np.concatenate([box_labels, objects_labels], axis=1)
            else:
                objects_points, objects_labels = b, box_labels
        if objects_points is None:
            raise ValueError("SAM needs a box or a point")
        return objects_points.astype(np.float32), objects_labels.astype(np.float32)
