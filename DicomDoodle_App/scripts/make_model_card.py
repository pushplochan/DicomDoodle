#!/usr/bin/env python3
"""
Makes a DicomDoodle model card: one file with the AI models, which users
download separately from the installer (core/ai/model_card.py explains how the
app finds, installs and uses it).

Every model is converted to ONNX, which ONNX Runtime runs faster than PyTorch
on a processor and which reaches a graphics card on Windows (DirectML); each
conversion is checked against the original before it goes into the card.

    # one environment that can load every model:
    python scripts/make_model_card.py build best.pt yoloe-11l-seg-pf.pt sam2.1_l.pt \\
        --version 1.0 --check-dicom "path/to/a DICOM series" --out DicomDoodle-ModelCard-1.0.ddcard

    # or step by step, each model in an environment whose Ultralytics can load it
    # (YOLO26 models need Ultralytics 8.4 or newer):
    python scripts/make_model_card.py convert best.pt --out work/ --check-dicom SERIES
    python scripts/make_model_card.py convert yoloe-11l-seg-pf.pt sam2.1_l.pt --out work/ --check-dicom SERIES
    python scripts/make_model_card.py pack work/*.onnx --version 1.0 --out DicomDoodle-ModelCard-1.0.ddcard

Needs: torch, ultralytics, onnx and onnxruntime. What goes in the card:
  * YOLO / YOLOE (.pt, Ultralytics): exported by Ultralytics, input size dynamic
    (so the input is letterboxed exactly as for the PyTorch model)
  * SAM 2 / 2.1 (.pt): two files, the image encoder (image -> features) and the
    prompt decoder (features + box / points -> mask); the box-to-points step and
    the choice among SAM 2's masks are inside the decoder, as in Ultralytics
  * .onnx files given as input go in as they are

The app does not depend on file names: it reads what each file is from its
content (core/ai/model_files.py). The metadata added to the ONNX files
(dicomdoodle_*) records the original file and its training date.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import zipfile

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP_DIR)

import numpy as np  # noqa: E402

from dicomdoodle.core.ai.model_card import CARD_EXTENSION, CARD_FORMAT, MANIFEST  # noqa: E402
from dicomdoodle.core.ai.model_files import (PURPOSE, SAM, SAM2_DECODER, SAM2_ENCODER, YOLO, YOLOE,  # noqa: E402
                                             ModelFile, inspect)

OPSET = 17


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def legacy_onnx_export():
    """PyTorch 2.9+ exports with its new exporter by default; Ultralytics 8.3 and the SAM 2
    wrappers here are made for the TorchScript exporter."""
    import torch.onnx

    if getattr(torch.onnx.export, "_dicomdoodle", False):
        return
    original = torch.onnx.export

    def export(*args, **kwargs):
        kwargs.setdefault("dynamo", False)
        return original(*args, **kwargs)

    export._dicomdoodle = True
    torch.onnx.export = export


def stamp(path: str, values: dict) -> None:
    """Add DicomDoodle's metadata to an ONNX file."""
    import onnx

    model = onnx.load(path)
    existing = {p.key: p for p in model.metadata_props}
    for key, value in values.items():
        if value in (None, ""):
            continue
        if key in existing:
            existing[key].value = str(value)
        else:
            entry = model.metadata_props.add()
            entry.key, entry.value = key, str(value)
    onnx.save(model, path)


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #

def convert_yolo(source: ModelFile, out_dir: str) -> str:
    from ultralytics import YOLO as UltralyticsYOLO
    from ultralytics import YOLOE as UltralyticsYOLOE

    legacy_onnx_export()
    work = tempfile.mkdtemp(prefix="dicomdoodle_card_")
    try:
        copy = os.path.join(work, os.path.basename(source.path))
        shutil.copy2(source.path, copy)                     # Ultralytics writes next to the checkpoint
        model = (UltralyticsYOLOE if source.kind == YOLOE else UltralyticsYOLO)(copy)
        exported = model.export(format="onnx", dynamic=True, simplify=False, opset=OPSET, half=False, device="cpu")
        stem = os.path.splitext(os.path.basename(source.path))[0]
        target = os.path.join(out_dir, f"{stem}.onnx")
        shutil.move(str(exported), target)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    stamp(target, {"dicomdoodle_title": source.title, "dicomdoodle_source": os.path.basename(source.path),
                   "dicomdoodle_source_sha256": sha256(source.path), "dicomdoodle_source_date": source.date,
                   "dicomdoodle_data": os.path.basename(source.details.get("data", "")) if source.kind == YOLO else "",
                   "dicomdoodle_metrics": repr(source.details.get("metrics") or {})})
    return target


def convert_sam2(source: ModelFile, out_dir: str):
    """SAM 2 -> image encoder + prompt decoder, computing exactly what Ultralytics' SAM2Predictor
    computes for box and point prompts (multimask_output=False)."""
    import torch
    from ultralytics.models.sam import build

    legacy_onnx_export()
    model = getattr(build, source.details["builder"])(source.path).eval()
    size = int(model.image_size)
    model.set_imgsz((size, size))
    trunk = model.image_encoder.trunk
    with torch.no_grad():                                   # the position embedding as a constant
        position = trunk._get_pos_embed((size // 4, size // 4)).detach().clone()
    trunk._get_pos_embed = lambda _hw: position

    class Encoder(torch.nn.Module):
        """SAM2Predictor.get_im_features: image -> (image_embed, high_res_feats_0, high_res_feats_1)."""

        def __init__(self):
            super().__init__()
            self.m = model
            self.sizes = [[size // (4 * i)] * 2 for i in (1, 2, 4)]

        def forward(self, image):
            backbone = self.m.forward_image(image)
            _, feats, _, _ = self.m._prepare_backbone_features(backbone)
            if self.m.directly_add_no_mem_embed:
                feats[-1] = feats[-1] + self.m.no_mem_embed
            out = [f.permute(1, 2, 0).reshape(1, -1, *s) for f, s in zip(feats[::-1], self.sizes[::-1])][::-1]
            return out[2], out[0], out[1]

    class Decoder(torch.nn.Module):
        """SAM2Predictor.prompt_inference for one object: points in 1024 space (box corners have
        labels 2 and 3) -> low-resolution mask logits and the predicted IoU."""

        def __init__(self):
            super().__init__()
            self.m = model
            pe = model.sam_prompt_encoder
            h, w = pe.image_embedding_size
            self.register_buffer("dense_pe", pe.get_dense_pe().detach().clone())
            self.register_buffer("no_mask", pe.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(1, -1, h, w)
                                 .detach().clone())
            self.register_buffer("scale", torch.tensor([float(pe.input_image_size[1]), float(pe.input_image_size[0])]))

        def forward(self, image_embed, high_res_feats_0, high_res_feats_1, point_coords, point_labels):
            pe = self.m.sam_prompt_encoder
            # PromptEncoder._embed_points with pad=True, the boolean-mask assignments as torch.where
            coords = torch.cat([point_coords + 0.5, torch.zeros_like(point_coords[:, :1])], dim=1)
            labels = torch.cat([point_labels, -torch.ones_like(point_labels[:, :1])], dim=1).unsqueeze(-1)
            embedding = pe.pe_layer._pe_encoding(coords / self.scale)
            embedding = torch.where(labels == -1, pe.not_a_point_embed.weight.expand_as(embedding), embedding)
            for k in range(4):
                embedding = embedding + (labels == k).to(embedding.dtype) * pe.point_embeddings[k].weight
            masks, iou, _, _ = self.m.sam_mask_decoder(
                image_embeddings=image_embed, image_pe=self.dense_pe, sparse_prompt_embeddings=embedding,
                dense_prompt_embeddings=self.no_mask, multimask_output=False, repeat_image=False,
                high_res_features=[high_res_feats_0, high_res_feats_1])
            return masks, iou

    stem = os.path.splitext(os.path.basename(source.path))[0]
    encoder_path = os.path.join(out_dir, f"{stem}-encoder.onnx")
    decoder_path = os.path.join(out_dir, f"{stem}-decoder.onnx")
    encoder, decoder = Encoder().eval(), Decoder().eval()
    image = torch.randn(1, 3, size, size)
    with torch.no_grad():
        feats = encoder(image)
        points, point_labels = torch.tensor([[[100.0, 120.0], [400.0, 380.0]]]), torch.tensor([[2.0, 3.0]])
        torch.onnx.export(encoder, (image,), encoder_path, input_names=["image"],
                          output_names=["image_embed", "high_res_feats_0", "high_res_feats_1"], opset_version=OPSET,
                          do_constant_folding=True)
        torch.onnx.export(decoder, (*feats, points, point_labels), decoder_path,
                          input_names=["image_embed", "high_res_feats_0", "high_res_feats_1", "point_coords",
                                       "point_labels"],
                          output_names=["masks", "iou_predictions"], opset_version=OPSET, do_constant_folding=True,
                          dynamic_axes={"point_coords": {1: "points"}, "point_labels": {1: "points"}})
    family = f"{stem}:{sha256(source.path)[:16]}"
    common = {"dicomdoodle_family": family, "dicomdoodle_source": os.path.basename(source.path),
              "dicomdoodle_source_sha256": sha256(source.path), "dicomdoodle_title": source.title,
              "image_size": size, "mask_threshold": float(model.mask_threshold)}
    stamp(encoder_path, {**common, "dicomdoodle_model": SAM2_ENCODER})
    stamp(decoder_path, {**common, "dicomdoodle_model": SAM2_DECODER})
    return encoder_path, decoder_path


# --------------------------------------------------------------------------- #
# Checking a conversion against the original
# --------------------------------------------------------------------------- #

def check_images(dicom_folder: str):
    """Slices for the checks: the middle third of a DICOM series as DicomDoodle shows it, or a phantom."""
    if dicom_folder:
        from dicomdoodle.core.dicom_series import DicomSeries
        series = DicomSeries.load(dicom_folder)
        picks = np.linspace(len(series) // 3, 2 * len(series) // 3, 6).astype(int)
        return [np.repeat(series.image(int(i))[:, :, None], 3, -1) for i in picks]
    print("  (no --check-dicom: checking on a synthetic image only)")
    yy, xx = np.mgrid[:512, :512]
    image = np.where(((yy - 256) / 200) ** 2 + ((xx - 256) / 170) ** 2 < 1, 90, 0).astype(np.uint8)
    image[np.hypot(yy - 220, xx - 300) < 30] = 230
    return [np.repeat(image[:, :, None], 3, -1)]


def check_yolo(source: ModelFile, onnx_path: str, images) -> str:
    from ultralytics import YOLO as UltralyticsYOLO
    from ultralytics import YOLOE as UltralyticsYOLOE

    from dicomdoodle.core.ai.onnx_models import OnnxYolo

    reference = (UltralyticsYOLOE if source.kind == YOLOE else UltralyticsYOLO)(source.path)
    converted = OnnxYolo(inspect(onnx_path), "cpu")
    worst, compared = 0.0, 0
    for image in images:
        for kwargs in ({}, {"conf": 0.01, "iou": 0.7, "max_det": 300}):
            a = reference.predict(image, verbose=False, device="cpu", **kwargs)[0].boxes
            b = converted.predict(image, **kwargs)[0].boxes
            if len(a) != len(b):
                raise SystemExit(f"CHECK FAILED: {onnx_path}: {len(a)} boxes with PyTorch, {len(b)} with ONNX")
            if len(a):
                worst = max(worst, float((a.xyxy - b.xyxy).abs().max()))
            compared += len(a)
    if worst > 1.0:
        raise SystemExit(f"CHECK FAILED: {onnx_path}: boxes differ by up to {worst:.2f} px")
    return f"{compared} boxes on {len(images)} images, largest difference {worst:.4f} px"


def check_sam(source: ModelFile, encoder: str, decoder: str, images) -> str:
    from dicomdoodle.core.ai.onnx_models import OnnxSam2
    from dicomdoodle.core.ai.torch_models import TorchModel

    reference = TorchModel(source, "cpu")
    converted = OnnxSam2(inspect(encoder), inspect(decoder), "cpu")
    worst = 0
    for image in images:
        h, w = image.shape[:2]
        for prompt in ({"bboxes": [w * 0.3, h * 0.3, w * 0.6, h * 0.65]}, {"points": [[w * 0.45, h * 0.5]], "labels": [1]}):
            a = reference.predict(image, **prompt)[0].masks.data[0].cpu().numpy() > 0
            b = converted.predict(image, **prompt)[0].masks.data[0].cpu().numpy() > 0
            worst = max(worst, int((a != b).sum()))
    if worst > 0.001 * images[0].shape[0] * images[0].shape[1]:
        raise SystemExit(f"CHECK FAILED: {encoder}: masks differ in up to {worst} pixels")
    return f"{2 * len(images)} masks, at most {worst} pixel(s) different"


def convert(paths, out_dir: str, dicom_folder: str = "") -> list:
    os.makedirs(out_dir, exist_ok=True)
    images = None
    written = []
    for path in paths:
        source = inspect(path)
        print(f"{os.path.basename(path)}: {PURPOSE.get(source.kind, source.kind)}, {source.title}")
        if source.engine == "onnx":
            target = os.path.join(out_dir, os.path.basename(path))
            if os.path.abspath(target) != os.path.abspath(path):
                shutil.copy2(path, target)
            written.append(target)
            continue
        images = images if images is not None else check_images(dicom_folder)
        started = time.time()
        if source.kind in (YOLO, YOLOE):
            target = convert_yolo(source, out_dir)
            print(f"  -> {os.path.basename(target)} ({time.time() - started:.0f} s); check: "
                  f"{check_yolo(source, target, images)}")
            written.append(target)
        elif source.kind == SAM and source.details.get("sam2"):
            encoder, decoder = convert_sam2(source, out_dir)
            print(f"  -> {os.path.basename(encoder)} + {os.path.basename(decoder)} ({time.time() - started:.0f} s); "
                  f"check: {check_sam(source, encoder, decoder, images)}")
            written += [encoder, decoder]
        else:
            raise SystemExit(f"{path}: only SAM 2 is converted to ONNX; put the .pt in the card as it is")
    return written


# --------------------------------------------------------------------------- #
# The card
# --------------------------------------------------------------------------- #

def _names(model: ModelFile) -> str:
    names = model.details.get("names") or {}
    return ", ".join(str(v) for v in list(names.values())[:5]) + (" …" if len(names) > 5 else "")


def model_card_text(name: str, version: str, entries) -> str:
    """MODEL_CARD.md: what is in the card, for people (the app reads the files themselves)."""
    lines = [f"# {name} {version}", "",
             "The AI models of DicomDoodle, a free tool for annotating brain lesions on MRI. Put this file in your "
             "Downloads folder (or open it with *File ▸ Load Model Card…*) and start DicomDoodle: it installs the "
             "models once and uses them for Auto Detect, Detect in Box, Find tumour range and the Detect button of "
             "the Multiplanar & 3D Viewer. Without a model card DicomDoodle works for manual annotation only.", "",
             "**Not a medical device.** The models suggest outlines for research and teaching; a person checks every "
             "outline before it is saved.", "",
             "| Purpose | Model | File | Format |", "|---|---|---|---|"]
    for entry in entries:
        lines.append(f"| {entry['purpose']} | {entry['title']} | `{entry['file']}` | ONNX |")
    lines += ["", "All models run with ONNX Runtime, on the processor (default) or a graphics card "
              "(DirectML on Windows, CUDA on NVIDIA), and give the same results as the original PyTorch models "
              "(checked when the card was made).", ""]
    seen = set()
    for entry in entries:
        if entry["purpose"] in seen:
            continue
        seen.add(entry["purpose"])
        lines += [f"## {entry['purpose']}: {entry['model_title']}", ""]
        lines += entry["about"]
        scores = [f"{k.replace('metrics/', '')} {v:.3f}" for k, v in (entry.get("metrics") or {}).items()
                  if k.startswith("metrics/") and k.endswith("(B)")]
        if scores:
            lines.append("- Validation during training: " + ", ".join(scores))
        lines += [f"- Original file: `{entry['source']}` (SHA-256 {entry['source_sha256'][:16]}…)", ""]
    lines += ["## Licences", "",
              "- YOLO and YOLOE models: made with Ultralytics, AGPL-3.0 (https://ultralytics.com/license).",
              "- SAM 2.1: Meta AI, Apache License 2.0 (https://github.com/facebookresearch/sam2).",
              "- DicomDoodle: GPL-3.0.", ""]
    return "\n".join(lines)


def _about(model: ModelFile) -> list:
    if model.kind == YOLO:
        data = model.details.get("data") or ""
        return [f"Finds the lesion: a box around the tumour on a slice (classes: {_names(model)}). "
                "DicomDoodle takes the most confident box.",
                f"- Trained: {model.date[:10] or 'unknown date'}" + (f", data set {os.path.basename(str(data))}" if data else ""),
                f"- Input: {model.details.get('imgsz', ['?'])[0]} px (the slice is letterboxed to it)"]
    if model.kind == YOLOE:
        return ["Used only when the lesion detector finds nothing: a prompt-free open-vocabulary detector "
                f"({len(model.details.get('names') or {}):,} everyday classes). DicomDoodle takes the box whose "
                "content is brightest.",
                f"- Input: {model.details.get('imgsz', ['?'])[0]} px"]
    return ["Turns a box (or a click) into the precise outline of the lesion. Two files: the image encoder "
            "(once per slice) and the prompt decoder (once per box or click).",
            "- Input: the slice resized to 1024 px"]


def pack(paths, out: str, version: str, name: str = "DicomDoodle Model Card") -> None:
    files = []
    for pattern in paths:
        files += sorted(glob.glob(pattern)) or [pattern]
    models = [inspect(p) for p in files]
    entries = []
    for model in models:
        meta = model.details.get("metadata", {})
        purpose = PURPOSE[SAM] if model.kind in (SAM2_ENCODER, SAM2_DECODER) else PURPOSE[model.kind]
        part = {SAM2_ENCODER: " (image encoder)", SAM2_DECODER: " (prompt decoder)"}.get(model.kind, "")
        entries.append({"file": f"models/{model.name}", "path": model.path, "purpose": purpose,
                        "title": model.title + part, "model_title": model.title, "format": model.engine.upper(),
                        "bytes": os.path.getsize(model.path), "sha256": sha256(model.path),
                        "source": meta.get("dicomdoodle_source", model.name),
                        "source_sha256": meta.get("dicomdoodle_source_sha256", ""),
                        "about": _about(model), "metrics": model.details.get("metrics")})
    order = {PURPOSE[YOLO]: 0, PURPOSE[YOLOE]: 1, PURPOSE[SAM]: 2}
    entries.sort(key=lambda e: (order.get(e["purpose"], 9), "decoder)" in e["title"], e["file"]))
    manifest = {
        "format": CARD_FORMAT, "format_version": 1, "name": name, "version": version,
        "created": datetime.date.today().isoformat(), "app": "DicomDoodle 2.1 or newer",
        "description": "The AI models of DicomDoodle: lesion detector, fallback detector and segmentation (SAM), "
                       "in ONNX format.",
        "files": [{k: e[k] for k in ("file", "purpose", "title", "format", "bytes", "sha256", "source",
                                     "source_sha256")} for e in entries],
    }
    if not out.endswith(CARD_EXTENSION):
        out += CARD_EXTENSION
    partial = out + ".partial"
    with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as card:
        card.writestr(MANIFEST, json.dumps(manifest, indent=2))
        card.writestr("MODEL_CARD.md", model_card_text(name, version, entries))
        for entry in entries:
            print(f"  adding {entry['file']} ({entry['bytes'] / 1e6:,.0f} MB)")
            card.write(entry["path"], entry["file"])
    os.replace(partial, out)
    print(f"{out}: {os.path.getsize(out) / 1e6:,.0f} MB")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "convert"):
        p = sub.add_parser(command)
        p.add_argument("models", nargs="+")
        p.add_argument("--out", required=True, help="the card file (build) or a folder (convert)")
        p.add_argument("--check-dicom", default="", help="a DICOM series folder to check the conversions on")
        if command == "build":
            p.add_argument("--version", required=True)
            p.add_argument("--name", default="DicomDoodle Model Card")
    p = sub.add_parser("pack")
    p.add_argument("models", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--name", default="DicomDoodle Model Card")
    args = parser.parse_args(argv)

    os.environ.setdefault("YOLO_AUTOINSTALL", "false")      # never pip-install behind the user's back
    if args.command == "convert":
        convert(args.models, args.out, args.check_dicom)
    elif args.command == "pack":
        pack(args.models, args.out, args.version, args.name)
    else:
        work = tempfile.mkdtemp(prefix="dicomdoodle_card_")
        try:
            pack(convert(args.models, work, args.check_dicom), args.out, args.version, args.name)
        finally:
            shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
