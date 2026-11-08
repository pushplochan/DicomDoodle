"""
What a model file is, read from its content — never from its name.

DicomDoodle uses three kinds of model (core/ai_models.py):

    "yolo"   the lesion detector: an Ultralytics YOLO detect / segment model
             (DicomDoodle's: YOLO26x-seg, one class "tumor")
    "yoloe"  the fallback detector, used when the lesion detector finds nothing:
             a prompt-free Ultralytics YOLOE model (thousands of everyday classes)
    "sam"    the segmenter: SAM / SAM 2, which turns a box or a click into a mask

inspect() recognizes them without running anything from the file:

  * ONNX files: Ultralytics exports (metadata "task", "names", "imgsz" ...) and
    DicomDoodle's SAM 2 exports, an image encoder plus a prompt decoder
    (metadata "dicomdoodle_model", or else their input and output names). Only
    the file's structure is read; the weights are skipped over.
  * PyTorch files (.pt / .pth): Ultralytics YOLO / YOLOE checkpoints and SAM /
    SAM 2 weights. The checkpoint's pickle is read with stand-ins for every
    class it names, so no code from the file runs; those class names also tell
    whether the installed Ultralytics can load it (YOLO26 needs Ultralytics 8.4).

The metadata DicomDoodle's model-card script (scripts/make_model_card.py) adds
to ONNX files: dicomdoodle_model (sam2_encoder / sam2_decoder), dicomdoodle_title,
dicomdoodle_family (links a SAM 2 encoder to its decoder), dicomdoodle_source
(the original file) and dicomdoodle_source_date (when the network was trained).
"""
from __future__ import annotations

import ast
import collections
import importlib
import os
import pickle
import re
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

YOLO, YOLOE, SAM = "yolo", "yoloe", "sam"                 # what the app asks for
SAM2_ENCODER, SAM2_DECODER = "sam2_encoder", "sam2_decoder"  # the two halves of an ONNX SAM 2
PURPOSE = {YOLO: "Lesion detector", YOLOE: "Fallback detector", SAM: "Segmentation"}
ENGINE_NAME = {"onnx": "ONNX Runtime", "torch": "PyTorch"}
MODEL_EXTENSIONS = (".onnx", ".pt", ".pth")

SAM2_ENCODER_OUTPUTS = {"image_embed", "high_res_feats_0", "high_res_feats_1"}
SAM2_DECODER_INPUTS = {"image_embed", "high_res_feats_0", "high_res_feats_1", "point_coords", "point_labels"}

_SAM2_SIZES = {12: ("t", "tiny"), 16: ("s", "small"), 24: ("b", "base+"), 48: ("l", "large")}
_SAM_SIZES = {12: ("b", "ViT-B"), 24: ("l", "ViT-L"), 32: ("h", "ViT-H")}


class NotAModel(ValueError):
    """The file is not a model DicomDoodle can use."""


@dataclass
class ModelFile:
    path: str
    kind: str                  # YOLO, YOLOE, SAM, SAM2_ENCODER or SAM2_DECODER
    engine: str                # "onnx" or "torch"
    title: str                 # e.g. "YOLO26x-seg · 1 class (tumor)"
    date: str = ""             # when the network was trained (ISO 8601) if known: newer models are preferred
    family: str = ""           # SAM 2 in ONNX: the encoder and the decoder of one model share it
    details: Dict[str, Any] = field(default_factory=dict)
    needs: Tuple[Tuple[str, str], ...] = ()    # PyTorch: the Ultralytics (module, class) pairs it refers to

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


def inspect(path: str) -> ModelFile:
    """What the model file at ``path`` is; raises NotAModel for anything else."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".onnx":
            return _inspect_onnx(path)
        if ext in (".pt", ".pth"):
            return _inspect_torch(path)
    except NotAModel:
        raise
    except Exception as exc:              # unreadable, truncated, not what its extension says ...
        raise NotAModel(f"{os.path.basename(path)}: {exc}") from exc
    raise NotAModel(f"{os.path.basename(path)}: not a model file")


def torch_loadable(model: ModelFile) -> Tuple[bool, str]:
    """Whether the installed Ultralytics knows every class a PyTorch checkpoint refers to."""
    for module_name, class_name in model.needs:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            return False, f"needs {module_name} (a newer Ultralytics)"
        if not hasattr(module, class_name):
            try:
                import ultralytics
                version = ultralytics.__version__
            except Exception:
                version = "?"
            return False, f"needs {class_name}, which Ultralytics {version} does not have (a newer version does)"
    return True, ""


# --------------------------------------------------------------------------- #
# ONNX: read the protocol buffer's structure, skip the weights
# --------------------------------------------------------------------------- #

def _varint(fh) -> int:
    result = shift = 0
    while True:
        byte = fh.read(1)
        if not byte:
            raise NotAModel("not an ONNX file (ends early)")
        result |= (byte[0] & 0x7F) << shift
        if byte[0] < 0x80:
            return result
        shift += 7
        if shift > 63:
            raise NotAModel("not an ONNX file")


def _fields(fh, end: int) -> Iterator[Tuple[int, int, Any]]:
    """(field number, wire type, value) of the protocol-buffer fields up to ``end``; the value of a
    length-delimited field is its (offset, length), so large fields are skipped, not read."""
    while fh.tell() < end:
        key = _varint(fh)
        number, wire = key >> 3, key & 7
        if wire == 0:
            yield number, wire, _varint(fh)
        elif wire == 1:
            fh.seek(8, 1)
        elif wire == 5:
            fh.seek(4, 1)
        elif wire == 2:
            length = _varint(fh)
            start = fh.tell()
            if start + length > end:
                raise NotAModel("not an ONNX file (a field runs past its end)")
            yield number, wire, (start, length)
            fh.seek(start + length)
        else:
            raise NotAModel("not an ONNX file")


def _strings(fh, span: Tuple[int, int]) -> Dict[int, str]:
    """The string fields of a small message, by field number (first occurrence)."""
    start, length = span
    fh.seek(start)
    found: Dict[int, str] = {}
    for number, wire, value in _fields(fh, start + length):
        if wire == 2 and number not in found and value[1] < 1 << 20:
            here = fh.tell()
            fh.seek(value[0])
            found[number] = fh.read(value[1]).decode("utf-8", "replace")
            fh.seek(here)
    return found


def onnx_header(path: str) -> Tuple[Dict[str, str], List[str], List[str]]:
    """An ONNX file's metadata, input names and output names, read without loading its weights."""
    meta: Dict[str, str] = {}
    inputs: List[str] = []
    outputs: List[str] = []
    graph = opset = False
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        for number, wire, value in _fields(fh, size):
            if number == 7 and wire == 2:                      # ModelProto.graph
                graph = True
                start, length = value
                for inner, inner_wire, span in _fields(fh, start + length):
                    if inner in (11, 12) and inner_wire == 2:   # GraphProto.input / .output
                        here = fh.tell()
                        name = _strings(fh, span).get(1, "")
                        fh.seek(here)
                        (inputs if inner == 11 else outputs).append(name)
            elif number == 8 and wire == 2:                    # ModelProto.opset_import
                opset = True
            elif number == 14 and wire == 2:                   # ModelProto.metadata_props
                here = fh.tell()
                entry = _strings(fh, value)
                fh.seek(here)
                meta[entry.get(1, "")] = entry.get(2, "")
    if not (graph and opset):
        raise NotAModel("not an ONNX file")
    return meta, inputs, outputs


def _literal(text: Optional[str], default=None):
    try:
        return ast.literal_eval(text) if text else default
    except (ValueError, SyntaxError):
        return default


def _ultralytics_title(name: str) -> str:
    """'yolo26x-seg' / 'YOLOe-11l-seg' -> 'YOLO26x-seg' / 'YOLOE-11l-seg'."""
    return re.sub(r"(?i)^yoloe?", lambda m: m.group(0).upper(), name)


def _class_text(names: Dict[int, str]) -> str:
    count = len(names)
    if count <= 3:
        listed = ", ".join(str(v) for v in names.values())
        return f"{count} class{'es' if count != 1 else ''} ({listed})"
    return f"{count:,} classes"


def _inspect_onnx(path: str) -> ModelFile:
    meta, inputs, outputs = onnx_header(path)
    role = meta.get("dicomdoodle_model", "")
    title = meta.get("dicomdoodle_title", "")
    date = meta.get("dicomdoodle_source_date", "")
    family = meta.get("dicomdoodle_family", "") or os.path.dirname(os.path.abspath(path))
    details = {"metadata": {k: v for k, v in meta.items() if len(v) < 400}, "inputs": inputs, "outputs": outputs}

    if role == SAM2_ENCODER or (not role and SAM2_ENCODER_OUTPUTS <= set(outputs)):
        return ModelFile(path, SAM2_ENCODER, "onnx", title or "SAM 2 image encoder", date, family, details)
    if role == SAM2_DECODER or (not role and SAM2_DECODER_INPUTS <= set(inputs)):
        return ModelFile(path, SAM2_DECODER, "onnx", title or "SAM 2 prompt decoder", date, family, details)

    task = meta.get("task", "")
    names = _literal(meta.get("names"), {})
    if task in ("detect", "segment") and isinstance(names, dict) and names and inputs:
        description = meta.get("description", "")
        match = re.search(r"Ultralytics (\S+) model", description)
        model_name = _ultralytics_title(match.group(1)) if match else "YOLO"
        kind = YOLOE if ("yoloe" in description.lower() or len(names) > 1000) else YOLO
        imgsz = _literal(meta.get("imgsz"), [640, 640])
        details.update(task=task, names=names, imgsz=list(imgsz) if isinstance(imgsz, (list, tuple)) else [imgsz] * 2,
                       stride=int(meta.get("stride", 32) or 32), end2end=meta.get("end2end", "") == "True",
                       data=meta.get("dicomdoodle_data", ""), metrics=_literal(meta.get("dicomdoodle_metrics"), {}))
        return ModelFile(path, kind, "onnx", title or f"{model_name} · {_class_text(names)}",
                         date or meta.get("date", ""), "", details)
    raise NotAModel(f"{os.path.basename(path)}: an ONNX file, but not a model DicomDoodle knows")


# --------------------------------------------------------------------------- #
# PyTorch: read the checkpoint's pickle with stand-ins, run nothing from it
# --------------------------------------------------------------------------- #

class _StandIn:
    """Takes the place of every class and function a checkpoint names."""

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        obj.args, obj.state = args, None
        return obj

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.state = state

    def __setitem__(self, key, value):         # objects pickled with their items (defaultdict ...)
        self.__dict__.setdefault("items", {})[key] = value

    def append(self, value):
        self.__dict__.setdefault("values", []).append(value)

    def extend(self, values):
        self.__dict__.setdefault("values", []).extend(values)


_SAFE = {("collections", "OrderedDict"): collections.OrderedDict, ("builtins", "set"): set,
         ("builtins", "frozenset"): frozenset, ("builtins", "slice"): slice, ("builtins", "complex"): complex,
         ("builtins", "bytearray"): bytearray, ("builtins", "dict"): dict, ("builtins", "list"): list,
         ("builtins", "tuple"): tuple}


class _Reader(pickle.Unpickler):
    def __init__(self, fh):
        super().__init__(fh)
        self.refs: Set[Tuple[str, str]] = set()
        self._types: Dict[Tuple[str, str], type] = {}

    def find_class(self, module, name):
        if (module, name) in _SAFE:
            return _SAFE[(module, name)]
        self.refs.add((module, name))
        key = (module, name)
        if key not in self._types:
            self._types[key] = type(name, (_StandIn,), {"__module__": module})
        return self._types[key]

    def persistent_load(self, pid):            # a tensor's storage: not read
        return None


def _read_checkpoint(path: str):
    if not zipfile.is_zipfile(path):
        raise NotAModel(f"{os.path.basename(path)}: not a PyTorch file of the current format")
    with zipfile.ZipFile(path) as archive:
        pickles = [n for n in archive.namelist() if n.endswith("/data.pkl") and n.count("/") == 1]
        if not pickles:
            raise NotAModel(f"{os.path.basename(path)}: not a PyTorch file")
        with archive.open(pickles[0]) as fh:
            reader = _Reader(fh)
            return reader.load(), reader.refs


def _state(obj) -> dict:
    return obj.state if isinstance(getattr(obj, "state", None), dict) else {}


def _inspect_torch(path: str) -> ModelFile:
    checkpoint, refs = _read_checkpoint(path)
    if not isinstance(checkpoint, dict):
        raise NotAModel(f"{os.path.basename(path)}: not a model checkpoint")
    model = checkpoint.get("ema") or checkpoint.get("model")
    if isinstance(model, _StandIn):
        return _ultralytics_checkpoint(path, checkpoint, model, refs)
    weights = checkpoint.get("model") if isinstance(checkpoint.get("model"), dict) else checkpoint
    return _sam_weights(path, weights)


def _ultralytics_checkpoint(path: str, checkpoint: dict, model: _StandIn, refs) -> ModelFile:
    model_class = type(model).__name__
    state = _state(model)
    names = state.get("names") if isinstance(state.get("names"), dict) else {}
    yaml = state.get("yaml") if isinstance(state.get("yaml"), dict) else {}
    train_args = checkpoint.get("train_args") if isinstance(checkpoint.get("train_args"), dict) else {}
    needs = tuple(sorted(r for r in refs if r[0].startswith("ultralytics.")))
    if model_class.startswith("YOLOE"):
        if not any(name == "LRPCHead" for _module, name in refs):
            raise NotAModel(f"{os.path.basename(path)}: a YOLOE model that needs prompts (DicomDoodle uses "
                            "the prompt-free “-pf” models)")
        kind = YOLOE
    elif model_class in ("DetectionModel", "SegmentationModel"):
        kind = YOLO
    else:
        raise NotAModel(f"{os.path.basename(path)}: an Ultralytics {model_class}, not a detector DicomDoodle uses")
    stem = os.path.splitext(os.path.basename(str(yaml.get("yaml_file") or model_class)))[0]
    if kind == YOLOE and not stem.endswith("-pf"):
        stem += "-pf"                                       # prompt-free (its yaml is the prompted model's)
    imgsz = train_args.get("imgsz", 640)
    details = {"names": names, "imgsz": [imgsz, imgsz] if isinstance(imgsz, int) else list(imgsz),
               "task": "segment" if "Seg" in model_class else "detect", "class": model_class,
               "ultralytics": str(checkpoint.get("version") or ""), "data": str(train_args.get("data") or "")}
    if isinstance(checkpoint.get("train_metrics"), dict):
        details["metrics"] = {k: v for k, v in checkpoint["train_metrics"].items() if isinstance(v, (int, float))}
    title = f"{_ultralytics_title(stem)} · {_class_text(names)}" if names else _ultralytics_title(stem)
    return ModelFile(path, kind, "torch", title, str(checkpoint.get("date") or ""), "", details, needs)


def _sam_weights(path: str, weights: dict) -> ModelFile:
    keys = [k for k in weights.keys() if isinstance(k, str)]

    def blocks(prefix: str) -> int:
        found = {int(k[len(prefix):].split(".")[0]) for k in keys
                 if k.startswith(prefix) and k[len(prefix):].split(".")[0].isdigit()}
        return max(found) + 1 if found else 0

    trunk = blocks("image_encoder.trunk.blocks.")
    if trunk in _SAM2_SIZES:
        size, size_name = _SAM2_SIZES[trunk]
        version = "2.1" if "no_obj_embed_spatial" in keys else "2"
        return ModelFile(path, SAM, "torch", f"SAM {version} {size_name}", "", "",
                         {"builder": f"build_sam2_{size}", "sam2": True})
    vit = blocks("image_encoder.blocks.")
    if vit in _SAM_SIZES and "image_encoder.pos_embed" in keys:
        size, size_name = _SAM_SIZES[vit]
        return ModelFile(path, SAM, "torch", f"SAM {size_name}", "", "",
                         {"builder": f"build_sam_vit_{size}", "sam2": False})
    if any(k.startswith("image_encoder.layers.") for k in keys) and any(k.startswith("mask_decoder.") for k in keys):
        return ModelFile(path, SAM, "torch", "MobileSAM", "", "", {"builder": "build_mobile_sam", "sam2": False})
    raise NotAModel(f"{os.path.basename(path)}: PyTorch weights, but not a model DicomDoodle knows")
